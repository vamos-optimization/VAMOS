"""Batch Comparison page — durable studies of several algorithms on several problems, run in the background."""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import panel as pn
import param

from vamos.ux.panel.components.background import BackgroundWork
from vamos.ux.panel.pages.experiment import problem_options
from vamos.ux.studio.algorithm_catalog import algorithm_label, studio_algorithms
from vamos.ux.studio.batch import HV_DESCRIPTION, csv_text, summary_latex
from vamos.ux.studio.batch_studies import (
    RunnerActions,
    batch_spec,
    check_plan,
    create_and_start,
    default_batch_folder,
    list_studies,
    runner_actions,
)
from vamos.ux.studio.data import RunRecord
from vamos.ux.studio.study_jobs import JobConflictError, JobOperation, JobStatus, job_dir_for, job_status, log_tail, start_job, stop_job
from vamos.ux.studio.study_progress import StudyProgress, format_duration, journal_marker, study_hypervolume, study_progress

POLL_MILLISECONDS = 3000


@dataclass(frozen=True)
class _Snapshot:
    """What one background read of a study found."""

    root: Path
    marker: int
    progress: StudyProgress
    job: JobStatus
    log: str
    succeeded: int
    hypervolume: tuple[list[dict[str, object]], list[dict[str, object]]] | None


class BatchState(param.Parameterized):
    """A batch form, the studies of one folder, and the monitor of the selected study."""

    # ---- New batch ----
    algorithms = param.ListSelector(default=["nsgaii"], objects=list(studio_algorithms()), doc="Algorithms to compare.")
    problems = param.ListSelector(default=["zdt1", "zdt2", "zdt3"], objects=list(problem_options().values()), doc="Registered problems.")
    n_runs = param.Integer(default=5, bounds=(1, 100), doc="Independent runs per algorithm and problem.")
    first_seed = param.Integer(default=0, bounds=(0, 2**31 - 1), doc="Seed of the first run; run i uses first seed + i.")
    default_population = param.Boolean(default=True, doc="Let every algorithm use its VAMOS default population size.")
    pop_size = param.Integer(default=100, bounds=(4, 100000))
    max_evaluations = param.Integer(default=25000, bounds=(100, 50_000_000))
    on_error = param.Selector(default="continue", objects=["continue", "fail_fast"], doc="What a failed run does to the batch.")
    batch_name = param.String(default="", doc="Folder name of the batch; empty for a timestamped name.")

    # ---- Studies ----
    study_dir = param.String(default="results", doc="Folder where batches are created and searched for.")
    studies = param.List(default=[], precedence=-1)
    selected = param.Selector(default=None, objects=[], allow_None=True, check_on_set=False, doc="Study to monitor.")
    confirm_resume = param.Boolean(default=False, doc="Confirm that no other process is running the study.")

    # ---- Outputs ----
    plan_text = param.String(default="", precedence=-1)
    plan_valid = param.Boolean(default=False, precedence=-1)
    status = param.String(default="Configure a batch, check its plan and create it.", precedence=-1)
    status_kind = param.Selector(default="info", objects=["info", "success", "warning", "danger"], precedence=-1)
    progress = param.Parameter(default=None, precedence=-1)
    job = param.Parameter(default=JobStatus("none"), precedence=-1)
    actions = param.Parameter(default=RunnerActions(), precedence=-1)
    log = param.String(default="", precedence=-1)
    summary = param.List(default=[], precedence=-1)
    runs = param.List(default=[], precedence=-1)

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._hv_key: tuple[str, int] | None = None
        self._marker: int | None = None
        self._runs: dict[str, RunRecord] = {}
        self._work: BackgroundWork[_Snapshot] = BackgroundWork()
        self._poller: Any = None

    # ---- the batch form ----

    def _report(self, message: str, kind: str = "info") -> None:
        self.status_kind, self.status = kind, message

    def output_folder(self) -> Path:
        base = Path(self.study_dir).expanduser()
        return base / self.batch_name.strip() if self.batch_name.strip() else default_batch_folder(base)

    def _spec(self) -> Any:
        return batch_spec(
            problems=self.problems,
            algorithms=self.algorithms,
            seeds=range(self.first_seed, self.first_seed + self.n_runs),
            max_evaluations=self.max_evaluations,
            pop_size=None if self.default_population else self.pop_size,
            on_error=self.on_error,
        )

    def check(self, event: Any = None) -> None:
        """Resolve the exact plan of the batch without writing anything."""
        if not self.problems or not self.algorithms:
            self.plan_valid, self.plan_text = False, "Choose at least one algorithm and one problem."
            return
        plan = check_plan(self._spec(), self.output_folder())
        self.plan_valid, self.plan_text = plan.valid, plan.text

    def launch(self, event: Any = None) -> None:
        """Create the batch study and run it in the background."""
        self.check()
        if not self.plan_valid:
            self._report("Fix the plan before creating the batch.", "warning")
            return
        output = self.output_folder()
        try:
            create_and_start(self._spec(), output)
        except Exception as exc:
            self._report(f"The batch could not be created: {exc}", "danger")
            return
        self.batch_name = ""
        self.refresh_studies(select=str(output.resolve()))
        self._report(f"Created {output.name} and started it in the background.", "success")

    # ---- the studies of the folder ----

    def refresh_studies(self, event: Any = None, *, select: str | None = None) -> None:
        listings = list_studies(Path(self.study_dir))
        self.studies = listings
        roots = [str(listing.root) for listing in listings]
        self.param.selected.objects = roots
        before = self.selected
        if select is not None:
            self.selected = select
        elif self.selected not in roots:
            candidates = [str(listing.root) for listing in listings if listing.job.active] or roots
            self.selected = candidates[0] if candidates else None
        if self.selected == before:  # A new selection has already refreshed the monitor.
            self.refresh(force=True)

    def study_options(self) -> dict[str, str]:
        return {listing.label(): str(listing.root) for listing in self.studies}

    # ---- the monitor ----

    @param.depends("selected", watch=True)
    def _on_select(self) -> None:
        self._hv_key, self._marker = None, None
        self.confirm_resume = False
        self.refresh(force=True)

    def refresh(self, *, force: bool = False) -> None:
        """Poll the selected study and its runner without blocking the page.

        The runner's files are cheap to read every time; the study itself is
        reloaded, in a background thread, only when its journal has grown.
        """
        if not self.selected:
            self.progress, self.job, self.actions, self.log, self.summary, self.runs = None, JobStatus("none"), RunnerActions(), "", [], []
            return
        root = Path(self.selected)
        previous = self.progress if isinstance(self.progress, StudyProgress) and self.progress.root == root else None
        if previous is not None and not force and journal_marker(root) == self._marker:
            job_dir = job_dir_for(root, previous.study_id)
            job = job_status(job_dir)
            if job != self.job:
                self.job, self.actions = job, runner_actions(previous, job)
            if job.active:
                self.log = log_tail(job_dir)
            return
        self._work.submit(lambda: self._read(root), self._apply)

    def _read(self, root: Path) -> _Snapshot:
        marker = journal_marker(root)  # Read first: a change during the load triggers another reload.
        progress = study_progress(root)
        job_dir = job_dir_for(root, progress.study_id)
        succeeded = int(progress.counts.get("succeeded", 0))
        hypervolume = None
        if self._hv_key != (str(root), succeeded):
            hypervolume = study_hypervolume(root, progress.failures, cache=self._runs) if succeeded else ([], [])
        return _Snapshot(root, marker, progress, job_status(job_dir), log_tail(job_dir), succeeded, hypervolume)

    def _apply(self, result: _Snapshot | Exception) -> None:
        if isinstance(result, Exception):
            self._report(f"Cannot read the study right now: {result}", "warning")
            return
        if str(result.root) != str(self.selected):
            return  # The selection changed while the study was being read.
        self._marker = result.marker
        self.progress, self.job, self.log = result.progress, result.job, result.log
        self.actions = runner_actions(result.progress, result.job)
        if result.hypervolume is not None:
            self._hv_key = (str(result.root), result.succeeded)
            self.summary, self.runs = result.hypervolume

    def _start(self, operation: JobOperation, *, retry_failed: bool = False) -> None:
        progress = self.progress
        if not isinstance(progress, StudyProgress):
            return
        if self.actions.needs_confirmation and not self.confirm_resume:
            self._report("Confirm that no other process is running this study before resuming it.", "warning")
            return
        try:
            start_job(progress.root, progress.study_id, operation, retry_failed=retry_failed)
        except (JobConflictError, OSError) as exc:
            self._report(str(exc), "warning")
            return
        self._report(f"Started the {operation} of {progress.root.name} in the background.", "success")
        self.refresh(force=True)

    def start(self, event: Any = None) -> None:
        self._start("run")

    def resume(self, event: Any = None) -> None:
        self._start("resume")

    def retry_failed(self, event: Any = None) -> None:
        self._start("retry")

    def stop(self, event: Any = None) -> None:
        progress = self.progress
        if not isinstance(progress, StudyProgress):
            return
        try:
            status = stop_job(job_dir_for(progress.root, progress.study_id))
        except (JobConflictError, OSError) as exc:
            self._report(str(exc), "warning")
            return
        self.job = status
        self.actions = runner_actions(progress, status)
        if status.phase == "stopped":
            self._report(f"Stopped {progress.root.name}. Resume it to continue; the interrupted run is repeated.", "info")
        elif status.phase == "stopping":
            self._report(f"Stop requested for {progress.root.name}; waiting for the worker to confirm.", "info")
        else:
            self._report(status.describe(), "warning")
        self.refresh(force=True)

    def start_polling(self) -> None:
        if self._poller is None and pn.state.curdoc is not None:
            self._poller = pn.state.add_periodic_callback(self.refresh, period=POLL_MILLISECONDS)

    def stop_polling(self) -> None:
        if self._poller is not None:
            self._poller.stop()
            self._poller = None
        self._work.shutdown()

    # ---- downloads ----

    def summary_csv(self) -> io.StringIO:
        return io.StringIO(csv_text(self.summary))

    def runs_csv(self) -> io.StringIO:
        return io.StringIO(csv_text(self.runs))

    def summary_tex(self) -> io.StringIO:
        return io.StringIO(summary_latex(self.summary))


def progress_markdown(progress: StudyProgress | None) -> str:
    if progress is None:
        return "*Select or create a batch to follow it here.*"
    counts = progress.counts
    lines = [
        f"### {progress.root.name}",
        f"**State:** {progress.state} · **{progress.finished} of {progress.tasks}** runs finished · "
        f"{counts.get('succeeded', 0)} succeeded · {counts.get('failed', 0)} failed · {counts.get('pending', 0)} pending",
        f"**Elapsed:** {format_duration(progress.elapsed_seconds)}"
        + (f" · **Time left (estimate):** ~{format_duration(progress.eta_seconds)}" if progress.eta_seconds is not None else ""),
    ]
    if progress.running:
        lines.append("**Running now:** " + "; ".join(row.label() for row in progress.running))
    lines += [f"**Issue:** {issue}" for issue in progress.issues]
    return "\n\n".join(lines)


def runner_markdown(job: JobStatus, actions: RunnerActions) -> str:
    text = job.describe()
    return f"{text}\n\n{actions.guidance}" if actions.guidance and actions.guidance != text else text


def _tabulator(rows: list[dict[str, object]], empty: str) -> Any:
    if not rows:
        return pn.pane.Markdown(empty)
    import pandas as pd

    return pn.widgets.Tabulator(pd.DataFrame(rows), sizing_mode="stretch_width", show_index=False, theme="fast", disabled=True)


def render_batch(
    study_dir: Path | None = None,
    open_results: Callable[[str], None] | None = None,
    open_compare: Callable[[str], None] | None = None,
    on_exit: Callable[[Callable[[], None]], None] | None = None,
) -> pn.Column:
    """Build the Batch Comparison page; it polls the selected study while it is shown."""
    state = BatchState(study_dir=str(study_dir or "results"))
    state.refresh_studies()
    state.start_polling()
    if on_exit is not None:
        on_exit(state.stop_polling)

    form = pn.Card(
        pn.widgets.CrossSelector.from_param(
            state.param.algorithms, name="Algorithms", options={algorithm_label(n): n for n in studio_algorithms()}
        ),
        pn.widgets.MultiChoice.from_param(state.param.problems, name="Problems", options=problem_options()),
        pn.widgets.IntInput.from_param(state.param.n_runs, name="Runs per algorithm and problem"),
        pn.widgets.IntInput.from_param(state.param.first_seed, name="First seed"),
        pn.widgets.Checkbox.from_param(state.param.default_population, name="Use each algorithm's default population size"),
        pn.widgets.IntInput.from_param(
            state.param.pop_size, name="Population size", disabled=pn.bind(lambda d: d, state.param.default_population)
        ),
        pn.widgets.IntInput.from_param(state.param.max_evaluations, name="Evaluations per run"),
        pn.widgets.Select.from_param(
            state.param.on_error,
            name="If a run fails",
            options={"Continue with the other runs": "continue", "Pause the batch": "fail_fast"},
        ),
        pn.widgets.TextInput.from_param(state.param.batch_name, name="Batch folder name (optional)"),
        title="New batch",
        sizing_mode="stretch_width",
    )
    check_btn = pn.widgets.Button(name="Check plan", button_type="light")
    check_btn.on_click(state.check)
    launch_btn = pn.widgets.Button(name="Create and run", button_type="primary")
    launch_btn.on_click(state.launch)
    plan = pn.pane.Markdown(pn.bind(lambda text: text, state.param.plan_text), sizing_mode="stretch_width")

    refresh_btn = pn.widgets.Button(name="Refresh", button_type="light")
    refresh_btn.on_click(lambda event: state.refresh_studies())
    studies = pn.Card(
        pn.widgets.TextInput.from_param(state.param.study_dir, name="Folder"),
        refresh_btn,
        pn.widgets.Select.from_param(
            state.param.selected, name="Study", options=pn.bind(lambda _: state.study_options(), state.param.studies)
        ),
        title="Batches and studies",
        sizing_mode="stretch_width",
    )
    status = pn.pane.Alert(pn.bind(lambda text: text, state.param.status), alert_type=pn.bind(lambda kind: kind, state.param.status_kind))

    def _button(label: str, kind: str, handler: Callable[[Any], None], enabled: Callable[[RunnerActions], bool]) -> Any:
        button = pn.widgets.Button(
            name=label,
            button_type=kind,
            disabled=pn.bind(lambda actions: not enabled(actions), state.param.actions),
            sizing_mode="fixed",
            width=180,
        )
        button.on_click(handler)
        return button

    def _open(event: Any) -> None:
        if open_results is not None and state.selected:
            open_results(str(state.selected))

    def _analyze(event: Any) -> None:
        if open_compare is not None and state.selected:
            open_compare(str(state.selected))

    controls = pn.FlexBox(
        _button("Stop", "danger", state.stop, lambda a: a.stop),
        _button("Start", "primary", state.start, lambda a: a.start),
        _button("Resume", "primary", state.resume, lambda a: a.resume),
        _button("Retry failed runs", "warning", state.retry_failed, lambda a: a.retry_failed),
        _button("Open in Explore Results", "success", _open, lambda a: a.open_results and open_results is not None),
        _button("Compare algorithms", "success", _analyze, lambda a: a.open_results and open_compare is not None),
    )
    confirm = pn.widgets.Checkbox.from_param(
        state.param.confirm_resume,
        name="No other process is running this study",
        visible=pn.bind(lambda actions: actions.needs_confirmation, state.param.actions),
    )
    progress_bar = pn.indicators.Progress(
        value=pn.bind(lambda p: p.finished if isinstance(p, StudyProgress) else 0, state.param.progress),
        max=pn.bind(lambda p: max(p.tasks, 1) if isinstance(p, StudyProgress) else 1, state.param.progress),
        bar_color="success",
        sizing_mode="stretch_width",
    )
    failures = pn.bind(
        lambda p: _tabulator(
            [{"Run": row.label(), "Failure": row.failure, "Retryable": row.retryable} for row in p.failures]
            if isinstance(p, StudyProgress)
            else [],
            "",
        ),
        state.param.progress,
    )
    has_results = pn.bind(lambda rows: not rows, state.param.summary)
    downloads = pn.Row(
        pn.widgets.FileDownload(callback=state.summary_csv, filename="vamos_batch_summary.csv", label="Summary CSV", disabled=has_results),
        pn.widgets.FileDownload(callback=state.runs_csv, filename="vamos_batch_runs.csv", label="Runs CSV", disabled=has_results),
        pn.widgets.FileDownload(callback=state.summary_tex, filename="vamos_batch_summary.tex", label="LaTeX table", disabled=has_results),
    )

    left = pn.Column(
        "## Batch Comparison",
        pn.pane.Markdown(
            "*A batch is a durable VAMOS study: it runs in a background process, survives closing the browser, and "
            "can be stopped, resumed, retried, or inspected with `vamos study`.*"
        ),
        form,
        pn.Row(check_btn, launch_btn),
        plan,
        status,
        studies,
        max_width=520,
        sizing_mode="stretch_width",
    )
    right = pn.Column(
        "## Monitor",
        pn.pane.Markdown(pn.bind(progress_markdown, state.param.progress), sizing_mode="stretch_width"),
        progress_bar,
        pn.pane.Markdown(pn.bind(runner_markdown, state.param.job, state.param.actions), sizing_mode="stretch_width"),
        confirm,
        controls,
        failures,
        "## Hypervolume",
        pn.pane.Markdown(HV_DESCRIPTION),
        pn.bind(lambda rows: _tabulator(rows, "*No completed runs yet.*"), state.param.summary),
        downloads,
        pn.Card(
            pn.pane.Str(pn.bind(lambda text: text or "(empty)", state.param.log)),
            title="Runner log",
            collapsed=True,
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_both",
    )
    return pn.Column(pn.Row(left, right, sizing_mode="stretch_both"), sizing_mode="stretch_both")
