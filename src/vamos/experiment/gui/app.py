"""NiceGUI pages of the experimental VAMOS GUI.

This module is imported only when ``vamos gui`` starts the server, after the CLI
has checked that the optional ``gui`` extra (NiceGUI and Plotly) is installed.
Pages are registered by :func:`build_app`, never at import time. All domain work
is delegated to the UI-independent service modules of this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from vamos.ux.visualization.interactive import FrontView, convergence_figure, front_figure, objective_labels, placeholder_figure

from .catalog import RunSummary, discover_runs, load_front
from .jobs import ENGINES, JobHandle, RunRequest, launch_job
from .monitor import ProgressMonitor, ProgressState
from .tables import run_table, solution_table, solutions_csv

POLL_SECONDS = 0.3
PLOT_STYLE = "height: 440px"
LIVE_UIREVISION = "vamos-live-front"
# Emit only the row indices stored in ``customdata``; the default Plotly event payload
# carries the full trace data for every selected point.
SELECTION_JS = "(event) => emit(event && event.points ? event.points.map((point) => point.customdata) : [])"


@dataclass(frozen=True)
class GuiSettings:
    """Directories used by one GUI server."""

    results_root: Path
    jobs_root: Path


def _ui() -> Any:
    from nicegui import ui

    return ui


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or value is None or value == "":
        raise ValueError(f"{label} is required.")
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"{label} must be an integer.")
    return int(number)


def _header() -> None:
    ui = _ui()
    with ui.header().classes("items-center gap-6"):
        ui.label("VAMOS").classes("text-xl font-bold")
        ui.link("Explorer", "/").classes("text-white")
        ui.link("New run", "/run").classes("text-white")
        ui.space()
        ui.badge("experimental", color="orange")


class FrontPanel:
    """Interactive objective-space view with projections, selection, table, and CSV export."""

    def __init__(self, F: npt.NDArray[np.float64], X: npt.NDArray[Any] | None, *, title: str) -> None:
        ui = _ui()
        self.F = F
        self.X = X
        self.title = title
        self.selected: list[int] = []
        n_obj = int(F.shape[1])
        axis_options = dict(enumerate(objective_labels(n_obj)))
        views = {"2d": "2-D projection"}
        if n_obj >= 3:
            views.update({"3d": "3-D", "parallel": "Parallel coordinates"})
        with ui.row().classes("items-end gap-4"):
            self.view = ui.select(views, value="2d", label="View", on_change=self.refresh).classes("w-48")
            self.axes = [
                ui.select(axis_options, value=k, label=f"Axis {k + 1}", on_change=self.refresh).classes("w-28")
                for k in range(min(3, n_obj))
            ]
        self.plot = ui.plotly(self._figure()).classes("w-full").style(PLOT_STYLE).mark("front-plot")
        self.plot.on("plotly_selected", self._on_points, js_handler=SELECTION_JS)
        self.plot.on("plotly_click", self._on_points, js_handler=SELECTION_JS)
        with ui.row().classes("items-center gap-4"):
            self.caption = ui.label(self._caption())
            ui.button("Download CSV", icon="download", on_click=self._download).props("flat")
        columns, rows = solution_table(F, X, [])
        self.table = ui.table(columns=columns, rows=rows, row_key="id", pagination=10).classes("w-full").mark("solutions-table")

    def _figure(self) -> Any:
        raw = str(self.view.value)
        view: FrontView = "3d" if raw == "3d" else "parallel" if raw == "parallel" else "2d"
        count = 3 if view == "3d" else 2
        axes = None if view == "parallel" else [int(select.value) for select in self.axes[:count]]
        return front_figure(self.F, view=view, axes=axes, title=self.title)

    def refresh(self) -> None:
        try:
            figure = self._figure()
        except ValueError as exc:
            _ui().notify(str(exc), type="warning")
            return
        self.plot.update_figure(figure)

    def _on_points(self, event: Any) -> None:
        values = event.args if isinstance(event.args, list) else []
        picked = {int(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)}
        self.selected = sorted(index for index in picked if 0 <= index < self.F.shape[0])
        _, rows = solution_table(self.F, self.X, self.selected)
        self.table.update_rows(rows)
        self.caption.set_text(self._caption())

    def _caption(self) -> str:
        total = int(self.F.shape[0])
        if not self.selected:
            return f"{total} solutions. Lasso or box-select (2-D) or click points to inspect them."
        return f"{len(self.selected)} of {total} solutions selected."

    def _download(self) -> None:
        content = solutions_csv(self.F, self.X, self.selected or None).encode("utf-8")
        _ui().download.content(content, "vamos-solutions.csv", "text/csv")


async def _verify(path: Path) -> None:
    from nicegui import run

    from vamos.experiment.artifacts import verify_run

    ui = _ui()
    try:
        report = await run.io_bound(verify_run, path)
    except Exception as exc:  # noqa: BLE001 - every verification failure is shown to the user
        ui.notify(f"Verification failed: {type(exc).__name__}: {exc}", type="negative", multi_line=True)
        return
    ui.notify(
        f"Integrity {report.artifact_integrity}; paths {report.path_safety}; replayability {report.effective_replayability}.",
        type="positive",
        multi_line=True,
    )


def _show_run(container: Any, summary: RunSummary) -> None:
    ui = _ui()
    with container:
        ui.label(f"{summary.algorithm} on {summary.problem}: {summary.relative_path}").classes("text-lg font-medium")
        if not summary.loadable:
            ui.label(summary.error or f"Run status is {summary.status!r}; there is no result to display.").classes("text-negative")
            return
        try:
            F, X = load_front(summary.path)
        except Exception as exc:  # noqa: BLE001 - shown to the user instead of breaking the page
            ui.label(f"Cannot load the result: {type(exc).__name__}: {exc}").classes("text-negative")
            return
        FrontPanel(F, X, title=f"{summary.algorithm} on {summary.problem}")
        ui.button("Verify integrity", icon="verified", on_click=lambda: _verify(summary.path)).props("outline")


def _explorer_page(settings: GuiSettings) -> None:
    ui = _ui()
    _header()
    summaries: list[RunSummary] = []
    with ui.column().classes("w-full p-4 gap-3"):
        with ui.row().classes("items-center gap-3"):
            ui.label("Results root:").classes("font-medium")
            ui.label(str(settings.results_root)).classes("font-mono text-sm")
            refresh_button = ui.button("Refresh", icon="refresh").props("flat")
        columns, rows = run_table([])
        table = ui.table(columns=columns, rows=rows, row_key="id", selection="single", pagination=15).classes("w-full").mark("runs-table")
        detail = ui.column().classes("w-full gap-3")

    def refresh() -> None:
        summaries[:] = discover_runs(settings.results_root)
        table.update_rows(run_table(summaries)[1])
        detail.clear()
        if not summaries:
            ui.notify(f"No canonical runs found under {settings.results_root}.", type="info")

    def show(event: Any) -> None:
        detail.clear()
        selection = getattr(event, "selection", None) or []
        index = selection[0].get("id") if selection else None
        if isinstance(index, int) and 0 <= index < len(summaries):
            _show_run(detail, summaries[index])

    refresh_button.on_click(refresh)
    table.on_select(show)
    refresh()


def _status_text(state: ProgressState) -> str:
    if state.status in ("failed", "cancelled"):
        reason = f": {state.error}" if state.error else ""
        return f"{state.status.capitalize()}{reason}"
    budget = state.max_evaluations
    evaluations = "?" if state.evaluations is None else f"{state.evaluations:,}"
    total = f" / {budget:,}" if budget is not None else ""
    return f"{state.status.capitalize()}: {evaluations}{total} evaluations, {state.front_size} non-dominated, {state.elapsed:.1f} s"


def _live_figure(state: ProgressState) -> Any:
    assert state.front is not None
    n_obj = state.n_obj or int(state.front.shape[1])
    view: FrontView = "2d" if n_obj == 2 else "3d" if n_obj == 3 else "parallel"
    return front_figure(state.front, view=view, title=f"Current front ({state.front_size} non-dominated)", uirevision=LIVE_UIREVISION)


def _finish(container: Any, job: JobHandle, state: ProgressState) -> None:
    ui = _ui()
    container.clear()
    with container:
        if state.status == "succeeded" and state.run_dir:
            run_dir = job.job_dir / state.run_dir
            try:
                F, X = load_front(run_dir)
            except Exception as exc:  # noqa: BLE001 - shown to the user
                ui.label(f"Cannot load the stored run: {type(exc).__name__}: {exc}").classes("text-negative")
                return
            ui.label(f"Stored canonical run: {run_dir}").classes("font-mono text-sm")
            FrontPanel(F, X, title="Final front (stored run)")
        elif state.status == "cancelled":
            ui.label("The run was cancelled; nothing was stored.")
        else:
            ui.label(state.error or "The run failed.").classes("text-negative")
            ui.label(f"Worker log: {job.log_path}").classes("font-mono text-sm")


def _run_page(settings: GuiSettings) -> None:
    from vamos.engine.algorithm.registry import get_algorithms_registry
    from vamos.foundation.problem.registry import available_problem_names

    ui = _ui()
    _header()
    problems = sorted(available_problem_names())
    session: dict[str, Any] = {"job": None, "monitor": None}
    with ui.row().classes("w-full p-4 gap-6 items-start no-wrap"):
        with ui.card().classes("w-80 shrink-0"):
            ui.label("Run configuration").classes("text-lg font-medium")
            problem = (
                ui.select(problems, value="zdt1" if "zdt1" in problems else problems[0], label="Problem", with_input=True)
                .classes("w-full")
                .mark("problem")
            )
            algorithm = (
                ui.select(["auto", *sorted(get_algorithms_registry())], value="nsgaii", label="Algorithm")
                .classes("w-full")
                .mark("algorithm")
            )
            budget = ui.number("Max evaluations", value=20000, min=1, step=1000, precision=0).classes("w-full").mark("max-evaluations")
            pop_size = (
                ui.number("Population size (blank: default)", value=None, min=2, step=1, precision=0).classes("w-full").mark("pop-size")
            )
            seed = ui.number("Seed", value=1, min=0, step=1, precision=0).classes("w-full").mark("seed")
            engine = ui.select(list(ENGINES), value="numpy", label="Engine").classes("w-full").mark("engine")
            with ui.row():
                launch = ui.button("Launch", icon="play_arrow").mark("launch")
                cancel = ui.button("Cancel", icon="stop", color="negative").mark("cancel")
            cancel.disable()
        with ui.column().classes("grow gap-3"):
            status = ui.label("Configure a run and press Launch.").mark("run-status")
            progress = ui.linear_progress(value=0.0, show_value=False)
            with ui.row().classes("w-full no-wrap gap-4"):
                live = (
                    ui.plotly(placeholder_figure("The current front appears here.")).classes("w-1/2").style(PLOT_STYLE).mark("live-front")
                )
                curve = (
                    ui.plotly(placeholder_figure("The progress hypervolume appears here."))
                    .classes("w-1/2")
                    .style(PLOT_STYLE)
                    .mark("progress-curve")
                )
            final = ui.column().classes("w-full gap-3")

    def start() -> None:
        try:
            request = RunRequest(
                problem=str(problem.value),
                algorithm=str(algorithm.value),
                max_evaluations=_as_int(budget.value, "Max evaluations"),
                pop_size=None if pop_size.value in (None, "") else _as_int(pop_size.value, "Population size"),
                seed=_as_int(seed.value, "Seed"),
                engine=str(engine.value),
            )
            job = launch_job(request, settings.jobs_root)
        except (ValueError, OSError) as exc:
            ui.notify(str(exc), type="negative", multi_line=True)
            return
        session.update(job=job, monitor=ProgressMonitor(job.events_path))
        final.clear()
        live.update_figure(placeholder_figure("Waiting for the first generation."))
        curve.update_figure(placeholder_figure("Waiting for progress.", uirevision="vamos-convergence"))
        progress.set_value(0.0)
        status.set_text(f"Started {job.job_dir.name}")
        launch.disable()
        cancel.enable()
        timer.activate()

    def tick() -> None:
        job, monitor = session.get("job"), session.get("monitor")
        if not isinstance(job, JobHandle) or not isinstance(monitor, ProgressMonitor):
            timer.deactivate()
            return
        changed = monitor.poll()
        code = job.returncode()
        if code is not None and not monitor.state.finished:
            changed = monitor.poll() or changed
            monitor.mark_exited(code, log_path=job.log_path)
            changed = True
        state = monitor.state
        if changed:
            status.set_text(_status_text(state))
            if state.fraction is not None:
                progress.set_value(state.fraction)
            if state.front is not None:
                live.update_figure(_live_figure(state))
            if state.convergence_x:
                curve.update_figure(convergence_figure(state.convergence_x, state.convergence_y, label="Progress hypervolume"))
        if state.finished:
            timer.deactivate()
            launch.enable()
            cancel.disable()
            _finish(final, job, state)

    def stop() -> None:
        job = session.get("job")
        if isinstance(job, JobHandle) and job.cancel():
            ui.notify("Run cancelled.", type="warning")
        tick()

    timer = ui.timer(POLL_SECONDS, tick, active=False)
    launch.on_click(start)
    cancel.on_click(stop)


def build_app(settings: GuiSettings) -> None:
    """Register the GUI pages on the NiceGUI application."""
    ui = _ui()

    def explorer() -> None:
        _explorer_page(settings)

    def new_run() -> None:
        _run_page(settings)

    ui.page("/", title="VAMOS GUI")(explorer)
    ui.page("/run", title="VAMOS GUI: new run")(new_run)


def run_gui(settings: GuiSettings, *, host: str, port: int, show: bool, native: bool = False) -> None:
    """Register the pages and serve them until interrupted."""
    build_app(settings)
    _ui().run(host=host, port=port, title="VAMOS GUI", show=show, native=native, reload=False, show_welcome_message=True)


__all__ = ["FrontPanel", "GuiSettings", "build_app", "run_gui"]
