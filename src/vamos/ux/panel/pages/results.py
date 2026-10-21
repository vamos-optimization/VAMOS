"""Results Explorer page — load saved studies, filter and compare fronts, rank solutions, export them and run a run again."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import panel as pn
import param

from vamos.ux.panel.components.plots import overlay_figure
from vamos.ux.studio.algorithm_catalog import algorithm_label
from vamos.ux.studio.batch import csv_text
from vamos.ux.studio.experiment_setup import ExperimentSetup, setup_from_run
from vamos.ux.studio.front_filters import filter_fronts, have_constraints, objective_bounds, solution_count
from vamos.ux.studio.ranking import MCDM_METHODS, front_summary_rows, fronts_for, problems_in, rank_fronts


def study_state_text(root: Path) -> str:
    """Describe a study's state from its data-only inspection, and what to do next."""
    from vamos.study_artifacts import load_study

    try:
        report = load_study(root).inspect().as_dict()
    except Exception:
        return ""
    counts_value = report.get("counts")
    counts: dict[str, Any] = counts_value if isinstance(counts_value, dict) else {}
    text = (
        f"The study is {report.get('state', 'unknown')}: {counts.get('succeeded', 0)} of {counts.get('tasks', 0)} tasks "
        f"succeeded, {counts.get('failed', 0)} failed, {counts.get('pending', 0)} pending."
    )
    if counts.get("failed") or counts.get("pending"):
        text += " Inspect it with `vamos study inspect` and continue it with `vamos study resume`."
    return text


class ResultsState(param.Parameterized):
    """Reactive state for the Results Explorer."""

    study_dir = param.String(default="results", doc="A study directory, or a folder to search for studies.")
    selected_study = param.Selector(default=None, objects=[], allow_None=True, doc="Study to load.")
    problem = param.Selector(default=None, objects=[], allow_None=True, doc="Problem whose fronts are shown.")
    mcdm_method = param.Selector(default="weighted_sum", objects=list(MCDM_METHODS), doc="MCDM ranking method.")
    top_k = param.Integer(default=10, bounds=(1, 1000), doc="Solutions shown per front.")
    feasible_only = param.Boolean(default=False, doc="Keep only the solutions that satisfy every constraint.")
    objective_ranges = param.List(default=[], doc="(low, high) per objective; None leaves that side open.")
    rerun = param.Selector(default=None, objects=[], allow_None=True, doc="Run of the problem to run again.")

    # ---- outputs ----
    status = param.String(default="Enter a study directory and click Find studies.", precedence=-1)
    status_kind = param.Selector(default="info", objects=["info", "success", "warning", "danger"], precedence=-1)
    fronts = param.List(default=[], precedence=-1)
    runs = param.List(default=[], precedence=-1)
    source = param.String(default="", precedence=-1)

    def __init__(self, open_experiment: Callable[[ExperimentSetup], None] | None = None, **params: Any) -> None:
        super().__init__(**params)
        self._open_experiment = open_experiment

    def _report(self, message: str, kind: str = "info") -> None:
        self.status_kind = kind
        self.status = message

    def find_studies(self, event: Any = None) -> None:
        """List the canonical studies at or below ``study_dir``."""
        from vamos.ux.studio.services import discover_study_directories

        base = Path(self.study_dir).expanduser()
        if not base.is_dir():
            self.param.selected_study.objects = []
            self.selected_study = None
            self._report(f"Directory not found: {base}", "danger")
            return
        found = [str(path) for path in discover_study_directories(base, limit=50)]
        self.param.selected_study.objects = found
        self.selected_study = found[0] if found else None
        if found:
            self._report(f"Found {len(found)} studies under {base}. Choose one and click Load study.")
        else:
            self._report(f"No canonical study (study-manifest.json) found under {base}.", "warning")

    def load_data(self, event: Any = None) -> None:
        """Load the selected study through the data-only readers."""
        from vamos.ux.studio.services import load_studio_data

        target = self.selected_study or self.study_dir
        try:
            runs, fronts = load_studio_data(Path(target).expanduser())
        except Exception as exc:
            self._report(f"Cannot load {target}: {exc}", "danger")
            return
        if not fronts:
            self._report(f"{target} has no completed runs to show. {study_state_text(Path(target).expanduser())}", "warning")
        else:
            self._report(f"Loaded {len(runs)} runs, {len(fronts)} fronts from {target}.", "success")
        self._set_fronts(runs, fronts, source=str(target))

    def focus(self, study_root: str) -> None:
        """Select and load one study, for example a batch that has just run."""
        objects = list(self.param.selected_study.objects or [])
        if study_root not in objects:
            self.param.selected_study.objects = [study_root, *objects]
        self.selected_study = study_root
        self.load_data()

    def load_demo(self, event: Any = None) -> None:
        """Load the small synthetic dataset, clearly labelled as such."""
        from vamos.ux.studio.services import build_demo_study_data

        runs, fronts = build_demo_study_data()
        self._set_fronts(runs, fronts, source="demo")
        self._report("Showing synthetic demo fronts, not results of a study.", "warning")

    def _set_fronts(self, runs: list[Any], fronts: list[Any], *, source: str) -> None:
        problems = problems_in(fronts)
        self.param.problem.objects = problems
        self.problem = problems[0] if problems else None
        self.source = source
        self.runs, self.fronts = runs, fronts

    @param.depends("problem", "runs", watch=True)
    def _sync_problem_selection(self) -> None:
        runs = [] if self.source == "demo" else [run for run in self.runs if run.problem_name == self.problem]
        self.param.rerun.objects = {f"{algorithm_label(run.algorithm_name)} · seed {run.seed}": run.experiment_id for run in runs}
        self.rerun = runs[0].experiment_id if runs else None
        self.objective_ranges = []

    def filtered_fronts(self) -> list[Any]:
        """The fronts of the chosen problem, with only the solutions that pass the filters."""
        return filter_fronts(fronts_for(self.fronts, self.problem), feasible_only=self.feasible_only, ranges=self.objective_ranges)

    def ranking(self) -> Any:
        return rank_fronts(self.filtered_fronts(), method=self.mcdm_method, top_k=self.top_k, problem=self.problem)

    def run_again(self, event: Any = None) -> None:
        """Open the chosen run's settings in Run Experiment."""
        run = next((run for run in self.runs if run.experiment_id == self.rerun), None)
        if run is None or self._open_experiment is None:
            self._report("Load a study and choose one of its runs to run it again.", "warning")
            return
        self._open_experiment(setup_from_run(run))

    @pn.depends("fronts", "problem", "feasible_only", "objective_ranges")
    def filter_summary(self) -> pn.pane.Markdown:
        fronts = fronts_for(self.fronts, self.problem)
        if not fronts:
            return pn.pane.Markdown("")
        text = f"Showing {solution_count(self.filtered_fronts()):,} of {solution_count(fronts):,} solutions."
        if self.feasible_only and not have_constraints(fronts):
            text += " These fronts store no constraint values, so every solution counts as feasible."
        return pn.pane.Markdown(text, sizing_mode="stretch_width")

    @pn.depends("fronts", "problem", "mcdm_method", "top_k", "feasible_only", "objective_ranges")
    def ranking_table(self) -> pn.Column | pn.pane.Markdown:
        if not self.fronts:
            return pn.pane.Markdown("*Load a study first.*")
        import pandas as pd

        ranking = self.ranking()
        items: list[Any] = [pn.pane.Alert(warning, alert_type="warning") for warning in ranking.warnings]
        if ranking.rows:
            items.append(
                pn.widgets.Tabulator(pd.DataFrame(ranking.rows), sizing_mode="stretch_width", show_index=False, theme="fast", disabled=True)
            )
        else:
            items.append(pn.pane.Markdown("*No solutions to rank.*"))
        return pn.Column(*items, sizing_mode="stretch_width")

    @pn.depends("fronts", "problem", "feasible_only", "objective_ranges")
    def pareto_overlay(self) -> pn.pane.Plotly | pn.pane.Markdown:
        if not fronts_for(self.fronts, self.problem):
            return pn.pane.Markdown("*No fronts loaded.*")
        fronts = self.filtered_fronts()
        if not fronts:
            return pn.pane.Markdown("*No solution passes the filters.*")
        title = f"Fronts on {self.problem}" + (" (synthetic demo data)" if self.source == "demo" else "")
        figure = overlay_figure([(algorithm_label(front.algorithm_name), front.points_F) for front in fronts], title=title)
        return pn.pane.Plotly(figure, sizing_mode="stretch_width")

    @pn.depends("fronts")
    def fronts_table(self) -> pn.widgets.Tabulator | pn.pane.Markdown:
        if not self.fronts:
            return pn.pane.Markdown("")
        import pandas as pd

        return pn.widgets.Tabulator(pd.DataFrame(front_summary_rows(self.fronts)), show_index=False, theme="fast", disabled=True)

    def ranking_csv(self) -> io.StringIO:
        return io.StringIO(csv_text(self.ranking().rows))


def _range_sliders(state: ResultsState) -> pn.Column:
    """One range slider per objective of the chosen problem; moving one filters its solutions."""
    box = pn.Column(sizing_mode="stretch_width")
    sliders: list[tuple[int, Any]] = []
    n_obj = [0]

    def update(_event: Any = None) -> None:
        ranges: list[tuple[float | None, float | None]] = [(None, None)] * n_obj[0]
        for index, slider in sliders:
            ranges[index] = (float(slider.value[0]), float(slider.value[1]))
        state.objective_ranges = ranges

    def rebuild(_event: Any = None) -> None:
        bounds = objective_bounds(fronts_for(state.fronts, state.problem))
        n_obj[0] = len(bounds)
        sliders.clear()
        for index, (low, high) in enumerate(bounds):
            if high <= low:
                continue  # A constant objective has nothing to filter.
            slider = pn.widgets.EditableRangeSlider(
                name=f"f{index + 1}", start=low, end=high, value=(low, high), step=(high - low) / 100, sizing_mode="stretch_width"
            )
            slider.param.watch(update, "value")
            sliders.append((index, slider))
        box.objects = [slider for _, slider in sliders]

    state.param.watch(rebuild, ["fronts", "problem"])
    rebuild()
    return box


def render_results(
    study_dir: Path | None = None,
    focus_study: str | None = None,
    open_experiment: Callable[[ExperimentSetup], None] | None = None,
) -> pn.Column:
    """Build and return the Results Explorer page layout; ``focus_study`` is loaded at once."""
    state = ResultsState(open_experiment=open_experiment, study_dir=str(study_dir or "results"))
    if Path(state.study_dir).expanduser().is_dir():
        state.find_studies()
    if focus_study:
        state.focus(focus_study)

    find_btn = pn.widgets.Button(name="Find studies", button_type="light")
    find_btn.on_click(state.find_studies)
    load_btn = pn.widgets.Button(name="Load study", button_type="primary")
    load_btn.on_click(state.load_data)
    demo_btn = pn.widgets.Button(name="Load demo data", button_type="light")
    demo_btn.on_click(state.load_demo)
    export_btn = pn.widgets.FileDownload(
        callback=state.ranking_csv,
        filename="vamos_ranked_solutions.csv",
        label="Export ranking (CSV)",
        disabled=pn.bind(lambda fronts: not fronts, state.param.fronts),
    )
    rerun_btn = pn.widgets.Button(
        name="Run again in Run Experiment", button_type="light", disabled=pn.bind(lambda run: run is None, state.param.rerun)
    )
    rerun_btn.on_click(state.run_again)
    status = pn.pane.Alert(
        pn.bind(lambda text: text, state.param.status),
        alert_type=pn.bind(lambda kind: kind, state.param.status_kind),
    )

    config = pn.Column(
        "## Explore Results",
        pn.widgets.TextInput.from_param(state.param.study_dir, name="Study directory or search folder"),
        pn.Row(find_btn, load_btn),
        pn.widgets.Select.from_param(state.param.selected_study, name="Study"),
        demo_btn,
        status,
        pn.layout.Divider(),
        pn.widgets.Select.from_param(state.param.problem, name="Problem"),
        pn.widgets.Select.from_param(
            state.param.mcdm_method, name="MCDM method", options={label: key for key, label in MCDM_METHODS.items()}
        ),
        pn.widgets.IntInput.from_param(state.param.top_k, name="Top solutions per front"),
        export_btn,
        pn.pane.Markdown("*Ranks compare the solutions of one front: objectives are normalized per front before scoring.*"),
        pn.layout.Divider(),
        "### Filters",
        pn.widgets.Checkbox.from_param(state.param.feasible_only, name="Feasible solutions only"),
        _range_sliders(state),
        state.filter_summary,
        pn.layout.Divider(),
        "### Run again",
        pn.widgets.Select.from_param(state.param.rerun, name="Run"),
        rerun_btn,
        pn.pane.Markdown("*Opens the run's problem, algorithm, population, budget and seed in Run Experiment.*"),
        sizing_mode="stretch_width",
        max_width=420,
    )

    output = pn.Column(
        "## Pareto Fronts",
        state.pareto_overlay,
        state.fronts_table,
        pn.layout.Divider(),
        "## Solution Rankings",
        state.ranking_table,
        sizing_mode="stretch_both",
    )
    return pn.Column(pn.Row(config, output, sizing_mode="stretch_both"), sizing_mode="stretch_both")
