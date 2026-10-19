"""Results Explorer page — load saved studies, compare fronts, rank solutions and export them."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import panel as pn
import param

from vamos.ux.panel.components.plots import overlay_figure
from vamos.ux.studio.algorithm_catalog import algorithm_label
from vamos.ux.studio.batch import csv_text
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

    # ---- outputs ----
    status = param.String(default="Enter a study directory and click Find studies.", precedence=-1)
    status_kind = param.Selector(default="info", objects=["info", "success", "warning", "danger"], precedence=-1)
    fronts = param.List(default=[], precedence=-1)
    runs = param.List(default=[], precedence=-1)
    source = param.String(default="", precedence=-1)

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
        self.runs, self.fronts, self.source = runs, fronts, source

    def ranking(self) -> Any:
        return rank_fronts(self.fronts, method=self.mcdm_method, top_k=self.top_k, problem=self.problem)

    @pn.depends("fronts", "problem", "mcdm_method", "top_k")
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

    @pn.depends("fronts", "problem")
    def pareto_overlay(self) -> pn.pane.Plotly | pn.pane.Markdown:
        fronts = fronts_for(self.fronts, self.problem)
        if not fronts:
            return pn.pane.Markdown("*No fronts loaded.*")
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


def render_results(study_dir: Path | None = None, focus_study: str | None = None) -> pn.Column:
    """Build and return the Results Explorer page layout; ``focus_study`` is loaded at once."""
    state = ResultsState(study_dir=str(study_dir or "results"))
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
