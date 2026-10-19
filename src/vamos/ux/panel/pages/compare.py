"""Statistical Comparison page — test whether the algorithms of a study differ, and export the evidence."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import panel as pn
import param

from vamos.ux.panel.components.background import BackgroundWork
from vamos.ux.panel.components.plots import a12_heatmap, attainment_figure, box_figure, cd_figure, cost_figure, overlay_figure
from vamos.ux.studio.batch import csv_text
from vamos.ux.studio.batch_studies import list_studies
from vamos.ux.studio.comparison import (
    HV,
    IGD_PLUS,
    INDICATOR_DIRECTIONS,
    Comparison,
    ComparisonData,
    compare_with_pivot,
    load_comparison_data,
    median_run,
    rank_algorithms,
    verdict_counts,
)
from vamos.ux.studio.comparison_export import comparison_html, comparison_latex, comparison_rows, ranking_latex, saes_csv, saes_metrics_csv
from vamos.ux.studio.comparison_stats import CORRECTIONS, attainment_surface_2d
from vamos.ux.studio.data import RunRecord

LEVELS = {"best": "best", "median": "median", "worst": "worst"}


def smallest_p_value(n1: int, n2: int) -> float:
    """The smallest two-sided p-value the exact rank-sum test can give with ``n1`` and ``n2`` runs."""
    return min(1.0, 2.0 / math.comb(n1 + n2, n1)) if n1 and n2 else 1.0


class CompareState(param.Parameterized):
    """One study, the indicator and pivot it is read with, and the tests computed from them."""

    study_dir = param.String(default="results", doc="Folder searched for studies.")
    studies = param.List(default=[], precedence=-1)
    selected = param.Selector(default=None, objects=[], allow_None=True, check_on_set=False, doc="Study to analyze.")
    indicator = param.Selector(default=HV, objects=[HV, IGD_PLUS], doc="Quality indicator.")
    pivot = param.Selector(default=None, objects=[], allow_None=True, check_on_set=False, doc="Algorithm the others are compared with.")
    correction = param.Selector(default="holm", objects=list(CORRECTIONS), doc="Multiple-comparison correction.")
    alpha = param.Selector(default=0.05, objects=[0.01, 0.05, 0.1], doc="Significance level.")
    problem = param.Selector(default=None, objects=[], allow_None=True, check_on_set=False, doc="Problem shown in Fronts and Attainment.")
    levels = param.ListSelector(default=["median"], objects=list(LEVELS), doc="Attainment surfaces to draw.")
    data = param.Parameter(default=None, precedence=-1)
    loading = param.Boolean(default=False, precedence=-1)
    status = param.String(default="Choose a study to compare its algorithms.", precedence=-1)
    status_kind = param.Selector(default="info", objects=["info", "success", "warning", "danger"], precedence=-1)

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._runs: dict[str, RunRecord] = {}
        self._requested: str | None = None
        self._work: BackgroundWork[ComparisonData] = BackgroundWork()

    def _report(self, message: str, kind: str = "info") -> None:
        self.status_kind, self.status = kind, message

    def refresh_studies(self, event: Any = None, *, select: str | None = None) -> None:
        self.studies = [listing for listing in list_studies(Path(self.study_dir)) if listing.finished]
        roots = [str(listing.root) for listing in self.studies]
        if select is not None and select not in roots:
            roots.insert(0, select)
        self.param.selected.objects = roots
        before = self.selected
        if select is not None:
            self.selected = select
        elif self.selected not in roots:
            self.selected = roots[0] if roots else None
        if event is not None and self.selected == before:
            self.load(force=True)  # Refresh rereads a study that may have gained runs.

    def study_options(self) -> dict[str, str]:
        return {listing.label(): str(listing.root) for listing in self.studies}

    @param.depends("selected", watch=True)
    def _on_select(self) -> None:
        self.load()

    def load(self, *, force: bool = False) -> None:
        """Read the selected study's runs and indicators in the background, once per study."""
        if not self.selected:
            self.data = None
            return
        root = Path(self.selected)
        loaded = isinstance(self.data, ComparisonData) and self.data.root == root
        if not force and (self._requested == str(root) or loaded):
            return
        self._requested = str(root)
        self.loading = True
        self._report(f"Reading {root.name}…")
        self._work.submit(lambda: load_comparison_data(root, cache=self._runs), lambda result: self._loaded(root, result))

    def _loaded(self, root: Path, result: ComparisonData | Exception) -> None:
        self.loading = False
        self._requested = None
        if str(root) != str(self.selected):
            return
        if isinstance(result, Exception):
            self.data = None
            self._report(f"Cannot analyze {root.name}: {result}", "danger")
            return
        self.param.indicator.objects = list(result.indicators) or [HV]
        if self.indicator not in self.param.indicator.objects:
            self.indicator = self.param.indicator.objects[0]
        self.param.pivot.objects = list(result.algorithms)
        if self.pivot not in result.algorithms:
            self.pivot = result.algorithms[0] if result.algorithms else None
        self.param.problem.objects = list(result.problems)
        if self.problem not in result.problems:
            self.problem = result.problems[0] if result.problems else None
        self.data = result
        runs = len(result.runs)
        self._report(f"{root.name}: {runs} runs of {len(result.algorithms)} algorithms on {len(result.problems)} problems.", "success")
        if len(result.algorithms) < 2:
            self._report(f"{root.name} has a single algorithm; there is nothing to compare.", "warning")

    # ---- derived results ----

    def comparisons(self) -> list[Comparison]:
        if not isinstance(self.data, ComparisonData) or self.pivot is None:
            return []
        return compare_with_pivot(self.data, self.indicator, self.pivot, correction=self.correction, alpha=float(self.alpha))

    def power_warning(self) -> str:
        """Say when the run counts make a significant per-problem result impossible."""
        data = self.data
        if not isinstance(data, ComparisonData) or self.pivot is None:
            return ""
        comparisons = self.comparisons()
        if not comparisons:
            return ""
        sizes = [
            (data.sample(self.indicator, item.problem, item.contender).size, data.sample(self.indicator, item.problem, item.pivot).size)
            for item in comparisons
        ]
        smallest = min(smallest_p_value(n1, n2) for n1, n2 in sizes)
        threshold = float(self.alpha) / (len(comparisons) if self.correction != "none" else 1)
        if smallest > threshold:
            return (
                f"With these run counts the smallest possible p-value is {smallest:.2g}, above {threshold:.2g}, "
                f"so no per-problem difference can be significant{' after the correction' if self.correction != 'none' else ''}. "
                "Add runs per algorithm and problem to detect differences."
            )
        return ""

    def describe(self) -> str:
        direction = "higher" if INDICATOR_DIRECTIONS[self.indicator] else "lower"
        return (
            f"{self.indicator} ({direction} is better) against **{self.pivot}**: two-sided Mann-Whitney test at "
            f"α = {self.alpha} with {CORRECTIONS[self.correction]} correction over {len(self.comparisons())} comparisons, "
            "and the Vargha-Delaney A12 effect size."
        )

    # ---- downloads ----

    def comparison_tex(self) -> io.StringIO:
        if not isinstance(self.data, ComparisonData) or self.pivot is None:
            return io.StringIO("")
        return io.StringIO(
            comparison_latex(self.data, self.indicator, self.comparisons(), self.pivot, correction=self.correction, alpha=float(self.alpha))
        )

    def ranking_tex(self) -> io.StringIO:
        if not isinstance(self.data, ComparisonData) or self.pivot is None:
            return io.StringIO("")
        return io.StringIO(ranking_latex(rank_algorithms(self.data, self.indicator, self.pivot), self.pivot, self.indicator))

    def comparisons_csv(self) -> io.StringIO:
        return io.StringIO(csv_text(comparison_rows(self.comparisons())))

    def saes_results(self) -> io.StringIO:
        return io.StringIO(saes_csv(self.data) if isinstance(self.data, ComparisonData) else "")

    def saes_metrics(self) -> io.StringIO:
        return io.StringIO(saes_metrics_csv(self.data) if isinstance(self.data, ComparisonData) else "")


def _table(rows: list[dict[str, object]]) -> Any:
    if not rows:
        return pn.pane.Markdown("*Nothing to show.*")
    import pandas as pd

    return pn.widgets.Tabulator(pd.DataFrame(rows), show_index=False, theme="fast", disabled=True, sizing_mode="stretch_width")


def verdict_view(state: CompareState) -> Any:
    comparisons = state.comparisons()
    if not comparisons:
        return pn.pane.Markdown("*Load a study with at least two algorithms.*")
    counts = verdict_counts(comparisons)
    problems = {item.contender: 0 for item in comparisons}
    for item in comparisons:
        problems[item.contender] += 1
    lines = [state.describe(), ""]
    for contender, tally in counts.items():
        lines.append(
            f"- **{contender}** is significantly better than {state.pivot} on **{tally['+']}**, worse on **{tally['-']}**, "
            f"and not significantly different on {tally['=']} of {problems[contender]} problems."
        )
    lines += ["", "*No significant difference is not evidence that two algorithms perform the same.*"]
    items: list[Any] = [pn.pane.Markdown("\n".join(lines), sizing_mode="stretch_width")]
    if warning := state.power_warning():
        items.insert(0, pn.pane.Alert(warning, alert_type="warning"))
    return pn.Column(*items, sizing_mode="stretch_width")


def comparisons_view(state: CompareState) -> Any:
    data, comparisons = state.data, state.comparisons()
    if not isinstance(data, ComparisonData) or not comparisons or state.pivot is None:
        return pn.pane.Markdown("*Load a study with at least two algorithms.*")
    details = [
        {key: (round(value, 4) if isinstance(value, float) else value) for key, value in row.items()}
        for row in comparison_rows(comparisons)
    ]
    return pn.Column(
        pn.pane.Markdown(
            "Median (IQR) of every algorithm; `+`/`-`/`=` compare it with the pivot. The darker cell is the best median, "
            "the lighter one the second best."
        ),
        pn.pane.HTML(comparison_html(data, state.indicator, comparisons, state.pivot), sizing_mode="stretch_width"),
        pn.Card(_table(details), title="p-values and effect sizes", collapsed=True, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )


def effect_view(state: CompareState) -> Any:
    data, comparisons = state.data, state.comparisons()
    if not isinstance(data, ComparisonData) or not comparisons or state.pivot is None:
        return pn.pane.Markdown("*Load a study with at least two algorithms.*")
    contenders = list(dict.fromkeys(item.contender for item in comparisons))
    problems = list(dict.fromkeys(item.problem for item in comparisons))
    effects = np.full((len(problems), len(contenders)), np.nan)
    labels = [["" for _ in contenders] for _ in problems]
    for item in comparisons:
        row, column = problems.index(item.problem), contenders.index(item.contender)
        effects[row, column] = item.effect
        labels[row][column] = f"{item.effect:.2f} {item.verdict}<br>{item.magnitude}"
    return pn.pane.Plotly(a12_heatmap(problems, contenders, effects, labels, pivot=state.pivot), sizing_mode="stretch_width")


def ranking_view(state: CompareState) -> Any:
    data = state.data
    if not isinstance(data, ComparisonData) or state.pivot is None or len(data.algorithms) < 2:
        return pn.pane.Markdown("*Load a study with at least two algorithms.*")
    ranking = rank_algorithms(data, state.indicator, state.pivot)
    items: list[Any] = [pn.pane.Alert(warning, alert_type="warning") for warning in ranking.warnings]
    if ranking.friedman is not None:
        statistic, p = ranking.friedman
        items.append(pn.pane.Markdown(f"Friedman test over {ranking.problems} problems: χ² = {statistic:.3g}, p = {p:.3g}."))
    if ranking.ranks:
        items.append(pn.pane.Plotly(cd_figure(ranking.algorithms, ranking.ranks, ranking.cd, ranking.groups), sizing_mode="stretch_width"))
        rows = []
        for index in np.argsort(ranking.ranks):
            name = ranking.algorithms[index]
            z, p, adjusted = ranking.against_pivot[index] if ranking.against_pivot else (0.0, 1.0, 1.0)
            rows.append(
                {
                    "Algorithm": name,
                    "Average rank": round(ranking.ranks[index], 3),
                    f"p vs {state.pivot}": "" if name == state.pivot else round(p, 4),
                    "Holm p": "" if name == state.pivot else round(adjusted, 4),
                }
            )
        items.append(_table(rows))
    return pn.Column(*items, sizing_mode="stretch_width")


def distribution_view(state: CompareState) -> Any:
    data = state.data
    if not isinstance(data, ComparisonData):
        return pn.pane.Markdown("*Load a study first.*")
    rows = [
        {"Problem": problem, "Algorithm": algorithm, state.indicator: float(value)}
        for (problem, algorithm), values in data.values.get(state.indicator, {}).items()
        for value in values
        if np.isfinite(value)
    ]
    return (
        pn.pane.Plotly(box_figure(rows, indicator=state.indicator), sizing_mode="stretch_width")
        if rows
        else pn.pane.Markdown("*No values.*")
    )


def fronts_view(state: CompareState) -> Any:
    data = state.data
    if not isinstance(data, ComparisonData) or state.problem is None:
        return pn.pane.Markdown("*Load a study first.*")
    fronts = []
    for algorithm in data.algorithms:
        run = median_run(data, state.indicator, state.problem, algorithm)
        if run is not None and run.feasible_F.size:
            fronts.append((f"{algorithm} (seed {run.seed})", run.feasible_F))
    if not fronts:
        return pn.pane.Markdown("*No feasible front on this problem.*")
    title = f"Front of the median-{state.indicator} run of each algorithm on {state.problem}"
    return pn.pane.Plotly(overlay_figure(fronts, title=title), sizing_mode="stretch_width")


def attainment_view(state: CompareState) -> Any:
    data = state.data
    if not isinstance(data, ComparisonData) or state.problem is None:
        return pn.pane.Markdown("*Load a study first.*")
    surfaces = []
    for algorithm in data.algorithms:
        fronts = [run.feasible_F for run in data.runs if run.problem == state.problem and run.algorithm == algorithm]
        if not fronts or fronts[0].shape[1] != 2:
            return pn.pane.Markdown("*Attainment surfaces are drawn for two-objective problems.*")
        n = len(fronts)
        levels = {"best": 1, "median": (n + 1) // 2, "worst": n}
        for label in state.levels:
            surfaces.append((algorithm, label, attainment_surface_2d(fronts, levels[label])))
    title = f"Empirical attainment surfaces on {state.problem} (median: attained by at least half of the runs)"
    return pn.pane.Plotly(attainment_figure(surfaces, title=title), sizing_mode="stretch_width")


def cost_view(state: CompareState) -> Any:
    data = state.data
    if not isinstance(data, ComparisonData) or not data.runtimes_ms:
        return pn.pane.Markdown("*No runtimes recorded.*")
    rows = [
        {
            "Problem": problem,
            "Algorithm": algorithm,
            "Median runtime (s)": round(float(np.median(values)) / 1000.0, 3),
            "Runs": int(values.size),
        }
        for (problem, algorithm), values in data.runtimes_ms.items()
    ]
    return pn.Column(pn.pane.Plotly(cost_figure(rows), sizing_mode="stretch_width"), _table(rows), sizing_mode="stretch_width")


def render_compare(study_dir: Path | None = None, focus_study: str | None = None) -> pn.Column:
    """Build the Statistical Comparison page; ``focus_study`` is analyzed at once."""
    state = CompareState(study_dir=str(study_dir or "results"))
    state.refresh_studies(select=focus_study or None)

    refresh = pn.widgets.Button(name="Refresh", button_type="light")
    refresh.on_click(state.refresh_studies)
    controls = pn.Column(
        "## Statistical Comparison",
        pn.widgets.TextInput.from_param(state.param.study_dir, name="Folder"),
        refresh,
        pn.widgets.Select.from_param(
            state.param.selected, name="Study", options=pn.bind(lambda _: state.study_options(), state.param.studies)
        ),
        pn.indicators.LoadingSpinner(
            value=pn.bind(lambda busy: busy, state.param.loading), size=30, visible=pn.bind(lambda busy: busy, state.param.loading)
        ),
        pn.pane.Alert(pn.bind(lambda text: text, state.param.status), alert_type=pn.bind(lambda kind: kind, state.param.status_kind)),
        pn.widgets.Select.from_param(state.param.indicator, name="Indicator"),
        pn.widgets.Select.from_param(state.param.pivot, name="Compare with (pivot)"),
        pn.widgets.Select.from_param(state.param.correction, name="Correction", options={label: key for key, label in CORRECTIONS.items()}),
        pn.widgets.Select.from_param(state.param.alpha, name="Significance level"),
        pn.widgets.Select.from_param(state.param.problem, name="Problem (fronts and attainment)"),
        pn.widgets.CheckButtonGroup.from_param(state.param.levels, name="Attainment surfaces"),
        pn.pane.Markdown("### Export"),
        pn.widgets.FileDownload(callback=state.comparison_tex, filename="vamos_comparison.tex", label="Comparison table (LaTeX)"),
        pn.widgets.FileDownload(callback=state.ranking_tex, filename="vamos_ranking.tex", label="Ranking table (LaTeX)"),
        pn.widgets.FileDownload(callback=state.comparisons_csv, filename="vamos_comparisons.csv", label="Comparisons (CSV)"),
        pn.widgets.FileDownload(callback=state.saes_results, filename="vamos_saes_results.csv", label="SAES results (CSV)"),
        pn.widgets.FileDownload(callback=state.saes_metrics, filename="vamos_saes_metrics.csv", label="SAES metrics (CSV)"),
        max_width=380,
        sizing_mode="stretch_width",
    )
    watched = (state.param.data, state.param.indicator, state.param.pivot, state.param.correction, state.param.alpha)
    tabs = pn.Tabs(
        ("Verdict", pn.bind(lambda *_: verdict_view(state), *watched)),
        ("Comparisons", pn.bind(lambda *_: comparisons_view(state), *watched)),
        ("Effect size", pn.bind(lambda *_: effect_view(state), *watched)),
        ("Ranking", pn.bind(lambda *_: ranking_view(state), state.param.data, state.param.indicator, state.param.pivot)),
        ("Distributions", pn.bind(lambda *_: distribution_view(state), state.param.data, state.param.indicator)),
        ("Fronts", pn.bind(lambda *_: fronts_view(state), state.param.data, state.param.indicator, state.param.problem)),
        ("Attainment", pn.bind(lambda *_: attainment_view(state), state.param.data, state.param.problem, state.param.levels)),
        ("Cost", pn.bind(lambda *_: cost_view(state), state.param.data)),
        dynamic=True,
        sizing_mode="stretch_both",
    )
    return pn.Column(pn.Row(controls, tabs, sizing_mode="stretch_both"), sizing_mode="stretch_both")
