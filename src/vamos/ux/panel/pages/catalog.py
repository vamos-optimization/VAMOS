"""Catalog page — browse the registered problems and built-in algorithms, and open one in Run Experiment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import panel as pn
import param

from vamos.ux.studio.algorithm_catalog import algorithm_label, studio_algorithms
from vamos.ux.studio.catalog import (
    ENCODINGS,
    ProblemEntry,
    algorithm_entries,
    filter_problems,
    parameter_rows,
    population_rule,
    problem_entries,
    problem_rows,
    representative_problem,
)
from vamos.ux.studio.experiment_setup import ExperimentSetup

ALL_ENCODINGS = "all"


class CatalogState(param.Parameterized):
    """Reactive state of the catalog: problem filters and selection, and the algorithm shown."""

    encoding = param.Selector(default=ALL_ENCODINGS, objects=[ALL_ENCODINGS, *ENCODINGS], doc="Problem encoding.")
    search = param.String(default="", doc="Text in a problem's key, name or description.")
    problem = param.String(default="", doc="Key of the selected problem.")
    algorithm = param.Selector(default="nsgaii", objects=list(studio_algorithms()), doc="Algorithm whose parameters are shown.")
    algorithm_encoding = param.Selector(default="real", objects=list(ENCODINGS), doc="Encoding whose operators are shown.")

    def __init__(self, open_experiment: Callable[[ExperimentSetup], None] | None = None, **params: Any) -> None:
        super().__init__(**params)
        self.entries = problem_entries()
        self._by_key = {entry.key: entry for entry in self.entries}
        self._algorithms = {entry.key: entry for entry in algorithm_entries()}
        self._open_experiment = open_experiment
        self._sync_algorithm_encodings()

    def visible_problems(self) -> list[ProblemEntry]:
        encoding = None if self.encoding == ALL_ENCODINGS else self.encoding
        return filter_problems(self.entries, encoding=encoding, text=self.search)

    def selected_problem(self) -> ProblemEntry | None:
        return self._by_key.get(self.problem)

    def detail_problem(self) -> ProblemEntry | None:
        """The problem whose dimensions the algorithm details use: the selected one when its encoding matches."""
        selected = self.selected_problem()
        if selected is not None and selected.encoding == self.algorithm_encoding:
            return selected
        return representative_problem(self.algorithm_encoding, self.entries)

    @param.depends("algorithm", watch=True)
    def _sync_algorithm_encodings(self) -> None:
        encodings = list(self._algorithms[self.algorithm].encodings)
        self.param.algorithm_encoding.objects = encodings
        if self.algorithm_encoding not in encodings:
            self.algorithm_encoding = encodings[0]

    def open_problem(self, event: Any = None) -> None:
        """Open the selected problem in Run Experiment."""
        entry = self.selected_problem()
        if entry is not None and self._open_experiment is not None:
            self._open_experiment(ExperimentSetup(problem=entry.key, source=f"the catalog ({entry.label})"))

    def open_algorithm(self, event: Any = None) -> None:
        """Open the algorithm in Run Experiment with the problem its details describe."""
        entry = self.detail_problem()
        if entry is not None and self._open_experiment is not None:
            self._open_experiment(
                ExperimentSetup(problem=entry.key, algorithm=self.algorithm, source=f"the catalog ({algorithm_label(self.algorithm)})")
            )

    @param.depends("problem")
    def problem_details(self) -> pn.pane.Markdown:
        entry = self.selected_problem()
        if entry is None:
            return pn.pane.Markdown("*Select a problem in the table to see its details.*", sizing_mode="stretch_width")
        objectives = f"{entry.n_obj} (adjustable)" if entry.n_obj_adjustable else str(entry.n_obj)
        front = "VAMOS ships its reference front" if entry.reference_front else "no reference front is shipped"
        algorithms = ", ".join(algorithm_label(name) for name in entry.algorithms) or "none"
        return pn.pane.Markdown(
            f"### {entry.label} (`{entry.key}`)\n\n{entry.description}\n\n"
            f"- **Encoding:** {entry.encoding}\n- **Variables:** {entry.n_var}\n- **Objectives:** {objectives}\n"
            f"- **Reference front:** {front} for {entry.n_obj} objectives\n- **Algorithms:** {algorithms}",
            sizing_mode="stretch_width",
        )

    @param.depends("algorithm", "algorithm_encoding", "problem")
    def algorithm_details(self) -> pn.Column:
        entry = self.detail_problem()
        if entry is None:
            return pn.Column(pn.pane.Markdown("*No registered problem has this encoding.*"))
        import pandas as pd

        rows = parameter_rows(self.algorithm, entry)
        summary = pn.pane.Markdown(
            f"### {algorithm_label(self.algorithm)} on {self.algorithm_encoding} problems\n\n"
            f"Defaults shown for **{entry.label}**: {entry.n_var} variables, {entry.n_obj} objectives. "
            f"{population_rule(self.algorithm, entry.n_obj)} Run Experiment and the Problem Builder offer these "
            "parameters under **Customize**; `vamos tune` searches the same space.",
            sizing_mode="stretch_width",
        )
        table: Any = (
            pn.widgets.Tabulator(pd.DataFrame(rows), show_index=False, theme="fast", disabled=True, sizing_mode="stretch_width")
            if rows
            else pn.pane.Markdown("*This algorithm has no configurable parameters for this encoding.*")
        )
        return pn.Column(summary, table, sizing_mode="stretch_width")


def render_catalog(open_experiment: Callable[[ExperimentSetup], None] | None = None) -> pn.Column:
    """Build the catalog page with a Problems tab and an Algorithms tab."""
    import pandas as pd

    state = CatalogState(open_experiment=open_experiment)
    table = pn.widgets.Tabulator(
        pd.DataFrame(problem_rows(state.visible_problems())),
        show_index=False,
        theme="fast",
        disabled=True,
        selectable=1,
        pagination="local",
        page_size=15,
        sizing_mode="stretch_width",
    )

    def refresh_table(_event: Any = None) -> None:
        table.selection = []
        table.value = pd.DataFrame(problem_rows(state.visible_problems()))

    def select_row(event: Any) -> None:
        visible = state.visible_problems()
        if event.new and event.new[0] < len(visible):
            state.problem = visible[event.new[0]].key

    state.param.watch(refresh_table, ["encoding", "search"])
    table.param.watch(select_row, "selection")
    count = pn.bind(
        lambda _encoding, _search: f"{len(state.visible_problems())} of {len(state.entries)} problems",
        state.param.encoding,
        state.param.search,
    )

    open_problem = pn.widgets.Button(
        name="Open in Run Experiment", button_type="primary", disabled=pn.bind(lambda key: not key, state.param.problem)
    )
    open_problem.on_click(state.open_problem)
    problems_tab = pn.Row(
        pn.Column(
            pn.Row(
                pn.widgets.Select.from_param(state.param.encoding, name="Encoding"),
                pn.widgets.TextInput.from_param(state.param.search, name="Search", placeholder="zdt, knapsack, constrained…"),
            ),
            pn.pane.Markdown(count),
            table,
            sizing_mode="stretch_width",
        ),
        pn.Column(state.problem_details, open_problem, sizing_mode="stretch_width", max_width=420),
        sizing_mode="stretch_width",
    )

    open_algorithm = pn.widgets.Button(name="Open in Run Experiment", button_type="primary")
    open_algorithm.on_click(state.open_algorithm)
    algorithm_options = {entry.label: entry.key for entry in algorithm_entries()}
    algorithms_tab = pn.Column(
        pn.Row(
            pn.widgets.Select.from_param(state.param.algorithm, name="Algorithm", options=algorithm_options),
            pn.widgets.Select.from_param(state.param.algorithm_encoding, name="Problem encoding"),
        ),
        state.algorithm_details,
        open_algorithm,
        sizing_mode="stretch_width",
    )
    return pn.Column(
        "## Problems & Algorithms",
        pn.Tabs(("Problems", problems_tab), ("Algorithms", algorithms_tab), dynamic=True, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )


__all__ = ["CatalogState", "render_catalog"]
