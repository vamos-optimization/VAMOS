"""Experiment page — run one algorithm on a registered problem and browse its generations."""

from __future__ import annotations

from typing import Any

import panel as pn
import param

from vamos.foundation.problem.registry import get_problem_specs, make_problem_selection
from vamos.ux.panel.components.algorithm_config import AlgorithmConfigurator
from vamos.ux.panel.components.plots import front_figure
from vamos.ux.studio.algorithm_form import algorithm_label
from vamos.ux.studio.services import DynamicsCallback, _run_algorithm


def problem_options() -> dict[str, str]:
    """Registered problems as ``{label: key}``, in registry order."""
    options: dict[str, str] = {}
    for key, spec in get_problem_specs().items():
        label = spec.label if spec.label.lower() == key else f"{spec.label} ({key})"
        options[label] = key
    return options


class ExperimentState(param.Parameterized):
    """Reactive state for one run and its generation history."""

    problem_name = param.Selector(default="zdt1", objects=list(problem_options().values()), doc="Registered problem key.")
    max_evaluations = param.Integer(default=25000, bounds=(100, 5_000_000))
    seed = param.Integer(default=0, bounds=(0, 2**31 - 1))

    # ---- Outputs ----
    running = param.Boolean(default=False, precedence=-1)
    status = param.String(default="Ready.", precedence=-1)
    problem_info = param.String(default="", precedence=-1)
    generation_history = param.List(default=[], precedence=-1)
    current_gen = param.Integer(default=0, bounds=(0, None), precedence=-1)
    result_F = param.Array(default=None, precedence=-1)

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.configurator = AlgorithmConfigurator()
        self._sync_problem()

    @param.depends("problem_name", watch=True)
    def _sync_problem(self) -> None:
        selection = make_problem_selection(self.problem_name)
        spec = selection.spec
        self.configurator.set_problem(encoding=spec.encoding, n_var=selection.n_var, n_obj=selection.n_obj)
        description = f" — {spec.description}" if spec.description else ""
        self.problem_info = (
            f"**{spec.label}**: {selection.n_var} variables, {selection.n_obj} objectives, {spec.encoding} encoding{description}"
        )

    @param.depends("current_gen", "generation_history")
    def generation_view(self) -> pn.pane.Plotly | pn.pane.Markdown:
        if not self.generation_history:
            return pn.pane.Markdown("*No generations recorded yet.*")
        idx = min(self.current_gen, len(self.generation_history) - 1)
        return pn.pane.Plotly(front_figure(self.generation_history[idx], title=f"Generation {idx}"), sizing_mode="stretch_width")

    def run_experiment(self, event: Any = None) -> None:
        """Run the configured algorithm and keep the population objectives of every generation."""
        self.running = True
        self.status = "Running..."
        self.generation_history = []
        self.current_gen = 0
        try:
            problem = make_problem_selection(self.problem_name).instantiate()
            resolved = self.configurator.resolve()
            algorithm = self.configurator.selected_algorithm
            callback = DynamicsCallback()
            _run_algorithm(
                problem,
                algorithm=algorithm,
                algorithm_config=resolved.config,
                termination=("max_evaluations", self.max_evaluations),
                seed=self.seed,
                engine="numpy",
                live_viz=callback,
            )
            history = list(callback.history)
            self.generation_history = history
            if history:
                self.current_gen = len(history) - 1
                self.result_F = history[-1]
            configuration = "customized configuration" if resolved.differences else "VAMOS default configuration"
            pop = resolved.config.to_dict().get("pop_size")
            self.status = f"Done — {algorithm_label(algorithm)} ({configuration}, population {pop}): {len(history)} generations recorded."
        except Exception as exc:
            self.status = f"Error: {exc}"
        finally:
            self.running = False


def render_experiment() -> pn.Column:
    """Build and return the Experiment page layout."""
    state = ExperimentState()

    problem_select = pn.widgets.Select.from_param(state.param.problem_name, name="Problem", options=problem_options())
    general = pn.Column(
        "### Problem",
        problem_select,
        pn.pane.Markdown(pn.bind(lambda text: text, state.param.problem_info), sizing_mode="stretch_width"),
        pn.widgets.IntInput.from_param(state.param.max_evaluations, name="Max evaluations"),
        pn.widgets.IntInput.from_param(state.param.seed, name="Seed"),
        pn.layout.Divider(),
        "### Algorithm",
        state.configurator.view(),
    )

    run_btn = pn.widgets.Button(name="Run Experiment", button_type="primary", disabled=pn.bind(lambda r: r, state.param.running))
    run_btn.on_click(state.run_experiment)
    status_pane = pn.pane.Alert(pn.bind(lambda s: s, state.param.status), alert_type="info")
    note = pn.pane.Markdown(
        "*The run executes in the Studio server process; the page updates when it finishes.*",
        sizing_mode="stretch_width",
    )

    gen_slider = pn.widgets.IntSlider.from_param(
        state.param.current_gen,
        name="Generation",
        start=0,
        end=pn.bind(lambda h: max(len(h) - 1, 1), state.param.generation_history),
    )

    config_col = pn.Column(
        "## Run Experiment",
        general,
        pn.layout.Divider(),
        run_btn,
        status_pane,
        note,
        sizing_mode="stretch_width",
        max_width=460,
    )
    viz_col = pn.Column("## Visualization", state.generation_view, gen_slider, sizing_mode="stretch_both")
    return pn.Column(pn.Row(config_col, viz_col, sizing_mode="stretch_both"), sizing_mode="stretch_both")
