"""Experiment page — run one algorithm on a registered problem and browse its generations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import panel as pn
import param

from vamos.foundation.problem.registry import get_problem_specs, make_problem_selection
from vamos.ux.panel.components.algorithm_config import DEFAULTS_MODE, AlgorithmConfigurator
from vamos.ux.panel.components.background import BackgroundRun
from vamos.ux.panel.components.plots import front_figure
from vamos.ux.studio.algorithm_form import algorithm_label
from vamos.ux.studio.experiment_setup import ExperimentSetup
from vamos.ux.studio.live_run import LiveRun
from vamos.ux.studio.services import _run_algorithm
from vamos.ux.studio.study_progress import format_duration


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
    n_var = param.Integer(default=30, bounds=(1, 1_000_000), doc="Number of decision variables.")
    n_obj = param.Integer(default=2, bounds=(1, 100), doc="Number of objectives, when the problem allows choosing it.")
    max_evaluations = param.Integer(default=25000, bounds=(100, 5_000_000))
    seed = param.Integer(default=0, bounds=(0, 2**31 - 1))

    # ---- Outputs ----
    running = param.Boolean(default=False, precedence=-1)
    status = param.String(default="Ready.", precedence=-1)
    setup_note = param.String(default="", precedence=-1)
    n_obj_adjustable = param.Boolean(default=False, precedence=-1)
    progress = param.Number(default=0.0, bounds=(0.0, 1.0), precedence=-1)
    problem_info = param.String(default="", precedence=-1)
    generation_history = param.List(default=[], precedence=-1)
    generation_numbers = param.List(default=[], precedence=-1)
    current_gen = param.Integer(default=0, bounds=(0, None), precedence=-1)
    result_F = param.Array(default=None, precedence=-1)

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.configurator = AlgorithmConfigurator()
        self._live: LiveRun | None = None
        self._run_label = ""
        self._syncing = False
        self._runner: BackgroundRun[dict[str, Any]] = BackgroundRun(on_tick=self._tick, on_done=self._done)
        self._sync_problem()

    # ---- problem and dimensions ----

    @param.depends("problem_name", watch=True)
    def _sync_problem(self) -> None:
        spec = get_problem_specs()[self.problem_name]
        self.n_obj_adjustable = spec.allow_n_obj_override
        self._set_dimensions(*spec.resolve_dimensions(n_var=None, n_obj=None))

    @param.depends("n_obj", watch=True)
    def _n_obj_changed(self) -> None:
        """A new number of objectives brings the problem's default number of variables for it."""
        if self._syncing:
            return
        try:
            dimensions = get_problem_specs()[self.problem_name].resolve_dimensions(n_var=None, n_obj=self.n_obj)
        except ValueError as exc:
            self.status = f"Error: {exc}"
            return
        self._set_dimensions(*dimensions)

    @param.depends("n_var", watch=True)
    def _n_var_changed(self) -> None:
        if not self._syncing:
            self._apply_dimensions()

    def _set_dimensions(self, n_var: int, n_obj: int) -> None:
        self._syncing = True
        try:
            self.param.update(n_var=n_var, n_obj=n_obj)
        finally:
            self._syncing = False
        self._apply_dimensions()

    def _apply_dimensions(self) -> None:
        try:
            selection = make_problem_selection(self.problem_name, n_var=self.n_var, n_obj=self.n_obj)
        except ValueError as exc:
            self.status = f"Error: {exc}"
            return
        spec = selection.spec
        self.configurator.set_problem(encoding=spec.encoding, n_var=selection.n_var, n_obj=selection.n_obj)
        description = f" — {spec.description}" if spec.description else ""
        self.problem_info = (
            f"**{spec.label}**: {selection.n_var} variables, {selection.n_obj} objectives, {spec.encoding} encoding{description}"
        )

    def apply_setup(self, setup: ExperimentSetup) -> None:
        """Load a problem from the catalog, or the settings of a stored run, into the page."""
        notes = list(setup.notes)
        if setup.problem not in self.param.problem_name.objects:
            self.setup_note = f"{setup.problem} is not a registered problem, so Run Experiment cannot open it."
            return
        self.problem_name = setup.problem
        if setup.n_obj is not None and setup.n_obj != self.n_obj:
            self.n_obj = setup.n_obj
        if setup.n_var is not None and setup.n_var != self.n_var:
            self.n_var = setup.n_var
        if setup.max_evaluations is not None:
            low, high = self.param.max_evaluations.bounds or (1, None)
            budget = max(setup.max_evaluations, low)
            if budget != setup.max_evaluations:
                notes.append(f"That run's budget of {setup.max_evaluations} evaluations is below this page's minimum of {low}.")
            self.max_evaluations = min(budget, high) if high is not None else budget
        if setup.seed is not None:
            self.seed = setup.seed
            notes.append("The seed is that run's; change it to run another replicate.")
        if setup.algorithm is not None:
            if setup.algorithm in self.configurator.algorithm.options.values():
                self.configurator.algorithm.value = setup.algorithm
            else:
                notes.append(f"{algorithm_label(setup.algorithm)} has no operators for this problem's encoding.")
        if setup.pop_size is not None:
            self.configurator.pop_size.value = setup.pop_size
        self.configurator.mode.value = DEFAULTS_MODE
        self.setup_note = " ".join([f"Loaded from {setup.source}." if setup.source else "", *notes]).strip()

    @param.depends("current_gen", "generation_history")
    def generation_view(self) -> pn.pane.Plotly | pn.pane.Markdown:
        if not self.generation_history:
            return pn.pane.Markdown("*No generations recorded yet.*")
        idx = min(self.current_gen, len(self.generation_history) - 1)
        number = self.generation_numbers[idx] if idx < len(self.generation_numbers) else idx
        return pn.pane.Plotly(front_figure(self.generation_history[idx], title=f"Generation {number}"), sizing_mode="stretch_width")

    def run_experiment(self, event: Any = None) -> None:
        """Start the configured algorithm in the background and follow it generation by generation."""
        if self.running:
            return
        try:
            problem = make_problem_selection(self.problem_name, n_var=self.n_var, n_obj=self.n_obj).instantiate()
            resolved = self.configurator.resolve()
        except Exception as exc:
            self.status = f"Error: {exc}"
            return
        algorithm = self.configurator.selected_algorithm
        configuration = "customized configuration" if resolved.differences else "VAMOS default configuration"
        self._run_label = f"{algorithm_label(algorithm)} ({configuration}, population {resolved.config.to_dict().get('pop_size')})"
        live = LiveRun(self.max_evaluations)
        self._live = live
        self.generation_history, self.generation_numbers, self.current_gen, self.progress = [], [], 0, 0.0
        self.running = True
        self.status = f"Running {self._run_label}…"
        self._runner.start(
            lambda: _run_algorithm(
                problem,
                algorithm=algorithm,
                algorithm_config=resolved.config,
                termination=("max_evaluations", self.max_evaluations),
                seed=self.seed,
                engine="numpy",
                live_viz=live,
            )
        )

    def stop(self, event: Any = None) -> None:
        """Ask the algorithm to stop at the end of its current generation."""
        if self._live is not None and self.running:
            self._live.request_stop()
            self.status = "Stopping at the end of the current generation…"

    def _tick(self) -> None:
        live = self._live
        if live is None:
            return
        self.progress = live.fraction
        snapshots = live.snapshots
        if len(snapshots) != len(self.generation_history):
            follow = self.current_gen >= len(self.generation_history) - 1
            self.generation_numbers = [number for number, _ in snapshots]
            self.generation_history = [front for _, front in snapshots]
            if follow:
                self.current_gen = len(snapshots) - 1
        if self.running and not live.stop_requested:
            self.status = (
                f"Running {self._run_label}: generation {live.generation}, "
                f"{live.evaluations:,} of {live.budget:,} evaluations, {format_duration(live.elapsed)}."
            )

    def _done(self, result: dict[str, Any] | Exception) -> None:
        self.running = False
        live = self._live
        if isinstance(result, Exception) or live is None:
            self.status = f"Error: {result}"
            return
        if self.generation_history:
            self.result_F = self.generation_history[-1]
        ending = "Stopped" if live.stop_requested else "Done"
        self.status = (
            f"{ending} — {self._run_label}: {live.evaluations:,} evaluations in {format_duration(live.elapsed)}, "
            f"{len(self.generation_history)} generations recorded."
        )


def render_experiment(on_exit: Callable[[Callable[[], None]], None] | None = None, setup: ExperimentSetup | None = None) -> pn.Column:
    """Build and return the Experiment page layout, loading ``setup`` when one is given."""
    state = ExperimentState()
    if on_exit is not None:
        on_exit(state.stop)  # Leaving the page stops its run.
    if setup is not None:
        state.apply_setup(setup)

    problem_select = pn.widgets.Select.from_param(state.param.problem_name, name="Problem", options=problem_options())
    setup_pane = pn.pane.Alert(
        pn.bind(lambda text: text, state.param.setup_note),
        alert_type="info",
        visible=pn.bind(lambda text: bool(text), state.param.setup_note),
        sizing_mode="stretch_width",
    )
    general = pn.Column(
        "### Problem",
        setup_pane,
        problem_select,
        pn.Row(
            pn.widgets.IntInput.from_param(state.param.n_var, name="Variables"),
            pn.widgets.IntInput.from_param(
                state.param.n_obj, name="Objectives", disabled=pn.bind(lambda adjustable: not adjustable, state.param.n_obj_adjustable)
            ),
        ),
        pn.pane.Markdown(pn.bind(lambda text: text, state.param.problem_info), sizing_mode="stretch_width"),
        pn.widgets.IntInput.from_param(state.param.max_evaluations, name="Max evaluations"),
        pn.widgets.IntInput.from_param(state.param.seed, name="Seed"),
        pn.layout.Divider(),
        "### Algorithm",
        state.configurator.view(),
    )

    run_btn = pn.widgets.Button(name="Run Experiment", button_type="primary", disabled=pn.bind(lambda r: r, state.param.running))
    run_btn.on_click(state.run_experiment)
    stop_btn = pn.widgets.Button(name="Stop", button_type="danger", disabled=pn.bind(lambda r: not r, state.param.running))
    stop_btn.on_click(state.stop)
    progress = pn.indicators.Progress(
        value=pn.bind(lambda f: int(round(100 * f)), state.param.progress), max=100, sizing_mode="stretch_width"
    )
    status_pane = pn.pane.Alert(pn.bind(lambda s: s, state.param.status), alert_type="info")
    note = pn.pane.Markdown(
        "*The run executes in a background thread of the Studio server and the front updates every second. "
        "Stop ends it after the current generation; for long or many runs, use Batch Comparison.*",
        sizing_mode="stretch_width",
    )

    gen_slider = pn.widgets.IntSlider.from_param(
        state.param.current_gen,
        name="Recorded front (the title shows its generation)",
        start=0,
        end=pn.bind(lambda h: max(len(h) - 1, 1), state.param.generation_history),
    )

    config_col = pn.Column(
        "## Run Experiment",
        general,
        pn.layout.Divider(),
        pn.Row(run_btn, stop_btn),
        progress,
        status_pane,
        note,
        sizing_mode="stretch_width",
        max_width=460,
    )
    viz_col = pn.Column("## Visualization", state.generation_view, gen_slider, sizing_mode="stretch_both")
    return pn.Column(pn.Row(config_col, viz_col, sizing_mode="stretch_both"), sizing_mode="stretch_both")
