"""Problem Builder page — define a problem, configure algorithm, run & view results."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import panel as pn
import param

from vamos.ux.panel.components.algorithm_config import AlgorithmConfigurator
from vamos.ux.panel.components.background import BackgroundRun
from vamos.ux.panel.components.plots import front_figure
from vamos.ux.panel.pages.problem_builder_ai import build_ai_generation_tab, build_trusted_code_controls
from vamos.ux.studio.algorithm_form import algorithm_label
from vamos.ux.studio.problem_builder_backend import (
    TRUSTED_LOCAL_CODE_WARNING,
    compile_constraint_function,
    compile_objective_function,
    example_objectives,
    generate_script,
    parse_bounds_text,
    run_preview_optimization,
)
from vamos.ux.studio.problem_builder_templates import _DEFAULT_TEMPLATE
from vamos.ux.studio.services import llm_generate_problem_code

PREVIEW_TIME_LIMIT_SECONDS = 10.0


class ProblemBuilderState(param.Parameterized):
    """Reactive state for the unified Problem Builder."""

    # ---- Problem Definition ----
    template_name = param.Selector(
        default=_DEFAULT_TEMPLATE,
        objects=list(example_objectives().keys()),
        doc="Selected problem template.",
    )
    objective_code = param.String(default="", doc="Python code for objectives.")
    n_var = param.Integer(default=5, bounds=(1, 100), doc="Number of decision variables.")
    n_obj = param.Integer(default=2, bounds=(2, 10), doc="Number of objectives.")
    bounds_text = param.String(default="0.0, 1.0", doc="Bounds: one line per var, or single line for all.")
    constraint_code = param.String(default="", doc="Optional constraint code.")
    n_constraints = param.Integer(default=1, bounds=(1, 50), doc="Number of values returned by the constraint code.")
    trusted_local_code = param.Boolean(
        default=False,
        doc="Explicit acknowledgement that the currently displayed Python code is trusted.",
    )

    # ---- Run Setup ----
    max_gen = param.Integer(default=250, bounds=(1, 10000), doc="Generations: the budget is population size x generations.")
    seed = param.Integer(default=0, bounds=(0, 2**31 - 1), doc="Random seed.")
    time_limit = param.Number(default=120.0, bounds=(10.0, 3600.0), doc="Wall-clock limit of a full experiment, in seconds.")

    # ---- AI Assistant ----
    ai_provider = param.Selector(
        default="gemini",
        objects=["gemini", "openai", "anthropic"],
        doc="LLM provider for code generation.",
    )
    ai_api_key = param.String(default="", doc="API key (optional if set via env var).")
    ai_description = param.String(default="", doc="Natural language problem description.")
    ai_status = param.String(default="", precedence=-1)

    # ---- Outputs ----
    preview_plot = param.Parameter(default=None, precedence=-1)
    generated_script = param.String(default="", precedence=-1)
    status_text = param.String(default="", precedence=-1)
    running = param.Boolean(default=False, precedence=-1)
    generation_history = param.List(default=[], precedence=-1)
    current_gen = param.Integer(default=0, bounds=(0, None), precedence=-1)
    result_F = param.Array(default=None, precedence=-1)

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.configurator = AlgorithmConfigurator()
        self._cancel: threading.Event | None = None
        self._mode = "preview"
        self._started = time.monotonic()
        self._runner: BackgroundRun[dict[str, Any]] = BackgroundRun(on_tick=self._tick, on_done=self._done)
        self._apply_template()
        self._sync_configurator()

    @property
    def algorithm(self) -> str:
        return self.configurator.selected_algorithm

    @param.depends("template_name", watch=True)
    def _apply_template(self) -> None:
        templates = example_objectives()
        t = templates.get(self.template_name, {})
        self.objective_code = t.get("code", "")
        self.n_var = int(t.get("n_var", 2))
        self.n_obj = int(t.get("n_obj", 2))
        self.bounds_text = t.get("bounds", "0.0, 1.0")
        self.constraint_code = t.get("constraint_code", "")
        self.n_constraints = int(t.get("n_constraints", 1))

    @param.depends("n_var", "n_obj", watch=True)
    def _sync_configurator(self) -> None:
        if hasattr(self, "configurator"):
            self.configurator.set_problem(encoding="real", n_var=self.n_var, n_obj=self.n_obj)

    @param.depends("objective_code", "constraint_code", watch=True)
    def _invalidate_code_trust(self) -> None:
        self.trusted_local_code = False

    def _parse_bounds(self) -> list[tuple[float, float]] | None:
        result = parse_bounds_text(self.bounds_text, self.n_var)
        if isinstance(result, str):
            self.status_text = f"Bounds error: {result}"
            return None
        return result

    def _constraint_count(self) -> int:
        return int(self.n_constraints) if self.constraint_code.strip() else 0

    # ---- Generation viewer ----

    @pn.depends("current_gen", "generation_history")
    def generation_view(self) -> pn.pane.Plotly | pn.pane.Markdown:
        if not self.generation_history:
            return pn.pane.Markdown("*No generations recorded yet.*")
        idx = min(self.current_gen, len(self.generation_history) - 1)
        return pn.pane.Plotly(front_figure(self.generation_history[idx], title=f"Generation {idx}"), sizing_mode="stretch_width")

    # ---- Actions ----

    def _prepare_run(self, *, record_history: bool, timeout_seconds: float) -> Callable[[], dict[str, Any]] | None:
        """Compile and check the reviewed code now; return the child-process run, or ``None``."""
        if not self.trusted_local_code:
            self.status_text = f"Execution disabled. {TRUSTED_LOCAL_CODE_WARNING}"
            return None
        self.status_text = "Compiling..."
        try:
            fn = compile_objective_function(self.objective_code, trusted_local_code=True)
        except Exception as exc:
            self.status_text = f"Compilation error: {exc}"
            return None

        bounds = self._parse_bounds()
        if bounds is None:
            return None

        constraints = None
        n_constraints = self._constraint_count()
        if n_constraints:
            try:
                constraints = compile_constraint_function(self.constraint_code, trusted_local_code=True)
            except Exception as exc:
                self.status_text = f"Constraint error: {exc}"
                return None

        try:
            resolved = self.configurator.resolve()
            pop_size = self.configurator.effective_pop_size()
        except Exception as exc:
            self.status_text = f"Configuration error: {exc}"
            return None
        cancel = threading.Event()
        self._cancel = cancel
        settings: dict[str, Any] = {
            "n_var": self.n_var,
            "n_obj": self.n_obj,
            "bounds": bounds,
            "algorithm": self.algorithm,
            "pop_size": pop_size,
            "budget": pop_size * self.max_gen,
            "seed": self.seed,
            "constraints": constraints,
            "n_constraints": n_constraints,
            "objective_code": self.objective_code,
            "constraint_code": self.constraint_code,
            "timeout_seconds": timeout_seconds,
            "trusted_local_code": True,
            "algorithm_config": resolved.config,
            "record_history": record_history,
            "cancel": cancel,
        }
        return lambda: run_preview_optimization(fn, **settings)

    def _start(self, mode: str, *, record_history: bool, timeout_seconds: float) -> None:
        if self.running:
            return
        work = self._prepare_run(record_history=record_history, timeout_seconds=timeout_seconds)
        if work is None:
            if mode == "preview":
                self.preview_plot = None
            return
        self._mode, self._started = mode, time.monotonic()
        if mode == "full":
            self.generation_history, self.current_gen = [], 0
        self.running = True
        self.status_text = f"Running with {algorithm_label(self.algorithm)}…"
        self._runner.start(work)

    def run_preview(self, event: Any = None) -> None:
        """Compile code, run a short optimization in a child process, update preview_plot."""
        self._start("preview", record_history=False, timeout_seconds=PREVIEW_TIME_LIMIT_SECONDS)

    def run_full_experiment(self, event: Any = None) -> None:
        """Run the defined problem with the full budget in a child process and collect the generation history."""
        self._start("full", record_history=True, timeout_seconds=float(self.time_limit))

    def stop(self, event: Any = None) -> None:
        """End the child process of the running preview or experiment."""
        if self.running and self._cancel is not None:
            self._cancel.set()
            self.status_text = "Stopping…"

    def _tick(self) -> None:
        if self.running and not (self._cancel is not None and self._cancel.is_set()):
            self.status_text = f"Running with {algorithm_label(self.algorithm)}… {time.monotonic() - self._started:.0f} s"

    def _done(self, result: dict[str, Any] | Exception) -> None:
        self.running = False
        if isinstance(result, Exception):
            self.status_text = "Stopped." if isinstance(result, InterruptedError) else f"Error: {result}"
            if self._mode == "preview":
                self.preview_plot = None
            return
        F_arr = np.asarray(result["F"])
        if self._mode == "preview":
            self.preview_plot = front_figure(F_arr, title="Pareto Front Preview")
            self.status_text = f"Preview complete — {F_arr.shape[0]} solutions found with {algorithm_label(self.algorithm)}."
            return
        history = list(result.get("history") or [result["F"]])
        self.generation_history = history
        self.current_gen = len(history) - 1
        self.result_F = F_arr
        self.status_text = f"Done — {len(history)} generations recorded with {algorithm_label(self.algorithm)}."

    def do_generate_script(self, event: Any = None) -> None:
        """Generate a standalone Python script for the configured problem."""
        bounds = self._parse_bounds()
        if bounds is None:
            return
        try:
            resolved = self.configurator.resolve()
            pop_size = self.configurator.effective_pop_size()
            customized = bool(resolved.differences)
            self.generated_script = generate_script(
                self.objective_code,
                name=self.template_name,
                n_var=self.n_var,
                n_obj=self.n_obj,
                bounds=bounds,
                algorithm=self.algorithm,
                budget=pop_size * self.max_gen,
                constraint_code=self.constraint_code,
                n_constraints=self._constraint_count(),
                pop_size=pop_size,
                algorithm_config=dict(resolved.config.to_dict()) if customized else None,
                config_class=type(resolved.config).__name__ if customized else None,
            )
            self.status_text = "Script generated with the customized configuration." if customized else "Script generated."
        except Exception as exc:
            self.status_text = f"Script generation error: {exc}"

    def ai_generate(self, event: Any = None) -> None:
        """Use an LLM to generate VAMOS code from natural language."""
        if not self.ai_description.strip():
            self.ai_status = "Please enter a problem description."
            return
        self.ai_status = f"Generating with {self.ai_provider}..."
        try:
            result = llm_generate_problem_code(self.ai_description, provider=self.ai_provider, api_key=self.ai_api_key)
            self.objective_code = result["objective_code"]
            self.constraint_code = result.get("constraint_code", "")
            self.n_var = int(result["n_var"])
            self.n_obj = int(result["n_obj"])
            self.bounds_text = str(result["bounds"])
            self.trusted_local_code = False
            self.ai_status = "Code generated but not executed. Review it in Problem Definition, then opt in explicitly."
        except Exception as exc:
            self.ai_status = f"Error: {exc}"


def render_problem_builder(on_exit: Callable[[Callable[[], None]], None] | None = None) -> pn.Column:
    """Build and return the Problem Builder page layout with tabs."""
    state = ProblemBuilderState()
    if on_exit is not None:
        on_exit(state.stop)  # Leaving the page stops its run.

    template_select = pn.widgets.Select.from_param(state.param.template_name, name="Template")
    code_editor = pn.widgets.CodeEditor.from_param(
        state.param.objective_code,
        language="python",
        theme="chrome",
        height=200,
        name="Objective code",
    )
    var_controls = pn.Column(
        pn.widgets.IntInput.from_param(state.param.n_var, name="Variables (n_var)"),
        pn.widgets.IntInput.from_param(state.param.n_obj, name="Objectives (n_obj)"),
        pn.widgets.TextAreaInput.from_param(state.param.bounds_text, name="Bounds (lo, hi per line)"),
    )
    constraint_editor = pn.widgets.CodeEditor.from_param(
        state.param.constraint_code,
        language="python",
        theme="chrome",
        height=100,
        name="Constraint code (optional)",
    )
    constraint_count = pn.widgets.IntInput.from_param(state.param.n_constraints, name="Constraint values returned (g(x) <= 0 is feasible)")
    trusted_code_warning, trusted_code_confirmation = build_trusted_code_controls(state)

    run_setup = pn.Column(
        pn.widgets.IntInput.from_param(state.param.max_gen, name="Generations (budget = population x generations)"),
        pn.widgets.IntInput.from_param(state.param.seed, name="Seed"),
        pn.widgets.FloatInput.from_param(state.param.time_limit, name="Time limit of a full experiment (s)"),
    )

    preview_btn = pn.widgets.Button(name="Quick Preview", button_type="primary", disabled=pn.bind(lambda r: r, state.param.running))
    preview_btn.on_click(state.run_preview)
    stop_btn = pn.widgets.Button(name="Stop", button_type="danger", disabled=pn.bind(lambda r: not r, state.param.running))
    stop_btn.on_click(state.stop)
    run_btn = pn.widgets.Button(
        name="Run Full Experiment",
        button_type="success",
        disabled=pn.bind(lambda r: r, state.param.running),
    )
    run_btn.on_click(state.run_full_experiment)
    generate_btn = pn.widgets.Button(name="Generate Script", button_type="warning")
    generate_btn.on_click(state.do_generate_script)

    status_pane = pn.pane.Alert(pn.bind(lambda t: t or "Ready.", state.param.status_text), alert_type="info")
    plot_pane = pn.pane.Plotly(pn.bind(lambda p: p, state.param.preview_plot), sizing_mode="stretch_both", min_height=400)
    gen_slider = pn.widgets.IntSlider.from_param(
        state.param.current_gen,
        name="Generation",
        start=0,
        end=pn.bind(lambda h: max(len(h) - 1, 1), state.param.generation_history),
    )
    script_pane = pn.pane.Str(pn.bind(lambda s: s, state.param.generated_script), sizing_mode="stretch_width")

    tab_ai = build_ai_generation_tab(state)
    tab_problem = pn.Column(
        "### Template",
        template_select,
        pn.layout.Divider(),
        "### Objective Function",
        code_editor,
        pn.layout.Divider(),
        "### Variables & Bounds",
        var_controls,
        pn.layout.Divider(),
        "### Constraints (optional)",
        constraint_editor,
        constraint_count,
        pn.layout.Divider(),
        trusted_code_warning,
        trusted_code_confirmation,
        sizing_mode="stretch_width",
    )
    tab_algorithm = pn.Column(
        "### Algorithm",
        state.configurator.view(),
        pn.layout.Divider(),
        "### Run",
        run_setup,
        sizing_mode="stretch_width",
    )
    tab_results = pn.Column(
        pn.Row(preview_btn, run_btn, stop_btn, generate_btn),
        status_pane,
        pn.pane.Markdown(
            f"*Quick Preview stops after {PREVIEW_TIME_LIMIT_SECONDS:.0f} s; Run Full Experiment uses the time limit "
            "from Algorithm & Run and records every generation. Both run the reviewed code in a child process.*"
        ),
        pn.layout.Divider(),
        "### Pareto Front Preview",
        plot_pane,
        pn.layout.Divider(),
        "### Generation History",
        state.generation_view,
        gen_slider,
        pn.layout.Divider(),
        "### Generated Script",
        script_pane,
        sizing_mode="stretch_both",
    )

    tabs = pn.Tabs(
        ("AI Assistant", tab_ai),
        ("Problem Definition", tab_problem),
        ("Algorithm & Run", tab_algorithm),
        ("Results", tab_results),
        sizing_mode="stretch_both",
        dynamic=False,
    )
    return pn.Column("## Problem Builder", pn.layout.Divider(), tabs, sizing_mode="stretch_both")
