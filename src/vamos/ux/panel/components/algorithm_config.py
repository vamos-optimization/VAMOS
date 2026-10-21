"""Algorithm configuration controls generated from the VAMOS tuning spaces."""

from __future__ import annotations

import math
from typing import Any

import panel as pn

from vamos.ux.studio.algorithm_form import (
    VAMOS_DEFAULT_CHOICE,
    AlgorithmForm,
    FormParameter,
    ResolvedConfiguration,
    algorithm_label,
    algorithms_for_encoding,
    build_algorithm_form,
    choice_label,
    default_algorithm_config,
    format_config_value,
    population_is_derived,
)

DEFAULTS_MODE = "VAMOS defaults"
CUSTOM_MODE = "Customize"
CHANGED_MARK = " ✎"


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _step(low: float, high: float) -> float:
    span = high - low
    if span <= 0:
        return 0.1
    return float(10 ** math.floor(math.log10(span / 100.0)))


class AlgorithmConfigurator:
    """Algorithm, population size and configuration controls for one problem.

    In *VAMOS defaults* mode the run uses the canonical default configuration.
    In *Customize* mode the controls come from the algorithm's tuning space and
    only the fields the user changes are applied on top of that default.
    """

    def __init__(self, *, algorithm: str = "nsgaii", pop_size: int = 100) -> None:
        self.algorithm = pn.widgets.Select(name="Algorithm", options={}, sizing_mode="stretch_width")
        self.pop_size = pn.widgets.IntInput(name="Population size", value=pop_size, start=4, end=100000, sizing_mode="stretch_width")
        self.mode = pn.widgets.RadioButtonGroup(options=[DEFAULTS_MODE, CUSTOM_MODE], value=DEFAULTS_MODE, button_type="light")
        self._population_note = pn.pane.Markdown("", visible=False, sizing_mode="stretch_width")
        self._fields = pn.Column(sizing_mode="stretch_width", visible=False)
        self._message = pn.pane.Alert("", alert_type="warning", visible=False, sizing_mode="stretch_width")
        self._summary = pn.pane.Markdown("", sizing_mode="stretch_width")
        self._details = pn.pane.Markdown("", sizing_mode="stretch_width")
        self._form: AlgorithmForm | None = None
        self._inputs: dict[str, Any] = {}
        self._rows: dict[str, Any] = {}
        self._encoding = "real"
        self._n_var: int | None = None
        self._n_obj = 2
        self._preferred = algorithm
        self.algorithm.param.watch(lambda _event: self._rebuild(), "value")
        self.pop_size.param.watch(lambda _event: self._rebuild(), "value")
        self.mode.param.watch(lambda _event: self._refresh(), "value")
        self.set_problem(encoding="real", n_var=None, n_obj=2)

    # ---- public API ----

    def view(self) -> pn.Column:
        return pn.Column(
            self.algorithm,
            self.pop_size,
            self._population_note,
            self.mode,
            self._fields,
            self._message,
            self._summary,
            pn.Card(self._details, title="Configuration to run", collapsed=True, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    @property
    def selected_algorithm(self) -> str:
        return str(self.algorithm.value or self._preferred)

    @property
    def form(self) -> AlgorithmForm | None:
        return self._form

    def set_problem(self, *, encoding: str | None, n_var: int | None, n_obj: int) -> None:
        """Offer the algorithms that support ``encoding`` and rebuild the form."""
        self._encoding = str(encoding or "real")
        self._n_var = n_var
        self._n_obj = int(n_obj)
        names = algorithms_for_encoding(self._encoding)
        current = self.algorithm.value or self._preferred
        options = {algorithm_label(name): name for name in names}
        value = current if current in names else (names[0] if names else None)
        with pn.io.hold():
            self.algorithm.options = options
            if self.algorithm.value != value:
                self.algorithm.value = value  # triggers a rebuild
                return
        self._rebuild()

    def values(self) -> dict[str, object]:
        return {name: widget.value for name, widget in self._inputs.items()}

    def set_values(self, values: dict[str, object]) -> None:
        for name, value in values.items():
            if name in self._inputs:
                self._inputs[name].value = value

    def resolve(self) -> ResolvedConfiguration:
        """Return the configuration the next run will use."""
        if self.mode.value == CUSTOM_MODE and self._form is not None:
            return self._form.resolve(self.values())
        if self._form is not None:
            return self._form.resolve()
        config = default_algorithm_config(
            self.selected_algorithm,
            encoding=self._encoding,
            n_var=self._n_var,
            n_obj=self._n_obj,
            pop_size=int(self.pop_size.value),
        )
        return ResolvedConfiguration(config=config)

    def effective_pop_size(self) -> int:
        pop = self.resolve().config.to_dict().get("pop_size")
        return int(pop) if isinstance(pop, int) else int(self.pop_size.value)

    # ---- form construction ----

    def _rebuild(self) -> None:
        algorithm = self.selected_algorithm
        same_setup = self._form is not None and self._form.algorithm == algorithm and self._form.encoding == self._encoding
        previous = self.values() if same_setup else {}
        derived = population_is_derived(algorithm, self._n_obj)
        self.pop_size.disabled = derived
        try:
            self._form = build_algorithm_form(
                algorithm,
                encoding=self._encoding,
                n_var=self._n_var,
                n_obj=self._n_obj,
                pop_size=int(self.pop_size.value),
            )
        except Exception as exc:
            self._form = None
            self._show_message(f"Cannot build the configuration of {algorithm_label(algorithm)}: {exc}")
        self._inputs = {}
        self._rows = {}
        if self._form is not None:
            for parameter in self._form.parameters:
                widget = self._make_input(parameter)
                if parameter.name in previous and self._accepts(parameter, previous[parameter.name]):
                    widget.value = previous[parameter.name]
                widget.param.watch(lambda _event: self._refresh(), "value")
                self._inputs[parameter.name] = widget
                self._rows[parameter.name] = pn.Row(widget, margin=(0, 0, 0, 18 * parameter.depth), sizing_mode="stretch_width")
        self._fields.objects = list(self._rows.values())
        self._refresh()

    def _make_input(self, parameter: FormParameter) -> Any:
        description = parameter.note or None
        if parameter.kind == "boolean":
            return pn.widgets.Checkbox(name=parameter.label, value=bool(parameter.start))
        if parameter.kind == "categorical":
            options = {choice_label(parameter, choice): choice for choice in parameter.choices}
            return pn.widgets.Select(
                name=parameter.label, options=options, value=parameter.start, description=description, sizing_mode="stretch_width"
            )
        low = parameter.low if parameter.low is not None else -math.inf
        high = parameter.high if parameter.high is not None else math.inf
        hint = f"Range {low:g} to {high:g}."
        description = f"{hint} {parameter.note}".strip()
        if parameter.kind == "int":
            return pn.widgets.IntInput(
                name=parameter.label,
                value=int(_as_float(parameter.start)),
                start=int(low),
                end=int(high),
                description=description,
                sizing_mode="stretch_width",
            )
        return pn.widgets.FloatInput(
            name=parameter.label,
            value=_as_float(parameter.start),
            start=low,
            end=high,
            step=_step(low, high),
            description=description,
            sizing_mode="stretch_width",
        )

    @staticmethod
    def _accepts(parameter: FormParameter, value: object) -> bool:
        if parameter.kind == "categorical":
            return value in parameter.choices
        if parameter.kind == "boolean":
            return isinstance(value, bool)
        if not isinstance(value, (int, float)) or parameter.low is None or parameter.high is None:
            return False
        return parameter.low <= float(value) <= parameter.high

    # ---- reactive refresh ----

    def _refresh(self) -> None:
        custom = self.mode.value == CUSTOM_MODE
        self._fields.visible = custom and self._form is not None
        self._update_population_note()
        if self._form is None:
            self._summary.object = ""
            self._details.object = ""
            return
        values = self.values()
        active = set(self._form.active_names(values))
        changed = set(self._form.changed_names(values)) if custom else set()
        for parameter in self._form.parameters:
            self._rows[parameter.name].visible = parameter.name in active
            self._inputs[parameter.name].name = parameter.label + (CHANGED_MARK if parameter.name in changed else "")
        try:
            resolved = self.resolve()
        except Exception as exc:
            self._show_message(f"This configuration cannot be built: {exc}")
            self._summary.object = ""
            self._details.object = ""
            return
        self._show_message(self._warnings(resolved, self._unsupported_defaults(values if custom else None)))
        self._summary.object = self._summary_text(resolved, custom)
        self._details.object = f"```json\n{resolved.as_json()}\n```"

    def _update_population_note(self) -> None:
        algorithm = self.selected_algorithm
        if not population_is_derived(algorithm, self._n_obj) or self._form is None:
            self._population_note.visible = False
            return
        pop = self._form.baseline.to_dict().get("pop_size")
        self._population_note.object = (
            f"*{algorithm_label(algorithm)} sets its population to the number of reference directions "
            f"or weight vectors: **{pop}** for {self._n_obj} objectives.*"
        )
        self._population_note.visible = True

    def _unsupported_defaults(self, values: dict[str, object] | None) -> list[FormParameter]:
        """Default operators that are not among the encoding's operators and are still selected."""
        if self._form is None:
            return []
        return [
            parameter
            for parameter in self._form.parameters
            if parameter.name in {"crossover", "mutation"}
            and parameter.start == VAMOS_DEFAULT_CHOICE
            and (values is None or values.get(parameter.name) == VAMOS_DEFAULT_CHOICE)
        ]

    def _warnings(self, resolved: ResolvedConfiguration, unsupported: list[FormParameter]) -> str:
        messages: list[str] = []
        if unsupported:
            names = ", ".join(f"{parameter.name} `{format_config_value(parameter.default_value)}`" for parameter in unsupported)
            verb = "is" if len(unsupported) == 1 else "are"
            messages.append(
                f"The VAMOS default {names} {verb} not among the {self._encoding} operators that the tuning space "
                f"offers for {algorithm_label(self.selected_algorithm)}. Choose *Customize* and select {self._encoding} operators."
            )
        if resolved.unapplied:
            names = ", ".join(f"`{name}`" for name in resolved.unapplied)
            messages.append(f"These changes have no effect on the configuration VAMOS builds: {names}.")
        return " ".join(messages)

    def _show_message(self, text: str) -> None:
        self._message.object = text
        self._message.visible = bool(text)

    @staticmethod
    def _summary_text(resolved: ResolvedConfiguration, custom: bool) -> str:
        if not custom:
            return "Runs use the **VAMOS default configuration**. Choose *Customize* to change operators and parameters."
        if not resolved.differences:
            return "No changes: runs use the **VAMOS default configuration**."
        lines = ["**Changed from the VAMOS default:**"]
        for difference in resolved.differences:
            lines.append(f"- `{difference.field}`: {format_config_value(difference.default)} → **{format_config_value(difference.value)}**")
        return "\n".join(lines)


__all__ = ["AlgorithmConfigurator", "CUSTOM_MODE", "DEFAULTS_MODE"]
