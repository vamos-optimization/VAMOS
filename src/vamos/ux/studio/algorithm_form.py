"""Panel-free algorithm configuration forms for VAMOS Studio.

A form is the grammar of the algorithm's tuning space for the problem encoding
(:mod:`vamos.engine.tuning.grammar`): it shows the grammar's tree, starts from
its defaults, which are the VAMOS default configuration, and edits each
parameter within the one range that tuning uses. Studio therefore offers
exactly the operators and conditional parameters that VAMOS can build.
Resolving a form applies only the configuration fields that the user changed
on top of the default configuration, so untouched settings keep their exact
default values.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal, cast

from vamos.engine.algorithm.config.base import _SerializableConfig
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.engine.tuning.grammar import Parameter, ParameterSpace
from vamos.engine.tuning.racing.bridge_assignments import config_from_assignment
from vamos.engine.tuning.racing.parameter_names import bridge_parameter_names
from vamos.ux.studio.algorithm_catalog import (
    ALGORITHM_LABELS,
    algorithm_label,
    algorithms_for_encoding,
    config_fields_for,
    config_space_for,
    default_algorithm_config,
    grammar_for,
    population_is_derived,
    studio_algorithms,
)

ParameterKind = Literal["real", "int", "categorical", "boolean"]


@dataclass(frozen=True)
class FormParameter:
    """One configurable parameter of an algorithm form."""

    name: str
    kind: ParameterKind
    start: object
    depth: int = 0
    choices: tuple[object, ...] = ()
    low: float | None = None
    high: float | None = None
    log: bool = False
    note: str = ""
    title: str | None = None

    @property
    def label(self) -> str:
        """The grammar's label for the parameter, or its name in words."""
        return self.title or self.name.replace("_", " ").capitalize()


@dataclass(frozen=True)
class ConfigDifference:
    """A configuration field whose resolved value differs from the VAMOS default."""

    field: str
    default: object
    value: object


@dataclass(frozen=True)
class ResolvedConfiguration:
    """The configuration Studio will run, and how it departs from the default."""

    config: AlgorithmConfigProtocol
    changed: tuple[str, ...] = ()
    differences: tuple[ConfigDifference, ...] = ()
    unapplied: tuple[str, ...] = ()

    def as_json(self) -> str:
        return json.dumps(self.config.to_dict(), indent=2, sort_keys=True, default=str)


@dataclass(frozen=True)
class AlgorithmForm:
    """A configuration form for one algorithm, problem encoding and dimensions."""

    algorithm: str
    encoding: str
    n_obj: int
    pop_size: int | None
    grammar: ParameterSpace
    parameters: tuple[FormParameter, ...]
    baseline: AlgorithmConfigProtocol

    def start_values(self) -> dict[str, object]:
        return {parameter.name: parameter.start for parameter in self.parameters}

    def active_names(self, values: Mapping[str, object]) -> tuple[str, ...]:
        """Return the active parameters, in form order, for the current values: the derivation of the grammar."""
        active = {parameter.name for parameter in self.grammar.active_parameters(values)}
        return tuple(parameter.name for parameter in self.parameters if parameter.name in active and parameter.name in values)

    def changed_names(self, values: Mapping[str, object]) -> tuple[str, ...]:
        starts = self.start_values()
        return tuple(name for name in self.active_names(values) if not _same_value(values[name], starts[name]))

    def resolve(self, values: Mapping[str, object] | None = None) -> ResolvedConfiguration:
        """Return the default configuration with the user's changes applied."""
        if values is None:
            return ResolvedConfiguration(config=self.baseline)
        changed = self.changed_names(values)
        if not changed:
            return ResolvedConfiguration(config=self.baseline)
        assignment = self._assignment(values)
        baseline_pop = self.baseline.to_dict().get("pop_size")
        assignment["pop_size"] = baseline_pop if isinstance(baseline_pop, int) else (self.pop_size or 100)
        if self.algorithm == "rvea":
            assignment["n_obj"] = self.n_obj
        tuned = config_from_assignment(self.grammar.variant, assignment).to_dict()
        default_dict = self.baseline.to_dict()
        merged = dict(default_dict)
        unapplied: list[str] = []
        for name in changed:
            fields = [
                config_field for config_field in config_fields_for(_bridge_name(name)) if config_field in merged and config_field in tuned
            ]
            fields = fields[:1] if fields[:1] == ["mutation_prob_factor"] else fields  # the factor's field, where there is one
            # A change that leaves every field it determines untouched has no effect on the run.
            if not any(_canonical(tuned[config_field]) != _canonical(default_dict[config_field]) for config_field in fields):
                unapplied.append(name)
            for config_field in fields:
                merged[config_field] = tuned[config_field]
        baseline_type = type(self.baseline)
        if not issubclass(baseline_type, _SerializableConfig):
            raise TypeError(f"{baseline_type.__name__} cannot be rebuilt from a configuration dictionary.")
        config = cast(AlgorithmConfigProtocol, baseline_type.from_dict(merged))
        return ResolvedConfiguration(
            config=config,
            changed=changed,
            differences=configuration_differences(self.baseline, config),
            unapplied=tuple(unapplied),
        )

    def _assignment(self, values: Mapping[str, object]) -> dict[str, object]:
        """The active values, for the tuning bridge."""
        return {name: values[name] for name in self.active_names(values)}


def _bridge_name(name: str) -> str:
    """The name the tuning bridges and the lookups below use for a space parameter (spaces use Evolver's names)."""
    return bridge_parameter_names(name)[0]


def configuration_differences(baseline: AlgorithmConfigProtocol, config: AlgorithmConfigProtocol) -> tuple[ConfigDifference, ...]:
    default_dict = baseline.to_dict()
    resolved = config.to_dict()
    return tuple(
        ConfigDifference(field=name, default=default_dict.get(name), value=resolved.get(name))
        for name in sorted(set(default_dict) | set(resolved))
        if _canonical(default_dict.get(name)) != _canonical(resolved.get(name))
    )


def build_algorithm_form(
    algorithm: str,
    *,
    encoding: str | None,
    n_var: int | None,
    n_obj: int,
    pop_size: int | None,
) -> AlgorithmForm | None:
    """Build the form of ``algorithm`` for a problem, or ``None`` without a tuning space."""
    grammar = grammar_for(algorithm, encoding)
    if grammar is None:
        return None
    baseline = default_algorithm_config(algorithm, encoding=encoding, n_var=n_var, n_obj=n_obj, pop_size=pop_size)
    form = AlgorithmForm(
        algorithm=algorithm,
        encoding=str(encoding or "real"),
        n_obj=int(n_obj),
        pop_size=pop_size,
        grammar=grammar,
        parameters=_tree_parameters(grammar),
        baseline=baseline,
    )
    form = replace(form, parameters=_population_starts(form))
    return replace(form, parameters=_effective_parameters(form))


def _tree_parameters(grammar: ParameterSpace) -> tuple[FormParameter, ...]:
    """Every parameter of the grammar in depth-first order, indented by its depth in the tree."""
    parameters: list[FormParameter] = []

    def visit(nodes: tuple[Parameter, ...], depth: int) -> None:
        for node in nodes:
            parameters.append(_form_parameter(node, depth))
            visit(node.global_sub_parameters, depth + 1)
            for production in node.productions:
                visit(production, depth + 1)

    visit(grammar.parameters, 0)
    return tuple(parameters)


def _form_parameter(node: Parameter, depth: int) -> FormParameter:
    """A form control that starts at the grammar's default and edits within its one range."""
    note, title = node.description or "", node.label
    if node.type == "categorical":
        return FormParameter(node.name, "categorical", node.default, depth, choices=node.values, note=note, title=title)
    if node.type == "boolean":
        return FormParameter(node.name, "boolean", node.default, depth, note=note, title=title)
    assert node.range is not None
    low, high = node.range
    kind: ParameterKind = "int" if node.type == "int" else "real"
    return FormParameter(node.name, kind, node.default, depth, low=float(low), high=float(high), log=node.log, note=note, title=title)


def _population_starts(form: AlgorithmForm) -> tuple[FormParameter, ...]:
    """Start a population parameter at the run's default where the default configuration follows the population.

    The grammar's defaults are the default configuration of 100 individuals: SPEA2's and SMPSO's
    ``archive_size`` is 100 there, and the population in any other run. A form for another
    population starts such a parameter at the baseline's value, so the untouched form shows what it
    runs and entering the grammar's default is a change. The control keeps the grammar's one range,
    which the run's value may lie outside of.
    """
    baseline = form.baseline.to_dict()
    parameters: list[FormParameter] = []
    for parameter in form.parameters:
        node = form.grammar.parameter(parameter.name)
        fields = config_fields_for(_bridge_name(parameter.name))
        value = baseline.get(fields[0]) if len(fields) == 1 else None
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        if node.role == "population" and parameter.kind in {"int", "real"} and numeric and not _same_value(value, parameter.start):
            start = int(cast(int, value)) if parameter.kind == "int" else float(cast(float, value))
            note = f"VAMOS's default follows the population: {start} here, {parameter.start} for 100 individuals."
            parameter = replace(parameter, start=start, note=" ".join(text for text in (note, parameter.note) if text))
        parameters.append(parameter)
    return tuple(parameters)


def _effective_parameters(form: AlgorithmForm) -> tuple[FormParameter, ...]:
    """Drop parameters that the algorithm's configuration builder ignores.

    A control that changes nothing would silently mislead; each parameter
    active in the starting configuration is probed with alternative values.
    """
    start = form.start_values()
    active = set(form.active_names(start))
    return tuple(parameter for parameter in form.parameters if parameter.name not in active or _has_effect(form, start, parameter))


def _has_effect(form: AlgorithmForm, start: Mapping[str, object], parameter: FormParameter) -> bool:
    for alternative in _alternatives(parameter):
        values = {**start, parameter.name: alternative}
        try:
            resolved = form.resolve(values)
        except Exception:
            return True  # Keep the control; resolving reports the error to the user.
        if parameter.name in resolved.changed and parameter.name not in resolved.unapplied:
            return True
    return False


def _alternatives(parameter: FormParameter) -> list[object]:
    if parameter.kind == "boolean":
        return [not parameter.start]
    if parameter.kind == "categorical":
        return [choice for choice in parameter.choices if choice != parameter.start][:2]
    bounds = [bound for bound in (parameter.high, parameter.low) if bound is not None]
    values: list[object] = [int(bound) if parameter.kind == "int" else bound for bound in bounds]
    return [value for value in values if not _same_value(value, parameter.start)]


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-15)
    return left == right


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def format_config_value(value: object) -> str:
    """Render one configuration value as ``name(key=value, ...)`` when it is an operator."""
    if isinstance(value, (tuple, list)) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], dict):
        params = ", ".join(f"{key}={_short(item)}" for key, item in value[1].items())
        return f"{value[0]}({params})" if params else value[0]
    if isinstance(value, dict):
        return ", ".join(f"{key}={_short(item)}" for key, item in value.items() if item is not None) or "{}"
    return _short(value)


def choice_label(parameter: FormParameter, choice: object) -> str:
    """How a form shows one choice of a categorical parameter, marking the default."""
    if choice == parameter.start:
        return f"{choice} (VAMOS default)"
    return str(choice)


def _short(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


__all__ = [
    "ALGORITHM_LABELS",
    "AlgorithmForm",
    "ConfigDifference",
    "FormParameter",
    "ResolvedConfiguration",
    "algorithm_label",
    "algorithms_for_encoding",
    "build_algorithm_form",
    "choice_label",
    "config_fields_for",
    "config_space_for",
    "configuration_differences",
    "default_algorithm_config",
    "format_config_value",
    "grammar_for",
    "population_is_derived",
    "studio_algorithms",
]
