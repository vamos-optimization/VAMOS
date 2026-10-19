"""Panel-free algorithm configuration forms for VAMOS Studio.

A form is generated from the algorithm's tuning configuration space for the
problem encoding, so Studio offers exactly the operators and conditional
parameters that VAMOS can build. Its starting values are the VAMOS default
configuration. Resolving a form applies only the configuration fields that the
user changed on top of that default, so untouched settings keep their exact
default values even where the tuning space would build them differently.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal, cast

from vamos.engine.algorithm.config.base import _SerializableConfig
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.engine.tuning.racing.bridge_assignments import config_from_assignment
from vamos.engine.tuning.racing.config_space import AlgorithmConfigSpace
from vamos.engine.tuning.racing.param_space import Boolean, Categorical, Int, ParamType, Real
from vamos.ux.studio.algorithm_catalog import (
    ALGORITHM_LABELS,
    algorithm_label,
    algorithms_for_encoding,
    config_fields_for,
    config_space_for,
    default_algorithm_config,
    population_is_derived,
    studio_algorithms,
)

ParameterKind = Literal["real", "int", "categorical", "boolean"]

# Shown as the first choice when the VAMOS default is not one of the space's choices.
VAMOS_DEFAULT_CHOICE = "VAMOS default"

# Space parameters whose default lives under another flattened config key.
_DEFAULT_KEY_ALIASES = {
    "de_cr": "crossover_cr",
    "de_f": "crossover_f",
    "pbi_theta": "aggregation_theta",
    "selection_pressure": "selection_size",
}
# Concrete equivalents of defaults that the default configuration leaves implicit.
_IMPLICIT_DEFAULTS: dict[str, object] = {
    "mutation_prob_factor": 1.0,
    "offspring_ratio": 1.0,
    "selection_size": 2,
    "selection_pressure": 2,
    "use_external_archive": False,
}
_OPERATOR_ALIASES = {"pm": "polynomial", "polynomial": "pm"}
_BRIDGE_REQUIRED_OPERATORS = {"crossover", "mutation", "aggregation", "selection"}
# Population size is a general setting of every Studio run, not a form field.
_GENERAL_PARAMETERS = {"pop_size"}


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
    from_default: bool = True
    note: str = ""
    default_value: object = None

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").capitalize()


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
    space: AlgorithmConfigSpace
    parameters: tuple[FormParameter, ...]
    baseline: AlgorithmConfigProtocol
    _order: tuple[ParamType, ...] = field(default=(), repr=False)

    def start_values(self) -> dict[str, object]:
        return {parameter.name: parameter.start for parameter in self.parameters}

    def active_names(self, values: Mapping[str, object]) -> tuple[str, ...]:
        """Return the active parameters, in form order, for the current values."""
        param_space = self.space.to_param_space()
        partial: dict[str, object] = {}
        for spec in self._order:
            if spec.name in values and param_space.is_active(spec.name, partial):
                partial[spec.name] = values[spec.name]
        return tuple(parameter.name for parameter in self.parameters if parameter.name in partial)

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
        tuned = config_from_assignment(self.space.algorithm_name, assignment).to_dict()
        default_dict = self.baseline.to_dict()
        merged = dict(default_dict)
        unapplied: list[str] = []
        for name in changed:
            fields = [config_field for config_field in config_fields_for(name) if config_field in merged and config_field in tuned]
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
        """Active values for the tuning bridge, with VAMOS-default choices made concrete."""
        parameters = {parameter.name: parameter for parameter in self.parameters}
        assignment: dict[str, object] = {}
        for name in self.active_names(values):
            value = values[name]
            if value == VAMOS_DEFAULT_CHOICE:
                # The bridge requires operator names; it builds its own default for the rest.
                value = parameters[name].default_value if name in _BRIDGE_REQUIRED_OPERATORS else None
                if value is None:
                    continue
            assignment[name] = value
        return assignment


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
    space = config_space_for(algorithm, encoding)
    if space is None:
        return None
    baseline = default_algorithm_config(algorithm, encoding=encoding, n_var=n_var, n_obj=n_obj, pop_size=pop_size)
    defaults = _flatten_defaults(baseline.to_dict())
    order = _ordered_parameters(space)
    parameters = tuple(_form_parameter(spec, depth, defaults) for spec, depth in order if spec.name not in _GENERAL_PARAMETERS)
    form = AlgorithmForm(
        algorithm=algorithm,
        encoding=str(encoding or "real"),
        n_obj=int(n_obj),
        pop_size=pop_size,
        space=space,
        parameters=parameters,
        baseline=baseline,
        _order=tuple(list(space.params) + _conditional_parameters(space)),
    )
    return replace(form, parameters=_effective_parameters(form))


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
        return [choice for choice in parameter.choices if choice not in (parameter.start, VAMOS_DEFAULT_CHOICE)][:2]
    bounds = [bound for bound in (parameter.high, parameter.low) if bound is not None]
    values: list[object] = [int(bound) if parameter.kind == "int" else bound for bound in bounds]
    return [value for value in values if not _same_value(value, parameter.start)]


def _conditional_parameters(space: AlgorithmConfigSpace) -> list[ParamType]:
    seen = {spec.name for spec in space.params}
    ordered: list[ParamType] = []
    for block in space.conditionals or []:
        for spec in block.params:
            if spec.name not in seen:
                ordered.append(spec)
                seen.add(spec.name)
    return ordered


def _ordered_parameters(space: AlgorithmConfigSpace) -> list[tuple[ParamType, int]]:
    """Place every conditional parameter right after the parameter it depends on."""
    children: dict[str, list[ParamType]] = {}
    for block in space.conditionals or []:
        children.setdefault(block.parent_name, []).extend(block.params)
    ordered: list[tuple[ParamType, int]] = []
    seen: set[str] = set()

    def visit(spec: ParamType, depth: int) -> None:
        if spec.name in seen:
            return
        seen.add(spec.name)
        ordered.append((spec, depth))
        for child in children.get(spec.name, []):
            visit(child, depth + 1)

    for spec in space.params:
        visit(spec, 0)
    for spec in _conditional_parameters(space):
        visit(spec, 1)
    return ordered


def _flatten_defaults(config: Mapping[str, object]) -> dict[str, object]:
    """Flatten ``(name, params)`` operator entries into ``<field>_<param>`` keys."""
    flat: dict[str, object] = {}
    for key, value in config.items():
        if isinstance(value, (tuple, list)) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], Mapping):
            flat[key] = value[0]
            for param_name, param_value in value[1].items():
                flat[f"{key}_{param_name}"] = param_value
                if value[0] == "mixed":
                    # Mixed operators nest per-type settings such as ``perm_crossover``.
                    flat.setdefault(str(param_name), param_value)
        elif not isinstance(value, Mapping):
            flat[key] = value
    return flat


def _form_parameter(spec: ParamType, depth: int, defaults: Mapping[str, object]) -> FormParameter:
    default = defaults.get(_DEFAULT_KEY_ALIASES.get(spec.name, spec.name))
    if default is None and spec.name in _IMPLICIT_DEFAULTS:
        default = _IMPLICIT_DEFAULTS[spec.name]
    if isinstance(spec, Boolean):
        return FormParameter(spec.name, "boolean", bool(default) if isinstance(default, bool) else False, depth)
    if isinstance(spec, Categorical):
        return _categorical_parameter(spec, depth, default)
    return _numeric_parameter(spec, depth, default)


def _categorical_parameter(spec: Categorical, depth: int, default: object) -> FormParameter:
    choices = tuple(spec.choices)
    if default in choices:
        return FormParameter(spec.name, "categorical", default, depth, choices=choices)
    alias = _OPERATOR_ALIASES.get(str(default))
    if alias in choices:
        return FormParameter(spec.name, "categorical", alias, depth, choices=choices)
    if default is None and depth > 0:
        # A sub-parameter of an inactive default option has no VAMOS default of its own.
        return FormParameter(spec.name, "categorical", choices[0], depth, choices=choices, from_default=False)
    shown = "automatic" if default is None else default
    note = f"The VAMOS default ({shown}) is not one of the choices of the tuning space; it is kept unless you choose another value."
    return FormParameter(
        spec.name,
        "categorical",
        VAMOS_DEFAULT_CHOICE,
        depth,
        choices=(VAMOS_DEFAULT_CHOICE, *choices),
        note=note,
        default_value=default,
    )


def _numeric_parameter(spec: Real | Int, depth: int, default: object) -> FormParameter:
    kind: ParameterKind = "int" if isinstance(spec, Int) else "real"
    low, high = float(spec.low), float(spec.high)
    if isinstance(default, (int, float)) and not isinstance(default, bool) and math.isfinite(float(default)):
        start: float | int = int(default) if kind == "int" else float(default)
        note = ""
        if not low <= float(start) <= high:
            note = f"The VAMOS default ({start:g}) is outside the tuning range [{low:g}, {high:g}]."
            low, high = min(low, float(start)), max(high, float(start))
        return FormParameter(spec.name, kind, start, depth, low=low, high=high, log=spec.log, note=note)
    middle = math.sqrt(low * high) if spec.log else (low + high) / 2.0
    start = int(round(middle)) if kind == "int" else middle
    return FormParameter(spec.name, kind, start, depth, low=low, high=high, log=spec.log, from_default=False)


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-15)
    return left == right


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


__all__ = [
    "ALGORITHM_LABELS",
    "AlgorithmForm",
    "ConfigDifference",
    "FormParameter",
    "ResolvedConfiguration",
    "VAMOS_DEFAULT_CHOICE",
    "algorithm_label",
    "algorithms_for_encoding",
    "build_algorithm_form",
    "config_fields_for",
    "config_space_for",
    "configuration_differences",
    "default_algorithm_config",
    "population_is_derived",
    "studio_algorithms",
]
