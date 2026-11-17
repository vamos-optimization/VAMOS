"""Configurations of a parameter space and their serializations.

A configuration is a derivation of the grammar: a value for every categorical node on its branch,
and for every terminal that the chosen productions activate. It is valid when it holds exactly the
active parameters, each inside its domain. Because the grammar is strict, the active parameters in
depth-first order determine the tree, so three serializations carry the same information:

- the canonical document, ``vamos.parameter-configuration`` 1.0.0, with the space's identity and
  the ordered ``values`` (:meth:`ParameterConfiguration.to_document`);
- the flat mapping of the values alone, which Optuna, SMAC3 and irace exchange and
  ``config_from_assignment`` reads (:meth:`ParameterConfiguration.to_values`);
- the command line of Evolver's ``VAR_CONF.txt``, ``--crossover sbx --sbx_distribution_index 20.0``,
  and its Evolver form with camelCase names and Evolver's values, ``--crossover SBX
  --sbxDistributionIndex 20.0`` (:meth:`ParameterConfiguration.to_command_line`).

Parsing any of them rebuilds the derivation and validates it.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from ..racing.parameter_names import canonical_name_from_evolver, evolver_value_candidates, from_evolver, to_evolver
from .model import (
    CONFIGURATION_SCHEMA,
    CONFIGURATION_SCHEMA_VERSION,
    Parameter,
    ParameterSpace,
    ParameterSpaceError,
    Scalar,
    SpaceIdentity,
    format_value,
    same_scalar,
)


@dataclass(frozen=True)
class ParameterConfiguration:
    """A valid configuration of a space: the space's identity and the active values in depth-first order."""

    space: SpaceIdentity
    values: tuple[tuple[str, Scalar], ...]

    def to_values(self) -> dict[str, Scalar]:
        """The flat mapping of the active parameters, in depth-first order."""
        return dict(self.values)

    def to_document(self) -> dict[str, Any]:
        """The canonical ``vamos.parameter-configuration`` document."""
        return {
            "schema": CONFIGURATION_SCHEMA,
            "schema_version": CONFIGURATION_SCHEMA_VERSION,
            "space": self.space.to_document(),
            "values": self.to_values(),
        }

    def to_command_line(self, *, evolver: bool = False) -> str:
        """``--name value`` pairs in depth-first order; with ``evolver``, Evolver's names and values where Evolver has them."""
        values: Mapping[str, Any] = to_evolver(self.to_values()) if evolver else self.to_values()
        return " ".join(f"--{name} {format_value(value)}" for name, value in values.items())


def configuration(space: ParameterSpace, values: Mapping[str, object]) -> ParameterConfiguration:
    """The configuration of ``space`` that ``values`` gives, which must hold exactly its active parameters."""
    ordered: list[tuple[str, Scalar]] = []

    def visit(parameters: tuple[Parameter, ...]) -> None:
        for parameter in parameters:
            if parameter.name not in values:
                raise ParameterSpaceError("configuration", parameter.name, "the parameter is active and has no value")
            value = _coerce(parameter, values[parameter.name])
            ordered.append((parameter.name, value))
            visit(parameter.children(value))

    visit(space.parameters)
    extra = [name for name in values if name not in dict(ordered)]
    if extra:
        known = set(space.names())
        unknown = [name for name in extra if name not in known]
        why = f"not parameters of {space.name!r}: {unknown}" if unknown else f"inactive for the values given: {extra}"
        raise ParameterSpaceError("configuration", ", ".join(extra), why)
    return ParameterConfiguration(space.identity, tuple(ordered))


def _coerce(parameter: Parameter, value: object) -> Scalar:
    """``value`` as the parameter's type: a real takes any number, an integer any integral number."""
    coerced: object = value
    if parameter.type == "real" and isinstance(value, numbers.Real) and not isinstance(value, bool):
        coerced = float(value)
    elif parameter.type == "int" and isinstance(value, numbers.Integral) and not isinstance(value, bool):
        coerced = int(value)
    if not parameter.contains(coerced):
        raise ParameterSpaceError("configuration", parameter.name, f"{value!r} is outside the domain ({_domain(parameter)})")
    if parameter.type == "categorical":
        return next(candidate for candidate in parameter.values if same_scalar(candidate, coerced))
    return cast(Scalar, coerced)


def _domain(parameter: Parameter) -> str:
    if parameter.type == "categorical":
        return f"one of {list(parameter.values)}"
    if parameter.type == "boolean":
        return "true or false"
    assert parameter.range is not None
    return f"{parameter.type} in [{parameter.range[0]}, {parameter.range[1]}]"


def default_configuration(space: ParameterSpace) -> ParameterConfiguration:
    """The space's default configuration (decision 2 of the design: the algorithm's default configuration)."""
    return configuration(space, space.default_values())


def configuration_from_document(space: ParameterSpace, document: object) -> ParameterConfiguration:
    """The configuration that a ``vamos.parameter-configuration`` document of ``space`` holds."""
    if not isinstance(document, Mapping):
        raise ParameterSpaceError("structure", "", "a configuration document is a mapping")
    if document.get("schema") != CONFIGURATION_SCHEMA or document.get("schema_version") != CONFIGURATION_SCHEMA_VERSION:
        expected = f"{CONFIGURATION_SCHEMA} {CONFIGURATION_SCHEMA_VERSION}"
        raise ParameterSpaceError(
            "structure", "schema", f"expected {expected}, got {document.get('schema')} {document.get('schema_version')}"
        )
    identity = document.get("space")
    if not isinstance(identity, Mapping) or dict(identity) != space.identity.to_document():
        raise ParameterSpaceError(
            "configuration", "space", f"the document is a configuration of {identity!r}, not of {space.identity.to_document()!r}"
        )
    values = document.get("values")
    if not isinstance(values, Mapping):
        raise ParameterSpaceError("structure", "values", "the values of a configuration are a mapping")
    return configuration(space, values)


def configuration_from_command_line(space: ParameterSpace, text: str, *, evolver: bool = False) -> ParameterConfiguration:
    """The configuration that ``--name value`` pairs give; with ``evolver``, pairs with Evolver's names and values."""
    tokens = text.split()
    if len(tokens) % 2 or any(not name.startswith("--") or len(name) < 3 for name in tokens[::2]):
        raise ParameterSpaceError("structure", "", f"expected --name value pairs, got {text!r}")
    raw: dict[str, str] = {}
    for name, value in zip(tokens[::2], tokens[1::2]):
        if name[2:] in raw:
            raise ParameterSpaceError("configuration", name[2:], "the parameter is given twice")
        raw[name[2:]] = value
    # Each parameter's spellings: one, or with ``evolver`` every VAMOS value that Evolver writes alike.
    spellings: dict[str, tuple[str, ...]] = {name: (value,) for name, value in raw.items()}
    if evolver:
        # The whole line at once, so that the settings Evolver implies apply (its IPBI always normalizes).
        spellings = {name: (str(value),) for name, value in from_evolver(raw).items()}
        for name, value in raw.items():
            canonical = canonical_name_from_evolver(name)
            read = spellings[canonical][0]
            spellings[canonical] = (read, *(str(other) for other in evolver_value_candidates(canonical, value) if str(other) != read))
    known = set(space.names())
    return configuration(
        space, {name: _parse_spellings(space.parameter(name), texts) if name in known else texts[0] for name, texts in spellings.items()}
    )


def _parse_spellings(parameter: Parameter, texts: tuple[str, ...]) -> object:
    """The first spelling that names a value of the domain: Evolver's ``polynomial`` is the integer spaces' ``pm`` too."""
    parsed = [_parse(parameter, text) for text in texts]
    return next((value for value in parsed if parameter.contains(value)), parsed[0])


def _parse(parameter: Parameter, text: str) -> object:
    if parameter.type == "categorical":
        return next((value for value in parameter.values if format_value(value) == text), text)
    if parameter.type == "boolean":
        lowered = text.lower()
        return {"true": True, "false": False}.get(lowered, text)
    try:
        return int(text) if parameter.type == "int" else float(text)
    except ValueError:
        return text


__all__ = [
    "ParameterConfiguration",
    "configuration",
    "configuration_from_command_line",
    "configuration_from_document",
    "default_configuration",
    "format_value",
]
