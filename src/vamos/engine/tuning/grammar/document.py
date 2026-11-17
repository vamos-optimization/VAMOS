"""Reading and writing ``vamos.parameter-space`` documents, and the rules G1 to G6 that every document follows.

:func:`parameter_space_from_document` turns a parsed JSON or YAML mapping into a :class:`ParameterSpace`
and rejects a document that breaks a rule with a :class:`ParameterSpaceError` naming the rule and the
place. Rule G7 runs the algorithm, so it is a separate check (:mod:`.default_check`).

:func:`parameter_space_document` writes the canonical form: the same document with every key in a
fixed order, real ranges and defaults as floats, and ``values`` as a list unless some value has a
production. The space's SHA-256 is that of the canonical JSON without the presentation fields
(``label``, ``description``), so editing documentation does not change the identity of a space.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any, cast

from vamos.engine.algorithm.variants import ALGORITHM_VARIANT_GROUPS
from vamos.foundation.encoding import ENCODINGS

from ..racing.param_space import PARAM_ROLES
from .model import (
    SCHEMA,
    SCHEMA_VERSION,
    Parameter,
    ParameterSpace,
    ParameterSpaceError,
    ParameterType,
    Scalar,
    format_value,
    same_scalar,
)

NAME_PATTERN = r"^[a-z][a-z0-9_]*$"
_TOP_KEYS = ("schema", "schema_version", "name", "algorithm", "encoding", "description", "parameters")
_NODE_KEYS = ("type", "default", "range", "log", "values", "global_sub_parameters", "role", "label", "description")
_TYPES: tuple[ParameterType, ...] = ("categorical", "real", "int", "boolean")
# Keys of other formats that make a parameter depend on an expression instead of its place in the tree.
_EXPRESSION_KEYS = {"condition", "conditions", "active_if", "depends_on", "parent", "parents", "when", "if"}
_NOT_IN_SCHEMA = {
    "forbidden": "the schema has no forbidden clauses: a space offers only configurations that run",
    "forbidden_clauses": "the schema has no forbidden clauses: a space offers only configurations that run",
    "limits": "the schema has one range per parameter, the same for tuning and for editing by hand",
}
# A command line separates its tokens by whitespace, so a categorical string value is one token.
_ONE_TOKEN = r"\S+"


def parameter_space_from_document(document: object) -> ParameterSpace:
    """The space that ``document`` (a parsed ``vamos.parameter-space`` mapping) describes, after rules G1 to G6."""
    if not isinstance(document, Mapping):
        raise ParameterSpaceError("structure", "", "a parameter space is a mapping")
    _check_keys(document, _TOP_KEYS, "")
    if document.get("schema") != SCHEMA:
        raise ParameterSpaceError("G6", "schema", f"expected {SCHEMA!r}, got {document.get('schema')!r}")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ParameterSpaceError("G6", "schema_version", f"VAMOS reads version {SCHEMA_VERSION}, got {document.get('schema_version')!r}")
    name = _text(document, "name", "")
    if not re.fullmatch(NAME_PATTERN, name):
        raise ParameterSpaceError("G1", "name", f"{name!r} does not match {NAME_PATTERN}")
    algorithm = _text(document, "algorithm", "")
    if algorithm not in ALGORITHM_VARIANT_GROUPS:
        raise ParameterSpaceError(
            "G6", "algorithm", f"{algorithm!r} is not a VAMOS algorithm with tuning spaces: {', '.join(ALGORITHM_VARIANT_GROUPS)}"
        )
    encoding = _text(document, "encoding", "")
    if encoding not in ENCODINGS:
        raise ParameterSpaceError("G6", "encoding", f"{encoding!r} is not a VAMOS encoding: {', '.join(ENCODINGS)}")
    variant = algorithm if encoding == "real" else f"{algorithm}_{encoding}"
    if variant not in ALGORITHM_VARIANT_GROUPS[algorithm]:
        raise ParameterSpaceError("G6", "encoding", f"VAMOS has no {encoding} variant of {algorithm}")
    description = _optional_text(document, "description", "")
    parameters = document.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ParameterSpaceError("structure", "parameters", "a space has a non-empty mapping of parameters")
    seen: dict[str, str] = {}
    nodes = _nodes(parameters, "parameters", seen)
    return ParameterSpace(name=name, algorithm=algorithm, encoding=encoding, parameters=nodes, description=description)


def _nodes(raw: Mapping[Any, Any], path: str, seen: dict[str, str]) -> tuple[Parameter, ...]:
    return tuple(_node(name, node, f"{path}.{name}", seen) for name, node in raw.items())


def _node(name: object, raw: object, path: str, seen: dict[str, str]) -> Parameter:
    if not isinstance(name, str) or not re.fullmatch(NAME_PATTERN, name):
        raise ParameterSpaceError("G1", path, f"parameter name {name!r} does not match {NAME_PATTERN}")
    if name in seen:
        raise _repeated(name, seen[name], path)
    seen[name] = path
    if not isinstance(raw, Mapping):
        raise ParameterSpaceError("structure", path, "a parameter is a mapping with its type and default")
    _check_keys(raw, _NODE_KEYS, path)
    kind = raw.get("type")
    if kind not in _TYPES:
        raise ParameterSpaceError("structure", f"{path}.type", f"expected one of {', '.join(_TYPES)}, got {kind!r}")
    role = _optional_text(raw, "role", path)
    if role is not None and role not in PARAM_ROLES:
        raise ParameterSpaceError("structure", f"{path}.role", f"expected one of {', '.join(sorted(PARAM_ROLES))}, got {role!r}")
    if "default" not in raw:
        raise ParameterSpaceError("G3", path, "every parameter has a default inside its domain")
    presentation: dict[str, Any] = {
        "role": role,
        "label": _optional_text(raw, "label", path),
        "description": _optional_text(raw, "description", path),
    }
    kind = cast(ParameterType, kind)
    if kind == "categorical":
        return _categorical(name, raw, path, seen, presentation)
    _refuse(raw, ("values", "global_sub_parameters"), path, f"a {kind} parameter is a terminal")
    if kind == "boolean":
        _refuse(raw, ("range", "log"), path, "a boolean parameter has no range")
        default = raw["default"]
        if not isinstance(default, bool):
            raise ParameterSpaceError("G3", f"{path}.default", f"{default!r} is not a boolean")
        return Parameter(name, "boolean", default, **presentation)
    low, high, log = _range(raw, kind, path)
    default = raw["default"]
    if isinstance(default, bool) or not isinstance(default, (int, float)) or (kind == "int" and not isinstance(default, int)):
        raise ParameterSpaceError("G3", f"{path}.default", f"{default!r} is not {'an integer' if kind == 'int' else 'a number'}")
    value: Scalar = int(default) if kind == "int" else float(default)
    parameter = Parameter(name, kind, value, range=(low, high), log=log, **presentation)
    if not parameter.contains(value):
        raise ParameterSpaceError("G3", f"{path}.default", f"{default!r} lies outside the range [{low}, {high}]")
    return parameter


def _categorical(name: str, raw: Mapping[str, Any], path: str, seen: dict[str, str], presentation: dict[str, Any]) -> Parameter:
    _refuse(raw, ("range", "log"), path, "a categorical parameter has values, not a range")
    raw_values = raw.get("values")
    values: list[Scalar] = []
    productions: list[tuple[Parameter, ...]] = []
    if isinstance(raw_values, Mapping):
        entries = list(raw_values.items())
    elif isinstance(raw_values, list):
        entries = [(value, None) for value in raw_values]
    else:
        raise ParameterSpaceError(
            "G5", f"{path}.values", "a categorical parameter has a list of values, or a mapping from each value to its production"
        )
    if not entries:
        raise ParameterSpaceError("G5", f"{path}.values", "the list of values is empty")
    for value, production in entries:
        where = f"{path}.values.{value}"
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ParameterSpaceError("G5", where, f"{value!r} is not a string, number or boolean")
        if isinstance(value, float) and not math.isfinite(value):
            raise ParameterSpaceError("G5", where, f"{value!r} is not finite: the canonical JSON of a space has no infinities or NaN")
        if isinstance(value, str) and not re.fullmatch(_ONE_TOKEN, value):
            raise ParameterSpaceError(
                "G5", where, f"{value!r} is not one command-line token: a string value is not empty and has no whitespace"
            )
        if any(same_scalar(value, other) for other in values):
            raise ParameterSpaceError("G5", where, f"{value!r} is listed twice")
        twin = next((other for other in values if format_value(other) == format_value(value)), None)
        if twin is not None:
            raise ParameterSpaceError(
                "G5", where, f"{twin!r} and {value!r} are written alike ({format_value(value)}): a command line could not tell them apart"
            )
        values.append(value)
        productions.append(_production(value, production, where, seen))
    for value, production in zip(values, productions):
        if production and not isinstance(value, str):
            raise ParameterSpaceError("G5", f"{path}.values.{value}", "a value with a production is a string")
    if any(productions):
        # The document writes this node's values as the keys of a mapping, which are text.
        for value in values:
            if not isinstance(value, str):
                raise ParameterSpaceError(
                    "G5", f"{path}.values.{value}", f"a categorical with productions has string values only: write {str(value)!r}"
                )
    default = raw["default"]
    globals_raw = raw.get("global_sub_parameters", {})
    if not isinstance(globals_raw, Mapping):
        raise ParameterSpaceError("structure", f"{path}.global_sub_parameters", "global sub-parameters are a mapping of parameters")
    sub_parameters = _nodes(globals_raw, f"{path}.global_sub_parameters", seen)
    parameter = Parameter(
        name,
        "categorical",
        default,
        values=tuple(values),
        productions=tuple(productions),
        global_sub_parameters=sub_parameters,
        **presentation,
    )
    if not parameter.contains(default):
        raise ParameterSpaceError("G3", f"{path}.default", f"{default!r} is not one of the values {list(values)}")
    return parameter


def _production(value: object, raw: object, path: str, seen: dict[str, str]) -> tuple[Parameter, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise ParameterSpaceError("structure", path, "a production is a mapping that may hold conditional_parameters")
    _check_keys(raw, ("conditional_parameters",), path)
    children = raw.get("conditional_parameters", {})
    if not isinstance(children, Mapping):
        raise ParameterSpaceError("structure", f"{path}.conditional_parameters", "conditional parameters are a mapping of parameters")
    return _nodes(children, f"{path}.conditional_parameters", seen)


def _range(raw: Mapping[str, Any], kind: str, path: str) -> tuple[float, float, bool]:
    bounds = raw.get("range")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ParameterSpaceError("G4", f"{path}.range", "a numeric parameter has a range [low, high]")
    if any(isinstance(end, bool) or not isinstance(end, (int, float)) for end in bounds):
        raise ParameterSpaceError(
            "G4", f"{path}.range", f"{bounds!r} has an end that is not a number (YAML reads 1e-8 as text: write 1.0e-8)"
        )
    if kind == "int" and not all(isinstance(end, int) for end in bounds):
        raise ParameterSpaceError("G4", f"{path}.range", f"an integer range has integer ends, got {bounds!r}")
    low, high = (int(end) for end in bounds) if kind == "int" else (float(end) for end in bounds)
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ParameterSpaceError("G4", f"{path}.range", f"the range has finite ends, got {bounds!r}")
    if not low <= high:
        raise ParameterSpaceError("G4", f"{path}.range", f"the range is not ordered: {low} > {high}")
    log = raw.get("log", False)
    if not isinstance(log, bool):
        raise ParameterSpaceError("structure", f"{path}.log", f"log is true or false, got {log!r}")
    if log and low <= 0:
        raise ParameterSpaceError("G4", f"{path}.range", f"a log scale needs a positive lower end, got {low}")
    return low, high, log


def _repeated(name: str, first: str, second: str) -> ParameterSpaceError:
    first_parent, second_parent = first.rsplit(".values.", 1)[0], second.rsplit(".values.", 1)[0]
    if ".values." in first and ".values." in second and first_parent == second_parent:
        return ParameterSpaceError(
            "G2",
            second,
            f"{name!r} is also at {first}: a parameter has one parent. A parameter that applies to some values of a node "
            "is duplicated under each of their productions, with its own name",
        )
    return ParameterSpaceError("G1", second, f"parameter names are unique in the space; {name!r} is also at {first}")


def _check_keys(raw: Mapping[Any, Any], allowed: tuple[str, ...], path: str) -> None:
    for key in raw:
        if key in allowed:
            continue
        where = f"{path}.{key}" if path else str(key)
        if key in _EXPRESSION_KEYS:
            raise ParameterSpaceError("G2", where, "a parameter's parent is its place in the tree; the schema has no condition expressions")
        if key in _NOT_IN_SCHEMA:
            raise ParameterSpaceError("structure", where, _NOT_IN_SCHEMA[key])
        raise ParameterSpaceError("structure", where, f"unknown key {key!r}; expected {', '.join(allowed)}")


def _refuse(raw: Mapping[str, Any], keys: tuple[str, ...], path: str, why: str) -> None:
    for key in keys:
        if key in raw:
            raise ParameterSpaceError("structure", f"{path}.{key}", why)


def _text(raw: Mapping[str, Any], key: str, path: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ParameterSpaceError("structure", f"{path}.{key}" if path else key, f"expected a non-empty string, got {value!r}")
    return value


def _optional_text(raw: Mapping[str, Any], key: str, path: str) -> str | None:
    return None if raw.get(key) is None else _text(raw, key, path)


def parameter_space_document(space: ParameterSpace, *, presentation: bool = True) -> dict[str, Any]:
    """The canonical ``vamos.parameter-space`` document of ``space``; without ``presentation``, no labels or descriptions."""
    document: dict[str, Any] = {"schema": SCHEMA, "schema_version": space.schema_version, "name": space.name}
    document |= {"algorithm": space.algorithm, "encoding": space.encoding}
    if presentation and space.description is not None:
        document["description"] = space.description
    document["parameters"] = _documents(space.parameters, presentation)
    return document


def _documents(parameters: tuple[Parameter, ...], presentation: bool) -> dict[str, Any]:
    return {parameter.name: _document(parameter, presentation) for parameter in parameters}


def _document(parameter: Parameter, presentation: bool) -> dict[str, Any]:
    node: dict[str, Any] = {"type": parameter.type, "default": parameter.default}
    if parameter.range is not None:
        node["range"] = list(parameter.range)
        if parameter.log:
            node["log"] = True
    if parameter.type == "categorical":
        if any(parameter.productions):
            node["values"] = {
                str(value): ({"conditional_parameters": _documents(production, presentation)} if production else {})
                for value, production in zip(parameter.values, parameter.productions)
            }
        else:
            node["values"] = list(parameter.values)
        if parameter.global_sub_parameters:
            node["global_sub_parameters"] = _documents(parameter.global_sub_parameters, presentation)
    if parameter.role is not None:
        node["role"] = parameter.role
    if presentation:
        node |= {key: value for key, value in (("label", parameter.label), ("description", parameter.description)) if value is not None}
    return node


def canonical_json(document: Mapping[str, Any]) -> str:
    """Compact JSON with the document's own key order, which is the depth-first order of the tree."""
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def space_sha256(space: ParameterSpace) -> str:
    """The SHA-256 of the canonical JSON form of ``space`` without its presentation fields."""
    return hashlib.sha256(canonical_json(parameter_space_document(space, presentation=False)).encode("utf-8")).hexdigest()


__all__ = ["NAME_PATTERN", "canonical_json", "parameter_space_document", "parameter_space_from_document", "space_sha256"]
