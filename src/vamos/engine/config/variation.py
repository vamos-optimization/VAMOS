"""
Shared helpers for variation/default handling across CLI, runner, and factories.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias, TypedDict

from vamos.foundation.encoding import normalize_encoding

OperatorParams: TypeAlias = dict[str, Any]
OperatorTuple: TypeAlias = tuple[str, OperatorParams]


class OperatorSpecDict(TypedDict, total=False):
    method: str
    name: str


OperatorSpecInput: TypeAlias = str | OperatorTuple | OperatorSpecDict | Mapping[str, object]
RepairSpecInput: TypeAlias = OperatorSpecInput | Literal["auto"]


class VariationOverrides(TypedDict, total=False):
    crossover: OperatorSpecInput
    mutation: OperatorSpecInput
    selection: OperatorSpecInput
    repair: RepairSpecInput
    aggregation: OperatorSpecInput
    weight_vectors: Mapping[str, object] | str
    use_numba_variation: bool
    archive_size: int
    k_neighbors: int
    indicator: str
    kappa: float
    inertia: float
    c1: float
    c2: float
    vmax_fraction: float
    n_partitions: int
    alpha: float
    adapt_freq: float


VariationConfig: TypeAlias = dict[str, object]


def normalize_operator_tuple(spec: object) -> OperatorTuple | None:
    """Normalize an operator spec to ``(name, params)``; ``None`` stays ``None``.

    The accepted forms are a name (``"sbx"``), a ``(name, params)`` pair whose ``params`` is a
    mapping, and a mapping with ``method`` (or ``name``) plus the parameters. A parameter set to
    ``None`` is dropped, so the operator's default applies. Any other value raises ``ValueError``.
    """
    if spec is None:
        return None
    return _operator_tuple(spec, key=None)


def _operator_tuple(spec: object, *, key: str | None) -> OperatorTuple:
    if isinstance(spec, str):
        return (_operator_name(spec, spec, key=key), {})
    if isinstance(spec, (tuple, list)):
        if len(spec) != 2:
            raise _invalid_operator(spec, key, "a pair must be (name, params)")
        name, params = spec
        if not isinstance(params, Mapping):
            raise _invalid_operator(spec, key, "its parameters must be a mapping")
        return (_operator_name(name, spec, key=key), _parameters(params))
    if isinstance(spec, Mapping):
        method = spec.get("method") or spec.get("name")
        params = {k: v for k, v in spec.items() if k not in {"method", "name"}}
        return (_operator_name(method, spec, key=key), _parameters(params))
    raise _invalid_operator(spec, key, "expected a name, a (name, params) pair or a mapping with 'method'")


def _operator_name(name: object, spec: object, *, key: str | None) -> str:
    if not isinstance(name, str) or not name.strip():
        raise _invalid_operator(spec, key, "the operator name must be a non-empty string")
    return name


def _parameters(params: Mapping[Any, object]) -> OperatorParams:
    return {str(name): value for name, value in params.items() if value is not None}


def _invalid_operator(spec: object, key: str | None, reason: str) -> ValueError:
    where = f" for '{key}'" if key is not None else ""
    return ValueError(f"Invalid operator spec{where}: {spec!r}; {reason}.")


def _is_auto_repair(spec: object) -> bool:
    return isinstance(spec, str) and spec.strip().lower() == "auto"


def ensure_operator_tuple(spec: object, *, key: str) -> OperatorTuple:
    if spec is None:
        raise _invalid_operator(spec, key, "an operator is required")
    return _operator_tuple(spec, key=key)


def ensure_operator_tuple_optional(spec: object, *, key: str) -> OperatorTuple | None:
    if spec is None:
        return None
    return ensure_operator_tuple(spec, key=key)


def normalize_variation_config(raw: Mapping[str, object] | None) -> VariationConfig | None:
    """Normalize the operators of a variation configuration to ``(name, params)`` tuples.

    An operator set to ``None`` is left out; one that cannot be read raises ``ValueError``.
    ``repair: "auto"`` stays ``"auto"``, and the other entries are kept unless they are ``None``.
    """
    if not raw:
        return None
    normalized: VariationConfig = {}
    known_op_keys = ("crossover", "mutation", "selection", "repair", "aggregation")
    for key in known_op_keys:
        value = raw.get(key)
        if key == "repair" and _is_auto_repair(value):
            normalized[key] = "auto"
            continue
        if value is not None:
            normalized[key] = ensure_operator_tuple(value, key=key)
    for key, value in raw.items():
        if key in normalized or key in known_op_keys:
            continue
        if value is not None:
            normalized[key] = value
    return normalized or None


def resolve_default_variation_config(
    encoding: str,
    overrides: Mapping[str, object] | None,
    *,
    defaults: Mapping[str, object] | None = None,
) -> VariationConfig:
    """Default variation configuration of ``encoding``, merged with ``overrides``.

    ``defaults`` replaces entries of the encoding's table, for an algorithm with its own operators.
    A ``crossover`` or ``mutation`` override that names the default operator keeps the default
    parameters it does not set: ``{"method": "sbx", "eta": 5}`` keeps the default ``prob``, and
    ``"sbx"`` alone is the default SBX. An override that names another operator replaces the
    default with exactly its parameters. A parameter set to ``None`` keeps its default, and
    ``repair: "auto"`` keeps the encoding's default repair (clipping for real variables, none
    otherwise).
    """
    normalized = normalize_encoding(encoding)
    base: VariationConfig
    if normalized == "real":
        base = {
            "crossover": ("sbx", {"prob": 0.9, "eta": 20.0}),
            "mutation": ("polynomial", {"prob": "1/n", "eta": 20.0}),
            "repair": ("clip", {}),
        }
    elif normalized == "binary":
        base = {
            "crossover": ("hux", {"prob": 0.9}),
            "mutation": ("bitflip", {"prob": "1/n"}),
        }
    elif normalized == "integer":
        base = {
            "crossover": ("uniform", {"prob": 0.9}),
            "mutation": ("reset", {"prob": "1/n"}),
        }
    elif normalized == "mixed":
        base = {
            "crossover": ("mixed", {"prob": 0.9}),
            "mutation": ("mixed", {"prob": "1/n"}),
        }
    elif normalized == "permutation":
        base = {
            "crossover": ("ox", {"prob": 0.9}),
            "mutation": ("swap", {"prob": 0.1}),
        }
    else:
        base = {
            "crossover": ("sbx", {"prob": 0.9, "eta": 20.0}),
            "mutation": ("polynomial", {"prob": 0.1, "eta": 20.0}),
        }

    if defaults:
        base.update(defaults)
    if overrides:
        for key in ("crossover", "mutation"):
            if key in overrides:
                base[key] = _merge_operator(base.get(key), overrides[key], key=key)
        if "repair" in overrides and not _is_auto_repair(overrides["repair"]):
            base["repair"] = overrides["repair"]
        for key, value in overrides.items():
            if key in {"crossover", "mutation", "repair"}:
                continue
            base[key] = value

    return base


# Names under which the default operators of the encoding tables are also registered.
_OPERATOR_ALIASES = {"pm": "polynomial"}


def _canonical_operator(name: str) -> str:
    lowered = name.strip().lower()
    return _OPERATOR_ALIASES.get(lowered, lowered)


def _merge_operator(default: object, override: object, *, key: str) -> OperatorTuple | None:
    """``override``, completed with the parameters of ``default`` when both name the same operator."""
    if override is None:
        return None
    name, params = ensure_operator_tuple(override, key=key)
    if default is not None:
        default_name, default_params = ensure_operator_tuple(default, key=key)
        if _canonical_operator(name) == _canonical_operator(default_name):
            return (name, {**default_params, **params})
    return (name, params)


def merge_variation_overrides(base: Mapping[str, object] | None, override: Mapping[str, object] | None) -> VariationConfig:
    """``base`` with the entries of ``override``, which replace whole entries.

    An override entry set to ``None`` or ``repair: "auto"`` keeps the inherited value. Operator
    parameters are completed later, against the algorithm's defaults, by
    :func:`resolve_default_variation_config`.
    """
    base = base or {}
    if not override:
        return dict(base)
    merged = dict(base)
    for key, value in override.items():
        if value is None:
            continue
        if key == "repair" and _is_auto_repair(value):
            continue
        merged[key] = value
    return merged
