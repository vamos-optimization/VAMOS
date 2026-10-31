"""
Shared helpers and operator registries for variation pipelines.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

from vamos.engine.operators.impl.mixed import mixed_crossover, mixed_mutation
from vamos.engine.operators.impl.registry import get_operator_registry
from vamos.engine.operators.policies.discrete_operator_maps import (
    BINARY_CROSSOVER_COMMON,
    BINARY_MUTATION_COMMON,
    INT_CROSSOVER_COMMON,
    INT_MUTATION_COMMON,
    PERM_CROSSOVER_COMMON,
    PERM_MUTATION_COMMON,
)
from vamos.engine.variation.protocol import CrossoverName, MutationName, RepairName
from vamos.foundation.encoding import EncodingLike, normalize_encoding
from vamos.foundation.registry import Registry

_REAL_CROSSOVER_KEYS = (
    "sbx",
    "blx_alpha",
    "blx_alpha_beta",
    "whole_arithmetic",
    "laplace",
    "fuzzy",
    "arithmetic",
    "de",
    "pcx",
    "undx",
    "simplex",
)
_REAL_MUTATION_KEYS = (
    "polynomial",
    "non_uniform",
    "gaussian",
    "uniform_reset",
    "cauchy",
    "uniform",
    "linked_polynomial",
    "levy_flight",
    "power_law",
)

REAL_CROSSOVER_METHODS: tuple[str, ...] = _REAL_CROSSOVER_KEYS
"""Real-coded crossover names, the table used by :func:`validate_operator_support`."""
REAL_MUTATION_METHODS: tuple[str, ...] = (*_REAL_MUTATION_KEYS, "pm")
"""Real-coded mutation names, including the ``pm`` alias of ``polynomial``."""
THREE_PARENT_REAL_CROSSOVERS: tuple[str, ...] = ("pcx", "undx", "simplex", "de")
"""Real crossovers that take three parents per mating group."""


def validate_pairwise_real_operators(
    cross_method: str,
    mut_method: str,
    *,
    algorithm_label: str,
    three_parent_allowed: Collection[str] = (),
) -> None:
    """Reject operators that a real-coded, two-parent mating scheme cannot run.

    Both names must be real-coded operators, and the crossover must take two
    parents unless it is listed in ``three_parent_allowed``. Algorithms that group
    parents by the crossover's arity use :class:`~vamos.engine.variation.VariationPipeline`
    instead.
    """
    cross = cross_method.lower()
    if cross not in REAL_CROSSOVER_METHODS:
        supported = ", ".join(name for name in REAL_CROSSOVER_METHODS if name not in THREE_PARENT_REAL_CROSSOVERS)
        raise ValueError(f"Unsupported crossover '{cross_method}' for real encoding in {algorithm_label}. Supported: {supported}.")
    if cross in THREE_PARENT_REAL_CROSSOVERS and cross not in three_parent_allowed:
        raise ValueError(
            f"{algorithm_label} mates parents two by two, but crossover '{cross_method}' needs three parents per group. "
            "Use a two-parent crossover, or NSGA-II, IBEA, AGE-MOEA or RVEA, which group parents by crossover arity."
        )
    if mut_method.lower() not in REAL_MUTATION_METHODS:
        supported = ", ".join(REAL_MUTATION_METHODS)
        raise ValueError(f"Unsupported mutation '{mut_method}' for real encoding in {algorithm_label}. Supported: {supported}.")


def _per_variable_numerator(text: str) -> float | None:
    """``k`` of a ``"k/n"`` expression, ignoring case and spaces; ``None`` for anything else."""
    head, slash, tail = text.partition("/")
    if not slash or tail.strip().lower() != "n":
        return None
    head = head.strip()
    if not head:
        return 1.0
    try:
        numerator = float(head)
    except ValueError:
        return None
    return numerator if math.isfinite(numerator) and numerator >= 0.0 else None


def resolve_prob_expression(value: object, n_var: int, default: float) -> float:
    """Probability from a number in [0, 1] or a ``"k/n"`` expression.

    ``"k/n"`` means ``k / n_var``, capped at 1; case and spaces do not matter, so
    ``"1/N"`` and ``" 1 / n "`` equal ``"1/n"``. ``None`` gives ``default``.
    Anything else, including a number outside [0, 1], raises ``ValueError``.
    """
    if value is None:
        return default
    if isinstance(value, str):
        numerator = _per_variable_numerator(value)
        if numerator is not None:
            return min(1.0, numerator / max(1, n_var))
    elif isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Invalid probability {value!r}: use a number in [0, 1] or an expression 'k/n'.")
    try:
        probability = float(value)
    except ValueError:
        raise ValueError(f"Invalid probability {value!r}: use a number in [0, 1] or an expression 'k/n'.") from None
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Invalid probability {value!r}: it must lie in [0, 1].")
    return probability


def _mutation_prob_factor(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"Invalid mutation prob_factor {value!r}: it must be a positive number.")
    return float(value)


def prepare_mutation_params(
    mut_params: dict[str, float | int | str | None],
    encoding: EncodingLike,
    n_var: int,
    prob_factor: float | None = None,
) -> dict[str, float | int | str | None]:
    """Mutation parameters with ``prob`` resolved for ``n_var`` variables.

    A ``prob_factor`` (argument, ``prob_factor`` or ``mutation_prob_factor`` key)
    sets ``prob = factor / n_var``, capped at 1, and is removed from the
    parameters; it must be positive. Otherwise ``prob`` goes through
    :func:`resolve_prob_expression`, and a missing one becomes ``1/n``
    (``2/n`` for permutations).
    """
    normalized = normalize_encoding(encoding)
    params = dict(mut_params)
    keyed_factor = params.pop("prob_factor", None)
    aliased_factor = params.pop("mutation_prob_factor", None)
    factor = next((value for value in (prob_factor, keyed_factor, aliased_factor) if value is not None), None)
    if factor is not None:
        params["prob"] = min(1.0, _mutation_prob_factor(factor) / max(1, n_var))
    elif "prob" in params:
        params["prob"] = resolve_prob_expression(params["prob"], n_var, 1.0)
    else:
        if normalized == "permutation":
            params["prob"] = min(1.0, 2.0 / max(1, n_var))
        else:
            params["prob"] = 1.0 / max(1, n_var)
    return params


if TYPE_CHECKING:
    PERM_CROSSOVER: Registry[Any]
    PERM_MUTATION: Registry[Any]
    BINARY_CROSSOVER: Registry[Any]
    BINARY_MUTATION: Registry[Any]
    INT_CROSSOVER: Registry[Any]
    INT_MUTATION: Registry[Any]
    MIXED_CROSSOVER: Registry[Any]
    MIXED_MUTATION: Registry[Any]
    REAL_CROSSOVER: Registry[Any]
    REAL_MUTATION: Registry[Any]

_REGISTRY_LABELS = {
    "PERM_CROSSOVER": "Permutation Crossover",
    "PERM_MUTATION": "Permutation Mutation",
    "BINARY_CROSSOVER": "Binary Crossover",
    "BINARY_MUTATION": "Binary Mutation",
    "INT_CROSSOVER": "Integer Crossover",
    "INT_MUTATION": "Integer Mutation",
    "MIXED_CROSSOVER": "Mixed Crossover",
    "MIXED_MUTATION": "Mixed Mutation",
    "REAL_CROSSOVER": "Real Crossover",
    "REAL_MUTATION": "Real Mutation",
}
_REGISTRIES: dict[str, Registry[Any]] = {}
_DEFAULTS_POPULATED = False


def _get_registry(name: str) -> Registry[Any]:
    registry = _REGISTRIES.get(name)
    if registry is None:
        registry = Registry(_REGISTRY_LABELS[name])
        _REGISTRIES[name] = registry
    return registry


def _populate(reg: Registry[Any], items: Mapping[str, object]) -> None:
    for key, value in items.items():
        reg.register(key, value)


def _populate_defaults() -> None:
    global _DEFAULTS_POPULATED
    if _DEFAULTS_POPULATED:
        return
    perm_crossover = _get_registry("PERM_CROSSOVER")
    if len(perm_crossover) > 0:
        _DEFAULTS_POPULATED = True
        return

    perm_mutation = _get_registry("PERM_MUTATION")
    binary_crossover = _get_registry("BINARY_CROSSOVER")
    binary_mutation = _get_registry("BINARY_MUTATION")
    int_crossover = _get_registry("INT_CROSSOVER")
    int_mutation = _get_registry("INT_MUTATION")
    mixed_crossover_registry = _get_registry("MIXED_CROSSOVER")
    mixed_mutation_registry = _get_registry("MIXED_MUTATION")
    real_crossover = _get_registry("REAL_CROSSOVER")
    real_mutation = _get_registry("REAL_MUTATION")

    _populate(
        perm_crossover,
        {
            **PERM_CROSSOVER_COMMON,
            "oxd": PERM_CROSSOVER_COMMON["order"],
        },
    )

    _populate(perm_mutation, PERM_MUTATION_COMMON)
    _populate(binary_crossover, {**BINARY_CROSSOVER_COMMON, "spx": BINARY_CROSSOVER_COMMON["one_point"]})
    _populate(binary_mutation, BINARY_MUTATION_COMMON)
    _populate(int_crossover, INT_CROSSOVER_COMMON)
    _populate(int_mutation, {**INT_MUTATION_COMMON, "pm": INT_MUTATION_COMMON["polynomial"]})

    _populate(
        mixed_crossover_registry,
        {
            "mixed": mixed_crossover,
            "uniform": mixed_crossover,
        },
    )

    _populate(
        mixed_mutation_registry,
        {
            "mixed": mixed_mutation,
            "gaussian": mixed_mutation,
        },
    )

    registry = get_operator_registry()
    _populate(real_crossover, {k: registry.get(k) for k in _REAL_CROSSOVER_KEYS})
    _populate(real_mutation, {k: registry.get(k) for k in _REAL_MUTATION_KEYS})
    real_mutation.register("pm", registry.get("polynomial"))
    _DEFAULTS_POPULATED = True


def validate_operator_support(encoding: EncodingLike, crossover: CrossoverName, mutation: MutationName) -> None:
    _populate_defaults()
    normalized = normalize_encoding(encoding)
    perm_crossover = _get_registry("PERM_CROSSOVER")
    perm_mutation = _get_registry("PERM_MUTATION")
    binary_crossover = _get_registry("BINARY_CROSSOVER")
    binary_mutation = _get_registry("BINARY_MUTATION")
    int_crossover = _get_registry("INT_CROSSOVER")
    int_mutation = _get_registry("INT_MUTATION")
    mixed_crossover = _get_registry("MIXED_CROSSOVER")
    mixed_mutation = _get_registry("MIXED_MUTATION")
    real_crossover = _get_registry("REAL_CROSSOVER")
    real_mutation = _get_registry("REAL_MUTATION")
    if normalized == "permutation":
        if crossover not in perm_crossover:
            raise ValueError(f"Unsupported crossover '{crossover}' for permutation encoding.")
        if mutation not in perm_mutation:
            raise ValueError(f"Unsupported mutation '{mutation}' for permutation encoding.")
    elif normalized == "binary":
        if crossover not in binary_crossover:
            raise ValueError(f"Unsupported crossover '{crossover}' for binary encoding.")
        if mutation not in binary_mutation:
            raise ValueError(f"Unsupported mutation '{mutation}' for binary encoding.")
    elif normalized == "integer":
        if crossover not in int_crossover:
            raise ValueError(f"Unsupported crossover '{crossover}' for integer encoding.")
        if mutation not in int_mutation:
            raise ValueError(f"Unsupported mutation '{mutation}' for integer encoding.")
    elif normalized == "mixed":
        if crossover not in mixed_crossover:
            raise ValueError(f"Unsupported crossover '{crossover}' for mixed encoding.")
        if mutation not in mixed_mutation:
            raise ValueError(f"Unsupported mutation '{mutation}' for mixed encoding.")
    else:
        if crossover not in real_crossover:
            raise ValueError(f"Unsupported crossover '{crossover}' for real encoding.")
        if mutation not in real_mutation:
            raise ValueError(f"Unsupported mutation '{mutation}' for real encoding.")


def ensure_supported_operator_names(
    encoding: EncodingLike,
    crossover: str,
    mutation: str,
) -> tuple[CrossoverName, MutationName]:
    cross_name = cast(CrossoverName, crossover)
    mut_name = cast(MutationName, mutation)
    validate_operator_support(encoding, cross_name, mut_name)
    return cross_name, mut_name


def ensure_supported_repair_name(encoding: EncodingLike, repair: str) -> RepairName:
    normalized = normalize_encoding(encoding)
    if normalized != "real":
        raise ValueError("Repair operators are only supported for real encoding.")
    return cast(RepairName, repair)


def __getattr__(name: str) -> Registry[Any]:
    if name in _REGISTRY_LABELS:
        _populate_defaults()
        return _get_registry(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "REAL_CROSSOVER_METHODS",
    "REAL_MUTATION_METHODS",
    "THREE_PARENT_REAL_CROSSOVERS",
    "validate_pairwise_real_operators",
    "resolve_prob_expression",
    "prepare_mutation_params",
    "validate_operator_support",
    "ensure_supported_operator_names",
    "ensure_supported_repair_name",
    "PERM_CROSSOVER",
    "PERM_MUTATION",
    "BINARY_CROSSOVER",
    "BINARY_MUTATION",
    "INT_CROSSOVER",
    "INT_MUTATION",
    "MIXED_CROSSOVER",
    "MIXED_MUTATION",
]
