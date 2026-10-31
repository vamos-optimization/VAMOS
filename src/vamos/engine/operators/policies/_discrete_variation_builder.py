"""Shared helpers for discrete and mixed variation-policy assembly."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from vamos.engine.operators.impl.integer import (
    creep_mutation,
    gaussian_integer_mutation,
    integer_polynomial_mutation,
    integer_sbx_crossover,
)
from vamos.engine.operators.impl.mixed import mixed_crossover, mixed_mutation
from vamos.engine.operators.policies.discrete_operator_maps import (
    BINARY_CROSSOVER_COMMON,
    BINARY_MUTATION_COMMON,
    INT_CROSSOVER_COMMON,
    INT_MUTATION_COMMON,
    PERM_CROSSOVER_COMMON,
    PERM_MUTATION_COMMON,
    BinaryCrossoverOp,
    BinaryMutationOp,
    IntCrossoverOp,
    IntMutationOp,
    PermCrossoverOp,
    PermMutationOp,
)
from vamos.engine.variation.helpers import resolve_prob_expression
from vamos.engine.variation.protocol import RepairConfigValue
from vamos.foundation.encoding import EncodingLike, normalize_encoding

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import VariationKernels

VariationFn: TypeAlias = Callable[..., NDArray[np.generic]]
RealOperatorBuilder: TypeAlias = Callable[
    [
        str,
        dict[str, Any],
        str,
        dict[str, Any],
        int,
        NDArray[np.generic],
        NDArray[np.generic],
        np.random.Generator,
        RepairConfigValue,
        "VariationKernels | None",
    ],
    tuple[VariationFn, VariationFn],
]


def unpack_variation_methods(
    config: Mapping[str, Any],
    *,
    default_crossover: str = "sbx",
    default_mutation: str = "polynomial",
) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    cross_method, cross_params = _normalize_operator_entry(config.get("crossover", (default_crossover, {})), default_crossover)
    mut_method, mut_params = _normalize_operator_entry(config.get("mutation", (default_mutation, {})), default_mutation)
    return cross_method, cross_params, mut_method, mut_params


def build_discrete_variation_operators(
    *,
    config: Mapping[str, Any],
    algorithm_label: str,
    encoding: EncodingLike,
    n_var: int,
    xl: NDArray[np.generic],
    xu: NDArray[np.generic],
    rng: np.random.Generator,
    real_builder: RealOperatorBuilder,
    mixed_spec: dict[str, NDArray[np.generic]] | None = None,
    binary_crossover: Mapping[str, BinaryCrossoverOp] = BINARY_CROSSOVER_COMMON,
    binary_mutation: Mapping[str, BinaryMutationOp] = BINARY_MUTATION_COMMON,
    integer_crossover: Mapping[str, IntCrossoverOp] = INT_CROSSOVER_COMMON,
    integer_mutation: Mapping[str, IntMutationOp] = INT_MUTATION_COMMON,
    permutation_crossover: Mapping[str, PermCrossoverOp] = PERM_CROSSOVER_COMMON,
    permutation_mutation: Mapping[str, PermMutationOp] = PERM_MUTATION_COMMON,
    variation_kernels: VariationKernels | None = None,
) -> tuple[VariationFn, VariationFn]:
    cross_method, cross_params, mut_method, mut_params = unpack_variation_methods(config)

    normalized = normalize_encoding(encoding)
    repair_cfg = cast(RepairConfigValue, config.get("repair", "auto"))
    if normalized != "real" and repair_cfg != "auto":
        raise ValueError("Repair operators are only supported for real encoding.")
    if normalized == "binary":
        return _build_binary_operators(
            algorithm_label,
            cross_method,
            cross_params,
            mut_method,
            mut_params,
            n_var,
            rng,
            binary_crossover,
            binary_mutation,
        )
    if normalized == "integer":
        return _build_integer_operators(
            algorithm_label,
            cross_method,
            cross_params,
            mut_method,
            mut_params,
            n_var,
            xl,
            xu,
            rng,
            integer_crossover,
            integer_mutation,
        )
    if normalized == "permutation":
        return _build_permutation_operators(
            algorithm_label,
            cross_method,
            cross_params,
            mut_method,
            mut_params,
            n_var,
            rng,
            permutation_crossover,
            permutation_mutation,
        )
    if normalized == "mixed":
        if mixed_spec is None:
            raise ValueError(f"{algorithm_label} mixed encoding requires problem.mixed_spec.")
        return _build_mixed_operators(
            algorithm_label,
            cross_method,
            cross_params,
            mut_method,
            mut_params,
            n_var,
            mixed_spec,
            rng,
        )
    if normalized == "real":
        return real_builder(cross_method, cross_params, mut_method, mut_params, n_var, xl, xu, rng, repair_cfg, variation_kernels)
    raise ValueError(f"{algorithm_label} does not support encoding '{normalized}'.")


def _normalize_operator_entry(value: Any, default_method: str) -> tuple[str, dict[str, Any]]:
    if isinstance(value, tuple):
        method, params = value
        return str(method).lower(), dict(params) if params else {}
    return default_method, dict(value or {})


def _build_binary_operators(
    algorithm_label: str,
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    n_var: int,
    rng: np.random.Generator,
    crossover_map: Mapping[str, BinaryCrossoverOp],
    mutation_map: Mapping[str, BinaryMutationOp],
) -> tuple[VariationFn, VariationFn]:
    if cross_method not in crossover_map:
        raise ValueError(f"Unsupported {algorithm_label} crossover '{cross_method}' for binary encoding.")
    if mut_method not in mutation_map:
        raise ValueError(f"Unsupported {algorithm_label} mutation '{mut_method}' for binary encoding.")

    cross_fn = crossover_map[cross_method]
    cross_prob = float(cross_params.get("prob", 0.9))
    mut_fn = mutation_map[mut_method]
    mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))

    def crossover(parents: NDArray[np.generic], _rng: np.random.Generator = rng) -> NDArray[np.generic]:
        return cross_fn(parents, cross_prob, _rng)

    def mutation(X_child: NDArray[np.generic], _rng: np.random.Generator = rng) -> NDArray[np.generic]:
        mut_fn(X_child, mut_prob, _rng)
        return X_child

    return crossover, mutation


def _build_integer_operators(
    algorithm_label: str,
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    n_var: int,
    xl: NDArray[np.generic],
    xu: NDArray[np.generic],
    rng: np.random.Generator,
    crossover_map: Mapping[str, IntCrossoverOp],
    mutation_map: Mapping[str, IntMutationOp],
) -> tuple[VariationFn, VariationFn]:
    if cross_method not in crossover_map:
        raise ValueError(f"Unsupported {algorithm_label} crossover '{cross_method}' for integer encoding.")
    if mut_method not in mutation_map:
        raise ValueError(f"Unsupported {algorithm_label} mutation '{mut_method}' for integer encoding.")

    cross_fn = crossover_map[cross_method]
    cross_prob = float(cross_params.get("prob", 0.9))
    mut_fn = mutation_map[mut_method]
    mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
    step = int(mut_params.get("step", 1))

    if cross_fn is integer_sbx_crossover:
        eta = float(cross_params.get("eta", 20.0))

        def crossover(parents: NDArray[np.generic], _rng: np.random.Generator = rng) -> NDArray[np.generic]:
            return integer_sbx_crossover(parents, cross_prob, eta, xl, xu, _rng)

    else:

        def crossover(parents: NDArray[np.generic], _rng: np.random.Generator = rng) -> NDArray[np.generic]:
            return cross_fn(parents, cross_prob, _rng)

    if mut_fn is creep_mutation:

        def mutation(X_child: NDArray[np.generic], _rng: np.random.Generator = rng) -> NDArray[np.generic]:
            creep_mutation(X_child, mut_prob, step, xl, xu, _rng)
            return X_child

    elif mut_fn is integer_polynomial_mutation:
        eta = float(mut_params.get("eta", 20.0))

        def mutation(X_child: NDArray[np.generic], _rng: np.random.Generator = rng) -> NDArray[np.generic]:
            integer_polynomial_mutation(X_child, mut_prob, eta, xl, xu, _rng)
            return X_child

    elif mut_fn is gaussian_integer_mutation:
        sigma = float(mut_params.get("sigma", 1.0))

        def mutation(X_child: NDArray[np.generic], _rng: np.random.Generator = rng) -> NDArray[np.generic]:
            gaussian_integer_mutation(X_child, mut_prob, sigma, xl, xu, _rng)
            return X_child

    else:

        def mutation(X_child: NDArray[np.generic], _rng: np.random.Generator = rng) -> NDArray[np.generic]:
            mut_fn(X_child, mut_prob, xl, xu, _rng)
            return X_child

    return crossover, mutation


def _build_permutation_operators(
    algorithm_label: str,
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    n_var: int,
    rng: np.random.Generator,
    crossover_map: Mapping[str, PermCrossoverOp],
    mutation_map: Mapping[str, PermMutationOp],
) -> tuple[VariationFn, VariationFn]:
    if cross_method not in crossover_map:
        raise ValueError(f"Unsupported {algorithm_label} crossover '{cross_method}' for permutation encoding.")
    if mut_method not in mutation_map:
        raise ValueError(f"Unsupported {algorithm_label} mutation '{mut_method}' for permutation encoding.")

    cross_fn = crossover_map[cross_method]
    cross_prob = float(cross_params.get("prob", 0.9))
    mut_fn = mutation_map[mut_method]
    mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))

    def crossover(parents: NDArray[np.generic], _rng: np.random.Generator = rng) -> NDArray[np.generic]:
        flat_shape = parents.shape
        parents_flat = parents.reshape(-1, flat_shape[-1])
        offspring_flat = cross_fn(parents_flat, cross_prob, _rng)
        return offspring_flat.reshape(flat_shape)

    def mutation(X_child: NDArray[np.generic], _rng: np.random.Generator = rng) -> NDArray[np.generic]:
        mut_fn(X_child, mut_prob, _rng)
        return X_child

    return crossover, mutation


def _build_mixed_operators(
    algorithm_label: str,
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    n_var: int,
    mixed_spec: dict[str, NDArray[np.generic]],
    rng: np.random.Generator,
) -> tuple[VariationFn, VariationFn]:
    if cross_method not in {"mixed", "uniform"}:
        raise ValueError(f"Unsupported {algorithm_label} crossover '{cross_method}' for mixed encoding.")
    if mut_method not in {"mixed", "gaussian"}:
        raise ValueError(f"Unsupported {algorithm_label} mutation '{mut_method}' for mixed encoding.")

    cross_prob = float(cross_params.get("prob", 0.9))
    mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))

    def crossover(parents: NDArray[np.generic], _rng: np.random.Generator = rng) -> NDArray[np.generic]:
        parent_shape = parents.shape
        parents_flat = parents.reshape(-1, parent_shape[-1])
        offspring_flat = mixed_crossover(parents_flat, cross_prob, mixed_spec, _rng)
        return offspring_flat.reshape(parent_shape)

    def mutation(X_child: NDArray[np.generic], _rng: np.random.Generator = rng) -> NDArray[np.generic]:
        mixed_mutation(X_child, mut_prob, mixed_spec, _rng)
        return X_child

    return crossover, mutation


__all__ = [
    "VariationFn",
    "build_discrete_variation_operators",
    "unpack_variation_methods",
]
