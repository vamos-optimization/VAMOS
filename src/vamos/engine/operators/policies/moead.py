# operators/policies/moead.py
"""Operator building for MOEA/D."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.components.utils import resolve_prob_expression
from vamos.engine.operators.impl.binary import one_point_crossover
from vamos.engine.operators.impl.real import VariationWorkspace
from vamos.engine.operators.policies._discrete_variation_builder import (
    VariationFn,
    build_discrete_variation_operators,
)
from vamos.engine.operators.policies._real_operator_builder import instantiate_operator
from vamos.engine.operators.policies.discrete_operator_maps import (
    BINARY_CROSSOVER_COMMON,
    BINARY_MUTATION_COMMON,
    INT_CROSSOVER_COMMON,
    INT_MUTATION_COMMON,
    PERM_CROSSOVER_COMMON,
    PERM_MUTATION_COMMON,
)
from vamos.engine.operators.policies.real_repair import apply_policy_repair, resolve_policy_repair
from vamos.engine.variation.protocol import RepairConfigValue
from vamos.foundation.encoding import EncodingLike

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import VariationKernels

VariationCrossoverFn = VariationFn
VariationMutationFn = VariationFn

BINARY_CROSSOVER: dict[str, Any] = {
    **BINARY_CROSSOVER_COMMON,
    "spx": one_point_crossover,
}
BINARY_MUTATION = {**BINARY_MUTATION_COMMON}
INT_CROSSOVER = {**INT_CROSSOVER_COMMON}
INT_MUTATION = {**INT_MUTATION_COMMON}
PERM_CROSSOVER = {**PERM_CROSSOVER_COMMON}
PERM_MUTATION = {**PERM_MUTATION_COMMON}


def build_variation_operators(
    cfg: dict[str, Any],
    encoding: EncodingLike,
    n_var: int,
    xl: NDArray[np.generic],
    xu: NDArray[np.generic],
    rng: np.random.Generator,
    mixed_spec: dict[str, NDArray[np.generic]] | None = None,
    *,
    variation_kernels: VariationKernels | None = None,
) -> tuple[VariationCrossoverFn, VariationMutationFn]:
    """Build crossover and mutation operators for MOEA/D."""
    return build_discrete_variation_operators(
        config=cfg,
        algorithm_label="MOEA/D",
        encoding=encoding,
        n_var=n_var,
        xl=xl,
        xu=xu,
        rng=rng,
        real_builder=_build_continuous_operators,
        mixed_spec=mixed_spec,
        binary_crossover=BINARY_CROSSOVER,
        binary_mutation=BINARY_MUTATION,
        integer_crossover=INT_CROSSOVER,
        integer_mutation=INT_MUTATION,
        permutation_crossover=PERM_CROSSOVER,
        permutation_mutation=PERM_MUTATION,
        variation_kernels=variation_kernels,
    )


def _build_continuous_operators(
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    n_var: int,
    xl: NDArray[np.generic],
    xu: NDArray[np.generic],
    rng: np.random.Generator,
    repair_cfg: RepairConfigValue,
    variation_kernels: VariationKernels | None = None,
) -> tuple[VariationCrossoverFn, VariationMutationFn]:
    """Build variation operators for continuous/real encoding."""
    method = (cross_method or "sbx").lower()
    workspace = VariationWorkspace()
    repair_operator = resolve_policy_repair("real", repair_cfg)
    assert repair_operator is not None

    if method in {"de", "differential", "differential_evolution"}:
        cr = float(cross_params.get("cr", cross_params.get("CR", 1.0)))
        f = float(cross_params.get("f", cross_params.get("F", 0.5)))

        def crossover(parents: NDArray[np.floating[Any]], _rng: np.random.Generator = rng) -> NDArray[np.generic]:
            parents_arr = np.asarray(parents)
            if parents_arr.ndim != 3 or parents_arr.shape[1] != 3:
                raise ValueError("DE crossover expects parents shaped (n_pairs, 3, n_var).")
            n_pairs, _, n_vars = parents_arr.shape
            if variation_kernels is not None and parents_arr.dtype == np.float64:
                # Same draws (uniform grid, then one forced index per child) and arithmetic.
                rand = _rng.random((n_pairs, n_vars))
                j_rand = _rng.integers(0, n_vars, size=n_pairs)
                child = variation_kernels.differential_crossover(parents_arr, f, cr, rand, j_rand)
            else:
                base = parents_arr[:, 2, :]
                p1 = parents_arr[:, 0, :]
                p2 = parents_arr[:, 1, :]
                mutant = base + f * (p1 - p2)
                rand = _rng.random((n_pairs, n_vars))
                mask = rand < cr
                j_rand = _rng.integers(0, n_vars, size=n_pairs)
                mask[np.arange(n_pairs), j_rand] = True
                child = np.where(mask, mutant, base)
            offspring = child[:, None, :]
            return apply_policy_repair(repair_operator, offspring, xl, xu, _rng)

    else:
        cross_kwargs = dict(cross_params)
        prob = cross_kwargs.pop("prob", None)
        cross_kwargs.setdefault("prob_crossover", 0.9 if prob is None else float(prob))
        cross_kwargs.setdefault("allow_inplace", True)
        cross_kwargs.setdefault("lower", xl)
        cross_kwargs.setdefault("upper", xu)
        cross_kwargs.setdefault("workspace", workspace)
        crossover_operator = instantiate_operator(method, cross_kwargs, label="crossover", variation_kernels=variation_kernels)

        def crossover(parents: NDArray[np.floating[Any]], _rng: np.random.Generator = rng) -> NDArray[np.generic]:
            offspring = np.asarray(crossover_operator(parents, _rng))
            return apply_policy_repair(repair_operator, offspring, xl, xu, _rng)

    mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
    mut_name = (mut_method or "polynomial").lower()

    mut_kwargs = dict(mut_params)
    mut_kwargs.pop("prob", None)
    mut_kwargs.setdefault("prob_mutation", mut_prob)
    mut_kwargs.setdefault("lower", xl)
    mut_kwargs.setdefault("upper", xu)
    mut_kwargs.setdefault("workspace", workspace)
    mutation_operator = instantiate_operator(mut_name, mut_kwargs, label="mutation", variation_kernels=variation_kernels)

    def mutation(X_child: NDArray[np.generic], _rng: np.random.Generator = rng) -> NDArray[np.generic]:
        mutated = np.asarray(mutation_operator(X_child, _rng))
        return apply_policy_repair(repair_operator, mutated, xl, xu, _rng)

    return crossover, mutation


__all__ = [
    "BINARY_CROSSOVER",
    "BINARY_MUTATION",
    "INT_CROSSOVER",
    "INT_MUTATION",
    "PERM_CROSSOVER",
    "PERM_MUTATION",
    "build_variation_operators",
]
