"""SMS-EMOA operator registration and building."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from vamos.engine.operators.policies._discrete_variation_builder import (
    VariationFn,
    build_discrete_variation_operators,
)
from vamos.engine.operators.policies._real_operator_builder import build_real_variation_pair
from vamos.engine.operators.policies.discrete_operator_maps import (
    BINARY_CROSSOVER_COMMON,
    BINARY_MUTATION_COMMON,
    INT_CROSSOVER_COMMON,
    INT_MUTATION_COMMON,
    PERM_CROSSOVER_COMMON,
    PERM_MUTATION_COMMON,
)
from vamos.engine.variation.protocol import RepairConfigValue
from vamos.foundation.encoding import EncodingLike

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import VariationKernels

__all__ = [
    "BINARY_CROSSOVER",
    "BINARY_MUTATION",
    "INT_CROSSOVER",
    "INT_MUTATION",
    "PERM_CROSSOVER",
    "PERM_MUTATION",
    "build_variation_operators",
]

VariationCrossoverFn = VariationFn
VariationMutationFn = VariationFn

BINARY_CROSSOVER = {**BINARY_CROSSOVER_COMMON}
BINARY_MUTATION = {**BINARY_MUTATION_COMMON}
INT_CROSSOVER = {**INT_CROSSOVER_COMMON}
INT_MUTATION = {**INT_MUTATION_COMMON}
PERM_CROSSOVER = {**PERM_CROSSOVER_COMMON}
PERM_MUTATION = {**PERM_MUTATION_COMMON}


def build_variation_operators(
    config: dict[str, Any],
    encoding: EncodingLike,
    n_var: int,
    xl: NDArray[np.generic],
    xu: NDArray[np.generic],
    rng: np.random.Generator,
    mixed_spec: dict[str, NDArray[np.generic]] | None = None,
    *,
    variation_kernels: VariationKernels | None = None,
) -> tuple[VariationCrossoverFn, VariationMutationFn]:
    """Build crossover and mutation operators for the given encoding."""
    return build_discrete_variation_operators(
        config=config,
        algorithm_label="SMSEMOA",
        encoding=encoding,
        n_var=n_var,
        xl=xl,
        xu=xu,
        rng=rng,
        real_builder=_build_real_operators,
        mixed_spec=mixed_spec,
        binary_crossover=BINARY_CROSSOVER,
        binary_mutation=BINARY_MUTATION,
        integer_crossover=INT_CROSSOVER,
        integer_mutation=INT_MUTATION,
        permutation_crossover=PERM_CROSSOVER,
        permutation_mutation=PERM_MUTATION,
        variation_kernels=variation_kernels,
    )


def _build_real_operators(
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
    """Build continuous (real) encoding operators via the operator registry."""
    base_crossover, base_mutation = build_real_variation_pair(
        cross_method=cross_method,
        cross_params=cross_params,
        mut_method=mut_method,
        mut_params=mut_params,
        n_var=n_var,
        xl=xl,
        xu=xu,
        rng=rng,
        repair_cfg=repair_cfg,
        variation_kernels=variation_kernels,
    )

    def crossover(parents: NDArray[np.generic]) -> NDArray[np.generic]:
        return base_crossover(parents, rng)

    def mutation(X_child: NDArray[np.generic]) -> NDArray[np.generic]:
        return base_mutation(X_child, rng)

    return crossover, mutation
