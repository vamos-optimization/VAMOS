"""Operator building for SPEA2."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

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
from vamos.foundation.encoding import EncodingLike, normalize_encoding

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import VariationKernels
    from vamos.foundation.problem.types import ProblemProtocol


VariationFnAlias = VariationFn

BINARY_CROSSOVER = {**BINARY_CROSSOVER_COMMON}
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
    problem: ProblemProtocol | None = None,
    *,
    variation_kernels: VariationKernels | None = None,
) -> tuple[VariationFnAlias, VariationFnAlias]:
    """Build variation operators for SPEA2."""
    mixed_spec = getattr(problem, "mixed_spec", None) if normalize_encoding(encoding) == "mixed" and problem is not None else None
    return build_discrete_variation_operators(
        config=cfg,
        algorithm_label="SPEA2",
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
) -> tuple[VariationFnAlias, VariationFnAlias]:
    return cast(
        tuple[VariationFnAlias, VariationFnAlias],
        build_real_variation_pair(
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
        ),
    )


__all__ = [
    "BINARY_CROSSOVER",
    "BINARY_MUTATION",
    "INT_CROSSOVER",
    "INT_MUTATION",
    "PERM_CROSSOVER",
    "PERM_MUTATION",
    "build_variation_operators",
]
