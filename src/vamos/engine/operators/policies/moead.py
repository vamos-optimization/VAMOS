# operators/policies/moead.py
"""Operator building for MOEA/D."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.components.utils import resolve_prob_expression
from vamos.engine.operators.impl.binary import one_point_crossover
from vamos.engine.operators.impl.real import ClampRepair, PolynomialMutation, SBXCrossover, VariationWorkspace
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
from vamos.engine.variation.helpers import validate_pairwise_real_operators
from vamos.engine.variation.protocol import RepairConfigValue, RepairOperator
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
    de_aliases = {"de", "differential", "differential_evolution"}
    # MOEA/D feeds DE three parents (the subproblem and two neighbours); other crossovers get pairs.
    validate_pairwise_real_operators(
        "de" if method in de_aliases else method,
        mut_method or "polynomial",
        algorithm_label="MOEA/D",
        three_parent_allowed={"de"},
    )
    workspace = VariationWorkspace()
    repair_operator = resolve_policy_repair("real", repair_cfg)
    assert repair_operator is not None

    crossover: DifferentialCrossoverStep | RepairedOperatorStep
    if method in de_aliases:
        crossover = DifferentialCrossoverStep(
            f=float(cross_params.get("f", cross_params.get("F", 0.5))),
            cr=float(cross_params.get("cr", cross_params.get("CR", 1.0))),
            repair=repair_operator,
            xl=xl,
            xu=xu,
            rng=rng,
            variation_kernels=variation_kernels,
        )
    else:
        cross_kwargs = dict(cross_params)
        prob = cross_kwargs.pop("prob", None)
        cross_kwargs.setdefault("prob_crossover", 0.9 if prob is None else float(prob))
        cross_kwargs.setdefault("allow_inplace", True)
        cross_kwargs.setdefault("lower", xl)
        cross_kwargs.setdefault("upper", xu)
        cross_kwargs.setdefault("workspace", workspace)
        crossover_operator = instantiate_operator(method, cross_kwargs, label="crossover", variation_kernels=variation_kernels)
        crossover = RepairedOperatorStep(crossover_operator, repair_operator, xl, xu, rng)

    mut_prob = resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
    mut_name = (mut_method or "polynomial").lower()

    mut_kwargs = dict(mut_params)
    mut_kwargs.pop("prob", None)
    mut_kwargs.setdefault("prob_mutation", mut_prob)
    mut_kwargs.setdefault("lower", xl)
    mut_kwargs.setdefault("upper", xu)
    mut_kwargs.setdefault("workspace", workspace)
    mutation_operator = instantiate_operator(mut_name, mut_kwargs, label="mutation", variation_kernels=variation_kernels)
    return crossover, RepairedOperatorStep(mutation_operator, repair_operator, xl, xu, rng)


class DifferentialCrossoverStep:
    """DE/rand/1/bin children ``base + f * (p1 - p2)`` of ``(p1, p2, base)`` parents, then the policy repair."""

    def __init__(
        self,
        *,
        f: float,
        cr: float,
        repair: RepairOperator,
        xl: NDArray[np.generic],
        xu: NDArray[np.generic],
        rng: np.random.Generator,
        variation_kernels: VariationKernels | None,
    ) -> None:
        self.f = f
        self.cr = cr
        self.repair = repair
        self.xl = xl
        self.xu = xu
        self.rng = rng
        self.variation_kernels = variation_kernels

    def __call__(self, parents: NDArray[np.floating[Any]], _rng: np.random.Generator | None = None) -> NDArray[np.generic]:
        rng = self.rng if _rng is None else _rng
        parents_arr = np.asarray(parents)
        if parents_arr.ndim != 3 or parents_arr.shape[1] != 3:
            raise ValueError("DE crossover expects parents shaped (n_pairs, 3, n_var).")
        n_pairs, _, n_vars = parents_arr.shape
        f, cr = self.f, self.cr
        if self.variation_kernels is not None and parents_arr.dtype == np.float64:
            # Same draws (uniform grid, then one forced index per child) and arithmetic.
            rand = rng.random((n_pairs, n_vars))
            j_rand = rng.integers(0, n_vars, size=n_pairs)
            child = self.variation_kernels.differential_crossover(parents_arr, f, cr, rand, j_rand)
        else:
            base = parents_arr[:, 2, :]
            p1 = parents_arr[:, 0, :]
            p2 = parents_arr[:, 1, :]
            mutant = base + f * (p1 - p2)
            rand = rng.random((n_pairs, n_vars))
            mask = rand < cr
            j_rand = rng.integers(0, n_vars, size=n_pairs)
            mask[np.arange(n_pairs), j_rand] = True
            child = np.where(mask, mutant, base)
        offspring = child[:, None, :]
        return apply_policy_repair(self.repair, offspring, self.xl, self.xu, rng)


class RepairedOperatorStep:
    """A registry crossover or mutation operator followed by the policy repair."""

    def __init__(
        self,
        operator: Callable[..., object],
        repair: RepairOperator,
        xl: NDArray[np.generic],
        xu: NDArray[np.generic],
        rng: np.random.Generator,
    ) -> None:
        self.operator = operator
        self.repair = repair
        self.xl = xl
        self.xu = xu
        self.rng = rng

    def __call__(self, values: NDArray[np.generic], _rng: np.random.Generator | None = None) -> NDArray[np.generic]:
        rng = self.rng if _rng is None else _rng
        result = np.asarray(self.operator(values, rng))
        return apply_policy_repair(self.repair, result, self.xl, self.xu, rng)


@dataclass(frozen=True)
class OneOffspringRealVariation:
    """Resolved parameters of a real DE or SBX, polynomial mutation and clip pipeline.

    The values are read from the operators that :func:`build_variation_operators`
    returned, so compiled code that consumes the same draws reproduces them exactly.
    ``crossover`` is ``"de"`` or ``"sbx"``; the parameters of the other crossover are unused.
    """

    crossover: str
    de_f: float
    de_cr: float
    sbx_prob: float
    sbx_eta: float
    sbx_prob_var: float
    sbx_lower: NDArray[np.float64]
    sbx_upper: NDArray[np.float64]
    pm_prob: float
    pm_eta: float
    pm_lower: NDArray[np.float64]
    pm_upper: NDArray[np.float64]
    clip_lower: NDArray[np.float64]
    clip_upper: NDArray[np.float64]


def one_offspring_real_variation(crossover: object, mutation: object, n_var: int) -> OneOffspringRealVariation | None:
    """Describe operators built here when a compiled one-offspring step can reproduce them.

    That needs DE or SBX crossover and polynomial mutation that both use compiled
    variation kernels, the workspace-backed mutation draws, the clip repair, and
    float bounds shaped ``(n_var,)`` with ``lower <= upper``. Any other pipeline
    returns ``None``; callers then keep the operators themselves.
    """
    if type(mutation) is not RepairedOperatorStep or type(mutation.operator) is not PolynomialMutation:
        return None
    pm = mutation.operator
    if pm.variation_kernels is None or pm.workspace is None:
        return None
    sbx: SBXCrossover | None = None
    if type(crossover) is DifferentialCrossoverStep:
        if crossover.variation_kernels is None:
            return None
    elif type(crossover) is RepairedOperatorStep and type(crossover.operator) is SBXCrossover:
        sbx = crossover.operator
        if sbx.variation_kernels is None:
            return None
    else:
        return None
    if not (_exact_type(crossover.repair, ClampRepair) and _exact_type(mutation.repair, ClampRepair)):
        return None
    if crossover.xl is not mutation.xl or crossover.xu is not mutation.xu:
        return None
    clip_lower = np.asarray(crossover.xl, dtype=float)
    clip_upper = np.asarray(crossover.xu, dtype=float)
    bounds = [(clip_lower, clip_upper), (pm.lower, pm.upper)]
    if sbx is not None:
        bounds.append((sbx.lower, sbx.upper))
    for lower, upper in bounds:
        if lower.shape != (n_var,) or upper.shape != (n_var,) or lower.dtype != np.float64 or upper.dtype != np.float64:
            return None
        if bool(np.any(lower > upper)):
            return None
    de = crossover if type(crossover) is DifferentialCrossoverStep else None
    return OneOffspringRealVariation(
        crossover="sbx" if sbx is not None else "de",
        de_f=de.f if de is not None else 0.0,
        de_cr=de.cr if de is not None else 0.0,
        sbx_prob=sbx.prob if sbx is not None else 0.0,
        sbx_eta=sbx.eta if sbx is not None else 0.0,
        sbx_prob_var=sbx.prob_var if sbx is not None else 0.0,
        sbx_lower=sbx.lower if sbx is not None else clip_lower,
        sbx_upper=sbx.upper if sbx is not None else clip_upper,
        pm_prob=pm.prob,
        pm_eta=pm.eta,
        pm_lower=pm.lower,
        pm_upper=pm.upper,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )


def _exact_type(value: object, cls: type[object]) -> bool:
    # Subclasses may change the arithmetic, so only the exact operator class is described.
    return type(value) is cls


__all__ = [
    "BINARY_CROSSOVER",
    "BINARY_MUTATION",
    "INT_CROSSOVER",
    "INT_MUTATION",
    "PERM_CROSSOVER",
    "PERM_MUTATION",
    "DifferentialCrossoverStep",
    "OneOffspringRealVariation",
    "RepairedOperatorStep",
    "build_variation_operators",
    "one_offspring_real_variation",
]
