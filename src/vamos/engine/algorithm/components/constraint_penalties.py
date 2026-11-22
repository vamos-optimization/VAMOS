"""The constraint-handling techniques that jMetal and Evolver do not have (see :mod:`.constraint_handling`).

Each ranks by Pareto dominance over modified objectives, as jMetal's self-adaptive penalty does: a static
penalty, Joines and Houck's non-stationary penalty, the overall violation as one more objective, and Ma and
Wang's shift-based penalty.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .constrained_ranking import RankArray, pareto_ranks, violation
from .constraint_technique import ConstraintHandlingTechnique, check_progress

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import KernelBackend


class StaticPenalty(ConstraintHandlingTechnique):
    """A static penalty: Pareto dominance over every objective plus ``coefficient`` times the overall violation.

    The overall violation is the sum of the positive constraint values, as the other techniques measure
    it. A row that violates no constraint keeps its objectives, so a feasible population ranks by Pareto
    dominance alone.
    """

    name = "static_penalty"

    def __init__(self, coefficient: float) -> None:
        self.coefficient = float(coefficient)

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        return pareto_ranks(kernel, penalized_objectives(F, self.coefficient * violation(G)))


class DynamicPenalty(ConstraintHandlingTechnique):
    """Joines and Houck's non-stationary penalty, as a ranking by Pareto dominance.

    J. A. Joines and C. R. Houck, "On the use of non-stationary penalty functions to solve nonlinear
    constrained optimization problems with GA's", Proc. First IEEE Conference on Evolutionary Computation,
    1994, pp. 579-584, section 3.1: the GA minimizes ``F = f + P(alpha, beta)`` with ``P(alpha, beta) =
    (C * k) ** alpha * SVC(beta)``, where ``k`` is the generation and ``SVC(beta)`` sums each violation
    raised to ``beta``. Each objective becomes ``f_i + (constant * k) ** alpha * SVC``, the same penalty
    for every objective. VAMOS stores every constraint as ``g_j <= 0`` and measures its violation as
    ``max(0, g_j)``, without the tolerance band that the paper gives each constraint.

    ``k`` counts the run's evaluations in populations, and is at least 1: the progress of each
    :meth:`update` (the evaluations before the offspring's, over the budget) times the budget over the
    population size, which :meth:`bind_run` gives. A generational algorithm ranks its initial population
    and its first replacement at ``k = 1`` and its ``k``-th replacement at ``k``; a steady-state one adds
    ``1 / population_size`` per offspring rather than 1.
    """

    name = "dynamic_penalty"
    needs_progress = True

    def __init__(self, constant: float, alpha: float, beta: float) -> None:
        self.constant = float(constant)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.population_size = 0
        self.max_evaluations = 0
        self.generation = 1.0

    def bind_run(self, population_size: int, max_evaluations: int) -> None:
        if population_size < 1 or max_evaluations < 1:
            raise ValueError(f"dynamic_penalty needs a positive population and budget; got {population_size} and {max_evaluations}.")
        self.population_size = int(population_size)
        self.max_evaluations = int(max_evaluations)

    def update(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], progress: float) -> None:
        progress = check_progress(progress, "dynamic_penalty")
        if self.population_size < 1:
            raise ValueError("dynamic_penalty counts generations: give it the run's population and budget with bind_run().")
        self.generation = max(1.0, progress * self.max_evaluations / self.population_size)

    @property
    def weight(self) -> float:
        """``(constant * k) ** alpha``, the multiplier of the violations at the current generation."""
        return float((self.constant * self.generation) ** self.alpha)

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        excess = np.maximum(np.asarray(G, dtype=np.float64), 0.0)
        severity = np.asarray(np.sum(excess**self.beta, axis=1), dtype=np.float64)
        return pareto_ranks(kernel, penalized_objectives(F, self.weight * severity))

    def state(self) -> dict[str, Any]:
        return {"generation": self.generation}

    def restore(self, state: Mapping[str, Any]) -> None:
        self.generation = float(state.get("generation", 1.0))


class ViolationObjective(ConstraintHandlingTechnique):
    """The overall violation as one more objective (Coello, 2000), with Dam and Truong's options.

    T. M. Dam and L. V. Truong, "A Generalized Framework for Multi-objective-based Constraint Handling Technique",
    International Journal of Control, Automation, and Systems 22(10):3240-3252, 2024, section 3:

    - ``violation_norm``: the violation ``V = ||v||_p^p`` of the positive constraint values ``v`` (Def. 7): their
      sum (``l1``, the default), the sum of their squares (``l2_squared``) or the largest (``linf``).
    - ``ratio_objective``: the constraint violation ratio ``R`` (Def. 6), the weighted share of the violated
      constraints, is one more objective before ``V``. The weights are equal, or with ``cdf_alpha`` above 0 those
      of the CDF (section 3.2), ``1 + cdf_alpha * f_j / sum_i f_i``, where ``f_j`` is how many rows violate
      constraint ``j``.
    - ``feasible_first``: Algorithm A's survival, a total order and NSGA-II only. The feasible rows come first, by
      fronts and crowding distance of the objectives, then the infeasible ones by ``R`` (with ``ratio_objective``)
      and then ``V``, in ascending order; equal values keep the order of the population.

    Otherwise the ranks are Pareto dominance over the objectives, ``R`` and ``V``. A feasible population, whose
    ``R`` and ``V`` are 0, ranks by Pareto dominance over its objectives.
    """

    name = "violation_objective"

    def __init__(
        self, violation_norm: str = "l1", *, ratio_objective: bool = False, cdf_alpha: float = 0.0, feasible_first: bool = False
    ) -> None:
        self.violation_norm = violation_norm
        self.ratio_objective = ratio_objective
        self.cdf_alpha = float(cdf_alpha)
        self.feasible_first = feasible_first
        self.total_order = feasible_first

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        F = np.asarray(F, dtype=np.float64)
        extra = violation_objectives(G, self.violation_norm, self.ratio_objective, self.cdf_alpha)
        if not self.feasible_first:
            return pareto_ranks(kernel, np.column_stack([F, extra]))
        rows = np.arange(F.shape[0])
        # From G, not from V: a violation whose square underflows has l2_squared V of 0, and it is infeasible all the same.
        feasible = np.all(np.asarray(G, dtype=np.float64) <= 0.0, axis=1)
        ranks, crowding = kernel.nsga2_ranking(np.ascontiguousarray(F[feasible])) if np.any(feasible) else (np.empty(0), np.empty(0))
        first = rows[feasible][np.lexsort((rows[feasible], -np.asarray(crowding, dtype=np.float64), np.asarray(ranks)))]
        infeasible = rows[~feasible]
        then = infeasible[np.lexsort((infeasible, *extra[~feasible][:, ::-1].T))]
        order = np.concatenate([first, then])
        positions = np.empty(order.size, dtype=np.int64)
        positions[order] = np.arange(order.size, dtype=np.int64)
        return positions


def violation_objectives(
    G: NDArray[Any], violation_norm: str = "l1", ratio_objective: bool = False, cdf_alpha: float = 0.0
) -> NDArray[np.float64]:
    """The objectives that ``violation_objective`` adds: ``[R, V]`` with ``ratio_objective``, otherwise ``[V]``."""
    excess = np.maximum(np.asarray(G, dtype=np.float64), 0.0)
    if violation_norm == "l1":
        measure = violation(G)
    elif violation_norm == "l2_squared":
        measure = np.asarray(np.sum(excess * excess, axis=1), dtype=np.float64)
    else:
        measure = np.asarray(excess.max(axis=1), dtype=np.float64) if excess.shape[1] else np.zeros(excess.shape[0])
    if not ratio_objective:
        return measure[:, None]
    violated = excess > 0.0
    counts = violated.sum(axis=0).astype(np.float64)
    total = counts.sum()
    weights = 1.0 + cdf_alpha * counts / total if total > 0.0 else np.ones(excess.shape[1])
    ratio = (violated * weights).sum(axis=1) / weights.sum() if excess.shape[1] else np.zeros(excess.shape[0])
    return np.column_stack([ratio, measure])


class ShiftPenalty(ConstraintHandlingTechnique):
    """Ma and Wang's shift-based penalty, ShiP: Pareto dominance over the shifted and penalized objectives.

    Z. Ma and Y. Wang, "Shift-Based Penalty for Evolutionary Constrained Multiobjective Optimization and Its
    Application", IEEE Transactions on Cybernetics 53(1):18-30, 2023, section III-B (:func:`shift_penalty_objectives`).
    ``constant`` is the paper's ``c``, 2 in its experiments.
    """

    name = "shift_penalty"

    def __init__(self, constant: float) -> None:
        self.constant = float(constant)

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        return pareto_ranks(kernel, shift_penalty_objectives(F, G, self.constant))


def shift_penalty_objectives(F: NDArray[Any], G: NDArray[Any], constant: float) -> NDArray[np.float64]:
    """ShiP's objectives (eqs. 11-13): an infeasible row moves towards its local feasible nadir, plus its violation.

    ``P_fea`` is the proportion of feasible rows. For an infeasible row ``u`` and each objective ``i``, ``z*_i``
    is the smallest ``f_i`` of the feasible rows whose ``f_i`` is greater than ``f_i(u)``, or the largest
    ``f_i`` of all the rows if there is none (the virtual feasible extreme value). The row's objective becomes
    ``f_i + constant * P_fea * (z*_i - f_i) + CV``, with ``CV`` the overall violation. Feasible rows, and every
    row while none is feasible, keep their objectives.
    """
    F = np.asarray(F, dtype=np.float64)
    measure = violation(G)
    feasible = measure <= 0.0
    shifted = F.copy()
    if F.shape[0] == 0 or not np.any(feasible) or np.all(feasible):
        return shifted
    share = float(np.count_nonzero(feasible)) / F.shape[0]
    infeasible = np.flatnonzero(~feasible)
    largest = F.max(axis=0)
    for i in range(F.shape[1]):
        values = np.sort(F[feasible, i])
        own = F[infeasible, i]
        above = np.searchsorted(values, own, side="right")
        nadir = np.where(above < values.size, values[np.minimum(above, values.size - 1)], largest[i])
        shifted[infeasible, i] = own + constant * share * (nadir - own) + measure[infeasible]
    return shifted


def penalized_objectives(F: NDArray[Any], penalty: NDArray[np.float64]) -> NDArray[np.float64]:
    """Every objective of each row plus that row's penalty."""
    return np.asarray(np.asarray(F, dtype=np.float64) + np.asarray(penalty, dtype=np.float64)[:, None], dtype=np.float64)


__all__ = [
    "DynamicPenalty",
    "ShiftPenalty",
    "StaticPenalty",
    "ViolationObjective",
    "penalized_objectives",
    "shift_penalty_objectives",
    "violation_objectives",
]
