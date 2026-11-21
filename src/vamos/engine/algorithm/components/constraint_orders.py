"""IDEA and ATM: constraint-handling techniques that order the population totally, for NSGA-II (see :mod:`.constraint_handling`).

Both papers define a selection, of which NSGA-II keeps the first ``N`` solutions; each technique returns a
total order whose first ``N`` rows are that selection, one row per front (``total_order``), so NSGA-II's
replacement, which keeps whole fronts, keeps exactly them, and its tournament compares the positions. Fronts
and crowding distances are VAMOS's, NSGA-II's: a front's crowding distance divides each difference by the
objective's range in the front and skips an objective without spread. Rows with the same front and crowding
distance keep the order of the population.

- ``infeasibility_driven``: T. Ray, H. K. Singh, A. Isaacs and W. Smith, "Infeasibility Driven Evolutionary
  Algorithm for Constrained Optimization", in E. Mezura-Montes (ed.), Constraint-Handling in Evolutionary
  Optimization, Studies in Computational Intelligence 198, Springer, 2009, pp. 145-165 (:class:`InfeasibilityDriven`).
- ``adaptive_tradeoff``: Y. Wang, Z. Cai, Y. Zhou and W. Zeng, "An Adaptive Tradeoff Model for Constrained
  Evolutionary Optimization", IEEE Transactions on Evolutionary Computation 12(1):80-92, 2008, section III, as
  J.-P. Li, Y. Wang, S. Yang and Z. Cai extend it to several objectives, "A Comparative Study of
  Constraint-Handling Techniques in Evolutionary Constrained Multiobjective Optimization", IEEE CEC 2016,
  pp. 4175-4182, section III-C (:class:`AdaptiveTradeoff`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .constrained_ranking import RankArray, pareto_ranks
from .constraint_technique import ConstraintHandlingTechnique

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import KernelBackend


class InfeasibilityDriven(ConstraintHandlingTechnique):
    """Ray, Singh, Isaacs and Smith's IDEA: the best infeasible solutions first, ``proportion`` of the population.

    The population splits into the feasible and the infeasible solutions, and each set is sorted by fronts and
    crowding distance over the objectives and the violation measure (:func:`violation_measure`), as one more
    objective (section 2). The order is the first ``N_inf`` infeasible solutions, then the feasible ones, then
    the other infeasible ones, so its first ``N`` rows are Algorithm 1's selection, with the paper's filling
    rule when either set falls short: ``N_inf`` infeasible solutions rank above the feasible ones.
    ``N_inf = proportion * N``, rounded half up, with ``N`` the population size of :meth:`bind_run`.
    """

    name = "infeasibility_driven"
    total_order = True

    def __init__(self, proportion: float) -> None:
        self.proportion = float(proportion)
        self.population_size = 0

    def bind_run(self, population_size: int, max_evaluations: int) -> None:
        self.population_size = int(population_size)

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        if self.population_size < 1:
            raise ValueError("infeasibility_driven keeps a proportion of the population: give it the population size with bind_run().")
        return infeasibility_driven_ranks(kernel, F, G, self.proportion, self.population_size)


def violation_measure(G: NDArray[Any]) -> NDArray[np.int64]:
    """IDEA's violation measure (section 3): the sum over the constraints of each row's violation rank.

    For each constraint, the rows that violate it rank by their violation in ascending order from 1, and the
    rows that satisfy it get 0. Equal violations share a rank, and the next one follows it (dense ranks).
    """
    excess = np.maximum(np.asarray(G, dtype=np.float64), 0.0)
    measure = np.zeros(excess.shape[0], dtype=np.int64)
    for column in excess.T:
        violated = column > 0.0
        if np.any(violated):
            levels = np.unique(column[violated])
            measure[violated] += np.searchsorted(levels, column[violated]) + 1
    return measure


def infeasibility_driven_ranks(
    kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], proportion: float, population_size: int
) -> RankArray:
    """IDEA's order of the rows, from 0 (see :class:`InfeasibilityDriven`)."""
    F = np.asarray(F, dtype=np.float64)
    measure = violation_measure(G)
    extended = np.column_stack([F, measure.astype(np.float64)])
    infeasible = _front_order(kernel, extended, np.flatnonzero(measure > 0))
    feasible = _front_order(kernel, extended, np.flatnonzero(measure == 0))
    quota = int(math.floor(proportion * population_size + 0.5))
    return _positions(np.concatenate([infeasible[:quota], feasible, infeasible[quota:]]))


class AdaptiveTradeoff(ConstraintHandlingTechnique):
    """Wang, Cai, Zhou and Zeng's ATM, in Li, Wang, Yang and Cai's version for several objectives.

    The phase depends on the feasible proportion of the population being ranked (:func:`adaptive_tradeoff_ranks`).
    ``phi``, the feasible proportion of the last population that survived, is what the technique learns from
    :meth:`survived`; it is 0 until the first replacement, as in Wang et al.'s Fig. 4, and checkpoints keep it.
    """

    name = "adaptive_tradeoff"
    total_order = True

    def __init__(self) -> None:
        self.phi = 0.0

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        return adaptive_tradeoff_ranks(kernel, F, G, self.phi)

    def survived(self, F: NDArray[Any], G: NDArray[Any]) -> None:
        feasible = np.all(np.asarray(G) <= 0.0, axis=1)
        self.phi = float(np.count_nonzero(feasible)) / max(1, feasible.size)

    def state(self) -> dict[str, Any]:
        return {"phi": self.phi}

    def restore(self, state: Mapping[str, Any]) -> None:
        self.phi = float(state.get("phi", 0.0))


def normalized_violation(G: NDArray[Any]) -> NDArray[np.float64]:
    """Li et al.'s ``C`` (eq. 3): the mean over the constraints of each violation over the largest in the population.

    A constraint that no row violates adds 0.
    """
    excess = np.maximum(np.asarray(G, dtype=np.float64), 0.0)
    if excess.shape[1] == 0:
        return np.zeros(excess.shape[0], dtype=np.float64)
    largest = excess.max(axis=0) if excess.shape[0] else np.zeros(excess.shape[1])
    scaled = np.divide(excess, largest, out=np.zeros_like(excess), where=largest > 0.0)
    return np.asarray(scaled.mean(axis=1), dtype=np.float64)


def adaptive_tradeoff_fitness(F: NDArray[Any], C: NDArray[np.float64], phi: float) -> NDArray[np.float64]:
    """ATM's fitness in its second phase, with feasible and infeasible rows (Li et al., eqs. 10-15).

    An infeasible row's objective ``f_i`` becomes ``max(phi * f_b_i + (1 - phi) * f_w_i, f_i)``, with ``f_b_i``
    and ``f_w_i`` the smallest and largest ``f_i`` of the feasible rows (eq. 12). Each objective is then
    normalized over all the rows (eq. 13), and the violation over the infeasible rows, 0 for the feasible
    ones (eq. 14); the fitness is their sum (eq. 15). A range of 0 normalizes to 0.
    """
    F = np.asarray(F, dtype=np.float64)
    feasible = C <= 0.0
    converted = F.copy()
    floor = phi * F[feasible].min(axis=0) + (1.0 - phi) * F[feasible].max(axis=0)
    converted[~feasible] = np.maximum(floor, F[~feasible])
    low, span = converted.min(axis=0), np.ptp(converted, axis=0)
    normalized = np.divide(converted - low, span, out=np.zeros_like(converted), where=span > 0.0)
    infeasible_C = C[~feasible]
    lowest, spread = infeasible_C.min(), np.ptp(infeasible_C)
    penalty = np.zeros(F.shape[0], dtype=np.float64)
    if spread > 0.0:
        penalty[~feasible] = (infeasible_C - lowest) / spread
    return np.asarray(normalized + penalty[:, None], dtype=np.float64)


def adaptive_tradeoff_ranks(kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], phi: float) -> RankArray:
    """ATM's order of the rows, from 0, by the feasible proportion of the rows.

    - None feasible (phase one): the hierarchical non-dominated selection of Wang et al.'s Fig. 3, over the
      objectives and Li et al.'s ``C`` (eq. 3). The non-dominated rows of those that remain are sorted by
      ``C`` in ascending order and the first half of them, rounded up, come next; until no row remains.
    - Some feasible (phase two): fronts and crowding distance of :func:`adaptive_tradeoff_fitness`.
    - All feasible (phase three): fronts and crowding distance of the objectives.
    """
    F = np.asarray(F, dtype=np.float64)
    C = normalized_violation(G)
    rows = np.arange(F.shape[0])
    feasible = C <= 0.0
    if np.all(feasible):
        return _positions(_front_order(kernel, F, rows))
    if np.any(feasible):
        return _positions(_front_order(kernel, adaptive_tradeoff_fitness(F, C, phi), rows))
    return _positions(_hierarchical_order(kernel, np.column_stack([F, C]), C))


def _hierarchical_order(kernel: KernelBackend, values: NDArray[np.float64], C: NDArray[np.float64]) -> NDArray[np.int64]:
    remaining = np.arange(values.shape[0])
    order: list[NDArray[np.int64]] = []
    while remaining.size:
        leading = remaining[pareto_ranks(kernel, values[remaining]) == 0]
        leading = leading[np.argsort(C[leading], kind="stable")]
        taken = leading[: (leading.size + 1) // 2]
        order.append(taken)
        remaining = remaining[~np.isin(remaining, taken)]
    return np.concatenate(order) if order else np.empty(0, dtype=np.int64)


def _front_order(kernel: KernelBackend, values: NDArray[np.float64], rows: NDArray[Any]) -> NDArray[np.int64]:
    """``rows`` sorted by their fronts among themselves, then by crowding distance in descending order."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return rows
    ranks, crowding = kernel.nsga2_ranking(np.ascontiguousarray(values[rows]))
    return rows[np.lexsort((np.arange(rows.size), -np.asarray(crowding, dtype=np.float64), np.asarray(ranks)))]


def _positions(order: NDArray[np.int64]) -> RankArray:
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(order.size, dtype=np.int64)
    return ranks


__all__ = [
    "AdaptiveTradeoff",
    "InfeasibilityDriven",
    "adaptive_tradeoff_fitness",
    "adaptive_tradeoff_ranks",
    "infeasibility_driven_ranks",
    "normalized_violation",
    "violation_measure",
]
