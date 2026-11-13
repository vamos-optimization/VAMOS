"""Rankings of the constraint-handling techniques (see :mod:`.constraint_handling`).

Each function returns the front of every row, as ``getRank`` of jMetal's ``Ranking`` would after
``compute``. Constraints follow VAMOS's convention, ``G <= 0`` feasible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import KernelBackend

FloatArray = NDArray[np.float64]
RankArray = NDArray[np.int64]


def violation(G: NDArray[Any]) -> FloatArray:
    """The sum of the positive parts of each row's constraint values: jMetal's overall violation, negated."""
    return np.asarray(np.sum(np.maximum(G, 0.0), axis=1), dtype=np.float64)


def violated_constraints(G: NDArray[Any]) -> FloatArray:
    """The number of constraints each row violates."""
    return np.asarray(np.count_nonzero(G > 0.0, axis=1), dtype=np.float64)


def pareto_ranks(kernel: KernelBackend, F: NDArray[Any]) -> RankArray:
    """The fronts of the non-dominated sorting of ``F``, ignoring the constraints."""
    if F.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    ranks, _ = kernel.nsga2_ranking(np.asarray(F, dtype=np.float64))
    return np.asarray(ranks, dtype=np.int64)


def ranks_by_violation(kernel: KernelBackend, F: NDArray[Any], measure: FloatArray, *, objectives_on_ties: bool) -> RankArray:
    """The fronts of Deb's feasibility rule over a measure of violation (0 for a feasible row).

    A row with a lower measure dominates one with a higher measure, and feasible rows dominate one
    another as in Pareto dominance. Rows with the same positive measure dominate one another as in Pareto
    dominance if ``objectives_on_ties`` (jMetal's ``ConstrainedDominance``), and not at all otherwise (its
    ``EpsilonConstrained``). This is the non-dominated sorting of jMetal's ``FastNonDominatedSortRanking``
    with that comparator: each group of rows with the same measure adds its own fronts after those of the
    groups with lower measures.
    """
    n = F.shape[0]
    ranks = np.empty(n, dtype=np.int64)
    feasible = measure <= 0.0
    offset = 0
    if np.any(feasible):
        rows = np.flatnonzero(feasible)
        feasible_ranks = pareto_ranks(kernel, F[rows])
        ranks[rows] = feasible_ranks
        offset = int(feasible_ranks.max()) + 1
    infeasible = np.flatnonzero(~feasible)
    if infeasible.size == 0:
        return ranks
    order = infeasible[np.argsort(measure[infeasible], kind="stable")]
    values = measure[order]
    starts = np.flatnonzero(np.concatenate(([True], values[1:] != values[:-1])))
    ends = np.concatenate((starts[1:], [order.size]))
    for start, end in zip(starts.tolist(), ends.tolist()):
        members = order[start:end]
        if objectives_on_ties and members.size > 1:
            group_ranks = pareto_ranks(kernel, F[members])
            ranks[members] = offset + group_ranks
            offset += int(group_ranks.max()) + 1
        else:
            ranks[members] = offset
            offset += 1
    return ranks


def stochastic_ranking_order(
    pareto_rank: RankArray,
    feasible: NDArray[np.bool_],
    measure: FloatArray,
    uniforms: FloatArray,
    probability: float,
) -> RankArray:
    """The order of Runarsson and Yao's stochastic bubble sort, as jMetal's ``StochasticRanking`` sorts.

    Each sweep compares adjacent rows: by Pareto rank (ignoring the constraints) if both are feasible or
    with ``probability``, and by violation otherwise. The probability draws ``uniforms`` in order, one per
    comparison that is not between two feasible rows; ``uniforms`` holds ``n * (n - 1)`` values, enough for
    ``n`` sweeps. The sort stops after a sweep without swaps.
    """
    n = pareto_rank.shape[0]
    order = list(range(n))
    rank = pareto_rank.tolist()
    is_feasible = feasible.tolist()
    value = measure.tolist()
    draws = uniforms.tolist()
    drawn = 0
    for _ in range(n):
        swapped = False
        for j in range(n - 1):
            first, second = order[j], order[j + 1]
            if is_feasible[first] and is_feasible[second]:
                by_rank = True
            else:
                by_rank = draws[drawn] < probability
                drawn += 1
            if (rank[second] < rank[first]) if by_rank else (value[second] < value[first]):
                order[j], order[j + 1] = second, first
                swapped = True
        if not swapped:
            break
    return np.asarray(order, dtype=np.int64)


def stochastic_ranking_ranks(order: RankArray, pareto_rank: RankArray, feasible: NDArray[np.bool_]) -> RankArray:
    """The fronts of a stochastic ranking: runs of consecutive feasible rows with the same Pareto rank.

    Each infeasible row is a front of its own, as in jMetal's ``StochasticRanking``.
    """
    n = order.size
    ranks = np.empty(n, dtype=np.int64)
    if n == 0:
        return ranks
    previous, current = order[:-1], order[1:]
    same = feasible[previous] & feasible[current] & (pareto_rank[previous] == pareto_rank[current])
    ranks[order] = np.concatenate(([0], np.cumsum(~same)))
    return ranks


def self_adaptive_penalty_objectives(F: NDArray[Any], G: NDArray[Any]) -> FloatArray:
    """The modified objectives of Woldesenbet, Yen and Tessema's self-adaptive penalty.

    As jMetal's ``SelfAdaptivePenalty``: the objectives are normalized to the population's range and
    the violation is the mean, over the constraints, of each violation over the largest in the
    population. With no feasible row only the violation counts.
    """
    F = np.asarray(F, dtype=np.float64)
    excess = np.maximum(np.asarray(G, dtype=np.float64), 0.0)
    n_constraints = excess.shape[1]
    largest = excess.max(axis=0) if excess.shape[0] else np.zeros(n_constraints)
    scaled = np.divide(excess, largest, out=np.zeros_like(excess), where=largest > 0.0)
    penalty = scaled.sum(axis=1) / n_constraints if n_constraints > 0 else np.zeros(F.shape[0])
    feasible = np.all(excess <= 0.0, axis=1)
    feasible_ratio = float(np.count_nonzero(feasible)) / F.shape[0]
    low, high = F.min(axis=0), F.max(axis=0)
    span = high - low
    normalized = np.divide(F - low, span, out=np.zeros_like(F), where=span > 0.0)
    column = penalty[:, None]
    if feasible_ratio == 0.0:
        distance = np.broadcast_to(column, F.shape)
        violation_penalty = np.zeros_like(F)
    else:
        distance = np.sqrt(normalized * normalized + column * column)
        violation_penalty = np.broadcast_to(column, F.shape)
    objective_penalty = np.where(feasible[:, None], 0.0, normalized)
    return np.asarray(distance + (1.0 - feasible_ratio) * violation_penalty + feasible_ratio * objective_penalty)


__all__ = [
    "pareto_ranks",
    "ranks_by_violation",
    "self_adaptive_penalty_objectives",
    "stochastic_ranking_order",
    "stochastic_ranking_ranks",
    "violated_constraints",
    "violation",
]
