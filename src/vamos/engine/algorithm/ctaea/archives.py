"""Subregions and the updates of the two archives of C-TAEA.

C-TAEA (Li, Chen, Fu and Yao, 2019) keeps a convergence archive (CA) and a diversity archive (DA)
of N solutions, one per weight vector. These functions transcribe jMetal's port (pull request
jMetal/jMetal#524, classes ``Subregions``, ``NonDominatedSorting``, ``ConvergenceArchiveUpdate``
and ``DiversityArchiveUpdate``), which follows the reference code where it differs from the
paper:

- a solution is associated with the weight vector of largest cosine similarity on its raw
  objectives, except in the truncation of the feasible solutions of the CA, where the objectives
  are first normalized with the ideal and nadir points of the set; the paper normalizes before
  every association;
- the ideal point of each Tchebycheff function is that of the set it is applied to;
- every tie goes to the solution that comes first.

Archives are handled as row indices into the candidates, in the order the candidates are given
(the current archive, then the offspring), and every selection is returned in jMetal's order,
which later decides the ties. The arithmetic follows jMetal's operation by operation (sums in
objective order, the same divisions), so that the floating-point values, and the ties they
create, are the same. A point whose cosine similarities are not defined (the origin, or a
normalized point with a ``NaN``) is associated with the first weight vector. Objectives with
``NaN`` are not supported.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int_]


def _ordered_dot(a: NDArray[Any], b: NDArray[Any]) -> FloatArray:
    """``sum_j a[..., j] * b[..., j]`` added in the order of ``j`` from 0.0, as jMetal's loops."""
    total = np.zeros(np.broadcast_shapes(a.shape[:-1], b.shape[:-1]), dtype=float)
    for j in range(a.shape[-1]):
        total = total + a[..., j] * b[..., j]
    return total


def associate(points: NDArray[Any], weights: NDArray[Any]) -> IndexArray:
    """The subregion of each row: the weight vector with the largest cosine similarity, the first in a tie."""
    P = np.asarray(points, dtype=float)
    W = np.asarray(weights, dtype=float)
    if P.shape[0] == 0:
        return np.empty(0, dtype=int)
    point_norms = np.sqrt(_ordered_dot(P, P))
    weight_norms = np.sqrt(_ordered_dot(W, W))
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = _ordered_dot(P[:, None, :], W[None, :, :]) / (point_norms[:, None] * weight_norms[None, :])
    similarity = np.where(np.isnan(similarity), -np.inf, similarity)
    return np.asarray(np.argmax(similarity, axis=1), dtype=int)


def tchebycheff_in_own_subregion(points: NDArray[Any], weights: NDArray[Any]) -> FloatArray:
    """``max_j |f_j - z_j| / w_j`` of each row, with the weight vector of its subregion and the ideal point ``z`` of the rows."""
    P = np.asarray(points, dtype=float)
    W = np.asarray(weights, dtype=float)
    ideal = np.min(P, axis=0)
    return np.asarray(np.max(np.abs(P - ideal) / W[associate(P, W)], axis=1), dtype=float)


def dominance_matrix(points: NDArray[Any]) -> NDArray[np.bool_]:
    """``dom[i, j]`` is true when row ``i`` Pareto-dominates row ``j``: no component is worse and one is better."""
    P = np.asarray(points, dtype=float)
    worse = np.any(P[:, None, :] > P[None, :, :], axis=2)
    better = np.any(P[:, None, :] < P[None, :, :], axis=2)
    return np.asarray(~worse & better, dtype=bool)


def front_ranks(points: NDArray[Any]) -> IndexArray:
    """Non-dominated sorting by Pareto dominance alone: 0 for the non-dominated rows, 1 for the next front, and so on.

    Equal rows are in the same front; constraints play no part.
    """
    n = int(np.asarray(points).shape[0])
    ranks = np.full(n, -1, dtype=int)
    if n == 0:
        return ranks
    dom = dominance_matrix(points)
    count = dom.sum(axis=0).astype(np.int64)
    current = np.flatnonzero(count == 0)
    level = 0
    while current.size:
        ranks[current] = level
        count -= dom[current].sum(axis=0)
        count[current] = -1
        level += 1
        current = np.flatnonzero(count == 0)
    return ranks


def _add_fronts_until_full(members: IndexArray, ranks: IndexArray, selected: list[int], size: int) -> int:
    """Append the fronts 0, 1, ... of ``members`` to ``selected`` until it has ``size`` rows; returns the last front added."""
    max_front = int(ranks.max()) if ranks.size else -1
    front = -1
    while front < max_front and len(selected) < size:
        front += 1
        selected.extend(int(row) for row in members[ranks == front])
    return front


def _normalized(F: FloatArray) -> FloatArray:
    """``(f - ideal) / (nadir - ideal)`` with the set's own points, without safeguard: a flat objective gives ``NaN``."""
    ideal = np.min(F, axis=0)
    nadir = np.max(F, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray((F - ideal) / (nadir - ideal), dtype=float)


def _closest_rows(F: FloatArray) -> IndexArray:
    """Rows at the minimum non-zero distance from another row, with repetitions, column by column of the distance matrix.

    Zero distances (the diagonal and equal rows) are ignored; when every distance is zero, every
    pair is taken.
    """
    differences = F[:, None, :] - F[None, :, :]
    distance = np.sqrt(_ordered_dot(differences, differences))
    distance[distance == 0.0] = np.inf
    minimum = distance.min()
    # Column by column: the pairs (i, j) sorted by j, then i.
    _, rows = np.nonzero(np.ascontiguousarray(distance.T) == minimum)
    return np.asarray(rows, dtype=int)


def worst_of_the_most_crowded_subregion(F: NDArray[Any], weights: NDArray[Any]) -> int:
    """The row that the truncation of a set of feasible solutions removes.

    The rows are associated with the subregions on their normalized objectives; in the most
    crowded subregion (the first in a tie), the rows at the smallest Euclidean distance from
    another one (on the raw objectives) are the candidates, and the one with the largest
    Tchebycheff value, with its own subregion on the raw objectives and the ideal point of the
    candidates, is removed (the first in a tie).
    """
    objectives = np.asarray(F, dtype=float)
    W = np.asarray(weights, dtype=float)
    regions = associate(_normalized(objectives), W)
    most_crowded = int(np.argmax(np.bincount(regions, minlength=W.shape[0])))
    crowded = np.flatnonzero(regions == most_crowded)
    closest = crowded[_closest_rows(objectives[crowded])]
    return int(closest[int(np.argmax(tchebycheff_in_own_subregion(objectives[closest], W)))])


def _select_among_feasible(F: FloatArray, feasible: IndexArray, weights: FloatArray) -> list[int]:
    size = weights.shape[0]
    selected: list[int] = []
    _add_fronts_until_full(feasible, front_ranks(F[feasible]), selected, size)
    while len(selected) > size:
        del selected[worst_of_the_most_crowded_subregion(F[selected], weights)]
    return selected


def _complete_with_infeasible(
    F: FloatArray, violation: FloatArray, feasible: IndexArray, infeasible: IndexArray, weights: FloatArray
) -> list[int]:
    size = weights.shape[0]
    selected = [int(row) for row in feasible]
    if infeasible.size == 0:
        return selected
    criteria = np.column_stack([violation[infeasible], tchebycheff_in_own_subregion(F[infeasible], weights)])
    ranks = front_ranks(criteria)
    last = _add_fronts_until_full(infeasible, ranks, selected, size)
    excess = len(selected) - size
    if excess > 0:
        last_front = infeasible[ranks == last]
        # A stable sort by decreasing violation: ties keep their order.
        removed = {int(row) for row in last_front[np.argsort(-violation[last_front], kind="stable")[:excess]]}
        selected = [row for row in selected if row not in removed]
    return selected


def convergence_archive_update(F: NDArray[Any], violation: NDArray[Any], weights: NDArray[Any]) -> IndexArray:
    """Rows of the candidates (the current CA, then the offspring) that form the new CA, in jMetal's order.

    - If exactly N candidates are feasible (zero violation), they are the new CA.
    - If more are, the fronts of their non-dominated sorting are added until there are at least
      N, and :func:`worst_of_the_most_crowded_subregion` removes one row at a time until N remain.
    - If fewer are, all of them are kept, and the infeasible ones are sorted by non-dominated
      sorting on their violation and their Tchebycheff value (with their own subregion and the
      ideal point of the infeasible rows); the fronts are added until there are at least N, and
      the rows of the last front with the largest violation are removed (the first in a tie).

    N is the number of weight vectors; with fewer candidates, the CA keeps all of them.
    """
    objectives = np.asarray(F, dtype=float)
    cv = np.asarray(violation, dtype=float)
    W = np.asarray(weights, dtype=float)
    feasible_mask = cv == 0.0
    feasible = np.flatnonzero(feasible_mask)
    infeasible = np.flatnonzero(~feasible_mask)
    size = W.shape[0]
    if feasible.size == size:
        return np.asarray(feasible, dtype=int)
    if feasible.size > size:
        return np.asarray(_select_among_feasible(objectives, feasible, W), dtype=int)
    return np.asarray(_complete_with_infeasible(objectives, cv, feasible, infeasible, W), dtype=int)


def _best_of_the_subregion(F: FloatArray, members: IndexArray, weights: FloatArray) -> int:
    """The non-dominated member with the smallest Tchebycheff value (the ideal point of those members), the first in a tie."""
    non_dominated = members[front_ranks(F[members]) == 0]
    return int(non_dominated[int(np.argmin(tchebycheff_in_own_subregion(F[non_dominated], weights)))])


def diversity_archive_update(convergence_F: NDArray[Any], F: NDArray[Any], weights: NDArray[Any]) -> IndexArray:
    """Rows of the candidates (the current DA, then the offspring) that form the new DA, in selection order.

    The DA ignores the constraints. The candidates and the (already updated) CA are associated
    with the subregions on their raw objectives. In iteration ``itr = 1, 2, ...``, every subregion
    ``i`` in order with fewer than ``itr`` rows of the CA receives up to ``itr`` minus that number
    of candidates of the subregion, each time the best of its non-dominated candidates (see
    :func:`_best_of_the_subregion`), until the DA has N rows or no candidate is left.
    """
    objectives = np.asarray(F, dtype=float)
    W = np.asarray(weights, dtype=float)
    size = W.shape[0]
    regions = associate(objectives, W)
    in_convergence_archive = np.bincount(associate(np.asarray(convergence_F, dtype=float), W), minlength=size)
    left = np.bincount(regions, minlength=size)
    available = np.ones(objectives.shape[0], dtype=bool)
    remaining = int(objectives.shape[0])
    selected: list[int] = []
    iteration = 1
    while len(selected) < size and remaining:
        for region in range(size):
            if len(selected) >= size:
                break
            for _ in range(iteration - int(in_convergence_archive[region])):
                if not left[region]:
                    break
                best = _best_of_the_subregion(objectives, np.flatnonzero(available & (regions == region)), W)
                available[best] = False
                left[region] -= 1
                remaining -= 1
                if len(selected) < size:
                    selected.append(best)
        iteration += 1
    return np.asarray(selected, dtype=int)


__all__ = [
    "associate",
    "convergence_archive_update",
    "diversity_archive_update",
    "dominance_matrix",
    "front_ranks",
    "tchebycheff_in_own_subregion",
    "worst_of_the_most_crowded_subregion",
]
