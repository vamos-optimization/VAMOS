"""Constrained non-dominated sorting, crowding distance, truncation and tournament of ToP's phase 2.

Phase 2 of ToP (Liu and Wang, 2019) is NSGA-II with the constrained-domination principle. These
functions transcribe jMetal's port (pull request jMetal/jMetal#525, class ``ToPSelection``), which
follows PlatEMO's implementation:

- two feasible solutions are compared by Pareto dominance; otherwise the one with the smaller
  violation dominates the other, so every infeasible solution is behind every feasible one, and
  infeasible solutions with the same violation share a front whatever their objectives;
- the crowding distance sorts each front by each objective from the order of the solutions, ties
  keeping that order; the first and the last solution get an infinite distance, and each inner one
  adds the difference between its neighbours divided by the range of the objective in the front,
  so an objective whose values are all equal in a front gives ``NaN`` (zero divided by zero) to
  its inner solutions;
- the truncation of the last front takes the ``NaN`` distances first, then the largest ones, and
  the binary tournament takes ``NaN`` as the worst distance, as MATLAB sorts ``NaN`` values;
- every other tie goes to the solution that comes first.

Solutions are rows, and a selection is returned in the order of the rows. Violations are the
non-negative overall constraint violations of :func:`~vamos.engine.algorithm.top.phases.constraint_violation`;
objectives or violations with ``NaN`` are not supported.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.ctaea.archives import front_ranks

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int_]


def constrained_front_numbers(F: NDArray[Any], violation: NDArray[Any]) -> IndexArray:
    """Front number of each row, from 0, by non-dominated sorting with ToP's constrained dominance.

    The feasible rows (violation zero) are sorted by Pareto dominance into the first fronts; each
    distinct violation of the infeasible rows is a front of its own after them, in increasing
    order of violation. That is the ranking of jMetal's ``FastNonDominatedSortRanking`` with the
    dominance of ``ToPSelection.constrainedDominance``, which is unique for a given dominance
    relation.
    """
    objectives = np.asarray(F, dtype=float)
    cv = np.asarray(violation, dtype=float)
    fronts = np.zeros(cv.shape[0], dtype=int)
    feasible = cv == 0.0
    feasible_fronts = 0
    if feasible.any():
        ranks = front_ranks(objectives[feasible])
        fronts[feasible] = ranks
        feasible_fronts = int(ranks.max()) + 1
    if not feasible.all():
        _, levels = np.unique(cv[~feasible], return_inverse=True)
        fronts[~feasible] = feasible_fronts + np.asarray(levels, dtype=int).reshape(-1)
    return fronts


def crowding_distances(F: NDArray[Any], fronts: NDArray[Any]) -> FloatArray:
    """Crowding distance of each row within its front, as jMetal's ``ToPSelection.crowdingDistances``.

    For each objective in turn, the rows of each front are sorted by that objective from their
    order (ties keep it); the first and the last get an infinite distance, and each inner one adds
    ``(next - previous) / (last - first)``. An objective with a single value in a front gives
    ``NaN`` to its inner rows, and an infinite distance plus ``NaN`` is ``NaN``, unless a later
    objective makes the row an extreme again.
    """
    objectives = np.asarray(F, dtype=float)
    ranks = np.asarray(fronts, dtype=int)
    n = ranks.shape[0]
    distances = np.zeros(n, dtype=float)
    if n == 0:
        return distances
    rows = np.arange(n)
    for objective in range(objectives.shape[1]):
        values = objectives[:, objective]
        # By front, then by value; the row breaks the ties, as the stable sort of jMetal's list.
        order = np.lexsort((rows, values, ranks))
        sorted_fronts = ranks[order]
        sorted_values = values[order]
        start = np.searchsorted(sorted_fronts, sorted_fronts, side="left")
        end = np.searchsorted(sorted_fronts, sorted_fronts, side="right") - 1
        extreme = (rows == start) | (rows == end)
        inner = np.flatnonzero(~extreme)
        with np.errstate(divide="ignore", invalid="ignore"):
            increment = (sorted_values[inner + 1] - sorted_values[inner - 1]) / (sorted_values[end[inner]] - sorted_values[start[inner]])
        distances[order[inner]] += increment
        distances[order[extreme]] = np.inf
    return distances


def environmental_selection(F: NDArray[Any], violation: NDArray[Any], size: int) -> IndexArray:
    """The ``size`` rows that NSGA-II keeps, in the order of the rows, as jMetal's ``ToPSelection.select``.

    The fronts and crowding distances are computed on all the rows. The rows of the fronts that
    fit are kept; from the last front, which does not fit, those with ``NaN`` distances first, then
    those with the largest distances, ties going to the row that comes first.
    """
    objectives = np.asarray(F, dtype=float)
    cv = np.asarray(violation, dtype=float)
    n = cv.shape[0]
    if not 0 <= int(size) <= n:
        raise ValueError(f"ToP cannot select {size} rows from {n}.")
    if n == 0:
        return np.empty(0, dtype=int)
    fronts = constrained_front_numbers(objectives, cv)
    distances = crowding_distances(objectives, fronts)
    counts = np.bincount(fronts)
    cumulative = np.cumsum(counts)
    # The first front with which the selection reaches ``size`` rows (front 0 when size is 0).
    last_front = int(np.searchsorted(cumulative, int(size), side="left"))
    selected_before = int(cumulative[last_front] - counts[last_front])
    selected = fronts < last_front
    members = np.flatnonzero(fronts == last_front)
    member_distances = distances[members]
    is_nan = np.isnan(member_distances)
    descending = np.where(is_nan, 0.0, -member_distances)
    order = np.lexsort((members, descending, ~is_nan))
    selected[members[order[: int(size) - selected_before]]] = True
    return np.asarray(np.flatnonzero(selected), dtype=int)


def tournament_winners(fronts: NDArray[Any], distances: NDArray[Any], draws: NDArray[Any]) -> IndexArray:
    """Winners of binary tournaments between the rows of each pair of ``draws``, as jMetal's ``ToPSelection.binaryTournament``.

    The lower front number wins; with the same front, the larger crowding distance (``NaN`` loses
    to any other distance, two ``NaN`` are equal); with the same distance too, the row that comes
    first. Both candidates can be the same row.
    """
    ranks = np.asarray(fronts, dtype=int)
    crowding = np.asarray(distances, dtype=float)
    pairs = np.asarray(draws, dtype=int).reshape(-1, 2)
    first, second = pairs[:, 0], pairs[:, 1]
    first_distance, second_distance = crowding[first], crowding[second]
    first_nan, second_nan = np.isnan(first_distance), np.isnan(second_distance)
    # Negative when the first candidate's distance is better, positive when it is worse.
    by_distance = np.where(
        first_nan | second_nan,
        first_nan.astype(int) - second_nan.astype(int),
        np.where(second_distance < first_distance, -1, np.where(second_distance > first_distance, 1, 0)),
    )
    by_front = np.sign(ranks[first] - ranks[second])
    comparison = np.where(by_front != 0, by_front, np.where(by_distance != 0, by_distance, np.sign(first - second)))
    return np.asarray(np.where(comparison <= 0, first, second), dtype=int)


def top_tournament(fronts: NDArray[Any], distances: NDArray[Any], n_tournaments: int, rng: np.random.Generator) -> IndexArray:
    """Winners of ``n_tournaments`` binary tournaments, each between two rows drawn with replacement."""
    n = int(np.asarray(fronts).shape[0])
    if n == 0:
        raise ValueError("ToP cannot select parents from an empty population.")
    return tournament_winners(fronts, distances, rng.integers(0, n, size=(int(n_tournaments), 2)))


__all__ = [
    "constrained_front_numbers",
    "crowding_distances",
    "environmental_selection",
    "top_tournament",
    "tournament_winners",
]
