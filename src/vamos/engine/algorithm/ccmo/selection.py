"""Fitness, environmental selection and mating selection of CCMO.

CCMO (Tian, Zhang, Xiao, Zhang and Jin, 2021) evolves each of its two populations with the
fitness and the environmental selection of SPEA2 under a dominance relation in which the
constraint violation comes first. These functions follow jMetal's port (pull request
jMetal/jMetal#521), which follows the authors' reference code:

- a solution dominates another one if its violation is smaller or, with the same violation, if it
  dominates it in the Pareto sense, so two infeasible solutions with the same violation are also
  compared by Pareto dominance;
- the fitness is SPEA2's raw fitness ``R`` (the sum of the strengths of a solution's dominators)
  plus the density ``1 / (d_k + 2)``, where ``d_k`` is the Euclidean distance, in the objective
  space without normalization, to the ``k``-th nearest other solution, ``k = floor(sqrt(n))``;
- the environmental selection keeps the solutions with a fitness lower than 1 (those that no other
  one dominates), fills up with the next ones by fitness or removes the most crowded ones with
  SPEA2's truncation, and returns the selected solutions sorted by fitness;
- the mating selection is a binary tournament on the fitness.

Every tie is broken in favour of the solution that comes first, as the stable sorts of the
reference code do.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.spea2.helpers import strength_raw_fitness, truncate_lexicographic

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import SelectionKernels

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int_]


def constraint_first_dominance(F: NDArray[Any], violation: NDArray[Any]) -> NDArray[np.bool_]:
    """Dominance matrix of CCMO: ``dom[i, j]`` is true when solution ``i`` dominates solution ``j``.

    ``i`` dominates ``j`` if its constraint violation is smaller or, when both violations are
    equal, if it dominates ``j`` in the Pareto sense (no worse in every objective and better in
    one). A population without constraints passes zero violations, which leaves Pareto dominance.
    """
    objectives = np.asarray(F, dtype=float)
    cv = np.asarray(violation, dtype=float)
    smaller = cv[:, None] < cv[None, :]
    equal = cv[:, None] == cv[None, :]
    no_worse = np.all(objectives[:, None, :] <= objectives[None, :, :], axis=2)
    better = np.any(objectives[:, None, :] < objectives[None, :, :], axis=2)
    return np.asarray(smaller | (equal & no_worse & better), dtype=bool)


def objective_distances(F: NDArray[Any]) -> FloatArray:
    """Euclidean distances between the objective vectors (not normalized), with zeros on the diagonal."""
    objectives = np.asarray(F, dtype=float)
    return np.asarray(np.linalg.norm(objectives[:, None, :] - objectives[None, :, :], axis=2), dtype=float)


def ccmo_fitness(F: NDArray[Any], violation: NDArray[Any]) -> tuple[FloatArray, FloatArray]:
    """CCMO fitness of each solution and the objective distances it was computed from.

    The fitness, to be minimized, is ``R + 1 / (d_k + 2)``: ``R`` is the sum of the strengths (the
    number of solutions a solution dominates, see :func:`constraint_first_dominance`) of the
    solutions that dominate it, and ``d_k`` the distance to its ``k``-th nearest other solution,
    with ``k = floor(sqrt(n))``. A lone solution has density 0. The solutions that no other one
    dominates are those with a fitness lower than 1.
    """
    objectives = np.asarray(F, dtype=float)
    n = objectives.shape[0]
    distances = objective_distances(objectives)
    if n == 0:
        return np.empty(0, dtype=float), distances
    raw = strength_raw_fitness(constraint_first_dominance(objectives, violation))
    others = distances.copy()
    np.fill_diagonal(others, np.inf)
    k = math.isqrt(n)
    # Every row has n - 1 finite distances, and k <= n - 1 for n >= 2; with n = 1 the distance is infinite.
    kth = np.partition(others, kth=k - 1, axis=1)[:, k - 1]
    return np.asarray(raw + 1.0 / (kth + 2.0), dtype=float), distances


def ccmo_environmental_selection(
    F: NDArray[Any],
    violation: NDArray[Any],
    n_select: int,
    selection_kernels: SelectionKernels | None = None,
) -> tuple[IndexArray, FloatArray]:
    """Select ``n_select`` rows; returns them sorted by fitness, with the fitness each one has in the input.

    The rows with a fitness lower than 1 are selected. If they are fewer than ``n_select``, the
    rows with the lowest fitness are selected instead; if they are more, SPEA2's truncation removes,
    one at a time, the row whose sorted distances to the remaining ones are the lexicographically
    smallest. The selection is returned sorted by fitness, and the fitness, computed on the whole
    input, drives the next mating selection. Every tie keeps the earlier row first.
    """
    fitness, distances = ccmo_fitness(F, violation)
    n = fitness.shape[0]
    size = min(int(n_select), n)
    nondominated = np.flatnonzero(fitness < 1.0)
    if nondominated.size < size:
        selected = np.argsort(fitness, kind="stable")[:size]
    elif nondominated.size > size:
        keep = truncate_lexicographic(distances[np.ix_(nondominated, nondominated)], size, selection_kernels)
        selected = nondominated[keep]
    else:
        selected = nondominated
    selected = np.sort(selected)
    order = np.asarray(selected[np.argsort(fitness[selected], kind="stable")], dtype=int)
    return order, np.asarray(fitness[order], dtype=float)


def ccmo_tournament(fitness: NDArray[Any], n_tournaments: int, rng: np.random.Generator) -> IndexArray:
    """Winners of ``n_tournaments`` binary tournaments, with replacement, on the fitness.

    Each tournament draws two rows at random (possibly the same one); the lower fitness wins, and
    a tie goes to the row that comes first.
    """
    values = np.asarray(fitness, dtype=float)
    n = values.shape[0]
    if n == 0:
        raise ValueError("CCMO cannot select parents from an empty population.")
    drawn = rng.integers(0, n, size=(int(n_tournaments), 2))
    first, second = drawn[:, 0], drawn[:, 1]
    first_wins = (values[first] < values[second]) | ((values[first] == values[second]) & (first <= second))
    return np.asarray(np.where(first_wins, first, second), dtype=int)


__all__ = [
    "ccmo_environmental_selection",
    "ccmo_fitness",
    "ccmo_tournament",
    "constraint_first_dominance",
    "objective_distances",
]
