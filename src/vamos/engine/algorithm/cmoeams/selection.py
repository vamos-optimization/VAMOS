"""Environmental selection, truncation and mating selection of CMOEA-MS.

These functions transcribe jMetal's port (pull request jMetal/jMetal#526, class
``CMOEAMSSelection``), which follows the authors' implementation in PlatEMO:

- the environmental selection keeps the solutions whose fitness is lower than 1 (those that no
  other one dominates); if they are fewer than ``n``, the ``n`` solutions of lowest fitness, and if
  they are more, it removes the excess with the truncation on their objectives; the selected
  solutions keep their order;
- the truncation removes, one at a time, the solution whose cosine distances to the remaining
  ones, sorted, are the lexicographically smallest, the first one on a tie; the cosine distance is
  computed on the objectives in both stages;
- the binary tournament draws two solutions with replacement and takes the one ranked first by a
  stable sort of the fitness, so a tie goes to the solution that comes first and ``NaN`` loses.

Solutions are rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.spea2.helpers import truncate_lexicographic

from .fitness import cosine_distances, sorted_indices

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import SelectionKernels

IndexArray = NDArray[np.int_]

# Every cosine distance is in [0, 2] (up to round-off), so this value sorts a NaN distance after
# every finite one, as MATLAB and jMetal do, and before a row's infinite distance to itself.
_NAN_DISTANCE = 4.0


def truncation(F: NDArray[Any], k: int, selection_kernels: SelectionKernels | None = None) -> NDArray[np.bool_]:
    """Remove ``k`` rows as jMetal's ``CMOEAMSSelection.truncation``; true for each removed row.

    Each step sorts the cosine distances between the objective vectors of each remaining row and
    the other remaining ones, and removes the row whose sorted distances are the lexicographically
    smallest, the first one on a tie. A ``NaN`` distance (a zero objective vector) sorts after the
    others. Every row has as many distances as the others, so sorting a row's distance to itself
    last, after its ``NaN`` ones, as here, or before them, as jMetal does, takes the same decisions.
    """
    objectives = np.asarray(F, dtype=float)
    n = objectives.shape[0]
    if not 0 <= int(k) <= n:
        raise ValueError(f"CMOEA-MS cannot remove {k} of {n} solutions.")
    distances = cosine_distances(objectives)
    distances[np.isnan(distances)] = _NAN_DISTANCE
    kept = truncate_lexicographic(distances, n - int(k), selection_kernels)
    deleted = np.ones(n, dtype=bool)
    deleted[kept] = False
    return deleted


def environmental_selection(
    fitness: NDArray[Any], F: NDArray[Any], n_select: int, selection_kernels: SelectionKernels | None = None
) -> IndexArray:
    """The ``n_select`` rows that jMetal's ``CMOEAMSSelection.environmentalSelection`` keeps, in ascending order.

    The rows whose fitness is lower than 1 are kept. If they are fewer than ``n_select``, the
    ``n_select`` rows of lowest fitness are kept instead (a stable sort, ``NaN`` last); if they are
    more, :func:`truncation` removes the excess, on their objectives.
    """
    values = np.asarray(fitness, dtype=float)
    objectives = np.asarray(F, dtype=float)
    n = values.shape[0]
    if not 0 <= int(n_select) <= n:
        raise ValueError(f"CMOEA-MS cannot select {n_select} of {n} solutions.")
    selected = values < 1.0
    count = int(np.count_nonzero(selected))
    if count < n_select:
        selected[sorted_indices(values)[: int(n_select)]] = True
    elif count > n_select:
        candidates = np.flatnonzero(selected)
        selected[candidates[truncation(objectives[candidates], count - int(n_select), selection_kernels)]] = False
    return np.asarray(np.flatnonzero(selected), dtype=int)


def tournament_winners(fitness: NDArray[Any], draws: NDArray[Any]) -> IndexArray:
    """Winners of binary tournaments between the rows of each pair of ``draws``, as jMetal's ``CMOEAMSSelection.tournamentSelection``.

    The winner is the row ranked first by a stable sort of the fitness (``NaN`` last): the lower
    fitness, and on a tie the row that comes first. Both candidates can be the same row.
    """
    values = np.asarray(fitness, dtype=float)
    pairs = np.asarray(draws, dtype=int).reshape(-1, 2)
    rank = np.empty(values.shape[0], dtype=int)
    rank[sorted_indices(values)] = np.arange(values.shape[0])
    first, second = pairs[:, 0], pairs[:, 1]
    return np.asarray(np.where(rank[second] < rank[first], second, first), dtype=int)


def cmoeams_tournament(fitness: NDArray[Any], n_tournaments: int, rng: np.random.Generator) -> IndexArray:
    """Winners of ``n_tournaments`` binary tournaments, each between two rows drawn with replacement."""
    n = int(np.asarray(fitness).shape[0])
    if n == 0:
        raise ValueError("CMOEA-MS cannot select parents from an empty population.")
    return tournament_winners(fitness, rng.integers(0, n, size=(int(n_tournaments), 2)))


__all__ = ["cmoeams_tournament", "environmental_selection", "tournament_winners", "truncation"]
