"""ToR, the fitness of two rankings, as a constraint-handling technique (see :mod:`.constraint_handling`).

Z. Ma, Y. Wang and W. Song, "A New Fitness Function With Two Rankings for Evolutionary Constrained
Multiobjective Optimization", IEEE Transactions on Systems, Man, and Cybernetics: Systems 51(8):5005-5016,
2021, section III-A. Algorithm 1 sorts the population by the constrained-domination principle into levels,
and each level by crowding distance in descending order; ``R^c_i`` is the position of solution ``i`` in that
order, from 1. Algorithm 2 does the same with Pareto dominance, ignoring the constraints, and gives
``R^p_i``. Each solution's fitness is ``fit_i = alpha R^c_i + (1 - alpha) R^p_i`` (eq. 6), with ``alpha =
0.5 + 0.5 P_fea`` (eq. 7), where ``P_fea`` is the proportion of feasible solutions in the population being
ranked, and the best by fitness survive. The technique has no parameters.

Choices the paper leaves open:

- Solutions with the same crowding distance in a level, such as the two or more boundary solutions, whose
  distance is infinite, keep the order of the population, as the paper's Tables I-III do.
- Solutions with the same fitness are ordered by ``R^c``, which is unique, so the order is total.
- The crowding distance is NSGA-II's: the difference of the neighbours in each objective over the
  objective's range in the level. The paper's Tables I-III sum the differences without dividing them by the
  range, which changes some positions but not the solutions they select; :func:`two_rankings_fitness` takes
  another crowding distance to reproduce them.

The ranks that the technique returns are the positions in that total order, 0 to ``n - 1``, one solution
per front. The paper defines ToR for NSGA-II, whose survival, which keeps whole fronts, then keeps the best
by fitness; crowding distance never decides. NSGA-III, SMS-EMOA and AGE-MOEA reject the technique, since its
order would replace their niching, hypervolume contributions and survival scores.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .constrained_ranking import RankArray, pareto_ranks, ranks_by_violation, violation
from .constraint_technique import ConstraintHandlingTechnique

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import KernelBackend

#: A crowding distance: the distance of each row of a set of mutually non-dominated objective vectors.
Crowding = Callable[[NDArray[np.float64]], NDArray[np.float64]]


class TwoRankings(ConstraintHandlingTechnique):
    """Ma, Wang and Song's ToR: the best by ``alpha R^c + (1 - alpha) R^p``, one solution per front."""

    name = "two_rankings"
    total_order = True

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        return two_rankings_ranks(kernel, F, G)


def two_rankings_fitness(
    kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], crowding: Crowding | None = None
) -> tuple[RankArray, RankArray, NDArray[np.float64]]:
    """``R^c``, ``R^p`` (from 1) and the fitness of each row, with VAMOS's crowding distance or ``crowding``."""
    constrained, pareto, doubled = _two_rankings(kernel, F, G, crowding)
    return constrained, pareto, doubled / (2.0 * max(constrained.size, 1))


def two_rankings_ranks(kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], crowding: Crowding | None = None) -> RankArray:
    """The position of each row by fitness, then by ``R^c``, from 0, with VAMOS's crowding distance or ``crowding``."""
    constrained, _, doubled = _two_rankings(kernel, F, G, crowding)
    ranks = np.empty(constrained.size, dtype=np.int64)
    ranks[np.lexsort((constrained, doubled))] = np.arange(constrained.size, dtype=np.int64)
    return ranks


def _two_rankings(
    kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], crowding: Crowding | None
) -> tuple[RankArray, RankArray, RankArray]:
    """``R^c``, ``R^p`` and twice the population size times the fitness, an integer, so that equal fitness is exact.

    With ``k`` feasible rows of ``n``, ``2 n fit = n (R^c + R^p) + k (R^c - R^p)``.
    """
    F = np.asarray(F, dtype=np.float64)
    measure = violation(G)
    distance = crowding if crowding is not None else _kernel_crowding(kernel)
    constrained = _positions(F, ranks_by_violation(kernel, F, measure, objectives_on_ties=True), distance)
    pareto = _positions(F, pareto_ranks(kernel, F), distance)
    n, feasible = F.shape[0], int(np.count_nonzero(measure <= 0.0))
    return constrained, pareto, n * (constrained + pareto) + feasible * (constrained - pareto)


def _kernel_crowding(kernel: KernelBackend) -> Crowding:
    def crowding(F: NDArray[np.float64]) -> NDArray[np.float64]:
        if F.shape[0] <= 2:
            return np.full(F.shape[0], np.inf)
        return np.asarray(kernel.nsga2_ranking(F)[1], dtype=np.float64)

    return crowding


def _positions(F: NDArray[np.float64], levels: RankArray, crowding: Crowding) -> RankArray:
    """The position of each row, from 1, sorted by level and then by crowding distance in descending order."""
    n = F.shape[0]
    distance = np.empty(n, dtype=np.float64)
    for level in np.unique(levels):
        rows = np.flatnonzero(levels == level)
        distance[rows] = crowding(F[rows])
    positions = np.empty(n, dtype=np.int64)
    positions[np.lexsort((np.arange(n), -distance, levels))] = np.arange(1, n + 1, dtype=np.int64)
    return positions


__all__ = ["Crowding", "TwoRankings", "two_rankings_fitness", "two_rankings_ranks"]
