"""Restricted mating selection of C-TAEA: the proportions that choose the archives and the binary tournament.

These functions transcribe jMetal's port (pull request jMetal/jMetal#524, classes
``MatingProportions`` and ``CTAEATournamentSelection``), which follows the reference code where it
differs from the paper:

- the second parent comes from the CA with a probability equal to the proportion of
  non-dominated solutions of the CA sorted alone (``PC``); in Algorithm 4 of the paper it is the
  proportion ``Pc`` that also chooses the archive of the first parent;
- when both solutions of a tournament are infeasible, the winner is a third solution drawn at
  random from the archive, instead of one of the two at random, which gives the winner the same
  distribution.

All the sorting uses Pareto dominance on the objectives, without constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .archives import front_ranks

IndexArray = NDArray[np.int_]


@dataclass(frozen=True)
class MatingProportions:
    """The proportions of non-dominated solutions that choose the archive of each parent.

    ``convergence_share`` (``Pc``) and ``diversity_share`` (``Pd``) are the numbers of
    non-dominated solutions of the union ``CA + DA`` in its first half (the CA) and in its second
    half (the DA), divided by the size of the union. ``non_dominated_share_of_convergence_archive``
    (``PC``) is the proportion of non-dominated solutions of the CA sorted alone.
    """

    convergence_share: float
    diversity_share: float
    non_dominated_share_of_convergence_archive: float

    @property
    def first_parent_from_convergence_archive(self) -> bool:
        """The first parent of every offspring comes from the CA when ``Pc > Pd``, from the DA otherwise."""
        return self.convergence_share > self.diversity_share


def mating_proportions(convergence_F: NDArray[Any], diversity_F: NDArray[Any]) -> MatingProportions:
    """The proportions of a CA (not empty) and a DA; the union's first half is its first ``(n + 1) // 2`` rows."""
    convergence = np.asarray(convergence_F, dtype=float)
    if convergence.shape[0] == 0:
        raise ValueError("C-TAEA's mating proportions need a convergence archive with at least one solution.")
    union = np.vstack([convergence, np.asarray(diversity_F, dtype=float).reshape(-1, convergence.shape[1])])
    n = union.shape[0]
    non_dominated = front_ranks(union) == 0
    half = (n + 1) // 2
    in_first_half = int(np.count_nonzero(non_dominated[:half]))
    in_second_half = int(np.count_nonzero(non_dominated[half:]))
    in_convergence_archive = int(np.count_nonzero(front_ranks(convergence) == 0))
    return MatingProportions(in_first_half / n, in_second_half / n, in_convergence_archive / convergence.shape[0])


def tournament_winners(F: NDArray[Any], feasible: NDArray[Any], draws: NDArray[Any]) -> IndexArray:
    """Winners of binary tournaments on fixed draws: row ``k`` of ``draws`` holds the rows ``(a, b, c)`` of tournament ``k``.

    - if only one of ``a`` and ``b`` is feasible, it wins;
    - if both are feasible, ``a`` wins unless ``b`` Pareto-dominates it;
    - if both are infeasible, ``c`` wins.
    """
    objectives = np.asarray(F, dtype=float)
    ok = np.asarray(feasible, dtype=bool)
    picks = np.asarray(draws, dtype=int).reshape(-1, 3)
    first, second, third = picks[:, 0], picks[:, 1], picks[:, 2]
    first_ok, second_ok = ok[first], ok[second]
    worse = np.any(objectives[second] > objectives[first], axis=1)
    better = np.any(objectives[second] < objectives[first], axis=1)
    second_dominates = ~worse & better
    winner = np.where(
        first_ok & second_ok,
        np.where(second_dominates, second, first),
        np.where(first_ok, first, np.where(second_ok, second, third)),
    )
    return np.asarray(winner, dtype=int)


def ctaea_tournament(F: NDArray[Any], feasible: NDArray[Any], n_tournaments: int, rng: np.random.Generator) -> IndexArray:
    """Winners of ``n_tournaments`` binary tournaments in an archive (see :func:`tournament_winners`).

    Each tournament draws three rows uniformly with replacement, ``rng.integers(0, n, size=(n_tournaments, 3))``;
    the third is used only when the first two are infeasible.
    """
    n = int(np.asarray(F).shape[0])
    if n == 0:
        raise ValueError("C-TAEA cannot select parents from an empty archive.")
    return tournament_winners(F, feasible, rng.integers(0, n, size=(int(n_tournaments), 3)))


__all__ = [
    "MatingProportions",
    "ctaea_tournament",
    "mating_proportions",
    "tournament_winners",
]
