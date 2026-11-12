"""Differential-evolution style real-valued crossovers."""

from __future__ import annotations

import numpy as np

from ._crossover_base import Crossover
from .utils import ArrayLike, _check_nvars, _ensure_bounds


class DifferentialCrossover(Crossover):
    """Differential evolution DE/current/1/bin over a whole population.

    Each individual ``x_i`` is the base of its own mutant
    ``v = x_i + F * (x_r1 - x_r2)``, with ``r1`` and ``r2`` distinct and different
    from ``i``; binomial crossover with ``CR`` then takes each gene from ``v`` (at
    least one). The classic DE/rand/1/bin uses a random base ``x_r0`` instead.
    """

    def __init__(self, F: float = 0.5, CR: float = 0.9, *, lower: ArrayLike, upper: ArrayLike) -> None:
        self.F = float(F)
        self.CR = float(CR)
        self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, population: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        pop = self._as_population(population, name="population", copy=False)
        n_ind, n_vars = pop.shape
        _check_nvars(n_vars, self.lower)
        if n_ind < 3:
            raise ValueError("Differential crossover requires at least 3 individuals.")

        trial = pop.copy()
        all_indices = np.arange(n_ind)
        for i in range(n_ind):
            choices = np.delete(all_indices, i)
            r1, r2 = rng.choice(choices, size=2, replace=False)
            mutant = pop[i] + self.F * (pop[r1] - pop[r2])
            cross_mask = np.asarray(rng.random(n_vars) < self.CR, dtype=bool)
            cross_mask[rng.integers(n_vars)] = True
            trial[i, cross_mask] = mutant[cross_mask]
        return trial


class DEMatingCrossover(Crossover):
    """Differential evolution crossover for 3-parent mating groups ``[x, a, b]``.

    The mutant is ``x + F * (a - b)``, and binomial crossover with ``CR`` takes each
    gene of the child from it (at least one). ``prob_crossover`` is accepted for a
    uniform operator interface and ignored: every group is recombined.
    """

    def __init__(
        self,
        F: float = 0.5,
        CR: float = 0.9,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        prob_crossover: float | None = None,
        workspace: object | None = None,
        allow_inplace: bool = False,
    ) -> None:
        _ = prob_crossover, workspace, allow_inplace
        self.F = float(F)
        self.CR = float(CR)
        self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        groups = self._as_matings(parents, expected_parents=3, copy=False)
        n_groups, _, n_vars = groups.shape
        target = groups[:, 0, :]
        mutant = target + self.F * (groups[:, 1, :] - groups[:, 2, :])
        trial = target.copy()
        cross_mask = rng.random((n_groups, n_vars)) < self.CR
        cross_mask[np.arange(n_groups), rng.integers(n_vars, size=n_groups)] = True
        trial[cross_mask] = mutant[cross_mask]
        return trial[:, np.newaxis, :]


__all__ = ["DEMatingCrossover", "DifferentialCrossover"]
