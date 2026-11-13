from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.rng import require_generator


class TournamentSelection:
    """
    Simple tournament selection using a comparator.
    comparator(a, b) returns <0 if a better than b, >0 if b better, 0 if tie.
    """

    def __init__(
        self,
        tournament_size: int,
        comparator: Callable[[int, int], int],
        rng: np.random.Generator | None = None,
    ) -> None:
        if tournament_size <= 0:
            raise ValueError("tournament_size must be positive.")
        self.tournament_size = int(tournament_size)
        self.comparator = comparator
        self.rng = require_generator(rng, "TournamentSelection")

    def __call__(self, population: Sequence[object], n_parents: int) -> NDArray[np.int64]:
        pop_size = len(population)
        if pop_size == 0:
            raise ValueError("population is empty.")
        rng = self.rng
        selected = np.empty(n_parents, dtype=int)
        for i in range(n_parents):
            contenders = rng.integers(0, pop_size, size=self.tournament_size)
            best = contenders[0]
            for idx in contenders[1:]:
                cmp = self.comparator(idx, best)
                if cmp < 0:
                    best = idx
            selected[i] = best
        return selected


class RandomSelection:
    """Uniform random parent selection."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = require_generator(rng, "RandomSelection")

    def __call__(self, population: Sequence[object], n_parents: int) -> NDArray[np.int64]:
        pop_size = len(population)
        if pop_size == 0:
            raise ValueError("population is empty.")
        return self.rng.integers(0, pop_size, size=n_parents)


class BoltzmannSelection:
    """Boltzmann (softmax) selection converting fitness to probabilities.

    Probability of selecting individual *i* is:

        p_i = exp(fitness_i / T) / Σ exp(fitness_j / T)

    where *T* is the temperature.  Higher *T* flattens probabilities
    (more exploration); lower *T* concentrates on the best (exploitation).

    The *fitness* array must be provided at call time and should be oriented
    so that *higher is better* (e.g. negative rank).
    """

    def __init__(
        self,
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = float(temperature)
        self.rng = require_generator(rng, "BoltzmannSelection")

    def __call__(
        self,
        population: Sequence[object],
        n_parents: int,
        *,
        fitness: NDArray[np.float64] | None = None,
    ) -> NDArray[np.int64]:
        pop_size = len(population)
        if pop_size == 0:
            raise ValueError("population is empty.")
        if fitness is None:
            raise ValueError("BoltzmannSelection requires a fitness array.")
        f = np.asarray(fitness, dtype=float)
        # Numerically stable softmax
        f_shifted = f - f.max()
        exp_f = np.exp(f_shifted / self.temperature)
        probs = exp_f / exp_f.sum()
        return self.rng.choice(pop_size, size=n_parents, replace=True, p=probs)


class RankingSelection:
    """Linear ranking selection assigning probabilities by rank order.

    Individuals are sorted by fitness (higher is better) and assigned
    linearly interpolated selection probabilities controlled by *sp*
    (selective pressure, 1.0 – 2.0):

        p_i = (2 − sp + 2·(sp − 1)·(N − rank_i − 1) / (N − 1)) / N

    where *rank_i = 0* is the worst and *rank_i = N − 1* is the best.
    """

    def __init__(
        self,
        sp: float = 1.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not 1.0 <= sp <= 2.0:
            raise ValueError("Selective pressure sp must be in [1.0, 2.0].")
        self.sp = float(sp)
        self.rng = require_generator(rng, "RankingSelection")

    def __call__(
        self,
        population: Sequence[object],
        n_parents: int,
        *,
        fitness: NDArray[np.float64] | None = None,
    ) -> NDArray[np.int64]:
        pop_size = len(population)
        if pop_size == 0:
            raise ValueError("population is empty.")
        if fitness is None:
            raise ValueError("RankingSelection requires a fitness array.")
        f = np.asarray(fitness, dtype=float)
        # rank_order[i] = rank of individual i (0 = worst, N-1 = best)
        rank_order = np.argsort(np.argsort(f))
        if pop_size == 1:
            probs = np.array([1.0])
        else:
            probs = (2.0 - self.sp + 2.0 * (self.sp - 1.0) * rank_order / (pop_size - 1)) / pop_size
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum()
        return self.rng.choice(pop_size, size=n_parents, replace=True, p=probs)


class SUSSelection:
    """Stochastic Universal Sampling (SUS) with evenly spaced pointers.

    SUS is a low-variance alternative to roulette-wheel selection.
    A single random offset determines *n_parents* evenly spaced pointers
    along the cumulative fitness distribution, guaranteeing each
    individual is selected approximately proportionally to its fitness.

    The *fitness* array must be provided at call time (higher is better).
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = require_generator(rng, "SUSSelection")

    def __call__(
        self,
        population: Sequence[object],
        n_parents: int,
        *,
        fitness: NDArray[np.float64] | None = None,
    ) -> NDArray[np.int64]:
        pop_size = len(population)
        if pop_size == 0:
            raise ValueError("population is empty.")
        if fitness is None:
            raise ValueError("SUSSelection requires a fitness array.")
        f = np.asarray(fitness, dtype=float)
        # Shift so all values are positive
        f_shifted = f - f.min()
        total = f_shifted.sum()
        if total <= 0:
            # All equal fitness — uniform random
            return self.rng.choice(pop_size, size=n_parents, replace=True)
        cumsum = np.cumsum(f_shifted)
        step = total / n_parents
        start = self.rng.random() * step
        pointers = start + step * np.arange(n_parents)
        selected = np.searchsorted(cumsum, pointers, side="left")
        np.clip(selected, 0, pop_size - 1, out=selected)
        return np.asarray(selected, dtype=int)


__all__ = [
    "BoltzmannSelection",
    "RandomSelection",
    "RankingSelection",
    "SUSSelection",
    "TournamentSelection",
]
