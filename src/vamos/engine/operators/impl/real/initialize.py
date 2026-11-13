from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats.qmc import Halton, Sobol  # type: ignore[import-untyped]

from vamos.foundation.constraints.utils import compute_violation
from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.foundation.rng import require_generator


class LatinHypercubeInitializer:
    """
    Latin Hypercube Sampling initializer for real-valued variables.
    Generates n_solutions points inside [lower, upper].
    """

    def __init__(
        self,
        n_solutions: int,
        lower: np.ndarray[Any, Any],
        upper: np.ndarray[Any, Any],
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_solutions = int(n_solutions)
        if self.n_solutions <= 0:
            raise ValueError("n_solutions must be positive.")
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape.")
        if self.lower.ndim != 1:
            raise ValueError("Bounds must be one-dimensional.")
        self.n_var = self.lower.shape[0]
        self.rng = require_generator(rng, "LatinHypercubeInitializer")

    def __call__(self) -> np.ndarray[Any, Any]:
        n = self.n_solutions
        d = self.n_var
        samples = np.empty((n, d), dtype=float)
        for j in range(d):
            strata = (np.arange(n, dtype=float) + self.rng.random(n)) / n
            self.rng.shuffle(strata)
            samples[:, j] = strata
        span = self.upper - self.lower
        return np.asarray(self.lower + samples * span, dtype=float)


class ScatterSearchInitializer:
    """
    Scatter search-style initializer using an LHS base set and convex combinations.
    """

    def __init__(
        self,
        n_solutions: int,
        lower: np.ndarray[Any, Any],
        upper: np.ndarray[Any, Any],
        base_size: int = 20,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_solutions = int(n_solutions)
        if self.n_solutions <= 0:
            raise ValueError("n_solutions must be positive.")
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape.")
        if self.lower.ndim != 1:
            raise ValueError("Bounds must be one-dimensional.")
        self.base_size = max(1, int(base_size))
        self.rng = require_generator(rng, "ScatterSearchInitializer")

    def __call__(self) -> np.ndarray[Any, Any]:
        lhs = LatinHypercubeInitializer(self.base_size, self.lower, self.upper, rng=self.rng)
        base = lhs()
        if self.n_solutions <= self.base_size:
            return base[: self.n_solutions]
        needed = self.n_solutions - self.base_size
        children = np.empty((needed, self.lower.shape[0]), dtype=float)
        for i in range(needed):
            idx_a = self.rng.integers(0, self.base_size)
            idx_b = self.rng.integers(0, self.base_size)
            alpha = self.rng.random()
            child = alpha * base[idx_a] + (1.0 - alpha) * base[idx_b]
            np.clip(child, self.lower, self.upper, out=child)
            children[i] = child
        return np.vstack((base, children))[: self.n_solutions]


class SobolInitializer:
    """Sobol quasi-random sequence initializer for real-valued variables."""

    def __init__(
        self,
        n_solutions: int,
        lower: np.ndarray[Any, Any],
        upper: np.ndarray[Any, Any],
        rng: np.random.Generator | None = None,
        scramble: bool = True,
    ) -> None:
        self.n_solutions = int(n_solutions)
        if self.n_solutions <= 0:
            raise ValueError("n_solutions must be positive.")
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape.")
        if self.lower.ndim != 1:
            raise ValueError("Bounds must be one-dimensional.")
        self.n_var = self.lower.shape[0]
        self.scramble = scramble
        self.rng = require_generator(rng, "SobolInitializer")

    def __call__(self) -> np.ndarray[Any, Any]:
        seed = int(self.rng.integers(0, 2**31))
        sampler = Sobol(d=self.n_var, scramble=self.scramble, seed=seed)
        m = int(np.ceil(np.log2(max(self.n_solutions, 2))))
        n_pow2 = 2**m
        if n_pow2 < self.n_solutions:
            n_pow2 *= 2
        samples = sampler.random(n_pow2)[: self.n_solutions]
        span = self.upper - self.lower
        return np.asarray(self.lower + samples * span, dtype=float)


class HaltonInitializer:
    """Halton quasi-random sequence initializer for real-valued variables."""

    def __init__(
        self,
        n_solutions: int,
        lower: np.ndarray[Any, Any],
        upper: np.ndarray[Any, Any],
        rng: np.random.Generator | None = None,
        scramble: bool = True,
    ) -> None:
        self.n_solutions = int(n_solutions)
        if self.n_solutions <= 0:
            raise ValueError("n_solutions must be positive.")
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape.")
        if self.lower.ndim != 1:
            raise ValueError("Bounds must be one-dimensional.")
        self.n_var = self.lower.shape[0]
        self.scramble = scramble
        self.rng = require_generator(rng, "HaltonInitializer")

    def __call__(self) -> np.ndarray[Any, Any]:
        seed = int(self.rng.integers(0, 2**31))
        sampler = Halton(d=self.n_var, scramble=self.scramble, seed=seed)
        samples = sampler.random(self.n_solutions)
        span = self.upper - self.lower
        return np.asarray(self.lower + samples * span, dtype=float)


class OppositionBasedInitializer:
    """Opposition-Based Learning (OBL) initializer for real-valued variables.

    Generates a random population and its opposition-based counterparts
    (opposite = lower + upper - x), evaluates all 2*n_solutions candidates,
    and selects the best n_solutions: feasible candidates first, then by
    non-domination rank and crowding distance.

    An algorithm counts the 2*n_solutions evaluations against its budget and
    keeps the chosen candidates' objective and constraint values: it calls
    :meth:`candidates`, evaluates them with its own backend and keeps the rows
    that :meth:`select` returns. Calling the initializer evaluates the
    candidates with ``problem.evaluate`` and returns only the chosen decision
    vectors.
    """

    def __init__(
        self,
        n_solutions: int,
        lower: np.ndarray[Any, Any],
        upper: np.ndarray[Any, Any],
        problem: Any,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_solutions = int(n_solutions)
        if self.n_solutions <= 0:
            raise ValueError("n_solutions must be positive.")
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape.")
        if self.lower.ndim != 1:
            raise ValueError("Bounds must be one-dimensional.")
        self.n_var = self.lower.shape[0]
        self.problem = problem
        self.rng = require_generator(rng, "OppositionBasedInitializer")

    @property
    def n_candidates(self) -> int:
        """Number of candidates the initializer evaluates: the random points and their opposites."""
        return 2 * self.n_solutions

    def candidates(self) -> np.ndarray[Any, Any]:
        """Draw ``n_solutions`` random points and return them stacked over their opposites."""
        X = self.rng.uniform(self.lower, self.upper, size=(self.n_solutions, self.n_var))
        opposites = self.lower + self.upper - X
        np.clip(opposites, self.lower, self.upper, out=opposites)
        return np.vstack((X, opposites))

    def select(self, F: np.ndarray[Any, Any], G: np.ndarray[Any, Any] | None = None) -> np.ndarray[Any, Any]:
        """Indices of the ``n_solutions`` candidates to keep.

        Feasible candidates come first. When more are feasible than needed, they
        are chosen by non-domination rank and crowding distance; when fewer are,
        the least violated infeasible candidates complete the population.
        """
        n = self.n_solutions
        if G is None:
            return self._select_best(F, np.arange(F.shape[0]), n)
        violation = compute_violation(G)
        feasible = np.flatnonzero(violation <= 0.0)
        if feasible.size >= n:
            return self._select_best(F, feasible, n)
        infeasible = np.flatnonzero(violation > 0.0)
        least_violated = infeasible[np.argsort(violation[infeasible], kind="stable")]
        return np.concatenate([feasible, least_violated[: n - feasible.size]])

    def __call__(self) -> np.ndarray[Any, Any]:
        combined = self.candidates()
        F, G = evaluate_population_with_constraints(self.problem, combined)
        indices = self.select(F, G)
        return np.asarray(combined[indices], dtype=float)

    @staticmethod
    def _select_best(F: np.ndarray[Any, Any], rows: np.ndarray[Any, Any], n: int) -> np.ndarray[Any, Any]:
        F_rows = F[rows]
        if F_rows.shape[1] == 1:
            return np.asarray(rows[np.argsort(F_rows[:, 0], kind="stable")[:n]])
        from vamos.foundation.kernel.numpy_backend import (
            _compute_crowding,
            _fast_non_dominated_sort,
            _select_nsga2,
        )

        fronts, _rank = _fast_non_dominated_sort(F_rows)
        crowding = _compute_crowding(F_rows, fronts)
        return np.asarray(rows[_select_nsga2(fronts, crowding, n)])


__all__ = [
    "LatinHypercubeInitializer",
    "ScatterSearchInitializer",
    "SobolInitializer",
    "HaltonInitializer",
    "OppositionBasedInitializer",
]
