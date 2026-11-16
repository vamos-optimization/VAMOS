"""Blend and arithmetic real-valued crossover operators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._crossover_base import Crossover
from .utils import ArrayLike, VariationWorkspace, _ensure_bounds


class BLXAlphaCrossover(Crossover):
    """Blend crossover (BLX-alpha) with optional buffer reuse."""

    def __init__(
        self,
        alpha: float = 0.5,
        prob_crossover: float = 0.9,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
        allow_inplace: bool = False,
    ) -> None:
        self.alpha = float(alpha)
        self.prob = float(prob_crossover)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.workspace = workspace
        self.allow_inplace = bool(allow_inplace)

    def _rand(self, key: str, shape: tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        if self.workspace is None:
            return rng.random(shape)
        buf = self.workspace.request(key, shape, np.float64)
        rng.random(out=buf)
        return buf

    def _sample_mask(self, n_pairs: int, rng: np.random.Generator) -> NDArray[np.bool_]:
        if self.workspace is None:
            return rng.random(n_pairs) <= self.prob
        probs = self.workspace.request("blx_prob", (n_pairs,), np.float64)
        rng.random(out=probs)
        mask = self.workspace.request("blx_mask", (n_pairs,), np.bool_)
        np.less_equal(probs, self.prob, out=mask)
        return mask

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        parents_arr = self._as_matings(parents, copy=False, name="parents")
        offspring = parents_arr if self.allow_inplace else parents_arr.copy()
        self._check_bounds_match(offspring[:, 0, :], self.lower)
        n_pairs = offspring.shape[0]
        if n_pairs == 0:
            return offspring
        mask = self._sample_mask(n_pairs, rng)
        if not np.any(mask):
            return offspring
        active = offspring[mask]
        p1 = active[:, 0, :]
        p2 = active[:, 1, :]
        lo = np.minimum(p1, p2)
        hi = np.maximum(p1, p2)
        span = hi - lo
        lower = lo - self.alpha * span
        upper = hi + self.alpha * span
        width = upper - lower
        active[:, 0, :] = lower + self._rand("blx_rand1", lower.shape, rng) * width
        active[:, 1, :] = lower + self._rand("blx_rand2", lower.shape, rng) * width
        offspring[mask] = active
        return offspring


class BLXAlphaBetaCrossover(Crossover):
    """Blend crossover with two extensions (BLX-alpha-beta), positional as in jMetal.

    Each child gene is drawn uniformly from ``[lo - alpha * span, hi + beta * span]``, where
    ``lo`` and ``hi`` are the smaller and larger parent values and ``span = hi - lo``. The
    published operator (Eshelman, Mathias and Schaffer, 1997) extends ``alpha`` towards the
    better parent and ``beta`` towards the worse one, which needs the parents' fitness; this
    variant extends below and above instead, as jMetal's ``BLXAlphaBetaCrossover`` does. With
    ``alpha != beta`` it therefore drifts towards one bound, so both default to 0.5, jMetal's
    values, which make it BLX-0.5.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        prob_crossover: float = 0.9,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
        allow_inplace: bool = False,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.prob = float(prob_crossover)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.workspace = workspace
        self.allow_inplace = bool(allow_inplace)

    def _rand(self, key: str, shape: tuple[int, ...], rng: np.random.Generator) -> NDArray[np.float64]:
        if self.workspace is None:
            return rng.random(shape)
        buf = self.workspace.request(key, shape, np.float64)
        rng.random(out=buf)
        return buf

    def _sample_mask(self, n_pairs: int, rng: np.random.Generator) -> NDArray[np.bool_]:
        if self.workspace is None:
            return rng.random(n_pairs) <= self.prob
        probs = self.workspace.request("blxab_prob", (n_pairs,), np.float64)
        rng.random(out=probs)
        mask = self.workspace.request("blxab_mask", (n_pairs,), np.bool_)
        np.less_equal(probs, self.prob, out=mask)
        return mask

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        parents_arr = self._as_matings(parents, copy=False, name="parents")
        offspring = parents_arr if self.allow_inplace else parents_arr.copy()
        n_pairs = offspring.shape[0]
        if n_pairs == 0:
            return offspring
        mask = self._sample_mask(n_pairs, rng)
        if not np.any(mask):
            return offspring
        active = offspring[mask]
        p1 = active[:, 0, :]
        p2 = active[:, 1, :]
        lo = np.minimum(p1, p2)
        hi = np.maximum(p1, p2)
        span = hi - lo
        lower_range = lo - self.alpha * span
        upper_range = hi + self.beta * span
        width = upper_range - lower_range
        active[:, 0, :] = lower_range + self._rand("blxab_rand1", lower_range.shape, rng) * width
        active[:, 1, :] = lower_range + self._rand("blxab_rand2", lower_range.shape, rng) * width
        offspring[mask] = active
        return offspring


class ArithmeticCrossover(Crossover):
    """Arithmetic crossover mixing parents through random convex combinations."""

    def __init__(
        self,
        prob_crossover: float = 0.9,
        *,
        lower: ArrayLike | None = None,
        upper: ArrayLike | None = None,
        workspace: VariationWorkspace | None = None,
        allow_inplace: bool = False,
    ) -> None:
        self.prob = float(prob_crossover)

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        parents_arr = self._as_matings(parents, copy=False, name="parents")
        offspring = parents_arr.copy()
        n_pairs = offspring.shape[0]
        if n_pairs == 0:
            return offspring
        mask = np.asarray(rng.random(n_pairs) <= self.prob, dtype=bool)
        if not np.any(mask):
            return offspring
        lam = rng.random((int(mask.sum()), 1))
        p1 = offspring[mask, 0, :]
        p2 = offspring[mask, 1, :]
        offspring[mask, 0, :] = lam * p1 + (1.0 - lam) * p2
        offspring[mask, 1, :] = (1.0 - lam) * p1 + lam * p2
        return offspring


class WholeArithmeticCrossover(Crossover):
    """Whole arithmetic crossover with a fixed blending weight ``alpha``."""

    def __init__(
        self,
        alpha: float = 0.5,
        prob_crossover: float = 0.9,
        *,
        lower: ArrayLike | None = None,
        upper: ArrayLike | None = None,
        workspace: VariationWorkspace | None = None,
        allow_inplace: bool = False,
    ) -> None:
        self.alpha = float(alpha)
        self.prob = float(prob_crossover)

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        parents_arr = self._as_matings(parents, copy=False, name="parents")
        offspring = parents_arr.copy()
        n_pairs = offspring.shape[0]
        if n_pairs == 0:
            return offspring
        mask = rng.random(n_pairs) <= self.prob
        if not np.any(mask):
            return offspring
        p1 = offspring[mask, 0, :]
        p2 = offspring[mask, 1, :]
        offspring[mask, 0, :] = self.alpha * p1 + (1.0 - self.alpha) * p2
        offspring[mask, 1, :] = (1.0 - self.alpha) * p1 + self.alpha * p2
        return offspring


__all__ = [
    "ArithmeticCrossover",
    "BLXAlphaBetaCrossover",
    "BLXAlphaCrossover",
    "WholeArithmeticCrossover",
]
