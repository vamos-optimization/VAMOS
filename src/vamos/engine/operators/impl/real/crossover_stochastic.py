"""Distribution-driven real-valued crossover operators."""

from __future__ import annotations

import numpy as np

from ._crossover_base import Crossover
from .utils import ArrayLike, VariationWorkspace, _ensure_bounds


class LaplaceCrossover(Crossover):
    """Laplace crossover (Deep and Thakur, 2007), a variant in how it draws ``beta``.

    Each child is ``p + beta * |p1 - p2|`` with ``beta`` from a Laplace distribution of
    location ``a`` and scale ``b``. Here each child draws its own ``beta`` per gene;
    the paper uses one ``beta`` for both children, which places them symmetrically.
    """

    def __init__(
        self,
        a: float = 0.0,
        b: float = 0.5,
        prob_crossover: float = 0.9,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
        allow_inplace: bool = False,
    ) -> None:
        self.a = float(a)
        self.b = float(b)
        self.prob = float(prob_crossover)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.workspace = workspace
        self.allow_inplace = bool(allow_inplace)

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        parents_arr = self._as_matings(parents, copy=False, name="parents")
        offspring = parents_arr if self.allow_inplace else parents_arr.copy()
        n_pairs = offspring.shape[0]
        if n_pairs == 0:
            return offspring
        mask = rng.random(n_pairs) <= self.prob
        if not np.any(mask):
            return offspring
        active = offspring[mask]
        p1 = active[:, 0, :]
        p2 = active[:, 1, :]
        diff = np.abs(p1 - p2)
        u1 = rng.random(p1.shape)
        u2 = rng.random(p1.shape)
        eps = 1e-30
        beta1 = self.a + self.b * np.sign(u1 - 0.5) * np.log(np.maximum(1.0 - 2.0 * np.abs(u1 - 0.5), eps))
        beta2 = self.a + self.b * np.sign(u2 - 0.5) * np.log(np.maximum(1.0 - 2.0 * np.abs(u2 - 0.5), eps))
        active[:, 0, :] = p1 + beta1 * diff
        active[:, 1, :] = p2 + beta2 * diff
        offspring[mask] = active
        return offspring


class FuzzyCrossover(Crossover):
    """Fuzzy recombination crossover, a variant of Voigt, Mühlenbein and Cvetković (1995).

    Both children are drawn, gene by gene, from one triangular distribution over the
    parents' interval widened by ``d`` times its length on each side (clipped to the
    bounds), with its mode at the parents' midpoint. The paper draws each gene from
    two triangular distributions, one centred on each parent.
    """

    def __init__(
        self,
        d: float = 0.5,
        prob_crossover: float = 0.9,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
        allow_inplace: bool = False,
    ) -> None:
        self.d = float(d)
        self.prob = float(prob_crossover)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.workspace = workspace
        self.allow_inplace = bool(allow_inplace)

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        parents_arr = self._as_matings(parents, copy=False, name="parents")
        offspring = parents_arr if self.allow_inplace else parents_arr.copy()
        n_pairs = offspring.shape[0]
        if n_pairs == 0:
            return offspring
        mask = rng.random(n_pairs) <= self.prob
        if not np.any(mask):
            return offspring
        active = offspring[mask]
        p1 = active[:, 0, :]
        p2 = active[:, 1, :]
        lo_p = np.minimum(p1, p2)
        hi_p = np.maximum(p1, p2)
        span = hi_p - lo_p
        tri_lo = lo_p - self.d * span
        tri_hi = hi_p + self.d * span
        tri_mode = 0.5 * (p1 + p2)
        np.clip(tri_lo, self.lower, self.upper, out=tri_lo)
        np.clip(tri_hi, self.lower, self.upper, out=tri_hi)
        np.clip(tri_mode, tri_lo, tri_hi, out=tri_mode)
        degenerate = tri_hi - tri_lo < 1e-30
        width = np.where(degenerate, 1.0, tri_hi - tri_lo)
        frac_center = np.where(degenerate, 0.5, (tri_mode - tri_lo) / width)
        u1 = rng.random(p1.shape)
        u2 = rng.random(p1.shape)
        left1 = tri_lo + np.sqrt(u1 * width * frac_center * width)
        right1 = tri_hi - np.sqrt((1.0 - u1) * width * (1.0 - frac_center) * width)
        left2 = tri_lo + np.sqrt(u2 * width * frac_center * width)
        right2 = tri_hi - np.sqrt((1.0 - u2) * width * (1.0 - frac_center) * width)
        active[:, 0, :] = np.where(degenerate, tri_mode, np.where(u1 < frac_center, left1, right1))
        active[:, 1, :] = np.where(degenerate, tri_mode, np.where(u2 < frac_center, left2, right2))
        offspring[mask] = active
        return offspring


__all__ = ["FuzzyCrossover", "LaplaceCrossover"]
