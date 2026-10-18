"""SBX-family real-valued crossover operators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.kernel.operator_primitives import available_variation_kernels, sbx_crossover_pairs

from ._crossover_base import Crossover
from .utils import ArrayLike, VariationWorkspace, _ensure_bounds

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import VariationKernels


class SBXCrossover(Crossover):
    """Simulated Binary Crossover (SBX) operator."""

    supports_variation_kernels: ClassVar[bool] = True

    def __init__(
        self,
        prob_crossover: float = 0.9,
        eta: float = 10.0,
        prob_var: float = 0.5,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
        allow_inplace: bool = False,
        variation_kernels: VariationKernels | None = None,
    ) -> None:
        self.prob = float(prob_crossover)
        self.eta = float(eta)
        self.prob_var = float(prob_var)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.workspace = workspace
        self.allow_inplace = bool(allow_inplace)
        self.variation_kernels = variation_kernels

    def _rand(self, key: str, shape: tuple[int, ...], rng: np.random.Generator) -> ArrayLike:
        if self.workspace is None:
            return rng.random(shape)
        buf = self.workspace.request(key, shape, np.float64)
        rng.random(out=buf)
        return buf

    def _mask_pairs(self, n_pairs: int, rng: np.random.Generator) -> NDArray[np.bool_]:
        if self.workspace is None:
            return rng.random(n_pairs) <= self.prob
        probs = self.workspace.request("sbx_prob", (n_pairs,), np.float64)
        rng.random(out=probs)
        mask = self.workspace.request("sbx_mask", (n_pairs,), np.bool_)
        np.less_equal(probs, self.prob, out=mask)
        return mask

    def _buffer(self, key: str, shape: tuple[int, ...], dtype: Any) -> NDArray[Any]:
        if self.workspace is None:
            return np.empty(shape, dtype=dtype)
        return self.workspace.request(key, shape, dtype)

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        parents_arr = self._as_matings(parents, copy=False, name="parents")
        offspring = parents_arr if self.allow_inplace else parents_arr.copy()
        n_pairs, _, _ = offspring.shape
        if n_pairs == 0:
            return offspring
        self._check_bounds_match(offspring[:, 0, :], self.lower)
        if self.workspace is None:
            return sbx_crossover_pairs(
                offspring,
                rng=rng,
                lower=self.lower,
                upper=self.upper,
                prob_crossover=self.prob,
                eta=self.eta,
                prob_var=self.prob_var,
                inplace=True,
                kernels=self.variation_kernels,
            )

        kernels = self.variation_kernels
        if kernels is None and n_pairs == 1:
            kernels = available_variation_kernels()
        if kernels is not None:
            return kernels.sbx_crossover_pairs(
                offspring,
                rng=rng,
                lower=self.lower,
                upper=self.upper,
                prob_crossover=self.prob,
                eta=self.eta,
                prob_var=self.prob_var,
            )

        pair_mask = self._mask_pairs(n_pairs, rng)
        idx = np.flatnonzero(pair_mask)
        if idx.size == 0:
            return offspring
        parent1 = offspring[idx, 0, :]
        parent2 = offspring[idx, 1, :]
        base1 = parent1.copy()
        base2 = parent2.copy()
        eps = 1.0e-14
        y1 = np.minimum(parent1, parent2)
        y2 = np.maximum(parent1, parent2)
        diff = y2 - y1
        valid = diff > eps
        if self.prob_var >= 1.0:
            active = valid
        else:
            var_rand = self._rand("sbx_var", parent1.shape, rng)
            if self.workspace is None:
                var_mask = var_rand <= self.prob_var
            else:
                var_mask = self.workspace.request("sbx_var_mask", parent1.shape, np.bool_)
                np.less_equal(var_rand, self.prob_var, out=var_mask)
            active = valid & var_mask
        if not np.any(active):
            return offspring

        xl = self.lower.reshape(1, -1)
        xu = self.upper.reshape(1, -1)
        rand = self._rand("sbx_rand", parent1.shape, rng)
        betaq = self._buffer("sbx_betaq", parent1.shape, parent1.dtype)

        beta_valid = np.maximum(1.0 + (2.0 * (y1 - xl) / diff.clip(min=eps)), eps)
        alpha = np.maximum(2.0 - np.power(beta_valid, -(self.eta + 1.0)), eps)
        term = rand <= (1.0 / alpha)
        inv_eta = 1.0 / (self.eta + 1.0)
        betaq[term] = np.power(rand[term] * alpha[term], inv_eta)
        betaq[~term] = np.power(1.0 / (2.0 - rand[~term] * alpha[~term]), inv_eta)
        c1 = 0.5 * ((y1 + y2) - betaq * diff)

        beta_valid = np.maximum(1.0 + (2.0 * (xu - y2) / diff.clip(min=eps)), eps)
        alpha = np.maximum(2.0 - np.power(beta_valid, -(self.eta + 1.0)), eps)
        term = rand <= (1.0 / alpha)
        betaq[term] = np.power(rand[term] * alpha[term], inv_eta)
        betaq[~term] = np.power(1.0 / (2.0 - rand[~term] * alpha[~term]), inv_eta)
        c2 = 0.5 * ((y1 + y2) + betaq * diff)

        swap = self._rand("sbx_swap", parent1.shape, rng)
        if self.workspace is None:
            swap_mask = swap <= 0.5
        else:
            swap_mask = self.workspace.request("sbx_swap_mask", parent1.shape, np.bool_)
            np.less_equal(swap, 0.5, out=swap_mask)
        child1 = np.where(active, c1, base1)
        child2 = np.where(active, c2, base2)
        swap_mask = swap_mask & active
        offspring[idx, 0, :] = np.where(swap_mask, child2, child1)
        offspring[idx, 1, :] = np.where(swap_mask, child1, child2)
        return offspring


__all__ = ["SBXCrossover"]
