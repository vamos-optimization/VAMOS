from __future__ import annotations

from functools import cache
from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .backend import FloatArray, RealArray, VariationKernels


@cache
def available_variation_kernels() -> VariationKernels | None:
    """Return the compiled variation kernels when numba is installed and exact.

    Single-pair SBX has historically used compiled arithmetic whenever numba is
    importable, independently of the selected engine. The compiled kernels
    consume random draws exactly like the NumPy code below and are returned
    only when their arithmetic reproduces NumPy on this platform, so this
    changes speed, never results.
    """
    if find_spec("numba") is None:
        return None
    from ._numba_variation import verified_variation_kernels

    return verified_variation_kernels()


def _ensure_bounds(lower: float | RealArray, upper: float | RealArray, n_var: int) -> tuple[FloatArray, FloatArray]:
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    if lower_arr.ndim == 0 or (lower_arr.ndim == 1 and lower_arr.shape[0] == 1 and n_var > 1):
        lower_arr = np.full(n_var, float(lower_arr.reshape(-1)[0]))
    if upper_arr.ndim == 0 or (upper_arr.ndim == 1 and upper_arr.shape[0] == 1 and n_var > 1):
        upper_arr = np.full(n_var, float(upper_arr.reshape(-1)[0]))
    return lower_arr, upper_arr


def sbx_crossover_pairs(
    parents: RealArray,
    *,
    rng: np.random.Generator,
    lower: float | RealArray,
    upper: float | RealArray,
    prob_crossover: float = 0.9,
    eta: float = 10.0,
    prob_var: float = 0.5,
    inplace: bool = False,
    kernels: VariationKernels | None = None,
) -> FloatArray:
    offspring = parents if inplace else np.array(parents, copy=True, dtype=float)
    if offspring.ndim != 3 or offspring.shape[1] != 2:
        raise ValueError("SBX expects a mating tensor of shape (n_pairs, 2, n_var).")
    n_pairs, _, n_var = offspring.shape
    if n_pairs == 0:
        return offspring

    lower_arr, upper_arr = _ensure_bounds(lower, upper, n_var)

    if kernels is None and n_pairs == 1:
        kernels = available_variation_kernels()
    if kernels is not None and offspring.dtype == np.float64:
        return kernels.sbx_crossover_pairs(
            offspring,
            rng=rng,
            lower=lower_arr,
            upper=upper_arr,
            prob_crossover=float(prob_crossover),
            eta=float(eta),
            prob_var=float(prob_var),
        )

    pair_mask = rng.random(n_pairs) <= prob_crossover
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
    if prob_var >= 1.0:
        active = valid
    else:
        active = valid & (rng.random(parent1.shape) <= prob_var)
    if not np.any(active):
        return offspring

    xl = lower_arr.reshape(1, -1)
    xu = upper_arr.reshape(1, -1)
    rand = rng.random(parent1.shape)
    betaq = np.empty_like(parent1)

    beta_valid = 1.0 + (2.0 * (y1 - xl) / diff.clip(min=eps))
    beta_valid = np.maximum(beta_valid, eps)
    alpha = 2.0 - np.power(beta_valid, -(eta + 1.0))
    alpha = np.maximum(alpha, eps)
    term = rand <= (1.0 / alpha)
    inv_eta = 1.0 / (eta + 1.0)
    betaq[term] = np.power(rand[term] * alpha[term], inv_eta)
    betaq[~term] = np.power(1.0 / (2.0 - rand[~term] * alpha[~term]), inv_eta)
    c1 = 0.5 * ((y1 + y2) - betaq * diff)

    beta_valid = 1.0 + (2.0 * (xu - y2) / diff.clip(min=eps))
    beta_valid = np.maximum(beta_valid, eps)
    alpha = 2.0 - np.power(beta_valid, -(eta + 1.0))
    alpha = np.maximum(alpha, eps)
    term = rand <= (1.0 / alpha)
    betaq[term] = np.power(rand[term] * alpha[term], inv_eta)
    betaq[~term] = np.power(1.0 / (2.0 - rand[~term] * alpha[~term]), inv_eta)
    c2 = 0.5 * ((y1 + y2) + betaq * diff)

    swap_mask = (rng.random(parent1.shape) <= 0.5) & active
    child1 = np.where(active, c1, base1)
    child2 = np.where(active, c2, base2)
    new_child1 = np.where(swap_mask, child2, child1)
    new_child2 = np.where(swap_mask, child1, child2)

    offspring[idx, 0, :] = new_child1
    offspring[idx, 1, :] = new_child2
    return offspring


def polynomial_mutation_population(
    offspring: RealArray,
    *,
    rng: np.random.Generator,
    lower: float | RealArray,
    upper: float | RealArray,
    prob_mutation: float,
    eta: float = 20.0,
    rnd_mask: RealArray | None = None,
    rnd_delta: RealArray | None = None,
    inplace: bool = False,
    kernels: VariationKernels | None = None,
) -> FloatArray:
    X = offspring if inplace else np.array(offspring, copy=True, dtype=float)
    if X.ndim != 2:
        raise ValueError("Polynomial mutation expects a population matrix of shape (n_ind, n_var).")
    n_ind, n_var = X.shape
    if n_ind == 0:
        return X

    lower_arr, upper_arr = _ensure_bounds(lower, upper, n_var)
    active_mask = rnd_mask if rnd_mask is not None else rng.random(X.shape)
    delta_draws = rnd_delta if rnd_delta is not None else rng.random(X.shape)
    if kernels is not None and X.dtype == np.float64:
        kernels.polynomial_mutation(
            X,
            lower=lower_arr,
            upper=upper_arr,
            prob_mutation=float(prob_mutation),
            eta=float(eta),
            rnd_mask=np.asarray(active_mask, dtype=float),
            rnd_delta=np.asarray(delta_draws, dtype=float),
        )
        return X
    _polynomial_mutation_inplace(
        X,
        lower=lower_arr,
        upper=upper_arr,
        prob_mutation=float(prob_mutation),
        eta=float(eta),
        rnd_mask=np.asarray(active_mask, dtype=float),
        rnd_delta=np.asarray(delta_draws, dtype=float),
    )
    return X


def _polynomial_mutation_inplace(
    X: FloatArray,
    *,
    lower: FloatArray,
    upper: FloatArray,
    prob_mutation: float,
    eta: float,
    rnd_mask: FloatArray,
    rnd_delta: FloatArray,
) -> None:
    active = rnd_mask <= prob_mutation
    if not np.any(active):
        return

    valid_bounds = upper > lower
    if not np.any(valid_bounds):
        return
    active &= valid_bounds.reshape(1, -1)

    rows, cols = np.nonzero(active)
    if rows.size == 0:
        return

    y = X[rows, cols]
    yl = lower[cols]
    yu = upper[cols]
    span = yu - yl
    rnd = rnd_delta[rows, cols]
    delta1 = (y - yl) / span
    delta2 = (yu - y) / span
    deltaq = np.empty_like(y)
    mut_pow = 1.0 / (eta + 1.0)

    lower_branch = rnd <= 0.5
    if np.any(lower_branch):
        rnd_low = rnd[lower_branch]
        xy_low = 1.0 - delta1[lower_branch]
        val_low = 2.0 * rnd_low + (1.0 - 2.0 * rnd_low) * np.power(xy_low, eta + 1.0)
        deltaq[lower_branch] = np.power(val_low, mut_pow) - 1.0

    upper_branch = ~lower_branch
    if np.any(upper_branch):
        rnd_high = rnd[upper_branch]
        xy_high = 1.0 - delta2[upper_branch]
        val_high = 2.0 * (1.0 - rnd_high) + 2.0 * (rnd_high - 0.5) * np.power(xy_high, eta + 1.0)
        deltaq[upper_branch] = 1.0 - np.power(val_high, mut_pow)

    mutated = y + deltaq * span
    np.clip(mutated, yl, yu, out=mutated)
    X[rows, cols] = mutated


__all__ = [
    "available_variation_kernels",
    "polynomial_mutation_population",
    "sbx_crossover_pairs",
]
