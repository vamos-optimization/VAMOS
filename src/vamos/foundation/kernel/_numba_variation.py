"""Compiled SBX and polynomial-mutation arithmetic.

The vectorized NumPy operators are the reference semantics. These kernels never
draw random numbers: the Python wrappers draw every value from the run-owned
``numpy.random.Generator`` in the reference order and shape, then pass the draws
to nopython code that reproduces the reference arithmetic element by element.
"""

from __future__ import annotations

from functools import cache
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .numba_ops import njit

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
IndexArray: TypeAlias = NDArray[np.intp]

_SBX_EPS = 1.0e-14


@cache
def numpy_power_squares() -> bool:
    """Return whether ``np.power(x, 2.0)`` is evaluated as ``x * x``.

    NumPy 2 special-cases a scalar exponent of two; older releases call the
    platform ``pow``. Compiled kernels receive this flag so that ``eta == 1``
    reproduces the installed NumPy reference bit for bit.
    """
    probe = np.linspace(0.5, 3.0, 1 << 16)
    return bool(np.array_equal(np.power(probe, 2.0), probe * probe))


@njit(cache=True, error_model="numpy")
def _numpy_power(base: float, exponent: float, squares_exponent_two: bool) -> float:
    if squares_exponent_two and exponent == 2.0:
        return base * base
    return float(base**exponent)


@njit(cache=True, error_model="numpy")
def _numpy_power_vector(bases: FloatArray, exponent: float, squares_exponent_two: bool, out: FloatArray) -> None:
    for i in range(bases.shape[0]):
        out[i] = _numpy_power(bases[i], exponent, squares_exponent_two)


@cache
def compiled_power_matches_numpy() -> bool:
    """Return whether compiled ``pow`` reproduces ``np.power`` on this platform.

    NumPy may evaluate ``power`` with vendor SIMD libraries (for example SVML on
    AVX-512 Linux builds) whose results can differ from the C library by one
    ULP. Compiled variation is enabled only when a probe over SBX and
    polynomial-mutation style bases and exponents matches bit for bit.
    """
    bases = np.concatenate(
        (
            np.linspace(0.0, 2.0, 1025),
            np.geomspace(1.0e-12, 1.0e6, 1024),
            np.linspace(0.999, 1.001, 129),
        )
    )
    eta_values = np.concatenate((np.linspace(0.0, 100.0, 26), np.array([1.0, 2.0, 5.5, 15.0, 20.0, 30.0])))
    exponents = np.concatenate((eta_values + 1.0, -(eta_values + 1.0), 1.0 / (eta_values + 1.0), np.array([2.0, 0.5, -1.0])))
    out = np.empty_like(bases)
    with np.errstate(all="ignore"):
        for exponent in exponents:
            _numpy_power_vector(bases, float(exponent), numpy_power_squares(), out)
            if not np.array_equal(out, np.power(bases, float(exponent)), equal_nan=True):
                return False
    return True


@njit(cache=True, error_model="numpy")
def _sbx_mark_active(
    offspring: FloatArray,
    pair_index: IndexArray,
    var_draws: FloatArray,
    prob_var: float,
    use_var_draws: bool,
    active: BoolArray,
) -> bool:
    """Mark variables that SBX changes and report whether any variable changes."""
    n_var = offspring.shape[2]
    any_active = False
    for row in range(pair_index.shape[0]):
        pair = pair_index[row]
        for j in range(n_var):
            a = offspring[pair, 0, j]
            b = offspring[pair, 1, j]
            if a < b:
                diff = b - a
            else:
                diff = a - b
            is_active = diff > _SBX_EPS
            if is_active and use_var_draws:
                is_active = var_draws[row, j] <= prob_var
            active[row, j] = is_active
            if is_active:
                any_active = True
    return any_active


@njit(cache=True, error_model="numpy")
def _sbx_apply(
    offspring: FloatArray,
    pair_index: IndexArray,
    active: BoolArray,
    rand: FloatArray,
    swap: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    neg_eta_plus_one: float,
    inv_eta_plus_one: float,
    squares_exponent_two: bool,
) -> None:
    """Write SBX children in place for the selected pairs."""
    n_var = offspring.shape[2]
    for row in range(pair_index.shape[0]):
        pair = pair_index[row]
        for j in range(n_var):
            if not active[row, j]:
                continue
            a = offspring[pair, 0, j]
            b = offspring[pair, 1, j]
            if a < b:
                y1 = a
                y2 = b
            else:
                y1 = b
                y2 = a
            diff = y2 - y1
            r = rand[row, j]

            beta = 1.0 + (2.0 * (y1 - lower[j]) / diff)
            if beta < _SBX_EPS:
                beta = _SBX_EPS
            alpha = 2.0 - _numpy_power(beta, neg_eta_plus_one, squares_exponent_two)
            if alpha < _SBX_EPS:
                alpha = _SBX_EPS
            if r <= (1.0 / alpha):
                betaq = _numpy_power(r * alpha, inv_eta_plus_one, squares_exponent_two)
            else:
                betaq = _numpy_power(1.0 / (2.0 - r * alpha), inv_eta_plus_one, squares_exponent_two)
            c1 = 0.5 * ((y1 + y2) - betaq * diff)

            beta = 1.0 + (2.0 * (upper[j] - y2) / diff)
            if beta < _SBX_EPS:
                beta = _SBX_EPS
            alpha = 2.0 - _numpy_power(beta, neg_eta_plus_one, squares_exponent_two)
            if alpha < _SBX_EPS:
                alpha = _SBX_EPS
            if r <= (1.0 / alpha):
                betaq = _numpy_power(r * alpha, inv_eta_plus_one, squares_exponent_two)
            else:
                betaq = _numpy_power(1.0 / (2.0 - r * alpha), inv_eta_plus_one, squares_exponent_two)
            c2 = 0.5 * ((y1 + y2) + betaq * diff)

            if swap[row, j] <= 0.5:
                offspring[pair, 0, j] = c2
                offspring[pair, 1, j] = c1
            else:
                offspring[pair, 0, j] = c1
                offspring[pair, 1, j] = c2


@njit(cache=True, error_model="numpy")
def polynomial_mutation_apply(
    X: FloatArray,
    prob: float,
    eta: float,
    lower: FloatArray,
    upper: FloatArray,
    rnd_mask: FloatArray,
    rnd_delta: FloatArray,
    squares_exponent_two: bool,
) -> None:
    """Apply polynomial mutation in place from caller-provided draws."""
    n_ind, n_var = X.shape
    eta_plus_one = eta + 1.0
    mut_pow = 1.0 / eta_plus_one
    for i in range(n_ind):
        for j in range(n_var):
            if not (rnd_mask[i, j] <= prob):
                continue
            yl = lower[j]
            yu = upper[j]
            if not (yu > yl):
                continue
            y = X[i, j]
            span = yu - yl
            delta1 = (y - yl) / span
            delta2 = (yu - y) / span
            rnd = rnd_delta[i, j]
            if rnd <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * _numpy_power(xy, eta_plus_one, squares_exponent_two)
                deltaq = _numpy_power(val, mut_pow, squares_exponent_two) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * _numpy_power(xy, eta_plus_one, squares_exponent_two)
                deltaq = 1.0 - _numpy_power(val, mut_pow, squares_exponent_two)
            mutated = y + deltaq * span
            if mutated < yl:
                mutated = yl
            elif mutated > yu:
                mutated = yu
            X[i, j] = mutated


@njit(cache=True, error_model="numpy")
def differential_crossover_apply(
    parents: FloatArray,
    f: float,
    cr: float,
    rand: FloatArray,
    j_rand: NDArray[np.int64],
    out: FloatArray,
) -> None:
    """Binomial DE children from caller-provided draws (``parents[:, 2]`` is the base)."""
    for i in range(parents.shape[0]):
        forced = j_rand[i]
        for j in range(parents.shape[2]):
            base = parents[i, 2, j]
            if rand[i, j] < cr or j == forced:
                out[i, j] = base + f * (parents[i, 0, j] - parents[i, 1, j])
            else:
                out[i, j] = base


def sbx_crossover_pairs_compiled(
    offspring: FloatArray,
    *,
    rng: np.random.Generator,
    lower: FloatArray,
    upper: FloatArray,
    prob_crossover: float,
    eta: float,
    prob_var: float,
) -> FloatArray:
    """Apply SBX in place to a float64 mating tensor shaped ``(n_pairs, 2, n_var)``.

    Random draws follow the reference order: one pair-selection value per pair,
    then (when ``prob_var < 1``) one variable-selection grid for the selected
    pairs, and only when a variable changes, the SBX spread and swap grids.
    """
    n_pairs = offspring.shape[0]
    if n_pairs == 0:
        return offspring
    n_var = offspring.shape[2]
    pair_index = np.flatnonzero(rng.random(n_pairs) <= prob_crossover)
    if pair_index.size == 0:
        return offspring
    shape = (pair_index.size, n_var)
    use_var_draws = prob_var < 1.0
    var_draws = rng.random(shape) if use_var_draws else np.empty((0, 0), dtype=np.float64)
    active = np.empty(shape, dtype=np.bool_)
    if not _sbx_mark_active(offspring, pair_index, var_draws, float(prob_var), use_var_draws, active):
        return offspring
    rand = rng.random(shape)
    swap = rng.random(shape)
    _sbx_apply(
        offspring,
        pair_index,
        active,
        rand,
        swap,
        lower,
        upper,
        -(eta + 1.0),
        1.0 / (eta + 1.0),
        numpy_power_squares(),
    )
    return offspring


def polynomial_mutation_from_draws(
    X: FloatArray,
    *,
    lower: FloatArray,
    upper: FloatArray,
    prob_mutation: float,
    eta: float,
    rnd_mask: FloatArray,
    rnd_delta: FloatArray,
) -> None:
    """Apply polynomial mutation in place to a float64 population from given draws."""
    polynomial_mutation_apply(
        X,
        float(prob_mutation),
        float(eta),
        lower,
        upper,
        rnd_mask,
        rnd_delta,
        numpy_power_squares(),
    )


class NumbaVariationKernels:
    """Compiled variation arithmetic exposed through ``KernelBackend.variation_kernels``."""

    name = "numba"

    def sbx_crossover_pairs(
        self,
        offspring: FloatArray,
        *,
        rng: np.random.Generator,
        lower: FloatArray,
        upper: FloatArray,
        prob_crossover: float,
        eta: float,
        prob_var: float,
    ) -> FloatArray:
        return sbx_crossover_pairs_compiled(
            offspring,
            rng=rng,
            lower=lower,
            upper=upper,
            prob_crossover=prob_crossover,
            eta=eta,
            prob_var=prob_var,
        )

    def polynomial_mutation(
        self,
        X: FloatArray,
        *,
        lower: FloatArray,
        upper: FloatArray,
        prob_mutation: float,
        eta: float,
        rnd_mask: FloatArray,
        rnd_delta: FloatArray,
    ) -> None:
        polynomial_mutation_from_draws(
            X,
            lower=lower,
            upper=upper,
            prob_mutation=prob_mutation,
            eta=eta,
            rnd_mask=rnd_mask,
            rnd_delta=rnd_delta,
        )

    def differential_crossover(
        self,
        parents: FloatArray,
        f: float,
        cr: float,
        rand: FloatArray,
        j_rand: NDArray[np.integer[Any]],
    ) -> FloatArray:
        out = np.empty((parents.shape[0], parents.shape[2]), dtype=np.float64)
        differential_crossover_apply(parents, float(f), float(cr), rand, np.asarray(j_rand, dtype=np.int64), out)
        return out


@cache
def numba_variation_kernels() -> NumbaVariationKernels:
    """Shared compiled variation kernels (created on first use, not at import)."""
    return NumbaVariationKernels()


def verified_variation_kernels() -> NumbaVariationKernels | None:
    """Return the compiled kernels only when they reproduce NumPy on this platform."""
    return numba_variation_kernels() if compiled_power_matches_numpy() else None


__all__ = [
    "numpy_power_squares",
    "numba_variation_kernels",
    "NumbaVariationKernels",
    "compiled_power_matches_numpy",
    "verified_variation_kernels",
    "polynomial_mutation_apply",
    "polynomial_mutation_from_draws",
    "sbx_crossover_pairs_compiled",
]
