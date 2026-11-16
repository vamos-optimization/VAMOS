from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .flags import should_use_numba_variation

# Integer genes and bounds; the precision of the integer type is free.
IntegerArray: TypeAlias = NDArray[np.integer[Any]]

_RANDOM_RESET_MASKED_JIT: Callable[[NDArray[np.generic], NDArray[np.generic], NDArray[np.generic]], None] | None = None
_RANDOM_RESET_MASKED_DISABLED = False


def _use_numba_variation() -> bool:
    return should_use_numba_variation()


def _get_random_reset_masked() -> Callable[[NDArray[np.generic], NDArray[np.generic], NDArray[np.generic]], None] | None:
    global _RANDOM_RESET_MASKED_JIT, _RANDOM_RESET_MASKED_DISABLED
    if _RANDOM_RESET_MASKED_DISABLED:
        return None
    if _RANDOM_RESET_MASKED_JIT is not None:
        return _RANDOM_RESET_MASKED_JIT
    if not _use_numba_variation():
        return None
    try:
        from numba import njit
    except ImportError:
        _RANDOM_RESET_MASKED_DISABLED = True
        return None

    @njit(cache=True)  # type: ignore[untyped-decorator]
    def _random_reset_masked(X: NDArray[np.generic], mask: NDArray[np.generic], rand_vals: NDArray[np.generic]) -> None:
        rows, cols = mask.shape
        for i in range(rows):
            for j in range(cols):
                if mask[i, j]:
                    X[i, j] = rand_vals[i, j]

    _RANDOM_RESET_MASKED_JIT = _random_reset_masked
    return _RANDOM_RESET_MASKED_JIT


def random_integer_population(
    pop_size: int, n_var: int, lower: IntegerArray, upper: IntegerArray, rng: np.random.Generator
) -> NDArray[np.int32]:
    """
    Generate integer-valued individuals within inclusive [lower, upper].
    """
    if pop_size <= 0 or n_var <= 0:
        raise ValueError("pop_size and n_var must be positive integers.")
    if lower.shape != (n_var,) or upper.shape != (n_var,):
        raise ValueError("lower/upper must be 1D arrays matching n_var.")
    return rng.integers(lower, upper + 1, size=(pop_size, n_var), dtype=np.int32)


def _as_pairs(X_parents: NDArray[np.generic]) -> tuple[NDArray[np.generic], int]:
    if X_parents.ndim == 3 and X_parents.shape[1] == 2:
        return X_parents, X_parents.shape[2]
    Np, D = X_parents.shape
    if Np == 0:
        return np.empty((0, 2, D), dtype=X_parents.dtype), D
    # Handle odd parent count by duplicating the last parent
    if Np % 2 != 0:
        X_parents = np.vstack([X_parents, X_parents[-1:]])
        Np += 1
    return X_parents.reshape(Np // 2, 2, D).copy(), D


def _reshape_offspring(pairs: NDArray[np.generic], parents: NDArray[np.generic]) -> NDArray[np.generic]:
    if parents.ndim == 2 and parents.shape[0] % 2 != 0:
        return pairs.reshape(-1, pairs.shape[2])[: parents.shape[0]]
    return pairs.reshape(parents.shape)


def uniform_integer_crossover(X_parents: NDArray[np.generic], prob: float, rng: np.random.Generator) -> NDArray[np.generic]:
    """
    Per-gene uniform crossover for integer vectors.
    """
    pairs, D = _as_pairs(X_parents)
    prob = float(np.clip(prob, 0.0, 1.0))
    if pairs.size == 0 or prob <= 0.0:
        return _reshape_offspring(pairs, X_parents)
    active = rng.random(pairs.shape[0]) <= prob
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return _reshape_offspring(pairs, X_parents)
    swap_mask = rng.random((idx.size, D)) < 0.5
    for row, mask in zip(idx, swap_mask):
        p1, p2 = pairs[row, 0], pairs[row, 1]
        child1 = np.where(mask, p1, p2)
        child2 = np.where(mask, p2, p1)
        pairs[row, 0], pairs[row, 1] = child1, child2
    return _reshape_offspring(pairs, X_parents)


def arithmetic_integer_crossover(
    X_parents: NDArray[np.generic],
    prob: float,
    rng: np.random.Generator,
    lower: NDArray[np.generic] | None = None,
    upper: NDArray[np.generic] | None = None,
) -> NDArray[np.generic]:
    """
    Integer arithmetic crossover with complementary children.

    Each mated pair draws ``lam`` uniformly from [0, 1) and gets the children
    ``lam * p1 + (1 - lam) * p2`` and ``(1 - lam) * p1 + lam * p2``, rounded to the
    nearest integer and clipped to the bounds, so the children add up to the
    parents.
    """
    pairs, _ = _as_pairs(X_parents)
    prob = float(np.clip(prob, 0.0, 1.0))
    if pairs.size == 0 or prob <= 0.0:
        return _reshape_offspring(pairs, X_parents)
    active = rng.random(pairs.shape[0]) <= prob
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return _reshape_offspring(pairs, X_parents)
    for row in idx:
        p1, p2 = pairs[row, 0].astype(float), pairs[row, 1].astype(float)
        lam = rng.random()
        children = np.rint(np.stack([lam * p1 + (1.0 - lam) * p2, (1.0 - lam) * p1 + lam * p2]))
        if lower is not None and upper is not None:
            children = np.clip(children, lower, upper)
        pairs[row] = children.astype(pairs.dtype, copy=False)
    return _reshape_offspring(pairs, X_parents)


def integer_sbx_crossover(
    X_parents: NDArray[np.generic],
    prob: float,
    eta: float,
    lower: NDArray[np.generic],
    upper: NDArray[np.generic],
    rng: np.random.Generator,
) -> NDArray[np.generic]:
    """
    Integer-valued SBX crossover: jMetalPy's SBX on the integer values, each child clamped to
    the bounds and rounded to the nearest integer.

    jMetal and jMetalPy truncate each child with ``int()`` instead, which pulls the children
    towards zero: with parents 3 and 4 in [0, 10] their mean is 3.25 rather than 3.5.
    """
    pairs, D = _as_pairs(X_parents)
    if pairs.size == 0 or D == 0:
        return _reshape_offspring(pairs, X_parents)
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return _reshape_offspring(pairs, X_parents)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape[0] != D or upper.shape[0] != D:
        raise ValueError("lower/upper must match chromosome length.")

    active = rng.random(pairs.shape[0]) <= prob
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return _reshape_offspring(pairs, X_parents)

    eps = 1.0e-14
    eta = float(eta)
    inv_eta = 1.0 / (eta + 1.0)

    for row in idx:
        p1 = pairs[row, 0].copy()
        p2 = pairs[row, 1].copy()
        for j in range(D):
            if rng.random() > 0.5:
                continue
            x1 = float(p1[j])
            x2 = float(p2[j])
            if abs(x1 - x2) <= eps:
                continue
            if x1 < x2:
                y1, y2 = x1, x2
            else:
                y1, y2 = x2, x1
            lb = float(lower[j])
            ub = float(upper[j])
            try:
                beta1 = 1.0 + (2.0 * (y1 - lb) / (y2 - y1))
                alpha1 = 2.0 - beta1 ** (-(eta + 1.0))
                rand_val = rng.random()
                if rand_val <= (1.0 / alpha1):
                    betaq1 = (rand_val * alpha1) ** inv_eta
                else:
                    betaq1 = (1.0 / (2.0 - rand_val * alpha1)) ** inv_eta
                c1 = 0.5 * (y1 + y2 - betaq1 * (y2 - y1))

                beta2 = 1.0 + (2.0 * (ub - y2) / (y2 - y1))
                alpha2 = 2.0 - beta2 ** (-(eta + 1.0))
                if rand_val <= (1.0 / alpha2):
                    betaq2 = (rand_val * alpha2) ** inv_eta
                else:
                    betaq2 = (1.0 / (2.0 - rand_val * alpha2)) ** inv_eta
                c2 = 0.5 * (y1 + y2 + betaq2 * (y2 - y1))
            except (ValueError, ZeroDivisionError):
                c1, c2 = y1, y2

            if c1 < lb:
                c1 = lb
            if c1 > ub:
                c1 = ub
            if c2 < lb:
                c2 = lb
            if c2 > ub:
                c2 = ub

            if rng.random() <= 0.5:
                p1[j] = np.rint(c2)
                p2[j] = np.rint(c1)
            else:
                p1[j] = np.rint(c1)
                p2[j] = np.rint(c2)

        pairs[row, 0] = p1
        pairs[row, 1] = p2

    return _reshape_offspring(pairs, X_parents)


def random_reset_mutation(X: IntegerArray, prob: float, lower: IntegerArray, upper: IntegerArray, rng: np.random.Generator) -> None:
    """
    Per-gene random reset to any value within bounds.
    """
    if X.size == 0:
        return
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return
    if lower.shape != upper.shape or lower.shape[0] != X.shape[1]:
        raise ValueError("lower/upper must match chromosome length.")
    mask = rng.random(X.shape) <= prob
    if not np.any(mask):
        return
    rand_vals = rng.integers(lower, upper + 1, size=X.shape, dtype=X.dtype)
    jit_fn = _get_random_reset_masked()
    if jit_fn is not None:
        jit_fn(X, mask, rand_vals)
    else:
        X[mask] = rand_vals[mask]


def creep_mutation(
    X: NDArray[np.generic], prob: float, step: int, lower: NDArray[np.generic], upper: NDArray[np.generic], rng: np.random.Generator
) -> None:
    """
    Small integer step mutation (+/- step).
    """
    if X.size == 0:
        return
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return
    if lower.shape != upper.shape or lower.shape[0] != X.shape[1]:
        raise ValueError("lower/upper must match chromosome length.")
    mask = rng.random(X.shape) <= prob
    if not np.any(mask):
        return
    deltas = rng.choice([-step, step], size=X.shape, replace=True)
    proposed = X.copy()
    proposed[mask] = proposed[mask] + deltas[mask]
    np.clip(proposed, lower, upper, out=proposed)
    X[:] = proposed


def gaussian_integer_mutation(
    X: NDArray[np.generic],
    prob: float,
    sigma: float,
    lower: NDArray[np.generic],
    upper: NDArray[np.generic],
    rng: np.random.Generator,
) -> None:
    """
    Per-gene Gaussian mutation for integer variables (round + clip).
    """
    if X.size == 0:
        return
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return
    if lower.shape != upper.shape or lower.shape[0] != X.shape[1]:
        raise ValueError("lower/upper must match chromosome length.")
    sigma = float(sigma)
    if sigma <= 0.0:
        return
    mask = rng.random(X.shape) <= prob
    if not np.any(mask):
        return
    noise = rng.normal(loc=0.0, scale=sigma, size=X.shape)
    proposed = np.rint(X.astype(float) + noise).astype(X.dtype, copy=False)
    np.clip(proposed, lower, upper, out=proposed)
    X[mask] = proposed[mask]


def boundary_integer_mutation(
    X: NDArray[np.generic],
    prob: float,
    lower: NDArray[np.generic],
    upper: NDArray[np.generic],
    rng: np.random.Generator,
) -> None:
    """
    Per-gene boundary mutation that snaps selected genes to lower/upper bounds.
    """
    if X.size == 0:
        return
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return
    if lower.shape != upper.shape or lower.shape[0] != X.shape[1]:
        raise ValueError("lower/upper must match chromosome length.")
    mask = rng.random(X.shape) <= prob
    if not np.any(mask):
        return
    boundary_pick_upper = rng.random(X.shape) < 0.5
    boundary_values = np.broadcast_to(lower, X.shape).copy()
    upper_values = np.broadcast_to(upper, X.shape)
    boundary_values[boundary_pick_upper] = upper_values[boundary_pick_upper]
    X[mask] = boundary_values[mask]


def integer_polynomial_mutation(
    X: NDArray[np.generic],
    prob: float,
    eta: float,
    lower: NDArray[np.generic],
    upper: NDArray[np.generic],
    rng: np.random.Generator,
) -> None:
    """
    Integer polynomial mutation aligned with jMetalPy (round + clamp).
    """
    if X.size == 0:
        return
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape[0] != X.shape[1] or upper.shape[0] != X.shape[1]:
        raise ValueError("lower/upper must match chromosome length.")

    rnd_mask = rng.random(X.shape)
    rnd_delta = rng.random(X.shape)
    mask = rnd_mask <= prob
    if not np.any(mask):
        return

    mut_pow = 1.0 / (float(eta) + 1.0)
    rows, cols = np.nonzero(mask)
    for i, j in zip(rows, cols):
        y = float(X[i, j])
        yl = float(lower[j])
        yu = float(upper[j])
        if yu <= yl:
            X[i, j] = int(yl)
            continue
        delta1 = (y - yl) / (yu - yl)
        delta2 = (yu - y) / (yu - yl)
        rnd = rnd_delta[i, j]
        if rnd <= 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (float(eta) + 1.0))
            deltaq = val**mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (float(eta) + 1.0))
            deltaq = 1.0 - val**mut_pow

        y += deltaq * (yu - yl)
        y = min(max(y, yl), yu)
        X[i, j] = int(round(y))


# === Adapters ===


class UniformIntegerCrossover:
    def __init__(self, prob: float = 0.9, **kwargs: Any) -> None:
        self.prob = float(prob)

    def __call__(self, parents: NDArray[np.generic], rng: np.random.Generator, **kwargs: Any) -> NDArray[np.generic]:
        return uniform_integer_crossover(parents, self.prob, rng)


class ArithmeticIntegerCrossover:
    def __init__(self, prob: float = 0.9, **kwargs: Any) -> None:
        self.prob = float(prob)

    def __call__(self, parents: NDArray[np.generic], rng: np.random.Generator, **kwargs: Any) -> NDArray[np.generic]:
        return arithmetic_integer_crossover(parents, self.prob, rng)


class IntegerSBXCrossover:
    def __init__(self, prob: float = 0.9, eta: float = 20.0, **kwargs: Any) -> None:
        self.prob = float(prob)
        self.eta = float(eta)

    def __call__(self, parents: NDArray[np.generic], rng: np.random.Generator, **kwargs: Any) -> NDArray[np.generic]:
        lower = kwargs.get("lower")
        upper = kwargs.get("upper")
        if lower is None or upper is None:
            raise ValueError("IntegerSBXCrossover requires 'lower' and 'upper' bounds in kwargs.")
        return integer_sbx_crossover(parents, self.prob, self.eta, lower, upper, rng)


class RandomResetMutation:
    def __init__(self, prob: float = 0.1, **kwargs: Any) -> None:
        self.prob = float(prob)

    def __call__(self, X: IntegerArray, rng: np.random.Generator, **kwargs: Any) -> None:
        # kwargs must typically contain 'lower' and 'upper'
        lower = kwargs.get("lower")
        upper = kwargs.get("upper")
        if lower is None or upper is None:
            raise ValueError("RandomResetMutation requires 'lower' and 'upper' bounds in kwargs.")
        random_reset_mutation(X, self.prob, lower, upper, rng)


class IntegerPolynomialMutation:
    def __init__(self, prob: float = 0.1, eta: float = 20.0, **kwargs: Any) -> None:
        self.prob = float(prob)
        self.eta = float(eta)

    def __call__(self, X: NDArray[np.generic], rng: np.random.Generator, **kwargs: Any) -> None:
        lower = kwargs.get("lower")
        upper = kwargs.get("upper")
        if lower is None or upper is None:
            raise ValueError("IntegerPolynomialMutation requires 'lower' and 'upper' bounds in kwargs.")
        integer_polynomial_mutation(X, self.prob, self.eta, lower, upper, rng)


class CreepMutation:
    def __init__(self, prob: float = 0.1, step: int = 1, **kwargs: Any) -> None:
        self.prob = float(prob)
        self.step = int(step)

    def __call__(self, X: NDArray[np.generic], rng: np.random.Generator, **kwargs: Any) -> None:
        lower = kwargs.get("lower")
        upper = kwargs.get("upper")
        if lower is None or upper is None:
            raise ValueError("CreepMutation requires 'lower' and 'upper' bounds in kwargs.")
        creep_mutation(X, self.prob, self.step, lower, upper, rng)


class GaussianIntegerMutation:
    def __init__(self, prob: float = 0.1, sigma: float = 1.0, **kwargs: Any) -> None:
        self.prob = float(prob)
        self.sigma = float(sigma)

    def __call__(self, X: NDArray[np.generic], rng: np.random.Generator, **kwargs: Any) -> None:
        lower = kwargs.get("lower")
        upper = kwargs.get("upper")
        if lower is None or upper is None:
            raise ValueError("GaussianIntegerMutation requires 'lower' and 'upper' bounds in kwargs.")
        gaussian_integer_mutation(X, self.prob, self.sigma, lower, upper, rng)


class BoundaryIntegerMutation:
    def __init__(self, prob: float = 0.1, **kwargs: Any) -> None:
        self.prob = float(prob)

    def __call__(self, X: NDArray[np.generic], rng: np.random.Generator, **kwargs: Any) -> None:
        lower = kwargs.get("lower")
        upper = kwargs.get("upper")
        if lower is None or upper is None:
            raise ValueError("BoundaryIntegerMutation requires 'lower' and 'upper' bounds in kwargs.")
        boundary_integer_mutation(X, self.prob, lower, upper, rng)


__all__ = [
    "random_integer_population",
    "uniform_integer_crossover",
    "arithmetic_integer_crossover",
    "integer_sbx_crossover",
    "random_reset_mutation",
    "creep_mutation",
    "gaussian_integer_mutation",
    "boundary_integer_mutation",
    "integer_polynomial_mutation",
    "UniformIntegerCrossover",
    "ArithmeticIntegerCrossover",
    "IntegerSBXCrossover",
    "RandomResetMutation",
    "IntegerPolynomialMutation",
    "CreepMutation",
    "GaussianIntegerMutation",
    "BoundaryIntegerMutation",
]
