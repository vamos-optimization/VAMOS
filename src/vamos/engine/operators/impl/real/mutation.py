"""Real-valued mutation operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.kernel.operator_primitives import polynomial_mutation_population

from .utils import (
    ArrayLike,
    RealOperator,
    VariationWorkspace,
    _check_nvars,
    _ensure_bounds,
)

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import VariationKernels


class Mutation(RealOperator, ABC):
    """Base class for real-coded mutation operators."""

    @abstractmethod
    def __call__(self, offspring: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        raise NotImplementedError


class PolynomialMutation(Mutation):
    """Standard polynomial mutation used in NSGA-II."""

    supports_variation_kernels: ClassVar[bool] = True

    def __init__(
        self,
        prob_mutation: float,
        eta: float = 20.0,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
        variation_kernels: VariationKernels | None = None,
    ) -> None:
        self.prob = float(prob_mutation)
        self.eta = float(eta)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.span = self.upper - self.lower
        self._span_safe = np.where(self.span == 0.0, 1.0, self.span)
        self.workspace = workspace
        self.variation_kernels = variation_kernels

    def _mask(self, shape: tuple[int, ...], rng: np.random.Generator) -> NDArray[np.bool_]:
        if self.workspace is None:
            return rng.random(shape) <= self.prob
        probs = self.workspace.request("pm_prob", shape, np.float64)
        rng.random(out=probs)
        mask = self.workspace.request("pm_mask", shape, np.bool_)
        np.less_equal(probs, self.prob, out=mask)
        return mask

    def _rand(self, key: str, shape: tuple[int, ...], rng: np.random.Generator) -> ArrayLike:
        if self.workspace is None:
            return rng.random(shape)
        buf = self.workspace.request(key, shape, np.float64)
        rng.random(out=buf)
        return buf

    def _buffer(self, key: str, shape: tuple[int, ...], dtype: Any) -> NDArray[Any]:
        if self.workspace is None:
            return np.empty(shape, dtype=dtype)
        return self.workspace.request(key, shape, dtype)

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        X = self._as_population(offspring, name="offspring")
        n_ind, n_var = X.shape
        if n_ind == 0:
            return X
        self._check_bounds_match(X, self.lower)
        if self.workspace is None:
            return polynomial_mutation_population(
                X,
                rng=rng,
                lower=self.lower,
                upper=self.upper,
                prob_mutation=self.prob,
                eta=self.eta,
                inplace=True,
                kernels=self.variation_kernels,
            )

        # Draw full random grids to keep RNG consumption aligned with regression fixtures:
        # first grid for activation mask, second grid for delta sampling.
        rnd_mask = rng.random((n_ind, n_var))
        rnd_delta = rng.random((n_ind, n_var))
        return polynomial_mutation_population(
            X,
            rng=rng,
            lower=self.lower,
            upper=self.upper,
            prob_mutation=self.prob,
            eta=self.eta,
            rnd_mask=rnd_mask,
            rnd_delta=rnd_delta,
            inplace=True,
            kernels=self.variation_kernels,
        )


class GaussianMutation(Mutation):
    """Gaussian mutation with optional bounds metadata."""

    def __init__(
        self,
        prob_mutation: float,
        sigma: float | ArrayLike,
        *,
        lower: ArrayLike | None = None,
        upper: ArrayLike | None = None,
        workspace: VariationWorkspace | None = None,
    ) -> None:
        self.prob = float(prob_mutation)
        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.ndim == 0:
            self.sigma = sigma_arr
        elif sigma_arr.ndim == 1:
            self.sigma = sigma_arr
        else:
            raise ValueError("sigma must be scalar or 1-D array.")
        if lower is None or upper is None:
            self.lower = self.upper = None
        else:
            self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        X = self._as_population(offspring, name="offspring")
        n_ind, n_var = X.shape
        if n_ind == 0:
            return X
        if self.lower is not None and self.upper is not None:
            _check_nvars(n_var, self.lower)
        mask = rng.random((n_ind, n_var)) <= self.prob
        if not np.any(mask):
            return X
        if np.ndim(self.sigma) == 0:
            noise = rng.normal(0.0, float(self.sigma), size=(n_ind, n_var))
        else:
            sigma = np.broadcast_to(self.sigma, (n_ind, n_var))
            noise = rng.normal(0.0, sigma)
        X[mask] += noise[mask]
        return X


class UniformResetMutation(Mutation):
    """Uniform reset mutation that resamples genes inside their bounds."""

    def __init__(
        self,
        prob_mutation: float,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
    ) -> None:
        self.prob = float(prob_mutation)
        self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        X = self._as_population(offspring, name="offspring")
        self._check_bounds_match(X, self.lower)
        shape = (X.shape[0], X.shape[1])
        mask = rng.random(shape) <= self.prob
        if not np.any(mask):
            return X
        lower = np.broadcast_to(self.lower, shape)
        upper = np.broadcast_to(self.upper, shape)
        resampled = lower + rng.random(shape) * (upper - lower)
        X[mask] = resampled[mask]
        return X


class NonUniformMutation(Mutation):
    """
    Non-uniform mutation that samples perturbations biased toward the bounds
    using the simplified formulation employed by our NSGA-II variant.
    """

    def __init__(
        self,
        prob_mutation: float,
        perturbation: float = 0.5,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
    ) -> None:
        self.prob = float(np.clip(prob_mutation, 0.0, 1.0))
        self.perturbation = max(float(perturbation), 1e-8)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.span = self.upper - self.lower
        self.workspace = workspace

    def _rand(self, key: str, shape: tuple[int, ...], rng: np.random.Generator) -> ArrayLike:
        if self.workspace is None:
            return rng.random(shape)
        buf = self.workspace.request(key, shape, np.float64)
        rng.random(out=buf)
        return buf

    def _mask(self, shape: tuple[int, ...], rng: np.random.Generator) -> NDArray[np.bool_]:
        if self.workspace is None:
            return rng.random(shape) <= self.prob
        probs = self.workspace.request("nu_prob", shape, np.float64)
        rng.random(out=probs)
        mask = self.workspace.request("nu_mask", shape, np.bool_)
        np.less_equal(probs, self.prob, out=mask)
        return mask

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        X = self._as_population(offspring, name="offspring")
        self._check_bounds_match(X, self.lower)
        mask = self._mask(X.shape, rng)
        if not np.any(mask):
            return X

        rand = self._rand("nu_rand", X.shape, rng)
        np.power(rand, self.perturbation, out=rand)
        delta = self.workspace.request("nu_delta", X.shape, np.float64) if self.workspace else np.empty(X.shape, dtype=float)
        np.subtract(1.0, rand, out=delta)
        delta *= self.span
        direction_rand = self._rand("nu_direction", X.shape, rng)
        direction = np.where(direction_rand <= 0.5, -1.0, 1.0)
        update = delta * direction
        if self.workspace is None:
            X += np.where(mask, update, 0.0)
        else:
            buffered_update = self.workspace.request("nu_update", X.shape, np.float64)
            buffered_update.fill(0.0)
            buffered_update[mask] = update[mask]
            X += buffered_update
        return X


class UniformMutation(Mutation):
    """Uniform mutation with bounded perturbation scaled by variable range."""

    def __init__(
        self,
        prob_mutation: float,
        perturb: float = 0.1,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator | None = None,
        workspace: VariationWorkspace | None = None,
    ) -> None:
        self.prob = float(prob_mutation)
        self.perturb = float(np.clip(perturb, 0.0, 1.0))
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.range = self.upper - self.lower
        self.rng = rng

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator | None = None) -> ArrayLike:
        arr = np.asarray(offspring, dtype=float)
        squeeze = False
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
            squeeze = True
        X = self._as_population(arr, name="offspring")
        n_ind, n_var = X.shape
        active_rng = rng or self.rng or np.random.default_rng()
        mask = active_rng.random((n_ind, n_var)) <= self.prob
        if not np.any(mask):
            return X[0] if squeeze else X
        delta = active_rng.uniform(-self.perturb, self.perturb, size=(n_ind, n_var))
        update = delta * self.range
        X = X.copy()
        X += np.where(mask, update, 0.0)
        return X[0] if squeeze else X


class LinkedPolynomialMutation(Mutation):
    """Linked polynomial mutation (Zille et al., 2016), as jMetal's ``LinkedPolynomialMutation``.

    The bounded polynomial mutation of :class:`PolynomialMutation`, except that each
    individual draws one random number ``u`` and uses it for all of its mutated
    variables. The step of each variable still depends on its distance to the bounds,
    so the linked variables move in the same direction and stay inside the bounds.

    Reference: Zille, H., Ishibuchi, H., Mostaghim, S. & Nojima, Y. (2016). Mutation
    operators based on variable grouping for multi-objective large-scale optimization.
    IEEE Symposium Series on Computational Intelligence (SSCI).
    """

    def __init__(
        self,
        prob_mutation: float,
        eta: float = 20.0,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator | None = None,
        workspace: VariationWorkspace | None = None,
    ) -> None:
        self.prob = float(prob_mutation)
        self.eta = float(eta)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.span = self.upper - self.lower
        self.rng = rng

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator | None = None) -> ArrayLike:
        arr = np.asarray(offspring, dtype=float)
        squeeze = False
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
            squeeze = True
        X = self._as_population(arr, name="offspring")
        self._check_bounds_match(X, self.lower)
        active_rng = rng or self.rng or np.random.default_rng()
        n_ind, n_var = X.shape
        rnd_mask = active_rng.random((n_ind, n_var))
        if not np.any(rnd_mask <= self.prob):
            return X[0] if squeeze else X
        # One draw per individual, shared by all of its mutated variables.
        u = active_rng.random((n_ind, 1))
        polynomial_mutation_population(
            X,
            rng=active_rng,
            lower=self.lower,
            upper=self.upper,
            prob_mutation=self.prob,
            eta=self.eta,
            rnd_mask=rnd_mask,
            rnd_delta=np.broadcast_to(u, (n_ind, n_var)),
            inplace=True,
        )
        return X[0] if squeeze else X


class CauchyMutation(Mutation):
    """Cauchy mutation with heavy-tailed perturbations."""

    def __init__(
        self,
        prob_mutation: float,
        gamma: float = 0.1,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
    ) -> None:
        self.prob = float(prob_mutation)
        self.gamma = float(gamma)
        self.lower, self.upper = _ensure_bounds(lower, upper)

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        X = self._as_population(offspring, name="offspring")
        self._check_bounds_match(X, self.lower)
        shape = (X.shape[0], X.shape[1])
        mask = rng.random(shape) <= self.prob
        if not np.any(mask):
            return X
        noise = rng.standard_cauchy(size=shape) * self.gamma
        X[mask] += noise[mask]
        return X


class LevyFlightMutation(Mutation):
    """Lévy flight mutation using Mantegna's algorithm for stable random walks.

    The step size follows a Lévy stable distribution approximated by:

        step = sigma_u · u / |v|^{1/beta}

    where *u ~ N(0, sigma_u²)*, *v ~ N(0, 1)*, and *sigma_u* is derived
    from the Gamma function to ensure the correct heavy-tailed behaviour.

    Parameters
    ----------
    prob_mutation : float
        Per-gene mutation probability.
    beta : float
        Stability index of the Lévy distribution (0 < beta <= 2).
        Default 1.5.
    scale : float
        Scaling factor for the step relative to the variable range.
        Default 0.01.
    lower, upper : array-like
        Decision variable bounds.

    References
    ----------
    Yang, X.-S. (2010). Nature-Inspired Metaheuristic Algorithms (2nd ed.).
    Mantegna, R.N. (1994). Fast, accurate algorithm for numerical
    simulation of Lévy stable stochastic processes. Physical Review E 49(5).
    """

    def __init__(
        self,
        prob_mutation: float,
        beta: float = 1.5,
        scale: float = 0.01,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
    ) -> None:
        self.prob = float(prob_mutation)
        self.beta = float(beta)
        self.scale = float(scale)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.span = self.upper - self.lower
        # Precompute sigma_u for Mantegna's algorithm
        from math import gamma, pi, sin

        num = gamma(1.0 + self.beta) * sin(pi * self.beta / 2.0)
        den = gamma((1.0 + self.beta) / 2.0) * self.beta * 2.0 ** ((self.beta - 1.0) / 2.0)
        self._sigma_u = (num / den) ** (1.0 / self.beta)

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        X = self._as_population(offspring, name="offspring")
        self._check_bounds_match(X, self.lower)
        n_ind, n_var = X.shape
        if n_ind == 0:
            return X

        mask = rng.random((n_ind, n_var)) <= self.prob
        if not np.any(mask):
            return X

        u = rng.normal(0.0, self._sigma_u, size=(n_ind, n_var))
        v = rng.normal(0.0, 1.0, size=(n_ind, n_var))
        eps = 1e-30
        step = u / (np.abs(v) ** (1.0 / self.beta) + eps)
        update = self.scale * step * self.span
        X[mask] += update[mask]
        return X


class PowerLawMutation(Mutation):
    """Power mutation of Deep and Thakur (2007), as jMetal's ``PowerLawMutation``.

    Each mutated gene draws a strength ``s = u**index`` with ``u ~ U(0, 1)`` and moves
    by that fraction of its distance to one bound: ``x - s*(x - lower)`` if ``t < r``,
    else ``x + s*(upper - x)``, with ``r ~ U(0, 1)`` and ``t = (x - lower)/(upper - lower)``.
    Children never leave the bounds and the expected step is zero. A larger ``index``
    gives smaller steps; ``index=1`` gives a uniform strength. The paper writes
    ``t = (x - lower)/(upper - x)``, which drifts towards the upper bound; jMetal and
    VAMOS use the relative position instead.

    Parameters
    ----------
    prob_mutation : float
        Per-gene mutation probability.
    index : float
        Index of mutation ``p`` (> 0) in ``s = u**p``. Default 1.0.
    lower, upper : array-like
        Decision variable bounds.

    References
    ----------
    Deep, K. & Thakur, M. (2007). A new mutation operator for real coded
    genetic algorithms. Applied Mathematics and Computation 193, 211–230.
    """

    def __init__(
        self,
        prob_mutation: float,
        index: float = 1.0,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
    ) -> None:
        self.prob = float(prob_mutation)
        self.index = float(index)
        if not self.index > 0.0:
            raise ValueError("PowerLawMutation index must be positive.")
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.span = self.upper - self.lower
        self._span_safe = np.where(self.span > 0.0, self.span, 1.0)

    def __call__(self, offspring: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        X = self._as_population(offspring, name="offspring")
        self._check_bounds_match(X, self.lower)
        n_ind, n_var = X.shape
        if n_ind == 0:
            return X

        mask = rng.random((n_ind, n_var)) <= self.prob
        if not np.any(mask):
            return X

        strength = np.power(rng.random((n_ind, n_var)), self.index)
        position = (X - self.lower) / self._span_safe
        downward = position < rng.random((n_ind, n_var))
        mutated = np.where(downward, X - strength * (X - self.lower), X + strength * (self.upper - X))
        np.clip(mutated, self.lower, self.upper, out=mutated)
        X[mask] = mutated[mask]
        return X


__all__ = [
    "CauchyMutation",
    "GaussianMutation",
    "LevyFlightMutation",
    "LinkedPolynomialMutation",
    "Mutation",
    "NonUniformMutation",
    "PolynomialMutation",
    "PowerLawMutation",
    "UniformMutation",
    "UniformResetMutation",
]
