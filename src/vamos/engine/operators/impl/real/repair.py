"""Bounds repair helpers for real-encoded individuals."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .utils import ArrayLike, RealOperator, _clip_population, _ensure_bounds


class Repair(RealOperator, ABC):
    """Base class for repair operators (bounds and feasibility helpers)."""

    @abstractmethod
    def __call__(
        self,
        x: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator,
    ) -> ArrayLike:
        raise NotImplementedError


class ClampRepair(Repair):
    """Repair operator that clamps variables to their bounds."""

    def __call__(
        self,
        x: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator,  # pylint: disable=unused-argument
    ) -> ArrayLike:
        lower_arr, upper_arr = _ensure_bounds(lower, upper)
        values = self._as_population(x, name="x")
        return _clip_population(values, lower_arr, upper_arr)


def _infinities_to_bounds(values: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> ArrayLike:
    """``values`` with ``+inf`` set to ``upper`` and ``-inf`` to ``lower``.

    Reflecting or wrapping an infinite distance has no meaning (the modulo gives NaN).
    """
    if not bool(np.isinf(values).any()):
        return values
    values = np.where(values == np.inf, np.broadcast_to(upper, values.shape), values)
    return np.where(values == -np.inf, np.broadcast_to(lower, values.shape), values)


class ReflectRepair(Repair):
    """Repair operator that reflects out-of-bounds values back into range.

    An infinite value goes to the bound on its side.
    """

    def __call__(
        self,
        x: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator,  # pylint: disable=unused-argument
    ) -> ArrayLike:
        lower_arr, upper_arr = _ensure_bounds(lower, upper)
        result = _infinities_to_bounds(self._as_population(x, name="x"), lower_arr, upper_arr)
        n_vars = lower_arr.shape[0]
        for j in range(n_vars):
            low = lower_arr[j]
            high = upper_arr[j]
            width = high - low
            if width <= 0.0:
                result[:, j] = low
                continue
            val = result[:, j] - low
            period = 2.0 * width
            val = np.mod(val, period)
            over = val > width
            val[over] = period - val[over]
            result[:, j] = val + low
        return result


class ResampleRepair(Repair):
    """Repair operator that resamples violated genes uniformly inside bounds."""

    def __call__(
        self,
        x: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator,
    ) -> ArrayLike:
        lower_arr, upper_arr = _ensure_bounds(lower, upper)
        result = self._as_population(x, name="x")
        mask_low = result < lower_arr
        mask_high = result > upper_arr
        mask = mask_low | mask_high
        if not np.any(mask):
            return result
        span = upper_arr - lower_arr
        resampled = lower_arr + rng.random(result.shape) * span
        result[mask] = resampled[mask]
        return result


class RoundRepair(Repair):
    """Repair operator that rounds genes to the nearest integer inside the bounds."""

    def __call__(
        self,
        x: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator,  # pylint: disable=unused-argument
    ) -> ArrayLike:
        lower_arr, upper_arr = _ensure_bounds(lower, upper)
        rounded = np.rint(self._as_population(x, name="x"))
        return _clip_population(rounded, lower_arr, upper_arr)


class WrappingRepair(Repair):
    """Repair operator that wraps out-of-bounds values toroidally into range.

    Values that exceed the upper bound re-enter from the lower bound and
    vice-versa, preserving the displacement modulo the domain width.  This is
    useful when the search space has periodic / cyclic semantics (e.g. angles).
    Values already inside the closed interval ``[lower, upper]`` are preserved,
    and an infinite value goes to the bound on its side.
    """

    def __call__(
        self,
        x: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator,  # pylint: disable=unused-argument
    ) -> ArrayLike:
        lower_arr, upper_arr = _ensure_bounds(lower, upper)
        result = _infinities_to_bounds(self._as_population(x, name="x").copy(), lower_arr, upper_arr)
        width = upper_arr - lower_arr
        mask = width > 0.0

        if np.any(mask):
            wrapped = lower_arr[mask] + np.mod(result[:, mask] - lower_arr[mask], width[mask])
            violated = (result[:, mask] < lower_arr[mask]) | (result[:, mask] > upper_arr[mask])
            result[:, mask] = np.where(violated, wrapped, result[:, mask])

        result[:, ~mask] = lower_arr[~mask]
        return result


class MidpointBaseRepair(Repair):
    """Repair operator that moves violated genes halfway from the interval midpoint.

    Violations below the lower bound are mapped to the midpoint between the
    lower bound and the interval midpoint; violations above the upper bound are
    mapped symmetrically on the upper side. This keeps repaired values feasible
    while avoiding the boundary-hugging behavior of pure clamping.

    Every violation therefore lands on one of two fixed points (a quarter and three
    quarters of the interval). This is not the midpoint-base repair of Biedrzycki,
    Arabas and Jagodziński (2019), which places the value halfway between the bound
    and the parent (base) vector.
    """

    def __call__(
        self,
        x: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator,  # pylint: disable=unused-argument
    ) -> ArrayLike:
        lower_arr, upper_arr = _ensure_bounds(lower, upper)
        result = self._as_population(x, name="x").copy()
        midpoint = 0.5 * (lower_arr + upper_arr)
        low_target = 0.5 * (lower_arr + midpoint)
        high_target = 0.5 * (midpoint + upper_arr)
        low_mask = result < lower_arr
        high_mask = result > upper_arr
        result = np.where(low_mask, low_target, result)
        result = np.where(high_mask, high_target, result)
        return result


__all__ = [
    "ClampRepair",
    "MidpointBaseRepair",
    "ReflectRepair",
    "Repair",
    "ResampleRepair",
    "RoundRepair",
    "WrappingRepair",
]
