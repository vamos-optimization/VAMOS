"""Shared utilities for real-valued evolutionary operators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import VariationKernels

ArrayLike: TypeAlias = NDArray[np.float64]


def with_variation_kernels(factory: object, kwargs: dict[str, Any], kernels: VariationKernels | None) -> dict[str, Any]:
    """Bind backend variation kernels when the operator class opts in.

    Only operators that declare ``supports_variation_kernels`` receive the
    kernels; every other registry entry keeps its constructor contract.
    """
    if kernels is not None and getattr(factory, "supports_variation_kernels", False):
        kwargs.setdefault("variation_kernels", kernels)
    return kwargs


def _ensure_bounds(lower: ArrayLike, upper: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """Validate bounds and return float arrays of identical shape."""
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    if lower_arr.shape != upper_arr.shape:
        raise ValueError("lower and upper bounds must have the same shape.")
    if lower_arr.ndim != 1:
        raise ValueError("Bounds must be one-dimensional arrays.")
    if np.any(lower_arr > upper_arr):
        raise ValueError("Each lower bound must be <= corresponding upper bound.")
    return lower_arr, upper_arr


def _clip_population(x: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> ArrayLike:
    """Return a clipped copy of x inside [lower, upper]."""
    clipped = np.clip(x, lower, upper)
    return np.asarray(clipped)


def _check_nvars(n_vars: int, bounds: ArrayLike) -> None:
    """Ensure that a bounds array matches the provided dimensionality."""
    if bounds.shape[0] != n_vars:
        raise ValueError("Bounds dimensionality does not match the individual size.")


class VariationWorkspace:
    """
    Simple buffer registry that hands out reusable NumPy arrays keyed by name.
    Operators can reuse temporary arrays across generations to avoid reallocations.
    """

    def __init__(self) -> None:
        self._buffers: dict[str, NDArray[Any]] = {}

    def request(self, key: str, shape: tuple[int, ...], dtype: Any) -> NDArray[Any]:
        dtype = np.dtype(dtype)
        buf = self._buffers.get(key)
        if buf is None or buf.shape != shape or buf.dtype != dtype:
            buf = np.empty(shape, dtype=dtype)
            self._buffers[key] = buf
        return buf


class RealOperator:
    """Common validation utilities shared by all real-coded operators."""

    @staticmethod
    def _as_matings(
        parents: ArrayLike,
        *,
        expected_parents: int = 2,
        copy: bool = False,
        name: str = "parents",
    ) -> ArrayLike:
        arr = np.asarray(parents, dtype=float)
        if arr.ndim != 3 or arr.shape[1] != expected_parents:
            raise ValueError(f"{name} must have shape (n_matings, {expected_parents}, n_vars).")
        return arr.copy() if copy else arr

    @staticmethod
    def _as_population(
        values: ArrayLike,
        *,
        name: str,
        copy: bool = True,
    ) -> ArrayLike:
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{name} must have shape (n_individuals, n_vars).")
        return arr.copy() if copy else arr

    @staticmethod
    def _check_bounds_match(matrix: ArrayLike, bounds: ArrayLike) -> None:
        _check_nvars(matrix.shape[-1], bounds)


__all__ = [
    "ArrayLike",
    "RealOperator",
    "VariationWorkspace",
    "_check_nvars",
    "_clip_population",
    "_ensure_bounds",
    "with_variation_kernels",
]
