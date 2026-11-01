from __future__ import annotations

from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray


@overload
def pareto_filter(F: NDArray[np.float64] | None, *, return_indices: Literal[False] = False) -> NDArray[np.float64] | None: ...


@overload
def pareto_filter(F: NDArray[np.float64] | None, *, return_indices: Literal[True]) -> tuple[NDArray[np.float64], NDArray[np.int_]]: ...


def pareto_filter(
    F: NDArray[np.float64] | None, *, return_indices: bool = False
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.int_]] | None:
    """
    Return the non-dominated subset of points (first Pareto front).

    Parameters
    ----------
    F : NDArray[np.float64] | None
        Objective array of shape ``(n_solutions, n_objectives)``.
    return_indices : bool, default ``False``
        Whether to also return indices of the Pareto front in ``F``.

    Returns
    -------
    NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.int_]] | None
        Front array, or ``(front, indices)`` when ``return_indices`` is true.

    Raises
    ------
    ValueError
        If ``F`` is not empty and not two-dimensional, or if a row contains NaN,
        which is neither better nor worse than any value. Infinite values are ordered.
    """
    if F is None:
        if return_indices:
            return np.empty((0, 0)), np.array([], dtype=int)
        return None
    F = np.asarray(F)
    if F.size and F.ndim != 2:
        raise ValueError(f"pareto_filter expects a 2D array of shape (n_solutions, n_objectives); got {F.ndim} dimension(s).")
    if F.size and np.issubdtype(F.dtype, np.floating) and np.isnan(F).any():
        raise ValueError("pareto_filter cannot rank rows that contain NaN objective values.")
    if F.size == 0 or F.ndim < 2:
        if return_indices:
            n = int(F.shape[0]) if F.ndim > 0 else 0
            idx: NDArray[np.int_] = np.arange(n, dtype=int)
            return F, idx
        return F
    from vamos.foundation.kernel.numpy_backend import _fast_non_dominated_sort

    fronts, _ = _fast_non_dominated_sort(F)
    if not fronts or not fronts[0]:
        if return_indices:
            idx = np.arange(int(F.shape[0]), dtype=int)
            return F, idx
        return F
    idx = np.asarray(fronts[0], dtype=int)
    front = F[idx]
    return (front, idx) if return_indices else front
