"""
Utility helpers for constraint handling.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def compute_violation(G: ArrayLike | None, *, n: int | None = None, eps: float = 0.0) -> NDArray[np.float64]:
    """Sum per solution of each constraint's excess over ``eps``; assumes G shape (N, n_constraints), g<=0 satisfied.

    When *G* is ``None`` (unconstrained), ``n`` must be provided so the output
    length is explicit.

    *eps* is the feasibility tolerance of :func:`is_feasible`: the violation is
    ``sum(max(g - eps, 0))``, so a solution has zero violation exactly when it is
    feasible with the same ``eps`` (default ``0.0``).
    """
    if G is None:
        if n is None:
            raise ValueError("compute_violation() requires 'n' when G is None.")
        return np.zeros(n, dtype=float)
    values = np.asarray(G, dtype=float)
    positive = np.maximum(values - eps, 0.0) if eps else np.maximum(values, 0.0)
    return np.asarray(np.sum(positive, axis=1), dtype=float)


def is_feasible(G: ArrayLike | None, *, n: int | None = None, eps: float = 0.0) -> NDArray[np.bool_]:
    """Boolean feasibility mask; assumes G shape (N, n_constraints).

    When *G* is ``None`` (unconstrained), ``n`` must be provided so the output
    length is explicit.

    *eps* is a feasibility tolerance: constraints with ``g(x) <= eps`` are
    treated as satisfied (default ``0.0``).
    """
    if G is None:
        if n is None:
            raise ValueError("is_feasible() requires 'n' when G is None.")
        return np.ones(n, dtype=bool)
    return np.asarray(np.all(np.asarray(G) <= eps, axis=1), dtype=bool)


__all__ = ["compute_violation", "is_feasible"]
