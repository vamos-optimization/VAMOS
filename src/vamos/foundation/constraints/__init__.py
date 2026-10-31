from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .utils import compute_violation, is_feasible


@dataclass
class ConstraintInfo:
    G: NDArray[np.float64]
    cv: NDArray[np.float64]
    feasible_mask: NDArray[np.bool_]


def compute_constraint_info(G: NDArray[np.float64] | None, eps: float = 0.0) -> ConstraintInfo:
    """Compute aggregate constraint violation and feasibility mask.

    Parameters
    ----------
    G : NDArray[np.float64] | None
        Constraint values where ``<= 0`` means satisfied.
    eps : float, default ``0.0``
        Feasibility tolerance. Constraints ``<= eps`` are treated as satisfied.
    """
    if G is None:
        empty_G = np.zeros((0, 0), dtype=float)
        return ConstraintInfo(
            G=empty_G,
            cv=np.zeros(0, dtype=float),
            feasible_mask=np.ones(0, dtype=bool),
        )
    G = np.asarray(G, dtype=float)
    if G.ndim != 2:
        raise ValueError("G must be a 2D array of shape (n_points, n_constraints).")
    positive = np.maximum(G - eps, 0.0)
    cv = np.asarray(np.sum(positive, axis=1), dtype=float)
    feasible = np.asarray(np.all(G <= eps, axis=1), dtype=bool)
    return ConstraintInfo(G=G, cv=cv, feasible_mask=feasible)


__all__ = [
    "ConstraintInfo",
    "compute_constraint_info",
    "compute_violation",
    "is_feasible",
]
