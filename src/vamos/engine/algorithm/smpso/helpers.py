"""SMPSO helper functions.

This module contains utility functions for SMPSO:
- Repair strategy resolution
- Dominance comparison for personal best updates
- Personal best update logic with constraint handling
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, cast

import numpy as np

from vamos.engine.operators.impl.registry import get_operator_registry
from vamos.foundation.constraints.utils import compute_violation, is_feasible

__all__ = [
    "resolve_repair",
    "dominates",
    "update_personal_bests",
]


_SMPSO_REPAIR_NAMES = (
    "clip",
    "clamp",
    "reflect",
    "random",
    "resample",
    "round",
    "wrap",
    "wrapping",
    "midpoint",
    "midpoint_base",
    "gradient",
)


@lru_cache(maxsize=1)
def _repair_map() -> dict[str, type[Any] | None]:
    registry = get_operator_registry()
    return {name: cast(type[Any] | None, registry.get(name, None)) for name in _SMPSO_REPAIR_NAMES}


def resolve_repair(cfg: Any | None) -> Any | None:
    """Resolve repair strategy from configuration.

    Parameters
    ----------
    cfg : Any or None
        Repair configuration. Can be:
        - None: No repair
        - tuple: (method, params)
        - dict: {"method": ..., **params}
        - str: Method name

    Returns
    -------
    Any or None
        Repair operator instance or None.

    Raises
    ------
    ValueError
        If repair strategy is unknown.
    """
    if cfg is None:
        return None
    if isinstance(cfg, tuple):
        method, params = cfg
        params = dict(params)
    elif isinstance(cfg, dict):
        cfg = dict(cfg)
        method = cfg.pop("method", cfg.pop("name", cfg.pop("type", None)))
        params = cfg
    else:
        method, params = str(cfg), {}

    if method is None:
        return None
    normalized = str(method).lower()
    if normalized in {"none", "off", "disabled"}:
        return None
    cls = _repair_map().get(normalized)
    if cls is None:
        raise ValueError(f"Unknown repair strategy '{method}' for SMPSO.")
    return cls(**params)


def dominates(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Check Pareto dominance (a dominates b).

    Parameters
    ----------
    a : np.ndarray
        Objective values, shape (n, n_obj).
    b : np.ndarray
        Objective values, shape (n, n_obj).

    Returns
    -------
    np.ndarray
        Boolean array where True means a[i] dominates b[i].
    """
    leq = a <= b
    lt = a < b
    return np.asarray(np.all(leq, axis=1) & np.any(lt, axis=1), dtype=bool)


def update_personal_bests(
    X: np.ndarray[Any, Any],
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any] | None,
    pbest_X: np.ndarray[Any, Any],
    pbest_F: np.ndarray[Any, Any],
    pbest_G: np.ndarray[Any, Any] | None,
    constraint_mode: str,
) -> None:
    """Update personal best positions based on dominance.

    For constrained problems, uses feasibility-first approach:
    1. Feasible always beats infeasible
    2. Among feasible, dominance applies
    3. Among infeasible, lower constraint violation wins

    Parameters
    ----------
    X : np.ndarray
        Current positions.
    F : np.ndarray
        Current objective values.
    G : np.ndarray or None
        Current constraint values.
    pbest_X : np.ndarray
        Personal best positions (updated in-place).
    pbest_F : np.ndarray
        Personal best objectives (updated in-place).
    pbest_G : np.ndarray or None
        Personal best constraints (updated in-place).
    constraint_mode : str
        Constraint handling mode.
    """
    if constraint_mode and constraint_mode != "none" and G is not None and pbest_G is not None:
        feas_new = is_feasible(G)
        feas_old = is_feasible(pbest_G)
        cv_new = compute_violation(G)
        cv_old = compute_violation(pbest_G)

        better = np.zeros(X.shape[0], dtype=bool)
        # Feasible beats infeasible
        better |= feas_new & ~feas_old
        # Among feasible, dominance wins
        better |= feas_new & feas_old & dominates(F, pbest_F)
        # Among infeasible, lower violation wins
        better |= (~feas_new) & (~feas_old) & (cv_new < cv_old)
        update_idx = better
    else:
        # Without constraints, update if current is not dominated by personal best
        update_idx = ~dominates(pbest_F, F)

    pbest_X[update_idx] = X[update_idx]
    pbest_F[update_idx] = F[update_idx]
    if pbest_G is not None and G is not None:
        pbest_G[update_idx] = G[update_idx]
