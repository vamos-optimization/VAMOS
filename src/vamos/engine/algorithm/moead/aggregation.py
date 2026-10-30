from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

import numpy as np

# The compiled neighbourhood replacement owns the aggregation ids.
from vamos.foundation.kernel.moead_kernels import AGG_MODIFIED_TCHEBYCHEFF, AGG_PBI, AGG_TCHEBYCHEFF, AGG_WEIGHTED_SUM

Array: TypeAlias = np.ndarray[Any, Any]
AggregatorFn: TypeAlias = Callable[[Array, Array, Array], Array]

ZERO_WEIGHT_EPS: float = 1e-4

_AGGREGATION_IDS: dict[str, int] = {
    "tchebycheff": AGG_TCHEBYCHEFF,
    "tchebychef": AGG_TCHEBYCHEFF,
    "tschebyscheff": AGG_TCHEBYCHEFF,
    "weighted_sum": AGG_WEIGHTED_SUM,
    "weightedsum": AGG_WEIGHTED_SUM,
    "penaltyboundaryintersection": AGG_PBI,
    "penalty_boundary_intersection": AGG_PBI,
    "pbi": AGG_PBI,
    "modifiedtchebycheff": AGG_MODIFIED_TCHEBYCHEFF,
    "modified_tchebycheff": AGG_MODIFIED_TCHEBYCHEFF,
}


def tchebycheff(fvals: Array, weights: Array, ideal: Array) -> Array:
    """Tchebycheff aggregation: max(w * |f - z*|)."""
    diff = np.abs(fvals - ideal)
    weighted = np.where(weights == 0, ZERO_WEIGHT_EPS * diff, weights * diff)
    return np.asarray(np.max(weighted, axis=-1), dtype=float)


def weighted_sum(fvals: Array, weights: Array, ideal: Array) -> Array:
    """Weighted sum aggregation: sum(w * f)."""
    _ = ideal
    return np.asarray(np.sum(weights * fvals, axis=-1), dtype=float)


def pbi(fvals: Array, weights: Array, ideal: Array, theta: float = 5.0) -> Array:
    """Penalty boundary intersection (PBI) aggregation."""
    diff = fvals - ideal
    norm_w = np.linalg.norm(weights, axis=-1, keepdims=True)
    norm_w = np.where(norm_w > 0, norm_w, 1.0)
    w_unit = weights / norm_w
    d1 = np.abs(np.sum(diff * w_unit, axis=-1))
    proj = (d1[..., None]) * w_unit
    d2 = np.linalg.norm(diff - proj, axis=-1)
    return np.asarray(d1 + theta * d2, dtype=float)


def modified_tchebycheff(fvals: Array, weights: Array, ideal: Array, rho: float = 0.001) -> Array:
    """Modified Tchebycheff: max component plus weighted L1 term."""
    diff = np.abs(fvals - ideal)
    weighted = np.where(weights == 0, ZERO_WEIGHT_EPS * diff, weights * diff)
    return np.asarray(np.max(weighted, axis=-1) + rho * np.sum(weighted, axis=-1), dtype=float)


def build_aggregator(name: str, params: Mapping[str, object]) -> AggregatorFn:
    """Build aggregation function from name and parameters."""
    method = name.lower()
    if method in {"tchebycheff", "tchebychef", "tschebyscheff"}:
        return tchebycheff
    if method in {"weighted_sum", "weightedsum"}:
        return weighted_sum
    if method in {"penaltyboundaryintersection", "penalty_boundary_intersection", "pbi"}:
        theta_raw = params.get("theta", 5.0)
        theta = float(theta_raw) if isinstance(theta_raw, (int, float, str)) else 5.0

        def _agg(fvals: Array, weights: Array, ideal: Array) -> Array:
            return pbi(fvals, weights, ideal, theta)

        return _agg
    if method in {"modifiedtchebycheff", "modified_tchebycheff"}:
        rho_raw = params.get("rho", 0.001)
        rho = float(rho_raw) if isinstance(rho_raw, (int, float, str)) else 0.001

        def _agg(fvals: Array, weights: Array, ideal: Array) -> Array:
            return modified_tchebycheff(fvals, weights, ideal, rho)

        return _agg
    raise ValueError(f"Unsupported aggregation method '{name}'.")


def resolve_aggregation_spec(name: str, params: Mapping[str, object]) -> tuple[int, float, float]:
    """Resolve aggregation ID and parameters for fast paths."""
    method = name.lower()
    agg_id = _AGGREGATION_IDS.get(method, -1)

    theta_raw = params.get("theta", 5.0)
    theta = float(theta_raw) if isinstance(theta_raw, (int, float, str)) else 5.0

    rho_raw = params.get("rho", 0.001)
    rho = float(rho_raw) if isinstance(rho_raw, (int, float, str)) else 0.001

    return agg_id, theta, rho


__all__ = [
    "AGG_MODIFIED_TCHEBYCHEFF",
    "AGG_PBI",
    "AGG_TCHEBYCHEFF",
    "AGG_WEIGHTED_SUM",
    "AggregatorFn",
    "ZERO_WEIGHT_EPS",
    "build_aggregator",
    "modified_tchebycheff",
    "pbi",
    "resolve_aggregation_spec",
    "tchebycheff",
    "weighted_sum",
]
