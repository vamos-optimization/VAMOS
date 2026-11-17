"""MOEA/D's aggregation (scalarizing) functions.

Each function maps objective vectors ``fvals`` and weight vectors ``weights`` (row by row) to the
value of the subproblem, to be minimized, with the ideal point ``ideal`` as reference.

**VAMOS 1.4.0:** two more functions and the normalization of the objectives, after
jMetal 7.8 (``org.uma.jmetal.util.aggregationfunction.impl``, which Evolver's
``AggregationFunctionParameter`` builds):

- ``augmented_tchebycheff`` (jMetal's ``AugmentedTschebyscheff``, Steuer and Choo, 1983):
  ``max_i w_i |f_i - z*_i| + rho sum_i w_i |f_i - z*_i|``, a zero weight counting
  ``1e-4 |f_i - z*_i|``, with jMetal's ``rho = 1e-4``. This is the formula of VAMOS's
  ``modified_tchebycheff``, whose ``rho`` is ``1e-3``; jMetal's ``ModifiedTschebyscheff``
  (``max_i |f_i - z*_i| / w_i``) is another function, ``inverse_weight_tchebycheff`` below.
- ``inverse_weight_tchebycheff`` (jMetal's ``ModifiedTschebyscheff``, which Evolver calls
  ``modifiedTschebyscheff``): ``max_i |f_i - z*_i| / w_i``, a zero weight counting
  ``1e-7 |f_i - z*_i|``. Up to a positive factor per weight vector, which leaves the comparisons
  of a subproblem unchanged, it is the Tchebycheff function with the weights that Qi et al.
  (2014, Eq. 3) call the WS-transformation, ``(1 / w_i) / sum_j (1 / w_j)``. It has no parameter.
- ``inverted_pbi`` (jMetal's ``InvertedPenaltyBoundaryIntersection``; Sato, 2014): with the
  displacement ``u = z^nad - f`` from the nadir point, ``d1 = |u . w| / ||w||`` and
  ``d2 = ||u - d1 w / ||w|| ||``, the value ``theta d2 - d1`` (Sato's IPBI maximizes
  ``d1 - theta d2``), with jMetal's ``theta = 0.1``.
- ``normalize=True`` divides each displacement by ``z^nad_i - z*_i + epsilon``, as every jMetal
  aggregation function does with ``normalizeObjectives``: ``(f_i - z*_i) / (z^nad_i - z*_i + epsilon)``
  (the weighted sum uses it in place of ``f_i``) and, for the inverted PBI,
  ``(z^nad_i - f_i) / (z^nad_i - z*_i + epsilon)``. ``epsilon`` defaults to jMetal's ``1e-6``.

The nadir point is the estimate of :mod:`vamos.engine.algorithm.moead.nadir`, which MOEA/D keeps
only when an aggregation needs it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import numpy as np

# The compiled neighbourhood replacement owns the aggregation ids.
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.kernel.moead_kernels import (
    AGG_AUGMENTED_TCHEBYCHEFF,
    AGG_INVERSE_WEIGHT_TCHEBYCHEFF,
    AGG_INVERTED_PBI,
    AGG_MODIFIED_TCHEBYCHEFF,
    AGG_PBI,
    AGG_TCHEBYCHEFF,
    AGG_WEIGHTED_SUM,
)

Array: TypeAlias = np.ndarray[Any, Any]


class AggregatorFn(Protocol):
    """An aggregation of objective vectors; ``nadir`` is needed by the normalization and the inverted PBI."""

    def __call__(self, fvals: Array, weights: Array, ideal: Array, nadir: Array | None = None) -> Array: ...


ZERO_WEIGHT_EPS: float = 1e-4
#: The factor of a zero weight's difference in jMetal's ModifiedTschebyscheff.
INVERSE_ZERO_WEIGHT_EPS: float = 1e-7
#: jMetal's epsilon of the normalization (``AggregationFunction.epsilon``).
NORMALIZATION_EPSILON: float = 1e-6

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
    "augmented_tchebycheff": AGG_AUGMENTED_TCHEBYCHEFF,
    "inverted_pbi": AGG_INVERTED_PBI,
    "inverse_weight_tchebycheff": AGG_INVERSE_WEIGHT_TCHEBYCHEFF,
}
# The parameters of the aggregations added after 1.3.0, which reject any other.
_PARAMETERS: dict[int, tuple[str, ...]] = {
    AGG_AUGMENTED_TCHEBYCHEFF: ("rho", "normalize", "epsilon"),
    AGG_INVERTED_PBI: ("theta", "normalize", "epsilon"),
    AGG_INVERSE_WEIGHT_TCHEBYCHEFF: ("normalize", "epsilon"),
}


def aggregation_methods() -> tuple[str, ...]:
    """Aggregation names MOEA/D accepts, in lower case, aliases included."""
    return tuple(_AGGREGATION_IDS)


@dataclass(frozen=True)
class AggregationSpec:
    """A resolved aggregation: its id in the compiled update, its parameters and its normalization."""

    method_id: int
    theta: float
    rho: float
    normalize: bool
    epsilon: float

    @property
    def needs_nadir(self) -> bool:
        """Whether the aggregation reads the nadir point: with normalization, and the inverted PBI always."""
        return self.normalize or self.method_id == AGG_INVERTED_PBI


def _number(raw: object, default: float) -> float:
    return float(raw) if isinstance(raw, (int, float, str)) else default


def _check_parameter(label: str, value: object, *, strict: bool) -> float:
    number = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.nan
    if not math.isfinite(number) or (number <= 0.0 if strict else number < 0.0):
        raise ConfigurationError(f"{label} must be finite and {'>' if strict else '>='} 0; got {value!r}.")
    return number


def resolve_aggregation(name: str, params: Mapping[str, object]) -> AggregationSpec:
    """Validate an aggregation and resolve its parameters.

    Raises ``ConfigurationError`` for an unknown name, a ``normalize`` that is not a boolean, an
    ``epsilon`` without normalization or not positive, and, for the aggregations added after 1.3.0,
    an unknown parameter or a negative ``theta`` or ``rho``.
    """
    method = name.lower()
    if method not in _AGGREGATION_IDS:
        raise ConfigurationError(f"Unsupported aggregation method '{name}'.")
    method_id = _AGGREGATION_IDS[method]
    allowed = _PARAMETERS.get(method_id)
    if allowed is not None:
        unknown = [key for key in params if key not in allowed]
        if unknown:
            raise ConfigurationError(f"aggregation {method!r} takes {', '.join(allowed)}; got {', '.join(map(repr, unknown))}.")
    normalize = params.get("normalize", False)
    if not isinstance(normalize, (bool, np.bool_)):
        raise ConfigurationError(f"aggregation parameter normalize must be True or False; got {normalize!r}.")
    epsilon = NORMALIZATION_EPSILON
    if "epsilon" in params:
        if not normalize:
            raise ConfigurationError("aggregation parameter epsilon applies only with normalize=True.")
        epsilon = _check_parameter("aggregation parameter epsilon", params["epsilon"], strict=True)
    theta_default = 0.1 if method_id == AGG_INVERTED_PBI else 5.0
    rho_default = 0.0001 if method_id == AGG_AUGMENTED_TCHEBYCHEFF else 0.001
    theta = _number(params.get("theta", theta_default), theta_default)
    rho = _number(params.get("rho", rho_default), rho_default)
    if method_id == AGG_INVERTED_PBI and "theta" in params:
        theta = _check_parameter(f"aggregation {method!r} parameter theta", params["theta"], strict=False)
    if method_id == AGG_AUGMENTED_TCHEBYCHEFF and "rho" in params:
        rho = _check_parameter(f"aggregation {method!r} parameter rho", params["rho"], strict=False)
    return AggregationSpec(method_id=method_id, theta=theta, rho=rho, normalize=bool(normalize), epsilon=epsilon)


def tchebycheff(fvals: Array, weights: Array, ideal: Array) -> Array:
    """Tchebycheff aggregation: max(w * |f - z*|)."""
    diff = np.abs(fvals - ideal)
    weighted = np.where(weights == 0, ZERO_WEIGHT_EPS * diff, weights * diff)
    return np.asarray(np.max(weighted, axis=-1), dtype=float)


def weighted_sum(fvals: Array, weights: Array, ideal: Array) -> Array:
    """Weighted sum aggregation: sum(w * f)."""
    _ = ideal
    return np.asarray(np.sum(weights * fvals, axis=-1), dtype=float)


def _pbi_distances(diff: Array, weights: Array) -> tuple[Array, Array]:
    """``d1``, the projection of ``diff`` on the weight direction, and ``d2``, its distance to it."""
    norm_w = np.linalg.norm(weights, axis=-1, keepdims=True)
    norm_w = np.where(norm_w > 0, norm_w, 1.0)
    w_unit = weights / norm_w
    d1 = np.abs(np.sum(diff * w_unit, axis=-1))
    proj = (d1[..., None]) * w_unit
    d2 = np.linalg.norm(diff - proj, axis=-1)
    return d1, d2


def pbi(fvals: Array, weights: Array, ideal: Array, theta: float = 5.0) -> Array:
    """Penalty boundary intersection (PBI) aggregation."""
    d1, d2 = _pbi_distances(fvals - ideal, weights)
    return np.asarray(d1 + theta * d2, dtype=float)


def modified_tchebycheff(fvals: Array, weights: Array, ideal: Array, rho: float = 0.001) -> Array:
    """Modified Tchebycheff: max component plus weighted L1 term."""
    diff = np.abs(fvals - ideal)
    weighted = np.where(weights == 0, ZERO_WEIGHT_EPS * diff, weights * diff)
    return np.asarray(np.max(weighted, axis=-1) + rho * np.sum(weighted, axis=-1), dtype=float)


def augmented_tchebycheff(fvals: Array, weights: Array, ideal: Array, rho: float = 0.0001) -> Array:
    """Augmented Tchebycheff (jMetal's ``AugmentedTschebyscheff``): the modified formula with jMetal's ``rho = 1e-4``."""
    return modified_tchebycheff(fvals, weights, ideal, rho)


def inverse_weight_tchebycheff(fvals: Array, weights: Array, ideal: Array) -> Array:
    """jMetal's ``ModifiedTschebyscheff``: ``max(|f - z*| / w)``, a zero weight counting ``1e-7 |f - z*|``."""
    diff = np.abs(fvals - ideal)
    weights = np.broadcast_to(weights, diff.shape)
    weighted = np.divide(diff, weights, out=INVERSE_ZERO_WEIGHT_EPS * diff, where=weights != 0)
    return np.asarray(np.max(weighted, axis=-1), dtype=float)


def inverted_pbi(fvals: Array, weights: Array, nadir: Array, theta: float = 0.1) -> Array:
    """Inverted PBI (Sato, 2014; jMetal's ``InvertedPenaltyBoundaryIntersection``): ``theta d2 - d1`` of ``z^nad - f``."""
    d1, d2 = _pbi_distances(nadir - fvals, weights)
    return np.asarray(theta * d2 - d1, dtype=float)


def _require_nadir(nadir: Array | None) -> Array:
    if nadir is None:
        raise ValueError("This aggregation needs the nadir point.")
    return nadir


def build_aggregator(name: str, params: Mapping[str, object]) -> AggregatorFn:
    """Build the aggregation of ``name`` and ``params``; see :func:`resolve_aggregation`."""
    spec = resolve_aggregation(name, params)
    method_id, theta, rho, epsilon = spec.method_id, spec.theta, spec.rho, spec.epsilon

    def base(fvals: Array, weights: Array, ideal: Array) -> Array:
        if method_id == AGG_TCHEBYCHEFF:
            return tchebycheff(fvals, weights, ideal)
        if method_id == AGG_WEIGHTED_SUM:
            return weighted_sum(fvals, weights, ideal)
        if method_id == AGG_PBI:
            return pbi(fvals, weights, ideal, theta)
        if method_id == AGG_INVERSE_WEIGHT_TCHEBYCHEFF:
            return inverse_weight_tchebycheff(fvals, weights, ideal)
        return modified_tchebycheff(fvals, weights, ideal, rho)

    if method_id == AGG_INVERTED_PBI:

        def _inverted(fvals: Array, weights: Array, ideal: Array, nadir: Array | None = None) -> Array:
            reference = _require_nadir(nadir)
            if not spec.normalize:
                return inverted_pbi(fvals, weights, reference, theta)
            d1, d2 = _pbi_distances((reference - fvals) / (reference - ideal + epsilon), weights)
            return np.asarray(theta * d2 - d1, dtype=float)

        return _inverted

    if spec.normalize:

        def _normalized(fvals: Array, weights: Array, ideal: Array, nadir: Array | None = None) -> Array:
            reference = _require_nadir(nadir)
            # (f - z*) / (z^nad - z* + epsilon), measured from the origin, which is then the ideal point.
            return base((fvals - ideal) / (reference - ideal + epsilon), weights, np.zeros_like(ideal))

        return _normalized

    def _plain(fvals: Array, weights: Array, ideal: Array, nadir: Array | None = None) -> Array:
        return base(fvals, weights, ideal)

    return _plain


def resolve_aggregation_spec(name: str, params: Mapping[str, object]) -> tuple[int, float, float]:
    """Resolve aggregation ID and parameters for fast paths."""
    method = name.lower()
    agg_id = _AGGREGATION_IDS.get(method, -1)
    theta_default = 0.1 if agg_id == AGG_INVERTED_PBI else 5.0
    rho_default = 0.0001 if agg_id == AGG_AUGMENTED_TCHEBYCHEFF else 0.001
    return agg_id, _number(params.get("theta", theta_default), theta_default), _number(params.get("rho", rho_default), rho_default)


__all__ = [
    "AGG_AUGMENTED_TCHEBYCHEFF",
    "AGG_INVERSE_WEIGHT_TCHEBYCHEFF",
    "AGG_INVERTED_PBI",
    "AGG_MODIFIED_TCHEBYCHEFF",
    "AGG_PBI",
    "AGG_TCHEBYCHEFF",
    "AGG_WEIGHTED_SUM",
    "NORMALIZATION_EPSILON",
    "AggregationSpec",
    "AggregatorFn",
    "INVERSE_ZERO_WEIGHT_EPS",
    "ZERO_WEIGHT_EPS",
    "aggregation_methods",
    "augmented_tchebycheff",
    "build_aggregator",
    "inverse_weight_tchebycheff",
    "inverted_pbi",
    "modified_tchebycheff",
    "pbi",
    "resolve_aggregation",
    "resolve_aggregation_spec",
    "tchebycheff",
    "weighted_sum",
]
