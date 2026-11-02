"""MooCore-backed quality indicators with one input contract.

Every indicator requires finite fronts, reference fronts and reference points
(``ValueError`` otherwise) whose objective counts agree. The hypervolume family
(``hv``, ``hv_contributions``, ``hv_approx``, ``whv_rect``) requires an explicit
reference point and ignores points outside the reference box, like
:func:`vamos.foundation.quality_indicators.hypervolume.hypervolume`.

Definitions follow MooCore:

- ``igd`` is the mean distance from each reference point to the front (p = 1,
  as Coello and Cortés and pymoo define it). jMetal's default IGD,
  ``sqrt(sum d^2) / n``, is a different quantity.
- ``epsilon_mult`` requires strictly positive values in the front and in the
  reference front.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:  # pragma: no cover - optional dependency
    import moocore as _moocore
except Exception:  # pragma: no cover - optional dependency
    _moocore = None

FloatArray = NDArray[np.float64]


def has_moocore() -> bool:
    return _moocore is not None


@dataclass
class IndicatorResult:
    value: float | FloatArray
    details: dict[str, Any] = field(default_factory=dict)


class QualityIndicator(Protocol):
    name: str

    def compute(
        self,
        front: ArrayLike,
        reference_front: ArrayLike | None = None,
        maximise: bool | Sequence[bool] | None = None,
        **kwargs: Any,
    ) -> IndicatorResult: ...


def _require_moocore() -> Any:
    if _moocore is None:
        raise ImportError("MooCore is required for this indicator but is not installed.")
    return _moocore


def _normalize_maximise(maximise: bool | Sequence[bool] | None) -> bool | Sequence[bool]:
    return False if maximise is None else maximise


def _finite_matrix(values: ArrayLike, *, label: str) -> FloatArray:
    data = np.ascontiguousarray(values, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"{label} must be a 2D array of shape (n_points, n_objectives).")
    if not np.isfinite(data).all():
        raise ValueError(f"{label} must contain finite numbers only (no NaN or inf).")
    return data


def _front(front: ArrayLike) -> FloatArray:
    return _finite_matrix(front, label="front")


def _reference_front(reference_front: ArrayLike | None, n_obj: int) -> FloatArray:
    if reference_front is None:
        raise ValueError("reference_front is required.")
    data = _finite_matrix(reference_front, label="reference_front")
    if data.shape[0] == 0:
        raise ValueError("reference_front must contain at least one point.")
    if data.shape[1] != n_obj:
        raise ValueError(f"reference_front has {data.shape[1]} objectives; the front has {n_obj}.")
    return data


def _reference_point(reference_point: ArrayLike | None, n_obj: int, *, name: str) -> FloatArray:
    if reference_point is None:
        raise ValueError(f"{name} requires an explicit reference_point; VAMOS does not derive one from the front.")
    ref = np.asarray(reference_point, dtype=float)
    if ref.ndim != 1 or ref.shape[0] != n_obj:
        raise ValueError(f"reference_point must be a 1D array with one value per objective ({n_obj}).")
    if not np.isfinite(ref).all():
        raise ValueError("reference_point must contain finite numbers only (no NaN or inf).")
    return ref


@dataclass
class HVIndicator(QualityIndicator):
    reference_point: ArrayLike | None = None
    name: str = "hv"

    def compute(
        self,
        front: ArrayLike,
        reference_front: ArrayLike | None = None,
        maximise: bool | Sequence[bool] | None = None,
        **_: Any,
    ) -> IndicatorResult:
        moocore = _require_moocore()
        F = _front(front)
        ref = _reference_point(self.reference_point, F.shape[1], name=self.name)
        val = float(moocore.hypervolume(F, ref=ref, maximise=_normalize_maximise(maximise)))
        return IndicatorResult(value=val, details={"reference_point": ref})


@dataclass
class HVContributionsIndicator(QualityIndicator):
    reference_point: ArrayLike | None = None
    name: str = "hv_contributions"

    def compute(
        self,
        front: ArrayLike,
        reference_front: ArrayLike | None = None,
        maximise: bool | Sequence[bool] | None = None,
        **_: Any,
    ) -> IndicatorResult:
        moocore = _require_moocore()
        F = _front(front)
        ref = _reference_point(self.reference_point, F.shape[1], name=self.name)
        contrib = np.asarray(moocore.hv_contributions(F, ref=ref, maximise=_normalize_maximise(maximise)), dtype=float)
        return IndicatorResult(value=contrib, details={"reference_point": ref})


@dataclass
class _ReferenceFrontIndicator(QualityIndicator):
    """An indicator measured against a reference front with one MooCore function."""

    reference_front: ArrayLike
    name: str = ""

    def _value(self, moocore: Any, front: FloatArray, ref: FloatArray, maximise: bool | Sequence[bool], **kwargs: Any) -> float:
        raise NotImplementedError

    def compute(
        self,
        front: ArrayLike,
        reference_front: ArrayLike | None = None,
        maximise: bool | Sequence[bool] | None = None,
        **kwargs: Any,
    ) -> IndicatorResult:
        moocore = _require_moocore()
        F = _front(front)
        ref = _reference_front(reference_front if reference_front is not None else self.reference_front, F.shape[1])
        val = self._value(moocore, F, ref, _normalize_maximise(maximise), **kwargs)
        return IndicatorResult(value=val, details={"reference_front": ref})


@dataclass
class IGDIndicator(_ReferenceFrontIndicator):
    """Mean distance from each reference point to the front (p = 1)."""

    name: str = "igd"

    def _value(self, moocore: Any, front: FloatArray, ref: FloatArray, maximise: bool | Sequence[bool], **kwargs: Any) -> float:
        return float(moocore.igd(front, ref=ref, maximise=maximise))


@dataclass
class IGDPlusIndicator(_ReferenceFrontIndicator):
    name: str = "igd_plus"

    def _value(self, moocore: Any, front: FloatArray, ref: FloatArray, maximise: bool | Sequence[bool], **kwargs: Any) -> float:
        return float(moocore.igd_plus(front, ref=ref, maximise=maximise))


@dataclass
class EpsilonAdditiveIndicator(_ReferenceFrontIndicator):
    name: str = "epsilon_additive"

    def _value(self, moocore: Any, front: FloatArray, ref: FloatArray, maximise: bool | Sequence[bool], **kwargs: Any) -> float:
        return float(moocore.epsilon_additive(front, ref=ref, maximise=maximise))


@dataclass
class EpsilonMultiplicativeIndicator(_ReferenceFrontIndicator):
    """Multiplicative epsilon; the front and the reference front must be strictly positive."""

    name: str = "epsilon_multiplicative"

    def _value(self, moocore: Any, front: FloatArray, ref: FloatArray, maximise: bool | Sequence[bool], **kwargs: Any) -> float:
        return float(moocore.epsilon_mult(front, ref=ref, maximise=maximise))


@dataclass
class AvgHausdorffIndicator(_ReferenceFrontIndicator):
    name: str = "avg_hausdorff"

    def _value(self, moocore: Any, front: FloatArray, ref: FloatArray, maximise: bool | Sequence[bool], **kwargs: Any) -> float:
        p = kwargs.get("p", 1.0)
        return float(moocore.avg_hausdorff_dist(front, ref, maximise=maximise, p=p))

    def compute(
        self,
        front: ArrayLike,
        reference_front: ArrayLike | None = None,
        maximise: bool | Sequence[bool] | None = None,
        **kwargs: Any,
    ) -> IndicatorResult:
        result = super().compute(front, reference_front, maximise, **kwargs)
        result.details["p"] = kwargs.get("p", 1.0)
        return result


@dataclass
class HVApproxIndicator(QualityIndicator):
    reference_point: ArrayLike | None = None
    name: str = "hv_approx"

    def compute(
        self,
        front: ArrayLike,
        reference_front: ArrayLike | None = None,
        maximise: bool | Sequence[bool] | None = None,
        **kwargs: Any,
    ) -> IndicatorResult:
        moocore = _require_moocore()
        F = _front(front)
        ref = _reference_point(self.reference_point, F.shape[1], name=self.name)
        samples = int(kwargs.get("samples", 10000))
        val = float(moocore.hv_approx(F, ref=ref, maximise=_normalize_maximise(maximise), nsamples=samples))
        return IndicatorResult(value=val, details={"reference_point": ref, "samples": samples})


@dataclass
class WHVRectIndicator(QualityIndicator):
    reference_point: ArrayLike | None = None
    rectangles: ArrayLike | None = None
    name: str = "whv_rect"

    def compute(
        self,
        front: ArrayLike,
        reference_front: ArrayLike | None = None,
        maximise: bool | Sequence[bool] | None = None,
        **kwargs: Any,
    ) -> IndicatorResult:
        moocore = _require_moocore()
        F = _front(front)
        ref = _reference_point(self.reference_point, F.shape[1], name=self.name)
        rects_source = self.rectangles if self.rectangles is not None else kwargs.get("rectangles")
        if rects_source is None:
            raise ValueError("whv_rect requires rectangles to be provided.")
        rects = np.asarray(rects_source, dtype=float)
        if rects.size == 0:
            raise ValueError("whv_rect requires rectangles to be provided.")
        val = float(moocore.whv_rect(F, rects, ref=ref, maximise=_normalize_maximise(maximise)))
        return IndicatorResult(value=val, details={"reference_point": ref})


# Every accepted spelling of an indicator name, lower-cased, and its canonical name.
_INDICATOR_ALIASES = {
    "hv": "hv",
    "hypervolume": "hv",
    "hv_contrib": "hv_contributions",
    "hv_contributions": "hv_contributions",
    "igd": "igd",
    "igd+": "igd_plus",
    "igd_plus": "igd_plus",
    "eps": "epsilon_additive",
    "epsilon": "epsilon_additive",
    "epsilon_add": "epsilon_additive",
    "epsilon_additive": "epsilon_additive",
    "epsilon_mult": "epsilon_multiplicative",
    "eps_mult": "epsilon_multiplicative",
    "epsilon_multiplicative": "epsilon_multiplicative",
    "avg_hausdorff": "avg_hausdorff",
    "hausdorff": "avg_hausdorff",
    "hv_approx": "hv_approx",
    "whv_rect": "whv_rect",
    "weighted_hv": "whv_rect",
}

# Canonical names of the indicators that need a reference point or a reference front.
REFERENCE_POINT_INDICATORS = ("hv", "hv_contributions", "hv_approx", "whv_rect")
REFERENCE_FRONT_INDICATORS = ("igd", "igd_plus", "epsilon_additive", "epsilon_multiplicative", "avg_hausdorff")

_INDICATOR_CLASSES: dict[str, type[QualityIndicator]] = {
    "hv": HVIndicator,
    "hv_contributions": HVContributionsIndicator,
    "igd": IGDIndicator,
    "igd_plus": IGDPlusIndicator,
    "epsilon_additive": EpsilonAdditiveIndicator,
    "epsilon_multiplicative": EpsilonMultiplicativeIndicator,
    "avg_hausdorff": AvgHausdorffIndicator,
    "hv_approx": HVApproxIndicator,
    "whv_rect": WHVRectIndicator,
}


def canonical_indicator_name(name: str) -> str:
    """The canonical name of ``name`` or any accepted alias, case-insensitive."""
    try:
        return _INDICATOR_ALIASES[name.strip().lower()]
    except KeyError:
        raise ValueError(f"Unknown indicator '{name}'.") from None


def get_indicator(name: str, **kwargs: Any) -> QualityIndicator:
    return _INDICATOR_CLASSES[canonical_indicator_name(name)](**kwargs)


__all__ = [
    "IndicatorResult",
    "QualityIndicator",
    "REFERENCE_FRONT_INDICATORS",
    "REFERENCE_POINT_INDICATORS",
    "canonical_indicator_name",
    "has_moocore",
    "get_indicator",
    "HVIndicator",
    "HVContributionsIndicator",
    "IGDIndicator",
    "IGDPlusIndicator",
    "EpsilonAdditiveIndicator",
    "EpsilonMultiplicativeIndicator",
    "AvgHausdorffIndicator",
    "HVApproxIndicator",
    "WHVRectIndicator",
]
