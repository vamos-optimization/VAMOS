"""Exact hypervolume for minimization fronts, with MooCore or libhv when installed.

Every path follows one contract. Points and the reference point must be finite
(``ValueError`` otherwise), the reference point is required, and points that do
not lie inside the reference box (``point <= reference`` in every objective)
are ignored: they add no volume and their contribution is 0, as in MooCore.
Contributions follow MooCore's ``ignore_dominated`` semantics: a dominated or
duplicated point contributes 0, and the others are measured against the
non-dominated points only.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _is_finite_array(arr: FloatArray) -> bool:
    return bool(np.isfinite(arr).all())


def compute_hypervolume(F: ArrayLike, ref_point: Sequence[float]) -> float:
    """Compute (exact) hypervolume for 2D minimization fronts.

    Parameters
    - F: array-like shape (n_points, 2) of objective values (minimization)
    - ref_point: sequence of length 2 with reference point (worst values)

    Returns
    - hypervolume (float)

    Notes
    - This implementation supports 2-objective minimization problems only.
    - Points outside the reference box are ignored, as in :func:`hypervolume`.
    - For higher-dimensional hypervolume, use :func:`hypervolume`.
    """
    points = np.asarray(F, dtype=float)
    ref = np.asarray(ref_point, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("compute_hypervolume currently supports 2D fronts only")

    if not _is_finite_array(points) or not _is_finite_array(ref):
        raise ValueError("F and ref_point must contain finite numbers")

    # Filter points inside the reference box (dominated by ref in ALL objectives)
    mask = np.all(points <= ref, axis=1)
    pts = points[mask]

    if pts.size == 0:
        return 0.0

    # Remove points that are dominated by others (keep Pareto front)
    # For 2D minimization: sort by f1 ascending, then keep those with strictly decreasing f2
    idx = np.argsort(pts[:, 0])
    sorted_pts = pts[idx]

    # Sweep from left (small f1) to right, keeping non-dominated in f2
    pareto: list[tuple[float, float]] = []
    best_f2 = np.inf
    for x, y in sorted_pts:
        if y < best_f2:
            pareto.append((x, y))
            best_f2 = y

    pareto_arr = np.asarray(pareto, dtype=float)

    # Compute hypervolume as sum of rectangles between successive Pareto points
    hv = 0.0
    prev_f1 = ref[0]
    # iterate pareto points in reverse (from worst f1 to best) to form rectangles
    for x, y in pareto_arr[::-1]:
        width = prev_f1 - x
        height = ref[1] - y
        if width > 0 and height > 0:
            hv += width * height
        prev_f1 = x

    return float(max(hv, 0.0))


# General hypervolume utilities (moocore/libhv fallback).
_MOOCORE = None
_LIBHV_MODULE = None
_LIBHV_HV_FN = None
_LIBHV_CLASS = None
_OPTIONAL_LOADED = False


def _load_optional_backends() -> None:
    global _MOOCORE, _LIBHV_MODULE, _LIBHV_HV_FN, _LIBHV_CLASS, _OPTIONAL_LOADED
    if _OPTIONAL_LOADED:
        return
    _OPTIONAL_LOADED = True
    try:  # Prefer the MooCore C backend when available.
        import moocore as moocore_module
    except ImportError:  # pragma: no cover - optional dependency
        moocore_module = None
    _MOOCORE = moocore_module

    try:  # pragma: no cover - optional dependency
        import libhv as libhv_module
    except ImportError:
        libhv_module = None

    if libhv_module is None:
        _LIBHV_MODULE = None
        _LIBHV_HV_FN = None
        _LIBHV_CLASS = None
        return

    hv_fn = getattr(libhv_module, "hypervolume", None)
    hv_cls = getattr(libhv_module, "HyperVolume", None)
    if hv_fn is None and hv_cls is None:
        _LIBHV_MODULE = None
        _LIBHV_HV_FN = None
        _LIBHV_CLASS = None
        return

    _LIBHV_MODULE = libhv_module
    _LIBHV_HV_FN = hv_fn
    _LIBHV_CLASS = hv_cls


def hypervolume(points: ArrayLike, reference_point: ArrayLike, *, allow_ref_expand: bool = False) -> float:
    """Exact hypervolume of ``points`` (minimization) bounded by ``reference_point``.

    Points outside the reference box are ignored. ``allow_ref_expand`` is accepted
    for compatibility only: it is ignored with a ``FutureWarning`` and will be removed
    in VAMOS 2.0.
    """
    data = _validated_points(points)
    ref = _validate_reference_point(data, reference_point, allow_ref_expand=allow_ref_expand)
    inside = data[_inside_mask(data, ref)]
    if inside.shape[0] == 0:
        return 0.0
    _load_optional_backends()
    if _MOOCORE is not None:
        return float(_MOOCORE.hypervolume(np.ascontiguousarray(inside, dtype=np.float64), ref))
    if _LIBHV_MODULE is not None:
        return float(_hypervolume_with_libhv(inside, ref))
    return _hypervolume_impl(inside, ref)


def hypervolume_contributions(
    points: ArrayLike,
    reference_point: ArrayLike,
    *,
    allow_ref_expand: bool = False,
) -> FloatArray:
    """Exclusive hypervolume of each point, with MooCore's ``ignore_dominated`` semantics.

    A point outside the reference box, a dominated point and a duplicated point
    contribute 0; the other contributions are measured against the non-dominated
    points inside the box. ``allow_ref_expand`` is ignored with a ``FutureWarning``.
    """
    data = _validated_points(points)
    ref = _validate_reference_point(data, reference_point, allow_ref_expand=allow_ref_expand)
    contributions = np.zeros(data.shape[0], dtype=float)
    mask = _inside_mask(data, ref)
    if not mask.any():
        return contributions
    inside = data if mask.all() else data[mask]
    _load_optional_backends()
    if _MOOCORE is not None:
        values = np.asarray(_MOOCORE.hv_contributions(np.ascontiguousarray(inside, dtype=np.float64), ref), dtype=float)
    elif inside.shape[1] == 2:
        values = _hypervolume_contributions_2d(inside, ref)
    else:
        values = _hypervolume_contributions_generic(inside, ref)
    if mask.all():
        return values
    contributions[mask] = values
    return contributions


def _validated_points(points: ArrayLike) -> FloatArray:
    data = np.asarray(points, dtype=float)
    if data.ndim != 2:
        raise ValueError("points must be a 2D array.")
    if not _is_finite_array(data):
        raise ValueError("points must contain finite numbers.")
    return data


def _inside_mask(points: FloatArray, ref: FloatArray) -> NDArray[np.bool_]:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    return np.asarray(np.all(points <= ref, axis=1), dtype=bool)


def _hypervolume_with_libhv(points: FloatArray, ref: FloatArray) -> float:
    data = np.ascontiguousarray(points, dtype=np.float64)
    ref_arr = np.asarray(ref, dtype=np.float64)
    if _LIBHV_HV_FN is not None:
        return float(_LIBHV_HV_FN(data, ref_arr))
    hv_class = _LIBHV_CLASS
    if hv_class is None:  # pragma: no cover - guarded by initialization
        raise RuntimeError("libhv is imported without HyperVolume support.")
    hv_obj = hv_class(ref_arr)
    compute = getattr(hv_obj, "compute", None) or getattr(hv_obj, "compute_from_points", None)
    if compute is None:
        raise AttributeError("libhv.HyperVolume object lacks a compute method.")
    return float(compute(data))


def _hypervolume_impl(points: FloatArray, reference_point: ArrayLike) -> float:
    if points.shape[0] == 0:
        return 0.0
    ref = np.asarray(reference_point, dtype=float)
    n_obj = points.shape[1]
    if n_obj == 1:
        widths = np.maximum(ref[0] - points[:, 0], 0.0)
        return float(np.max(widths))
    if n_obj == 2:
        return _hypervolume_2d(points, ref)
    if n_obj == 3:
        return _hypervolume_3d(points, ref)
    return _hypervolume_recursive(points, ref)


def _validate_reference_point(
    points: FloatArray,
    reference_point: ArrayLike,
    *,
    allow_ref_expand: bool = False,
) -> FloatArray:
    if allow_ref_expand:
        warnings.warn(
            "allow_ref_expand is ignored: points outside the reference box are left out of the hypervolume. "
            "The argument will be removed in VAMOS 2.0.",
            FutureWarning,
            stacklevel=3,
        )
    ref = np.asarray(reference_point, dtype=float)
    if ref.ndim != 1:
        raise ValueError("reference_point must be a 1D array.")
    if not _is_finite_array(ref):
        raise ValueError("reference_point must contain finite numbers.")
    if points.shape[0] and ref.shape[0] != points.shape[1]:
        raise ValueError("reference_point dimensionality mismatch.")
    return ref


def _hypervolume_2d(points: FloatArray, ref: FloatArray) -> float:
    if points.shape[0] == 0:
        return 0.0
    order = np.argsort(points[:, 0], kind="mergesort")
    sorted_points = points[order]
    widths = np.maximum(ref[0] - sorted_points[:, 0], 0.0)
    prev_f2 = np.minimum.accumulate(np.concatenate(([ref[1]], sorted_points[:-1, 1])))
    heights = np.maximum(prev_f2 - sorted_points[:, 1], 0.0)
    return float(np.sum(widths * heights))


def _hypervolume_3d(points: FloatArray, ref: FloatArray) -> float:
    order = np.argsort(points[:, 2])
    sorted_points = points[order]
    hv = 0.0
    prev_f3 = ref[2]
    for end in range(sorted_points.shape[0] - 1, -1, -1):
        f3 = sorted_points[end, 2]
        height = max(prev_f3 - f3, 0.0)
        if height <= 0.0:
            continue
        slice_points = sorted_points[: end + 1, :2]
        slab = _hypervolume_2d(slice_points, ref[:2])
        hv += slab * height
        prev_f3 = f3
    return float(hv)


def _hypervolume_recursive(points: FloatArray, ref: FloatArray) -> float:
    """
    Recursive slicing algorithm that works for >= 4 objectives.
    """
    if points.size == 0:
        return 0.0
    n_obj = points.shape[1]
    if n_obj == 1:
        widths = np.maximum(ref[0] - points[:, 0], 0.0)
        return float(np.max(widths)) if widths.size else 0.0

    order = np.argsort(points[:, n_obj - 1])
    sorted_points = points[order]

    hv = 0.0
    bound = ref[n_obj - 1]
    while sorted_points.shape[0] > 0:
        current = sorted_points[-1, n_obj - 1]
        height = bound - current
        if height > 0.0:
            reduced = sorted_points[:, : n_obj - 1]
            hv += _hypervolume_recursive(reduced, ref[: n_obj - 1]) * height
            bound = current
        sorted_points = sorted_points[:-1]
    return float(hv)


def _hypervolume_contributions_2d(points: FloatArray, ref: FloatArray) -> FloatArray:
    if points.shape[0] == 0:
        return np.empty(0, dtype=float)

    unique_points, inverse, counts = np.unique(points, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)
    unique_contribs = np.zeros(unique_points.shape[0], dtype=float)

    if unique_points.shape[0]:
        order = np.lexsort((unique_points[:, 1], unique_points[:, 0]))
        sorted_points = unique_points[order]

        prev_min = np.concatenate(([np.inf], np.minimum.accumulate(sorted_points[:-1, 1])))
        nd_sorted = sorted_points[:, 1] < prev_min
        nd_order = order[nd_sorted]
        if nd_order.size:
            nd_points = unique_points[nd_order]
            x = nd_points[:, 0]
            y = nd_points[:, 1]
            x_next = np.concatenate((x[1:], [ref[0]]))
            y_prev = np.concatenate(([ref[1]], y[:-1]))
            nd_contrib = np.maximum(x_next - x, 0.0) * np.maximum(y_prev - y, 0.0)
            unique_contribs[nd_order] = nd_contrib

    contributions = unique_contribs[inverse]
    contributions[counts[inverse] > 1] = 0.0
    return contributions


def _hypervolume_contributions_generic(points: FloatArray, ref: FloatArray) -> FloatArray:
    """Contributions for three or more objectives without MooCore, with MooCore's semantics.

    Dominated points contribute 0 and do not shrink the others' exclusive volume;
    duplicates contribute 0 because each covers the other. Rounding noise is clipped at 0.
    """
    from .pareto import pareto_filter

    contributions = np.zeros(points.shape[0], dtype=float)
    front, front_idx = pareto_filter(points, return_indices=True)
    hv_full = _hypervolume_impl(front, ref)
    for position, index in enumerate(front_idx):
        hv_without = _hypervolume_impl(np.delete(front, position, axis=0), ref)
        contributions[index] = max(hv_full - hv_without, 0.0)
    return contributions
