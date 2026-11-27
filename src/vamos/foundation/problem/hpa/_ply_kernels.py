"""The kernels of the sizing of the flange plies (:mod:`.plies`), under the rules of :mod:`._kernels`."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._kernel_helpers import block_sum, count_below, first_at_least

Array = np.ndarray[Any, Any]


def segment_peaks(y0: Array, start: float, end: float, strain: Array, strain_zerolift: Array) -> tuple[int, int, int, float, int, float]:
    """The grid range ``[a, b)`` of a segment, and ``numpy.argmax`` and ``max`` of the absolute strains on it.

    As NumPy's, the maximum is NaN, at the first NaN, when there is one. For an empty range, the peaks are 0.
    """
    a = first_at_least(y0, start)
    b = max(a, count_below(y0, end))
    peaks = np.zeros(2)
    where = np.zeros(2, np.int64)
    for s in range(2):
        values = strain if s == 0 else strain_zerolift
        for k in range(a, b):
            value = np.abs(values[k])
            if np.isnan(value):
                peaks[s], where[s] = value, k - a
                break
            if k == a or value > peaks[s]:
                peaks[s], where[s] = value, k - a
    return a, b, where[0], peaks[0], where[1], peaks[1]


def adjust(y0: Array, i_root: int, i_tip: int, start: float, end: float, limits: Array) -> tuple[float, float]:
    """:func:`.plies._adjust`: the root and tip at grid indices ``i_root`` and ``i_tip``; ``limits`` is updated."""
    n = y0.size
    root = y0[i_root]
    tip = y0[i_tip]
    if i_root > 0 and y0[i_root - 1] < start:
        root = start
    if i_tip > 0 and y0[i_tip - 1] < start:
        tip = start
    if i_tip + 1 < n and y0[i_tip + 1] >= end:
        tip = end
    if tip - root > 0:
        if root > limits[0]:
            root = limits[0]
        else:
            limits[0] = root
        if tip < limits[1]:
            tip = limits[1]
        else:
            limits[1] = tip
    return root, tip


def cut(
    a: int,
    b: int,
    moment_flight: Array,
    moment_zerolift: Array,
    axial_force: Array,
    EI: Array,
    EA: Array,
    EI_ply: Array,
    EA_ply: Array,
    f: int,
    diameter: Array,
    ratio_flight: float,
    ratio_parked: float,
    strain_max: float,
) -> tuple[int, int]:
    """The grid indices of the root and tip of a flange cut (:func:`.plies._cut`) on its range ``[a, b)``."""
    n = EI.size
    first = -1
    last = -1
    for k in range(a, b):
        strain = np.abs(moment_flight[k] / (EI[k] - EI_ply[k, f]) * 0.5 * diameter[k]) + axial_force[k] / (EA[k] - EA_ply[k, f])
        parked = np.abs(moment_zerolift[k] / (EI[k] - EI_ply[k, f]) * 0.5 * diameter[k])
        flight_ = strain / ratio_flight
        ground = parked / ratio_parked
        # NumPy's maximum: NaN when either is NaN.
        largest = flight_ if flight_ >= ground else ground
        if np.isnan(flight_) or np.isnan(ground):
            largest = np.nan
        if largest >= strain_max:
            if first < 0:
                first = k
            last = k
    i_min, i_max = a, b - 1
    if b <= a:  # numpy.argmax of no True is 0, from either end
        i_min, i_max = 0, n - 1
    if first < 0:
        return i_min, i_min
    return min(max(first, i_min), i_max), min(max(last + 1, i_min), i_max)


def cut_flanges(
    plies: Array,
    j_start: int,
    first: bool,
    limits: Array,
    start: float,
    end: float,
    y0: Array,
    moment_flight: Array,
    moment_zerolift: Array,
    axial_force: Array,
    EI: Array,
    EA: Array,
    EI_ply: Array,
    EA_ply: Array,
    diameter: Array,
    ratio_flight: float,
    ratio_parked: float,
    strain_max: float,
) -> int:
    """Cut the flanges ``j_start, ..., 4`` of a segment, as the loop of :func:`.plies.optimize_plies` does.

    It stops at the first flange that is needed, ``first`` still true, when more than four flanges were cut off
    before it: it returns its index, so that the caller updates the wing and cuts that flange again with ``first``
    false. Otherwise, it aligns the flanges (:func:`.plies._align`) and returns -1. ``limits`` is updated.
    """
    n = y0.size
    rows = plies.shape[0]
    for j in range(j_start, 3, -1):
        f = j - 4
        a = first_at_least(y0, plies[j, 2] * (end - start) + start)
        b = max(a, count_below(y0, plies[j, 3] * (end - start) + start))
        i_root, i_tip = cut(
            a, b, moment_flight, moment_zerolift, axial_force, EI, EA, EI_ply, EA_ply, f, diameter, ratio_flight,
            ratio_parked, strain_max,
        )  # fmt: skip
        root, tip = adjust(y0, i_root, i_tip, start, end, limits)
        if first and tip > root:
            first = False
            if rows - 1 - j > 4:
                return j
        plies[j, 2] = (root - start) / (end - start)
        plies[j, 3] = (tip - start) / (end - start)
        for k in range(a, b):
            if y0[k] < root or y0[k] >= tip:
                EI[k] -= EI_ply[k, f]
                EA[k] -= EA_ply[k, f]
        EI[n - 1] = EI[n - 2]
        EA[n - 1] = EA[n - 2]
    align(plies, start, end, y0, EI, EA, EI_ply, EA_ply)
    return -1


def align(plies: Array, start: float, end: float, y0: Array, EI: Array, EA: Array, EI_ply: Array, EA_ply: Array) -> None:
    """:func:`.plies._align`: one end of every flange to an end of its pipe, restoring the stiffness it adds."""
    n = y0.size
    rows = plies.shape[0]
    roots = np.empty(rows)
    tips = np.empty(rows)
    count = 0
    for j in range(rows):
        if plies[j, 3] > plies[j, 2]:
            roots[count] = plies[j, 2]
            tips[count] = 1.0 - plies[j, 2]
            count += 1
    sum_root = block_sum(roots, count)
    sum_tip = block_sum(tips, count)
    if not sum_root * sum_tip > 0:
        return
    a = first_at_least(y0, start)
    b = max(a, count_below(y0, end))
    to_root = sum_root <= sum_tip
    for j in range(rows - 1, 3, -1):
        if to_root:
            if not (plies[j, 3] > plies[j, 2] and plies[j, 2] > 0):
                continue
            root = plies[j, 2] * (end - start) + start
            for k in range(a, b):
                if y0[k] < root:
                    EI[k] += EI_ply[k, j - 4]
                    EA[k] += EA_ply[k, j - 4]
        else:
            if not (plies[j, 3] > plies[j, 2] and plies[j, 3] < 1):
                continue
            tip = plies[j, 3] * (end - start) + start
            for k in range(a, b):
                if y0[k] >= tip:
                    EI[k] += EI_ply[k, j - 4]
                    EA[k] += EA_ply[k, j - 4]
        EI[n - 1] = EI[n - 2]
        EA[n - 1] = EA[n - 2]
    for j in range(rows):
        if plies[j, 3] > plies[j, 2]:
            if to_root:
                plies[j, 2] = 0.0
            else:
                plies[j, 3] = 1.0


def widen_trials(
    dphi: Array,
    sin_trial: Array,
    sin_phi: Array,
    Ex: Array,
    tr3_1: Array,
    tr_1: Array,
    tr3_2: Array,
    M1: float,
    AF1: float,
    M2: float,
    EI1: float,
    EA1: float,
    EI2: float,
    diameter1: float,
    diameter2: float,
    ratio_flight: float,
    ratio_parked: float,
    strain_max: float,
) -> int:
    """The first trial widening of :func:`.plies._widen` whose strains fit, or -1.

    Trial ``r`` widens the flanges by ``dphi[r]`` radians; ``sin_trial[r]`` is ``sin(phi + dphi[r])`` and
    ``sin_phi`` is ``sin(phi)``. The sums over the flanges are NumPy's (:func:`block_sum`).
    """
    m = sin_phi.size
    terms_1 = np.empty(m)
    terms_A = np.empty(m)
    terms_2 = np.empty(m)
    for r in range(dphi.size):
        for c in range(m):
            gain = dphi[r] + sin_trial[r, c] - sin_phi[c]
            terms_1[c] = Ex[c] * (gain * tr3_1[c])
            terms_A[c] = Ex[c] * (2.0 * dphi[r] * tr_1[c])
            terms_2[c] = Ex[c] * (gain * tr3_2[c])
        strain1 = (np.abs(M1 / (EI1 + block_sum(terms_1, m)) * 0.5 * diameter1) + AF1 / (EA1 + block_sum(terms_A, m))) / ratio_flight
        strain2 = np.abs(M2 / (EI2 + block_sum(terms_2, m)) * 0.5 * diameter2) / ratio_parked
        # Python's max(strain1, strain2) is strain2 only when strain2 > strain1.
        if (strain2 if strain2 > strain1 else strain1) <= strain_max:
            return r
    return -1
