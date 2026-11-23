"""The loops of the HPA analysis that Numba compiles, when it is installed.

Each kernel does, element by element and in the same order, the additions, subtractions, multiplications, divisions
and comparisons of the NumPy code it replaces in :mod:`.wing`, so that it gives the same bits. The powers and
trigonometric functions stay in NumPy and Python, and the kernels take their values: Numba computes ``x ** 2`` as
``x * x``, which is not always the C library's ``pow(x, 2.0)`` that NumPy's scalars and Python's floats use (on
Windows they differ in about one square in two thousand). Without Numba, :mod:`.wing` runs its NumPy code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

Array = np.ndarray[Any, Any]


def beam_column(L: Array, H: Array, F: Array, T: Array, EI: Array, M: Array, S: Array) -> None:
    """Fill ``M`` and ``S`` with the beam-column recurrence of :func:`.wing.moment_with_axial_force`.

    ``H[i]`` is ``(0.5 * L[i]) ** 2``, computed by the caller as the reference code computes it.
    """
    for i in range(L.size - 1, -1, -1):
        TL_EI = T[i] * H[i] / EI[i]
        S[i] = (F[i] + (1.0 - TL_EI) * S[i + 1] - T[i] * L[i] * M[i + 1] / EI[i]) / (1.0 + TL_EI)
        M[i] = (0.5 * L[i] * F[i] + (1.0 - TL_EI) * M[i + 1] + L[i] * S[i + 1]) / (1.0 + TL_EI)


def ply_weights(
    y0: Array,
    half_diameter: Array,
    segment: Array,
    y_div: Array,
    plies: Array,
    phi: Array,
    span: float,
    density: float,
    coef: float,
    beam_weight: Array,
    wing_weight: Array,
) -> None:
    """Add the weight of every ply to ``beam_weight`` and ``wing_weight`` (:meth:`.wing.Wing.compute_wing_weight`).

    ``plies`` has one row per ply, ``[axis, half width angle, root, tip, thickness]``, of segment ``segment``;
    ``phi`` is the half width angle in radians.
    """
    n = y0.size
    for p in range(plies.shape[0]):
        i = segment[p]
        start, end = y_div[i], y_div[i + 1]
        root, tip, thickness = plies[p, 2], plies[p, 3], plies[p, 4]
        low = root * (end - start) + start
        high = tip * (end - start) + start
        largest = 0.0
        for k in range(n):
            if y0[k] >= low and y0[k] <= high:
                r = half_diameter[k]
                if r > largest:
                    largest = r
                length = y0[k + 1] - y0[k] if k < n - 1 else 0.0
                wing_weight[k] += density * 2.0 * phi[p] * r * thickness * 0.5 * span * length * coef
        beam_weight[i] += density * 2.0 * phi[p] * largest * thickness * 0.5 * span * (tip - root) * (end - start)


def flange_stiffness(
    y0: Array,
    half_diameter: Array,
    cubed_half_diameter: Array,
    low: float,
    high: float,
    flange: int,
    Ex: float,
    inertia: float,
    area: float,
    thickness: float,
    EI: Array,
    EA: Array,
    EI_ply: Array,
    EA_ply: Array,
) -> None:
    """Add the stiffness of one flange ply between ``low`` and ``high`` (:meth:`.wing.Wing.stiffness`).

    ``inertia`` is ``(phi + sin(phi)) * thickness`` and ``area`` is ``Ex * 2 * phi``, as the reference code
    multiplies them; ``cubed_half_diameter`` is ``half_diameter ** 3`` computed by NumPy.
    """
    for k in range(y0.size):
        if y0[k] >= low and y0[k] < high:
            Iy = inertia * cubed_half_diameter[k]
            EA_k = area * half_diameter[k] * thickness
            EI[k] += Ex * Iy
            EA[k] += EA_k
            EI_ply[k, flange] += Ex * Iy
            EA_ply[k, flange] += EA_k


def integrate(values: Array, y0: Array, span: float, out: Array) -> None:
    """``out`` = the cumulative trapezoidal integral of :func:`.wing._integrate`, summed as NumPy's ``cumsum``."""
    out[0] = 0.0
    for k in range(1, values.size):
        out[k] = out[k - 1] + 0.5 * (values[k] + values[k - 1]) * (y0[k] - y0[k - 1]) * span * 0.5


def cut_range(
    moment: Array,
    moment_zerolift: Array,
    axial_force: Array,
    EI: Array,
    EA: Array,
    EI_flange: Array,
    EA_flange: Array,
    diameter: Array,
    ratio_flight: float,
    ratio_parked: float,
    inside: Array,
    strain_max: float,
) -> tuple[int, int]:
    """The grid indices of the root and tip of a flange cut (:func:`.plies._cut`), before their adjustments."""
    n = inside.size
    i_min = -1
    i_max = -1
    first = -1
    last = -1
    for k in range(n):
        if not inside[k]:
            continue
        if i_min < 0:
            i_min = k
        i_max = k
        strain = np.abs(moment[k] / (EI[k] - EI_flange[k]) * 0.5 * diameter[k]) + axial_force[k] / (EA[k] - EA_flange[k])
        parked = np.abs(moment_zerolift[k] / (EI[k] - EI_flange[k]) * 0.5 * diameter[k])
        flight = strain / ratio_flight
        ground = parked / ratio_parked
        # NumPy's maximum: NaN when either is NaN.
        largest = flight if flight >= ground else ground
        if np.isnan(flight) or np.isnan(ground):
            largest = np.nan
        if largest >= strain_max:
            if first < 0:
                first = k
            last = k
    if i_min < 0:  # numpy.argmax of no True is 0, from either end
        i_min, i_max = 0, n - 1
    if first < 0:
        return i_min, i_min
    return min(max(first, i_min), i_max), min(max(last + 1, i_min), i_max)


@dataclass(frozen=True)
class Kernels:
    beam_column: Callable[..., None]
    ply_weights: Callable[..., None]
    flange_stiffness: Callable[..., None]
    integrate: Callable[..., None]
    cut_range: Callable[..., tuple[int, int]]


_ENABLED = [True]


def set_enabled(enabled: bool) -> None:
    """Use the compiled kernels when Numba is installed (the default), or always the NumPy code."""
    _ENABLED[0] = bool(enabled)


@lru_cache(maxsize=1)
def _compile() -> Kernels | None:
    try:
        from numba import njit
    except ImportError:
        return None
    jit = njit(cache=True, error_model="numpy")
    return Kernels(jit(beam_column), jit(ply_weights), jit(flange_stiffness), jit(integrate), jit(cut_range))


def kernels() -> Kernels | None:
    """The compiled kernels, or ``None`` without Numba or when they are disabled."""
    return _compile() if _ENABLED[0] else None


__all__ = ["Kernels", "beam_column", "cut_range", "flange_stiffness", "integrate", "kernels", "ply_weights", "set_enabled"]
