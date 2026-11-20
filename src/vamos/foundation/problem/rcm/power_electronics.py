"""RCM30-RCM35: synchronous optimal pulse-width modulation (SOPWM) of 3- to 13-level inverters.

As jMetal's ``AbstractRCMSelectiveHarmonicElimination``: the variables are switching angles in degrees, in
``[0, 90]``. f1 is the weighted distortion of the harmonics of orders 5 to 97 that are not multiples of 2 or
3, f2 the squared distance of the fundamental from the modulation index, and the constraints keep the angles
increasing, ``x_i - x_{i+1} + 1e-6 <= 0``. The problems differ in their number of angles, modulation index and
sign pattern. Sums run in jMetal's order: the evaluation is vectorized over the harmonics and the angles, and each
sum is a running sum from the first term to the last (``np.cumsum``), which adds the terms as jMetal's loops do.
"""

from __future__ import annotations

import functools

import numpy as np

from vamos.foundation.problem.rcm._common import Array, Entry

_HARMONIC_ORDERS = (
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    25,
    29,
    31,
    35,
    37,
    41,
    43,
    47,
    49,
    53,
    55,
    59,
    61,
    65,
    67,
    71,
    73,
    77,
    79,
    83,
    85,
    91,
    95,
    97,
)

# The constraints keep consecutive angles at least this far apart (degrees).
ORDER_THRESHOLD = 1e-6

# Modulation index and sign pattern of each problem; the number of angles is the length of the pattern.
_PATTERNS: dict[str, tuple[float, str]] = {
    "rcm30": (0.32, "+-+-+-+-+-+-+-+-+-+-+-+-+"),
    "rcm31": (0.32, "+-++-+-+--+-++-+-+--+-++-"),
    "rcm32": (0.36, "+-+++---++--+++---++--+++"),
    "rcm33": (0.32, "++++-+---+--++++-+---+--++++-+"),
    "rcm34": (0.3333, "+-+++---++++--+---++++-++--+--"),
    "rcm35": (0.32, "+++-+-+-++++----+-+-++++---+-+"),
}


# Rows evaluated together: the harmonics of a block of rows take 31 x 30 values per row.
_BLOCK = 1024


@functools.cache
def _harmonic_weights() -> tuple[Array, float]:
    """``k^4`` of each harmonic order, as a column, and ``sum 1/k^4``, added from the first order to the last."""
    fourth = [float(k) ** 4 for k in _HARMONIC_ORDERS]
    weights = 0.0
    for value in fourth:
        weights = weights + 1.0 / value
    column = np.array(fourth)[:, None]
    column.setflags(write=False)
    return column, weights


def _running_total(terms: Array, axis: int) -> Array:
    """The sum of ``terms`` along ``axis``, added from the first term to the last as a loop does."""
    return np.asarray(np.take(np.cumsum(terms, axis=axis), -1, axis=axis))


def _sopwm(key: str, x: Array) -> tuple[list[Array], list[Array]]:
    modulation_index, pattern = _PATTERNS[key]
    signs = np.array([1.0 if symbol == "+" else -1.0 for symbol in pattern])[:, None]
    fourth, weights = _harmonic_weights()
    orders = np.array(_HARMONIC_ORDERS, dtype=float)[:, None, None]
    angles = x * np.pi / 180.0
    fundamental = _running_total(signs * np.cos(angles), axis=0)
    distortion = np.empty(x.shape[1])
    for start in range(0, x.shape[1], _BLOCK):
        block = angles[:, start : start + _BLOCK]
        # harmonic k = sum_i s_i cos(k x_i); distortion = sum_k harmonic^2 / k^4
        harmonics = _running_total(signs * np.cos(orders * block), axis=1)
        distortion[start : start + _BLOCK] = _running_total(harmonics * harmonics / fourth, axis=0)
    f1 = np.sqrt(distortion) / np.sqrt(weights)
    f2 = (fundamental - modulation_index) ** 2
    return [f1, f2], list(-(x[:-1] - x[1:] + ORDER_THRESHOLD))


def _rcm30(x: Array) -> tuple[list[Array], list[Array]]:
    return _sopwm("rcm30", x)


def _rcm31(x: Array) -> tuple[list[Array], list[Array]]:
    return _sopwm("rcm31", x)


def _rcm32(x: Array) -> tuple[list[Array], list[Array]]:
    return _sopwm("rcm32", x)


def _rcm33(x: Array) -> tuple[list[Array], list[Array]]:
    return _sopwm("rcm33", x)


def _rcm34(x: Array) -> tuple[list[Array], list[Array]]:
    return _sopwm("rcm34", x)


def _rcm35(x: Array) -> tuple[list[Array], list[Array]]:
    return _sopwm("rcm35", x)


PROBLEMS: dict[str, Entry] = {
    "rcm30": (_rcm30, (0.0,) * 25, (90.0,) * 25, 2, 24),
    "rcm31": (_rcm31, (0.0,) * 25, (90.0,) * 25, 2, 24),
    "rcm32": (_rcm32, (0.0,) * 25, (90.0,) * 25, 2, 24),
    "rcm33": (_rcm33, (0.0,) * 30, (90.0,) * 30, 2, 29),
    "rcm34": (_rcm34, (0.0,) * 30, (90.0,) * 30, 2, 29),
    "rcm35": (_rcm35, (0.0,) * 30, (90.0,) * 30, 2, 29),
}

__all__ = ["ORDER_THRESHOLD", "PROBLEMS"]
