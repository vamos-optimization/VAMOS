"""Types and helpers shared by the RCM problems."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

Array = np.ndarray[Any, Any]
# A problem's formula: the rows of ``x`` are the variables; it returns the objectives and jMetal's
# constraints, ``>= 0`` feasible.
Formula = Callable[[Array], tuple[list[Array], list[Array]]]
# formula, lower bounds, upper bounds, number of objectives, number of constraints
Entry = tuple[Formula, tuple[float, ...], tuple[float, ...], int, int]

# jMetal relaxes each equality constraint h(x) = 0 of the suite to |h(x)| <= EQUALITY_TOLERANCE.
EQUALITY_TOLERANCE = 1e-4


def java_round(x: Array) -> Array:
    """Java's ``Math.round``: the nearest integer, halves rounded up, as ``floor(x + 1/2)`` without its rounding error."""
    lower = np.floor(x)
    return np.asarray(lower + (x - lower >= 0.5))


def relaxed_equality(h: Array) -> Array:
    """jMetal's constraint for the equality ``h = 0``: ``1e-4 - |h|``, ``>= 0`` feasible."""
    return np.asarray(EQUALITY_TOLERANCE - np.abs(h))


__all__ = ["Array", "EQUALITY_TOLERANCE", "Entry", "Formula", "java_round", "relaxed_equality"]
