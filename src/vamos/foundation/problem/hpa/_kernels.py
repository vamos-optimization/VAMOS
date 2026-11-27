"""The loops of the HPA analysis that Numba compiles, when it is installed.

Each kernel does, element by element and in the same order, the additions, subtractions, multiplications, divisions
and comparisons of the NumPy code it replaces in :mod:`.wing` and :mod:`.plies`, so that it gives the same bits. The
powers and trigonometric functions stay in NumPy and Python, and the kernels take their values: Numba computes
``x ** 2`` as ``x * x``, which is not always the C library's ``pow(x, 2.0)`` that NumPy's scalars and Python's floats
use (on Windows they differ in about one square in two thousand). Where the NumPy code sums with ``numpy.sum``, the
kernels add in NumPy's pairwise order (:func:`block_sum`); where it accumulates (``cumsum``) or adds arrays in turn,
they add in turn. A mask of the NumPy code is a range of the spanwise grid, which ascends: the kernels find its ends
by bisection, with the mask's comparisons, and skip the zeros that the NumPy code adds outside it, which change no
sum of weights or stiffnesses (none is ``-0.0``). Without Numba, :mod:`.wing` runs its NumPy code.

The kernels of the wing's analysis are in :mod:`._wing_kernels`, those of the ply sizing in :mod:`._ply_kernels`;
the helpers they share in :mod:`._kernel_helpers`. The helpers (the functions that are no field of :class:`Kernels`) are compiled into
the kernels that call them; Numba's cache of a kernel follows the changes of its own module only, so a change of a
helper of another module needs the cached kernels (``__pycache__/*.nbi`` and ``*.nbc``) removed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ._kernel_helpers import BLOCK, block_sum, count_at_most, count_below, first_at_least

SHARED_HELPERS: tuple[Callable[..., Any], ...] = (block_sum, first_at_least, count_below, count_at_most)


@dataclass(frozen=True)
class Kernels:
    wing_weight: Callable[..., None]
    local_lift: Callable[..., None]
    zerolift: Callable[..., None]
    flight: Callable[..., float]
    lifting_line: Callable[..., None]
    induced_angles: Callable[..., None]
    entire_plies: Callable[..., None]
    stiffness: Callable[..., None]
    segment_peaks: Callable[..., tuple[int, int, int, float, int, float]]
    cut_flanges: Callable[..., int]
    widen_trials: Callable[..., int]
    profile_drag: Callable[..., int]


_ENABLED = [True]


def set_enabled(enabled: bool) -> None:
    """Use the compiled kernels when Numba is installed (the default), or always the NumPy code."""
    _ENABLED[0] = bool(enabled)


@lru_cache(maxsize=1)
def _compile() -> Kernels | None:
    try:
        from numba import extending, njit
    except ImportError:
        return None
    from ._ply_kernels import adjust, align, cut, cut_flanges, segment_peaks, widen_trials
    from ._wing_kernels import (
        beam_column,
        entire_plies,
        flight,
        grid_interval,
        induced_angles,
        integrate,
        lifting_line,
        local_lift,
        moment,
        profile_drag,
        stiffness,
        wing_weight,
        zerolift,
    )

    for helper in (*SHARED_HELPERS, integrate, moment, beam_column, grid_interval, adjust, cut, align):
        extending.register_jitable(error_model="numpy")(helper)
    jit = njit(cache=True, error_model="numpy")
    compiled = (
        wing_weight,
        local_lift,
        zerolift,
        flight,
        lifting_line,
        induced_angles,
        entire_plies,
        stiffness,
        segment_peaks,
        cut_flanges,
        widen_trials,
        profile_drag,
    )
    return Kernels(*(jit(kernel) for kernel in compiled))


def kernels() -> Kernels | None:
    """The compiled kernels, or ``None`` without Numba or when they are disabled."""
    return _compile() if _ENABLED[0] else None


__all__ = ["BLOCK", "Kernels", "block_sum", "count_at_most", "count_below", "first_at_least", "kernels", "set_enabled"]
