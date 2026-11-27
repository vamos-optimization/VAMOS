"""The helpers that the kernels of :mod:`._wing_kernels` and :mod:`._ply_kernels` share (:mod:`._kernels`)."""

from __future__ import annotations

from typing import Any

import numpy as np

Array = np.ndarray[Any, Any]

#: The block of NumPy's pairwise summation, ``PW_BLOCKSIZE`` in ``numpy/_core/src/umath/loops_utils.h.src``.
BLOCK = 128


def block_sum(a: Array, n: int) -> float:
    """``numpy.sum(a[:n])`` for ``n <= BLOCK``: ``0.0`` plus NumPy's pairwise sum, which is one block up to 128."""
    res: float = -0.0
    if n < 8:
        for i in range(n):
            res += a[i]
        return 0.0 + res
    r0, r1, r2, r3, r4, r5, r6, r7 = a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
    i = 8
    while i < n - n % 8:
        r0 += a[i]
        r1 += a[i + 1]
        r2 += a[i + 2]
        r3 += a[i + 3]
        r4 += a[i + 4]
        r5 += a[i + 5]
        r6 += a[i + 6]
        r7 += a[i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[i]
        i += 1
    return 0.0 + res


def first_at_least(y0: Array, value: float) -> int:
    """The first index of ``y0 >= value`` (``y0`` ascends), or ``y0.size`` when there is none (also for NaN)."""
    low, high = 0, y0.size
    while low < high:
        mid = (low + high) // 2
        if y0[mid] >= value:
            high = mid
        else:
            low = mid + 1
    return low


def count_below(y0: Array, value: float) -> int:
    """The number of ``y0 < value`` (``y0`` ascends): they are the first ones; none for NaN."""
    low, high = 0, y0.size
    while low < high:
        mid = (low + high) // 2
        if y0[mid] < value:
            low = mid + 1
        else:
            high = mid
    return low


def count_at_most(y0: Array, value: float) -> int:
    """The number of ``y0 <= value`` (``y0`` ascends): they are the first ones; none for NaN."""
    low, high = 0, y0.size
    while low < high:
        mid = (low + high) // 2
        if y0[mid] <= value:
            low = mid + 1
        else:
            high = mid
    return low


__all__ = ["BLOCK", "block_sum", "count_at_most", "count_below", "first_at_least"]
