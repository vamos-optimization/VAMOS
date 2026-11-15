"""Which rows an update of a few rows keeps, computed by insertion rather than by sorting the archive."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

BoolArray: TypeAlias = NDArray[np.bool_]

# Updates of at most this many rows into clean contents are inserted: each row costs a scan of the
# archive, so a large batch is cheaper to merge and sort.
INSERTION_ROWS = 16
# Up to this many dropped rows, the rows after them are shifted by blocks rather than gathered.
_SHIFTED_DROPS = 16


def _all_columns(F: NDArray[np.float64], row: NDArray[np.float64], compare: Callable[..., Any]) -> BoolArray:
    """``np.all(compare(F, row), axis=1)``, one column at a time, which is faster for few objectives."""
    out = np.asarray(compare(F[:, 0], row[0]))
    for j in range(1, F.shape[1]):
        out &= compare(F[:, j], row[j])
    return out


def insertion(archived_F: NDArray[np.float64], new_F: NDArray[np.float64]) -> tuple[BoolArray, BoolArray]:
    """The archived rows that the new rows dominate, and the new rows that the update keeps.

    ``archived_F`` must be mutually non-dominated with distinct rows, and every value finite. The
    update keeps a row when no row of the archive and the new rows together dominates it and no
    earlier row equals it, which is what moocore's ``is_nondominated`` and the fallback non-dominated
    filter followed by exact deduplication select for finite values.
    """
    dominated = np.zeros(archived_F.shape[0], dtype=bool)
    accepted = np.zeros(new_F.shape[0], dtype=bool)
    for j in range(new_F.shape[0]):
        row = new_F[j]
        if _all_columns(archived_F, row, np.less_equal).any():
            # Dominated by, or equal to, an archived row, which no row that it dominates could then dominate.
            continue
        # No archived row is <= row, so the archived rows >= row are strictly worse somewhere: dominated.
        dominated |= _all_columns(archived_F, row, np.greater_equal)
        if new_F.shape[0] == 1:
            accepted[j] = True
            continue
        no_worse = np.all(new_F <= row, axis=1)
        better = np.any(new_F < row, axis=1)
        accepted[j] = not bool((no_worse & better).any()) and not bool((no_worse[:j] & ~better[:j]).any())
    return dominated, accepted


def drop_rows(arrays: Sequence[NDArray[Any]], size: int, dropped: NDArray[np.intp]) -> int:
    """Remove the rows ``dropped`` (increasing, below ``size``) from the first ``size`` rows of each array.

    The other rows keep their order and move up, in place. Returns the number of rows left.
    """
    if dropped.size > _SHIFTED_DROPS:
        kept = np.ones(size, dtype=bool)
        kept[dropped] = False
        rows = np.flatnonzero(kept)
        for array in arrays:
            array[: rows.size] = array[rows]
        return int(rows.size)
    bounds = [*dropped.tolist(), size]
    write = bounds[0]
    for start, stop in zip(bounds[:-1], bounds[1:]):
        length = stop - start - 1
        if length > 0:
            for array in arrays:
                array[write : write + length] = array[start + 1 : stop]
            write += length
    return write


__all__ = ["INSERTION_ROWS", "drop_rows", "insertion"]
