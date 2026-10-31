"""Compiled survival and archive-truncation loops.

Each kernel reproduces an iterative NumPy reference loop of an algorithm
(element order, tie-breaking by first index, floating-point accumulation order)
without drawing random numbers. Callers keep the NumPy reference for inputs the
kernel does not accept (for example non-finite objective values).
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._numba_ranking import _front_crowding
from .numba_ops import njit

if TYPE_CHECKING:
    from .backend import SMSEMOASurvival

FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.int64]


@njit(cache=True)
def _first_argmin(values: FloatArray, alive: NDArray[np.bool_]) -> int:
    """Index of the first minimum among alive entries; the first NaN wins like ``np.argmin``."""
    index = -1
    best = 0.0
    for j in range(values.shape[0]):
        if not alive[j]:
            continue
        value = values[j]
        if value != value:
            return j
        if index == -1 or value < best:
            index = j
            best = value
    return index


@njit(cache=True)
def _first_argmax(values: FloatArray, alive: NDArray[np.bool_]) -> int:
    """Index of the first maximum among alive entries; the first NaN wins like ``np.argmax``."""
    index = -1
    best = 0.0
    for j in range(values.shape[0]):
        if not alive[j]:
            continue
        value = values[j]
        if value != value:
            return j
        if index == -1 or value > best:
            index = j
            best = value
    return index


@njit(cache=True)
def iterative_crowding_truncation(F: FloatArray, target_size: int) -> IndexArray:
    """Remove the first least-crowded row until ``target_size`` rows remain.

    Mirrors ``while keep.size > target: keep = np.delete(keep, argmin(crowding(F[keep])))``
    for a single front with finite values.
    """
    n = F.shape[0]
    keep = np.arange(n, dtype=np.int64)
    size = n
    local_values = np.empty(n, dtype=np.float64)
    local_order = np.empty(n, dtype=np.int64)
    local_temp = np.empty(n, dtype=np.int64)
    local_dist = np.empty(n, dtype=np.float64)
    alive = np.ones(n, dtype=np.bool_)
    while size > target_size:
        _front_crowding(F, keep, 0, size, local_values, local_order, local_temp, local_dist)
        worst = _first_argmin(local_dist[:size], alive[:size])
        for pos in range(worst, size - 1):
            keep[pos] = keep[pos + 1]
        size -= 1
    return keep[:size].copy()


@njit(cache=True)
def greedy_two_nearest_selection(
    distances: FloatArray,
    remaining: IndexArray,
    best1: FloatArray,
    best2: FloatArray,
    scores: FloatArray,
    selected_count: int,
    crowd_dist: FloatArray,
) -> None:
    """AGE-MOEA greedy diversity scores: repeatedly select the remaining point with the
    largest sum of distances to its two nearest selected points and record that sum."""
    r = remaining.shape[0]
    alive = np.ones(r, dtype=np.bool_)
    b1 = best1.copy()
    b2 = best2.copy()
    sc = scores.copy()
    n_alive = r
    while n_alive > 0:
        index = _first_argmax(sc, alive)
        best = remaining[index]
        crowd_dist[best] = sc[index]
        alive[index] = False
        n_alive -= 1
        selected_count += 1
        if n_alive == 0:
            break
        if selected_count == 2:
            for j in range(r):
                if not alive[j]:
                    continue
                nd = distances[remaining[j], best]
                current = b1[j]
                # np.minimum / np.maximum propagate NaN
                if current != current or nd != nd:
                    lo = current if current != current else nd
                    hi = lo
                elif nd < current:
                    lo = nd
                    hi = current
                else:
                    lo = current
                    hi = nd
                b1[j] = lo
                b2[j] = hi
                sc[j] = lo + hi
            continue
        for j in range(r):
            if not alive[j]:
                continue
            nd = distances[remaining[j], best]
            if nd < b1[j]:
                b2[j] = b1[j]
                b1[j] = nd
            elif nd < b2[j]:
                b2[j] = nd
            sc[j] = b1[j] + b2[j]


@njit(cache=True)
def _sorted_distances(dist: FloatArray, active: IndexArray, size: int, i: int, out: FloatArray) -> None:
    """Sorted distances from active row ``i`` to the other active rows, written to ``out[: size - 1]``."""
    row = active[i]
    count = 0
    for j in range(size):
        if j == i:
            continue
        value = dist[row, active[j]]
        pos = count
        while pos > 0 and out[pos - 1] > value:
            out[pos] = out[pos - 1]
            pos -= 1
        out[pos] = value
        count += 1


@njit(cache=True)
def lexicographic_truncation(dist: FloatArray, keep: int) -> IndexArray:
    """SPEA2 truncation operator on finite distances; returns the kept rows in ascending order.

    Each step finds the active rows with the smallest nearest-neighbour distance
    and removes the one whose sorted distances to the other active rows are
    lexicographically smallest, the first such row on exact ties.
    """
    n = dist.shape[0]
    active = np.arange(n, dtype=np.int64)
    size = n
    nearest = np.empty(n, dtype=np.float64)
    best_row = np.empty(n, dtype=np.float64)
    other_row = np.empty(n, dtype=np.float64)
    while size > keep:
        smallest = np.inf
        for i in range(size):
            value = np.inf
            for j in range(size):
                if j != i and dist[active[i], active[j]] < value:
                    value = dist[active[i], active[j]]
            nearest[i] = value
            if value < smallest:
                smallest = value
        chosen = -1
        for i in range(size):
            if nearest[i] != smallest:
                continue
            if chosen < 0:
                chosen = i
                _sorted_distances(dist, active, size, i, best_row)
                continue
            _sorted_distances(dist, active, size, i, other_row)
            for pos in range(size - 1):
                if other_row[pos] != best_row[pos]:
                    if other_row[pos] < best_row[pos]:
                        chosen = i
                        best_row[: size - 1] = other_row[: size - 1]
                    break
        for i in range(chosen, size - 1):
            active[i] = active[i + 1]
        size -= 1
    return active[:size].copy()


@njit(cache=True)
def indicator_removal(fitness: FloatArray, contrib: FloatArray, target_size: int) -> IndexArray:
    """IBEA environmental selection: remove the first lowest-fitness row and add its
    indicator contribution to the fitness of the remaining rows, until ``target_size``."""
    n = fitness.shape[0]
    active = np.arange(n, dtype=np.int64)
    size = n
    active_fitness = np.empty(n, dtype=np.float64)
    alive = np.ones(n, dtype=np.bool_)
    while size > target_size:
        for i in range(size):
            active_fitness[i] = fitness[active[i]]
        worst_pos = _first_argmin(active_fitness[:size], alive[:size])
        worst = active[worst_pos]
        for i in range(size):
            if i != worst_pos:
                row = active[i]
                fitness[row] += contrib[row, worst]
            else:
                fitness[worst] += 0.0
        for pos in range(worst_pos, size - 1):
            active[pos] = active[pos + 1]
        size -= 1
    return active[:size].copy()


class NumbaSelectionKernels:
    """Compiled survival loops exposed through ``KernelBackend.selection_kernels``."""

    name = "numba"

    def iterative_crowding_truncation(self, F: FloatArray, target_size: int) -> IndexArray:
        return iterative_crowding_truncation(np.ascontiguousarray(F, dtype=np.float64), int(target_size))

    def greedy_two_nearest_selection(
        self,
        distances: FloatArray,
        remaining: NDArray[np.integer[Any]],
        best1: FloatArray,
        best2: FloatArray,
        scores: FloatArray,
        selected_count: int,
        crowd_dist: FloatArray,
    ) -> None:
        greedy_two_nearest_selection(
            distances,
            np.asarray(remaining, dtype=np.int64),
            best1,
            best2,
            scores,
            int(selected_count),
            crowd_dist,
        )

    def lexicographic_truncation(self, dist: FloatArray, keep: int) -> IndexArray:
        return lexicographic_truncation(np.ascontiguousarray(dist, dtype=np.float64), int(keep))

    def indicator_removal(self, fitness: FloatArray, contrib: FloatArray, target_size: int) -> IndexArray:
        return indicator_removal(fitness, contrib, int(target_size))

    def smsemoa_survival(
        self,
        *,
        pop_size: int,
        n_obj: int,
        n_constraints: int,
        ref_offset: float,
        ref_adaptive: bool,
        eliminate_duplicates: bool,
        check_feasibility: bool,
    ) -> SMSEMOASurvival:
        from ._numba_smsemoa import NumbaSMSEMOASurvival

        return NumbaSMSEMOASurvival(
            pop_size=pop_size,
            n_obj=n_obj,
            n_constraints=n_constraints,
            ref_offset=ref_offset,
            ref_adaptive=ref_adaptive,
            eliminate_duplicates=eliminate_duplicates,
            check_feasibility=check_feasibility,
        )


@cache
def numba_selection_kernels() -> NumbaSelectionKernels:
    """Shared compiled selection kernels (created on first use, not at import)."""
    return NumbaSelectionKernels()


__all__ = [
    "NumbaSelectionKernels",
    "greedy_two_nearest_selection",
    "indicator_removal",
    "iterative_crowding_truncation",
    "lexicographic_truncation",
    "numba_selection_kernels",
]
