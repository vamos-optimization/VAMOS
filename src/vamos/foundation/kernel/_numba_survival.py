"""Compiled survival and archive-truncation loops.

Each kernel reproduces an iterative NumPy reference loop of an algorithm
(element order, tie-breaking by first index, floating-point accumulation order)
without drawing random numbers. Callers keep the NumPy reference for inputs the
kernel does not accept (for example non-finite objective values).
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._numba_ranking import _front_crowding
from .numba_ops import njit

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
def kth_distance_truncation(dist: FloatArray, keep: int, k: int) -> IndexArray:
    """SPEA2 truncation by the current k-th nearest-neighbour distance (finite distances).

    Each step computes the k-th smallest distance of every active row (the zero
    self-distance included, as in the reference ``np.partition``), orders the
    active rows by a stable descending sort of that value, and drops the last one.
    """
    n = dist.shape[0]
    active = np.arange(n, dtype=np.int64)
    size = n
    kth_values = np.empty(n, dtype=np.float64)
    smallest = np.empty(k + 1, dtype=np.float64)
    order = np.empty(n, dtype=np.int64)
    temp = np.empty(n, dtype=np.int64)
    keys = np.empty(n, dtype=np.float64)
    reordered = np.empty(n, dtype=np.int64)
    while size > keep:
        active_k = k if k < size - 1 else size - 1
        for i in range(size):
            row = active[i]
            count = 0
            for j in range(size):
                value = dist[row, active[j]]
                if count < active_k + 1:
                    pos = count
                    count += 1
                elif value < smallest[active_k]:
                    pos = active_k
                else:
                    continue
                while pos > 0 and smallest[pos - 1] > value:
                    smallest[pos] = smallest[pos - 1]
                    pos -= 1
                smallest[pos] = value
            kth_values[i] = smallest[active_k]
        for i in range(size):
            keys[i] = -kth_values[i]
            order[i] = i
        _stable_sort_ascending(keys, order, temp, size)
        for i in range(size - 1):
            reordered[i] = active[order[i]]
        size -= 1
        for i in range(size):
            active[i] = reordered[i]
    return active[:size].copy()


@njit(cache=True)
def _stable_sort_ascending(values: FloatArray, order: IndexArray, temp: IndexArray, length: int) -> None:
    width = 1
    while width < length:
        left = 0
        while left < length:
            mid = min(left + width, length)
            right = min(left + 2 * width, length)
            i = left
            j = mid
            out = left
            while i < mid and j < right:
                if values[order[j]] < values[order[i]]:
                    temp[out] = order[j]
                    j += 1
                else:
                    temp[out] = order[i]
                    i += 1
                out += 1
            while i < mid:
                temp[out] = order[i]
                i += 1
                out += 1
            while j < right:
                temp[out] = order[j]
                j += 1
                out += 1
            for pos in range(left, right):
                order[pos] = temp[pos]
            left += 2 * width
        width *= 2


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

    def kth_distance_truncation(self, dist: FloatArray, keep: int, k: int) -> IndexArray:
        return kth_distance_truncation(np.ascontiguousarray(dist, dtype=np.float64), int(keep), int(k))

    def indicator_removal(self, fitness: FloatArray, contrib: FloatArray, target_size: int) -> IndexArray:
        return indicator_removal(fitness, contrib, int(target_size))


NUMBA_SELECTION_KERNELS = NumbaSelectionKernels()

__all__ = [
    "NUMBA_SELECTION_KERNELS",
    "NumbaSelectionKernels",
    "greedy_two_nearest_selection",
    "indicator_removal",
    "iterative_crowding_truncation",
    "kth_distance_truncation",
]
