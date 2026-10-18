from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .numba_ops import njit

FloatArray: TypeAlias = NDArray[np.float64]
RealArray: TypeAlias = NDArray[np.floating[Any]]
RankArray: TypeAlias = NDArray[np.int64]


@njit(cache=True)
def _fast_non_dominated_sort_ranks_2d_sorted(F_sorted: FloatArray, y_codes: RankArray) -> RankArray:
    N = F_sorted.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.int64)

    bit = np.zeros(int(y_codes.max()) + 2, dtype=np.int64)
    ranks_sorted = np.empty(N, dtype=np.int64)
    i = 0

    while i < N:
        j = i + 1
        x = F_sorted[i, 0]
        y = F_sorted[i, 1]
        while j < N and F_sorted[j, 0] == x and F_sorted[j, 1] == y:
            j += 1

        bit_idx = y_codes[i] + 1
        best = 0
        query_idx = bit_idx
        while query_idx > 0:
            if bit[query_idx] > best:
                best = bit[query_idx]
            query_idx -= query_idx & -query_idx

        rank = best
        for pos in range(i, j):
            ranks_sorted[pos] = rank

        update_value = rank + 1
        update_idx = bit_idx
        while update_idx < bit.shape[0]:
            if update_value > bit[update_idx]:
                bit[update_idx] = update_value
            update_idx += update_idx & -update_idx

        i = j

    return ranks_sorted


def fast_non_dominated_sort_ranks_2d(F: RealArray) -> RankArray:
    N = F.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.int64)

    order = np.lexsort((F[:, 1], F[:, 0]))
    F_sorted = F[order]
    _, y_codes = np.unique(F_sorted[:, 1], return_inverse=True)
    ranks_sorted = _fast_non_dominated_sort_ranks_2d_sorted(F_sorted, y_codes.astype(np.int64, copy=False))
    ranks = np.empty(N, dtype=np.int64)
    ranks[order] = ranks_sorted
    return ranks


@njit(cache=True)
def _sequential_search_ranks(F: RealArray, order: RankArray) -> RankArray:
    """Pareto ranks by sequential search over a lexicographic visiting order.

    ``order`` sorts the rows lexicographically by all objectives, so a row can
    only be dominated by rows visited before it. Each row joins the first front
    that holds no row dominating it; by transitivity that front is its rank.
    The ranks are those of the classic fast non-dominated sort.
    """
    n = order.shape[0]
    n_obj = F.shape[1]
    ranks = np.empty(n, dtype=np.int64)
    last_member = np.full(n, -1, dtype=np.int64)
    previous_member = np.full(n, -1, dtype=np.int64)
    n_fronts = 0
    for pos in range(n):
        row = order[pos]
        front = 0
        while front < n_fronts:
            dominated = False
            member = last_member[front]
            while member != -1:
                member_better = False
                member_dominates = True
                for obj in range(n_obj):
                    a = F[member, obj]
                    b = F[row, obj]
                    if a > b:
                        member_dominates = False
                        break
                    if a < b:
                        member_better = True
                if member_dominates and member_better:
                    dominated = True
                    break
                member = previous_member[member]
            if not dominated:
                break
            front += 1
        ranks[row] = front
        if front == n_fronts:
            n_fronts += 1
        previous_member[row] = last_member[front]
        last_member[front] = row
    return ranks


def fast_non_dominated_sort_ranks(F: RealArray) -> RankArray:
    if F.shape[1] == 2:
        return fast_non_dominated_sort_ranks_2d(F)
    if F.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort(F.T[::-1]).astype(np.int64, copy=False)
    return _sequential_search_ranks(F, order)


@njit(cache=True)
def _stable_merge_sort_order(values: FloatArray, order: RankArray, temp: RankArray, length: int) -> None:
    width = 1
    while width < length:
        left = 0
        while left < length:
            mid = left + width
            if mid > length:
                mid = length
            right = left + 2 * width
            if right > length:
                right = length

            i = left
            j = mid
            k = left

            while i < mid and j < right:
                left_idx = order[i]
                right_idx = order[j]
                left_val = values[left_idx]
                right_val = values[right_idx]

                if left_val <= right_val:
                    temp[k] = left_idx
                    i += 1
                else:
                    temp[k] = right_idx
                    j += 1
                k += 1

            while i < mid:
                temp[k] = order[i]
                i += 1
                k += 1

            while j < right:
                temp[k] = order[j]
                j += 1
                k += 1

            for pos in range(left, right):
                order[pos] = temp[pos]

            left += 2 * width

        width *= 2


@njit(cache=True)
def _front_crowding(
    F: RealArray,
    members: RankArray,
    start: int,
    front_size: int,
    local_values: FloatArray,
    local_order: RankArray,
    local_temp: RankArray,
    local_dist: FloatArray,
) -> None:
    """Crowding distances of ``members[start:start + front_size]`` into ``local_dist``."""
    for i in range(front_size):
        local_dist[i] = 0.0
    if front_size == 1:
        local_dist[0] = np.inf
        return

    for obj in range(F.shape[1]):
        for i in range(front_size):
            local_order[i] = i
            local_values[i] = F[members[start + i], obj]

        _stable_merge_sort_order(local_values, local_order, local_temp, front_size)
        local_dist[local_order[0]] = np.inf
        local_dist[local_order[front_size - 1]] = np.inf

        span = local_values[local_order[front_size - 1]] - local_values[local_order[0]]
        if span <= 0.0:
            continue

        for pos in range(1, front_size - 1):
            local_dist[local_order[pos]] += (local_values[local_order[pos + 1]] - local_values[local_order[pos - 1]]) / span


@njit(cache=True)
def _ranks_counting_order(ranks: RankArray) -> tuple[RankArray, RankArray]:
    """Row indices grouped by rank (ascending index within a rank) and rank offsets."""
    n = ranks.shape[0]
    max_rank = 0
    for i in range(n):
        if ranks[i] > max_rank:
            max_rank = ranks[i]
    offsets = np.zeros(max_rank + 2, dtype=np.int64)
    for i in range(n):
        offsets[ranks[i] + 1] += 1
    for rank in range(max_rank + 1):
        offsets[rank + 1] += offsets[rank]
    cursor = offsets[:-1].copy()
    grouped = np.empty(n, dtype=np.int64)
    for i in range(n):
        rank = ranks[i]
        grouped[cursor[rank]] = i
        cursor[rank] += 1
    return grouped, offsets


@njit(cache=True)
def compute_crowding_numba(F: RealArray, ranks: RankArray) -> FloatArray:
    N = F.shape[0]
    crowding = np.zeros(N, dtype=np.float64)
    if N == 0:
        return crowding

    front_indices, offsets = _ranks_counting_order(ranks)
    n_fronts = offsets.shape[0] - 1
    max_front_size = 0
    for rank in range(n_fronts):
        size = offsets[rank + 1] - offsets[rank]
        if size > max_front_size:
            max_front_size = size
    local_values = np.empty(max_front_size, dtype=np.float64)
    local_order = np.empty(max_front_size, dtype=np.int64)
    local_temp = np.empty(max_front_size, dtype=np.int64)
    local_dist = np.empty(max_front_size, dtype=np.float64)

    for rank in range(n_fronts):
        start = offsets[rank]
        front_size = offsets[rank + 1] - start
        if front_size == 0:
            continue
        _front_crowding(F, front_indices, start, front_size, local_values, local_order, local_temp, local_dist)
        for i in range(front_size):
            crowding[front_indices[start + i]] = local_dist[i]

    return crowding


@njit(cache=True)
def nsga2_survival_split(F: RealArray, ranks: RankArray, pop_size: int) -> tuple[RankArray, RankArray, FloatArray]:
    """Split NSGA-II survivors into complete fronts and the front that must be truncated.

    Returns the complete-front survivors (rank order, ascending index within a
    rank), the members of the splitting front (ascending index) and their
    crowding distances. The splitting front is empty when complete fronts fill
    ``pop_size`` exactly or the whole population survives.
    """
    grouped, offsets = _ranks_counting_order(ranks)
    n_fronts = offsets.shape[0] - 1
    selected = 0
    rank = 0
    while rank < n_fronts and selected + (offsets[rank + 1] - offsets[rank]) <= pop_size:
        selected += offsets[rank + 1] - offsets[rank]
        rank += 1
    complete = grouped[:selected].copy()
    if rank >= n_fronts or selected >= pop_size:
        return complete, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    start = offsets[rank]
    front_size = offsets[rank + 1] - start
    splitting = grouped[start : start + front_size].copy()
    local_values = np.empty(front_size, dtype=np.float64)
    local_order = np.empty(front_size, dtype=np.int64)
    local_temp = np.empty(front_size, dtype=np.int64)
    crowding = np.empty(front_size, dtype=np.float64)
    _front_crowding(F, grouped, start, front_size, local_values, local_order, local_temp, crowding)
    return complete, splitting, crowding


def fronts_from_ranks(ranks: NDArray[np.integer[Any]]) -> list[list[int]]:
    if ranks.size == 0:
        return []
    unique_ranks = np.unique(ranks)
    return [np.flatnonzero(ranks == rank).tolist() for rank in unique_ranks]


__all__ = [
    "compute_crowding_numba",
    "fast_non_dominated_sort_ranks",
    "fronts_from_ranks",
    "nsga2_survival_split",
]
