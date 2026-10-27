"""Compiled survival of the one-offspring (steady-state) NSGA-II step.

The evaluated child is inserted into the cached Pareto fronts of the population
and NSGA-II truncates the combined population back to its size, as the
reference incremental insertion and selection do. Crowding distances walk
per-objective row orders kept sorted across steps (ties in row order, as the
reference stable sort), so they accumulate the same differences in the same
order without sorting each front again. Callers keep their reference path when
:func:`nsga2_steady_insert` returns ``STEADY_STATE_UNSUPPORTED`` (a non-finite
objective value).
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .backend import STEADY_STATE_REJECTED, STEADY_STATE_UNSUPPORTED
from .numba_ops import njit

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

# Rows of the ``int_work`` workspace, shaped ``(INT_WORK_ROWS + n_obj, N + 1)``: the
# combined ranks, two scratch rows, the members of the truncated front, then the
# combined per-objective orders.
COMB_RANKS = 0
SCRATCH_A = 1
SCRATCH_B = 2
SPLIT_MEMBERS = 3
COMB_ORDERS = 4
INT_WORK_ROWS = 4
# Rows of the ``float_work`` workspace, shaped ``(2, N + 1)``.
COMB_CROWD = 0
SPLIT_CROWD = 1


@njit(cache=True)
def _dominates(F: FloatArray, a: int, b: int) -> bool:
    """Whether row ``a`` Pareto-dominates row ``b`` (minimization; NaN never dominates)."""
    strictly_better = False
    for k in range(F.shape[1]):
        if not (F[a, k] <= F[b, k]):
            return False
        if F[a, k] < F[b, k]:
            strictly_better = True
    return strictly_better


@njit(cache=True)
def _sorted_front_crowding(F: FloatArray, orders: IntArray, ranks: IntArray, rank: int, crowding: FloatArray) -> None:
    """Crowding distances of the rows of ``rank``, written into ``crowding``.

    ``orders[m]`` lists every row in ascending order of objective ``m`` with ties
    in ascending row order: the stable sort of the reference crowding. Walking
    it restricted to the front therefore visits the reference sorted order, and
    the distances accumulate over the objectives in the reference order.
    """
    n_rows = orders.shape[1]
    size = 0
    single = -1
    for i in range(n_rows):
        if ranks[i] == rank:
            crowding[i] = 0.0
            size += 1
            single = i
    if size == 0:
        return
    if size == 1:
        crowding[single] = np.inf
        return
    for m in range(F.shape[1]):
        sequence = orders[m]
        first = 0
        while ranks[sequence[first]] != rank:
            first += 1
        last = n_rows - 1
        while ranks[sequence[last]] != rank:
            last -= 1
        low = sequence[first]
        high = sequence[last]
        crowding[low] = np.inf
        crowding[high] = np.inf
        span = F[high, m] - F[low, m]
        if span <= 0.0:
            continue
        previous = low
        current = -1
        for t in range(first + 1, last + 1):
            row = sequence[t]
            if ranks[row] != rank:
                continue
            if current >= 0:
                crowding[current] += (F[row, m] - F[previous, m]) / span
                previous = current
            current = row


@njit(cache=True)
def nsga2_steady_insert(
    F: FloatArray,
    ranks: NDArray[np.integer[Any]],
    crowding: FloatArray,
    f_child: FloatArray,
    orders: IntArray,
    comb_F: FloatArray,
    int_work: IntArray,
    float_work: FloatArray,
) -> int:
    """Insert the evaluated child ``f_child[0]`` into the fronts of ``F`` (child at row ``N``).

    ``ranks`` and ``crowding`` are the cached Pareto ranks and per-front
    crowding distances of the population, and ``orders[m]`` its rows sorted by
    objective ``m`` (ties by row). The combined population goes to ``comb_F``
    and the workspaces (see :data:`INT_WORK_ROWS`): fronts whose membership
    changes get fresh crowding distances, every other front keeps its cached
    values. The members of the last front (the one NSGA-II truncates) are left
    in ascending row order with their crowding in ``float_work[SPLIT_CROWD]``,
    and their number is returned. Returns ``STEADY_STATE_REJECTED`` when a
    member of the worst front dominates the child (the reference keeps the
    population unchanged) and ``STEADY_STATE_UNSUPPORTED`` when an objective
    value is not finite.
    """
    n = F.shape[0]
    n_obj = F.shape[1]
    comb_ranks = int_work[COMB_RANKS]
    comb_orders = int_work[COMB_ORDERS:]
    comb_crowd = float_work[COMB_CROWD]
    split_crowd = float_work[SPLIT_CROWD]
    worst = 0
    for i in range(n):
        for k in range(n_obj):
            value = F[i, k]
            if not (value - value == 0.0):
                return STEADY_STATE_UNSUPPORTED
            comb_F[i, k] = value
        comb_ranks[i] = ranks[i]
        comb_crowd[i] = crowding[i]
        if ranks[i] > worst:
            worst = ranks[i]
    for k in range(n_obj):
        value = f_child[0, k]
        if not (value - value == 0.0):
            return STEADY_STATE_UNSUPPORTED
        comb_F[n, k] = value

    for i in range(n):
        if ranks[i] == worst and _dominates(comb_F, i, n):
            return STEADY_STATE_REJECTED

    # The fronts holding a dominator of the child are 0..r (transitivity), so the
    # child joins front r + 1: the first front with no member dominating it.
    child_rank = 0
    for i in range(n):
        if child_rank <= ranks[i] < worst and _dominates(comb_F, i, n):
            child_rank = ranks[i] + 1
    comb_ranks[n] = child_rank

    # Rows dominated by the rows that just moved down one front move down too.
    displaced = int_work[SCRATCH_A]
    following = int_work[SCRATCH_B]
    n_displaced = 0
    for i in range(n):
        if ranks[i] == child_rank and _dominates(comb_F, n, i):
            displaced[n_displaced] = i
            n_displaced += 1
    level = child_rank
    while n_displaced > 0:
        for t in range(n_displaced):
            comb_ranks[displaced[t]] = level + 1
        n_following = 0
        for i in range(n):
            if ranks[i] == level + 1:
                for t in range(n_displaced):
                    if _dominates(comb_F, displaced[t], i):
                        following[n_following] = i
                        n_following += 1
                        break
        displaced, following = following, displaced
        n_displaced = n_following
        level += 1

    # The child has the largest row index, so it follows every row with an equal value.
    for m in range(n_obj):
        value = comb_F[n, m]
        sequence = orders[m]
        low = 0
        high = n
        while low < high:
            middle = (low + high) >> 1
            if F[sequence[middle], m] <= value:
                low = middle + 1
            else:
                high = middle
        target = comb_orders[m]
        for t in range(low):
            target[t] = sequence[t]
        target[low] = n
        for t in range(low, n):
            target[t + 1] = sequence[t]

    for rank in range(child_rank, level + 1):
        _sorted_front_crowding(comb_F, comb_orders, comb_ranks, rank, comb_crowd)

    last = worst if worst > level else level
    members = int_work[SPLIT_MEMBERS]
    size = 0
    for i in range(n + 1):
        if comb_ranks[i] == last:
            members[size] = i
            split_crowd[size] = comb_crowd[i]
            size += 1
    return size


@njit(cache=True)
def nsga2_steady_replace(
    X: FloatArray,
    x_child: FloatArray,
    comb_F: FloatArray,
    int_work: IntArray,
    float_work: FloatArray,
    order: NDArray[np.integer[Any]],
    out_X: FloatArray,
    out_F: FloatArray,
    out_ranks: IntArray,
    out_crowd: FloatArray,
    out_orders: IntArray,
) -> None:
    """Write the NSGA-II survivors of the combined population left by :func:`nsga2_steady_insert`.

    Complete fronts come first in rank order with ascending row order, then the
    last front's members in the reverse of ``order`` (``np.argsort`` of their
    crowding) without its first entry, exactly as the reference selection. The
    survivors keep their ranks and crowding distances, except the truncated
    front, whose crowding is recomputed over its remaining members.
    ``out_orders`` receives the survivors' per-objective orders.
    """
    n = X.shape[0]
    n_var = X.shape[1]
    n_obj = comb_F.shape[1]
    size = order.shape[0]
    comb_ranks = int_work[COMB_RANKS]
    comb_orders = int_work[COMB_ORDERS:]
    comb_crowd = float_work[COMB_CROWD]
    split_members = int_work[SPLIT_MEMBERS]
    last = comb_ranks[split_members[0]]

    cursor = int_work[SCRATCH_A]
    position_of = int_work[SCRATCH_B]
    for rank in range(last + 1):
        cursor[rank] = 0
    for i in range(n + 1):
        cursor[comb_ranks[i]] += 1
    total = 0
    for rank in range(last):
        count = cursor[rank]
        cursor[rank] = total
        total += count

    for i in range(n + 1):
        rank = comb_ranks[i]
        if rank == last:
            continue
        position = cursor[rank]
        cursor[rank] += 1
        position_of[i] = position
        out_ranks[position] = rank
        out_crowd[position] = comb_crowd[i]

    position_of[split_members[order[0]]] = -1
    for t in range(size - 1):
        position = total + t
        position_of[split_members[order[size - 1 - t]]] = position
        out_ranks[position] = last

    for i in range(n):
        position = position_of[i]
        if position >= 0:
            for j in range(n_var):
                out_X[position, j] = X[i, j]
            for k in range(n_obj):
                out_F[position, k] = comb_F[i, k]
    position = position_of[n]
    if position >= 0:
        for j in range(n_var):
            out_X[position, j] = x_child[0, j]
        for k in range(n_obj):
            out_F[position, k] = comb_F[n, k]

    # Renumber the orders, drop the removed row and restore the position order among equal values.
    for m in range(n_obj):
        source = comb_orders[m]
        target = out_orders[m]
        k = 0
        for t in range(n + 1):
            position = position_of[source[t]]
            if position >= 0:
                target[k] = position
                k += 1
        for t in range(1, n):
            position = target[t]
            value = out_F[position, m]
            u = t
            while u > 0 and out_F[target[u - 1], m] == value and target[u - 1] > position:
                target[u] = target[u - 1]
                u -= 1
            target[u] = position

    if size > 1:
        _sorted_front_crowding(out_F, out_orders, out_ranks, last, out_crowd)


__all__ = ["INT_WORK_ROWS", "SPLIT_CROWD", "nsga2_steady_insert", "nsga2_steady_replace"]
