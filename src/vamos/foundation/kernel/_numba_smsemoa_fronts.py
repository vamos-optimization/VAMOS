"""Compiled front maintenance for steady-state SMS-EMOA survival.

Incremental insertion of one row into cached Pareto fronts (as
``incremental_insert_fronts``), membership tests, the stable objective orders
and crowding recomputation for changed fronts (as ``compute_front_crowding``).
Work happens in caller-provided buffers whose rows are named below.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .numba_ops import njit

FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.int64]
RankArray: TypeAlias = NDArray[np.integer[Any]]
BoolArray: TypeAlias = NDArray[np.bool_]

# Rows of the integer, boolean and float work buffers.
_GROUPED = 0
_OFFSETS = 1
_CURSOR = 2
_FIRST = 3
_LAST = 4
_PREV = 5
_CUR = 6
_TEMP = 7
_CHILD_POS = 8
_INFEASIBLE = 9
_CV_ORDER = 10
_CV_TEMP = 11

_DISPLACED = 0
_CHANGED = 1
_MEMBER = 2

_CONTRIB = 0
_VIOLATION = 1


@njit(cache=True)
def _dominates(F: FloatArray, a: int, b: int) -> bool:
    better = False
    for k in range(F.shape[1]):
        if F[a, k] > F[b, k]:
            return False
        if F[a, k] < F[b, k]:
            better = True
    return better


@njit(cache=True)
def _contains_row(M: FloatArray, row: FloatArray) -> bool:
    for p in range(M.shape[0]):
        same = True
        for k in range(M.shape[1]):
            if M[p, k] != row[k]:
                same = False
                break
        if same:
            return True
    return False


@njit(cache=True)
def _group_by_rank(ranks: RankArray, n: int, member: BoolArray, iwork: IndexArray) -> int:
    """Member rows grouped by rank (ascending within a rank) into ``iwork``; returns the number of ranks."""
    grouped = iwork[_GROUPED]
    offsets = iwork[_OFFSETS]
    cursor = iwork[_CURSOR]
    n_fronts = 0
    for p in range(n):
        if member[p] and ranks[p] + 1 > n_fronts:
            n_fronts = ranks[p] + 1
    for r in range(n_fronts + 1):
        offsets[r] = 0
    for p in range(n):
        if member[p]:
            offsets[ranks[p] + 1] += 1
    for r in range(n_fronts):
        offsets[r + 1] += offsets[r]
        cursor[r] = offsets[r]
    for p in range(n):
        if member[p]:
            grouped[cursor[ranks[p]]] = p
            cursor[ranks[p]] += 1
    return n_fronts


@njit(cache=True)
def _insert_child_ranks(
    F_work: FloatArray, ranks: RankArray, ranks_work: IndexArray, member: BoolArray, iwork: IndexArray, displaced: BoolArray
) -> None:
    """Ranks of the member rows of ``F_work`` (population, then the child) after inserting a member child.

    Mirrors ``incremental_insert_fronts``: the child joins the first front
    with no member dominating it; members it dominates move one front down,
    and each move pushes the members of the next front that a moved row
    dominates. Other rows keep their ranks.
    """
    n = ranks.shape[0]
    for p in range(n):
        ranks_work[p] = ranks[p]
    ranks_work[n] = -1
    if not member[n]:
        return
    n_fronts = _group_by_rank(ranks, n, member, iwork)
    grouped = iwork[_GROUPED]
    offsets = iwork[_OFFSETS]
    level = n_fronts
    for front in range(n_fronts):
        blocked = False
        for t in range(offsets[front], offsets[front + 1]):
            if _dominates(F_work, grouped[t], n):
                blocked = True
                break
        if not blocked:
            level = front
            break
    ranks_work[n] = level
    moved = False
    if level < n_fronts:
        for t in range(offsets[level], offsets[level + 1]):
            q = grouped[t]
            displaced[q] = _dominates(F_work, n, q)
            moved = moved or displaced[q]
    while moved:
        target = level + 1
        for t in range(offsets[level], offsets[level + 1]):
            if displaced[grouped[t]]:
                ranks_work[grouped[t]] = target
        if target >= n_fronts:
            break
        moved = False
        for t in range(offsets[target], offsets[target + 1]):
            q = grouped[t]
            hit = False
            for s in range(offsets[level], offsets[level + 1]):
                d = grouped[s]
                if displaced[d] and _dominates(F_work, d, q):
                    hit = True
                    break
            displaced[q] = hit
            moved = moved or hit
        level = target


@njit(cache=True)
def _upper_bound(F: FloatArray, order: IndexArray, n: int, column: int, value: float) -> int:
    """First position of ``order[:n]`` whose value in ``column`` exceeds ``value``."""
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if F[order[mid], column] > value:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit(cache=True)
def _recompute_crowding(
    F: FloatArray, ranks: RankArray, orders: IndexArray, crowding: FloatArray, changed: BoolArray, member: BoolArray, iwork: IndexArray
) -> None:
    """Crowding distances of the member rows of every changed rank, as ``compute_front_crowding``."""
    n = F.shape[0]
    top = changed.shape[0]
    first = iwork[_FIRST]
    last = iwork[_LAST]
    prev = iwork[_PREV]
    cur = iwork[_CUR]
    for p in range(n):
        if member[p] and changed[ranks[p]]:
            crowding[p] = 0.0
    for c in range(F.shape[1]):
        order = orders[c]
        for r in range(top):
            first[r] = -1
            prev[r] = -1
            cur[r] = -1
        for t in range(n):
            p = order[t]
            if member[p] and changed[ranks[p]]:
                if first[ranks[p]] < 0:
                    first[ranks[p]] = p
                last[ranks[p]] = p
        for r in range(top):
            if first[r] >= 0:
                crowding[first[r]] = np.inf
                crowding[last[r]] = np.inf
        for t in range(n):
            p = order[t]
            if not (member[p] and changed[ranks[p]]):
                continue
            r = ranks[p]
            middle = cur[r]
            if middle >= 0 and prev[r] >= 0:
                span = F[last[r], c] - F[first[r], c]
                if span > 0.0:
                    crowding[middle] += (F[p, c] - F[prev[r], c]) / span
            prev[r] = middle
            cur[r] = p
