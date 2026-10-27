"""Compiled measures for steady-state SMS-EMOA survival.

Exclusive two-objective hypervolume contributions of one front, constraint
violation sums and the violation ranks of infeasible rows, each with a one-time probe against the reference function
(``hypervolume_contributions`` with MooCore or its NumPy fallback, and
``compute_violation``). Callers use the compiled measure only when its probe
matches bit for bit and keep the reference otherwise.
"""

from __future__ import annotations

from functools import cache
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._numba_ranking import _stable_merge_sort_order
from ._numba_smsemoa_fronts import _CV_ORDER, _CV_TEMP, _INFEASIBLE, _VIOLATION
from .numba_ops import njit

FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.int64]
RankArray: TypeAlias = NDArray[np.integer[Any]]
BoolArray: TypeAlias = NDArray[np.bool_]


@njit(cache=True)
def _violation(g: FloatArray) -> float:
    """``np.sum(np.maximum(g, 0.0))`` for one row, with NumPy's pairwise order for up to 128 values."""
    n = g.shape[0]
    if n < 8:
        total = 0.0
        for j in range(n):
            total += g[j] if g[j] > 0.0 else 0.0
        return float(0.0 + total)
    acc = np.empty(8, dtype=np.float64)
    for j in range(8):
        acc[j] = g[j] if g[j] > 0.0 else 0.0
    i = 8
    while i < n - n % 8:
        for j in range(8):
            acc[j] += g[i + j] if g[i + j] > 0.0 else 0.0
        i += 8
    total = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
    while i < n:
        total += g[i] if g[i] > 0.0 else 0.0
        i += 1
    return float(0.0 + total)


@njit(cache=True)
def row_violations(G: FloatArray) -> FloatArray:
    out = np.empty(G.shape[0], dtype=np.float64)
    for p in range(G.shape[0]):
        out[p] = _violation(G[p])
    return out


@cache
def violation_matches_reference(n_constraints: int) -> bool:
    """Whether compiled violation sums equal ``compute_violation`` for ``n_constraints`` columns.

    NumPy sums each row in a fixed (pairwise) order; a mismatch keeps the
    reference survival for steps with infeasible rows.
    """
    from vamos.foundation.constraints.utils import compute_violation

    if n_constraints < 1 or n_constraints > 128:
        return False
    probe = np.random.Generator(np.random.PCG64(4099))
    G = probe.standard_normal((256, n_constraints))
    G *= probe.choice((1.0e-3, 1.0, 1.0e3), size=G.shape)
    G[::7] = 0.0
    G = np.ascontiguousarray(G)
    return bool(np.array_equal(row_violations(G), compute_violation(G)))


@njit(cache=True)
def _contributions_2d_sorted(F: FloatArray, members: IndexArray, k: int, ref: FloatArray, contrib: FloatArray) -> bool:
    """Exclusive two-objective contributions of a front given in ascending ``f1`` order.

    ``members[:k]`` must be mutually non-dominated, repeated points adjacent.
    Repeated points contribute zero; every other point contributes the
    rectangle up to its neighbours, ``max(x_next - x, 0) * max(y_prev - y, 0)``,
    written to ``contrib[row]``. Returns ``False`` without a result when the
    reference would reject or overflow (non-finite reference, a point beyond
    it, a non-finite product).
    """
    r0 = ref[0]
    r1 = ref[1]
    if not (np.isfinite(r0) and np.isfinite(r1)):
        return False
    for pos in range(k):
        if F[members[pos], 0] > r0 or F[members[pos], 1] > r1:
            return False
    y_prev = r1
    start = 0
    while start < k:
        x = F[members[start], 0]
        y = F[members[start], 1]
        end = start + 1
        while end < k and F[members[end], 0] == x and F[members[end], 1] == y:
            end += 1
        x_next = r0 if end == k else F[members[end], 0]
        width = x_next - x
        if not width >= 0.0:
            width = 0.0
        height = y_prev - y
        if not height >= 0.0:
            height = 0.0
        value = width * height
        if not np.isfinite(value):
            return False
        for t in range(start, end):
            contrib[members[t]] = value if end - start == 1 else 0.0
        y_prev = y
        start = end
    return True


@njit(cache=True)
def front_contributions_2d(F: FloatArray, ref: FloatArray) -> tuple[bool, FloatArray]:
    """Contributions of all rows of a two-objective front (``ok`` is ``False`` when deferred)."""
    k = F.shape[0]
    values = F[:, 0].copy()
    order = np.arange(k).astype(np.int64)
    temp = np.empty(k, dtype=np.int64)
    _stable_merge_sort_order(values, order, temp, k)
    contrib = np.zeros(k, dtype=np.float64)
    ok = _contributions_2d_sorted(F, order, k, ref, contrib)
    return ok, contrib


@cache
def front_contributions_2d_match_reference() -> bool:
    """Whether the compiled 2-D front contributions equal ``hypervolume_contributions`` bit for bit here.

    The reference is MooCore when it is installed and the NumPy fallback
    otherwise; the probe covers repeated points and points on the reference
    boundary. A mismatch keeps the reference call in the survival loop.
    """
    from vamos.foundation.quality_indicators.hypervolume import hypervolume_contributions

    probe = np.random.Generator(np.random.PCG64(1729))
    for size in (2, 3, 5, 17, 64, 101):
        for offset in (1.0, 0.1, 0.0):
            x = np.sort(probe.random(size)) * probe.choice((1.0, 1.0e-3, 4.0e3))
            y = np.sort(probe.random(size))[::-1] * probe.choice((1.0, 3.0e-2, 7.0e2))
            points = np.column_stack((x, y))
            if size > 2:
                points[1] = points[0]
            points = np.ascontiguousarray(points[probe.permutation(size)])
            ref = points.max(axis=0) + offset
            ok, actual = front_contributions_2d(points, ref)
            expected = np.asarray(hypervolume_contributions(points, ref), dtype=np.float64)
            if not ok or actual.tobytes() != expected.tobytes():
                return False
    return True


@njit(cache=True)
def _rank_infeasible_rows(
    G: FloatArray, ranks: RankArray, crowding: FloatArray, member: BoolArray, iwork: IndexArray, fwork: FloatArray
) -> None:
    """Infeasible rows ranked after the feasible fronts by violation (equal violations share a rank)."""
    n = G.shape[0]
    rows = iwork[_INFEASIBLE]
    violation = fwork[_VIOLATION]
    next_rank = 0
    count = 0
    for p in range(n):
        if member[p]:
            if ranks[p] + 1 > next_rank:
                next_rank = ranks[p] + 1
        else:
            violation[count] = _violation(G[p])
            rows[count] = count
            count += 1
    _stable_merge_sort_order(violation, rows, iwork[_CV_TEMP], count)
    infeasible = iwork[_CV_ORDER]
    count = 0
    for p in range(n):
        if not member[p]:
            infeasible[count] = p
            count += 1
    start = 0
    while start < count:
        end = start + 1
        while end < count and violation[rows[end]] == violation[rows[start]]:
            end += 1
        for t in range(start, end):
            ranks[infeasible[rows[t]]] = next_rank
            crowding[infeasible[rows[t]]] = 0.0
        next_rank += 1
        start = end


__all__ = [
    "front_contributions_2d",
    "front_contributions_2d_match_reference",
    "row_violations",
    "violation_matches_reference",
]
