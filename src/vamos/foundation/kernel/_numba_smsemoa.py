"""Compiled steady-state SMS-EMOA survival.

The reference is ``survival_selection`` in
``vamos.engine.algorithm.smsemoa.helpers`` with its selection cache
(``refresh_selection_cache``). These kernels keep the cached ranks and
crowding distances as arrays and reproduce each step bit for bit for finite
objective values:

* incremental insertion assigns the same Pareto ranks as the front lists;
* the exclusive two-objective contributions are the products the reference
  computes (MooCore's ``hvc2d`` or the NumPy fallback); other dimensions, and
  inputs the reference would reject, are left to the caller;
* crowding is recomputed, with the reference operations, for the fronts whose
  members change; every other front keeps its members in the same relative
  order, so its distances are unchanged;
* with infeasible rows, the row with the largest violation (the last one on
  ties) is removed, feasible rows keep Pareto ranks among themselves and
  infeasible rows are ranked by violation, as the reference cache does.

Each objective keeps a stable order of the population (ascending value, ties
by row), updated in linear time per step; filtering it by rank yields each
front in the order of the reference's stable per-front sort. The kernels never
draw random numbers and work in buffers allocated once per run.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._numba_smsemoa_fronts import (
    _CHANGED,
    _CHILD_POS,
    _CONTRIB,
    _DISPLACED,
    _MEMBER,
    _TEMP,
    _VIOLATION,
    _contains_row,
    _insert_child_ranks,
    _recompute_crowding,
    _upper_bound,
)
from ._numba_smsemoa_measures import (
    _contributions_2d_sorted,
    _rank_infeasible_rows,
    _violation,
    front_contributions_2d_match_reference,
    violation_matches_reference,
)
from .backend import SMSEMOA_CONTRIBUTIONS, SMSEMOA_DELEGATE, SMSEMOA_DISCARDED
from .numba_ops import njit

FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.int64]
RankArray: TypeAlias = NDArray[np.integer[Any]]
BoolArray: TypeAlias = NDArray[np.bool_]

DISCARDED = SMSEMOA_DISCARDED
DELEGATE = SMSEMOA_DELEGATE
CONTRIBUTIONS = SMSEMOA_CONTRIBUTIONS

# Entries of the settings vector.
_DUPLICATES = 0
_CONSTRAINED = 1
_ADAPTIVE = 2
_EXACT_2D = 3
_EXACT_VIOLATION = 4
_OFFSET = 5


@njit(cache=True)
def commit_survivor(
    X: FloatArray,
    F: FloatArray,
    G: FloatArray,
    x_child: FloatArray,
    f_child: FloatArray,
    g_child: FloatArray,
    ranks: RankArray,
    crowding: FloatArray,
    ranks_work: IndexArray,
    orders: IndexArray,
    iwork: IndexArray,
    bwork: BoolArray,
    fwork: FloatArray,
    remove: int,
) -> None:
    """Remove row ``remove`` (shifting later rows up), append the child and update the cache.

    ``bwork[_MEMBER]`` marks the feasible work rows (all rows without constraints).
    """
    n = X.shape[0]
    member = bwork[_MEMBER]
    top = 0
    for p in range(n + 1):
        if member[p] and ranks_work[p] > top:
            top = ranks_work[p]
    changed = bwork[_CHANGED, : top + 1]
    for r in range(top + 1):
        changed[r] = False
    for p in range(n):
        if member[p] and ranks_work[p] != ranks[p]:
            changed[ranks[p]] = True
            changed[ranks_work[p]] = True
    for p in (n, remove):
        if member[p]:
            changed[ranks_work[p]] = True
    for p in range(remove, n):
        if p < n - 1:
            for j in range(X.shape[1]):
                X[p, j] = X[p + 1, j]
            for j in range(F.shape[1]):
                F[p, j] = F[p + 1, j]
            for j in range(G.shape[1]):
                G[p, j] = G[p + 1, j]
            crowding[p] = crowding[p + 1]
        member[p] = member[p + 1]
        ranks_work[p] = ranks_work[p + 1]
    for j in range(X.shape[1]):
        X[n - 1, j] = x_child[j]
    for j in range(F.shape[1]):
        F[n - 1, j] = f_child[j]
    for j in range(G.shape[1]):
        G[n - 1, j] = g_child[j]
    crowding[n - 1] = 0.0
    any_infeasible = False
    for p in range(n):
        ranks[p] = ranks_work[p]
        any_infeasible = any_infeasible or not member[p]

    temp = iwork[_TEMP]
    child_pos = iwork[_CHILD_POS]
    for c in range(F.shape[1]):
        order = orders[c]
        size = 0
        for t in range(n):
            if t == child_pos[c]:
                temp[size] = n - 1
                size += 1
            q = order[t]
            if q != remove:
                temp[size] = q - 1 if q > remove else q
                size += 1
        if child_pos[c] == n:
            temp[size] = n - 1
        for t in range(n):
            order[t] = temp[t]
    if any_infeasible:
        _rank_infeasible_rows(G, ranks, crowding, member, iwork, fwork)
    _recompute_crowding(F, ranks, orders, crowding, changed, member, iwork)


@njit(cache=True)
def survival_step(
    X: FloatArray,
    F: FloatArray,
    G: FloatArray,
    x_child: FloatArray,
    f_child: FloatArray,
    g_child: FloatArray,
    ranks: RankArray,
    crowding: FloatArray,
    ref_point: FloatArray,
    F_work: FloatArray,
    ranks_work: IndexArray,
    worst: IndexArray,
    orders: IndexArray,
    iwork: IndexArray,
    bwork: BoolArray,
    fwork: FloatArray,
    settings: FloatArray,
) -> tuple[int, int]:
    """One SMS-EMOA survival step; returns ``(status, worst front size)``.

    ``status`` is the removed row (the population size when the child is
    rejected and the population is unchanged) or ``DISCARDED``, ``DELEGATE``
    (a non-finite child objective, or violations that may differ from NumPy's
    sums) or ``CONTRIBUTIONS``. ``F_work``, ``ranks_work``, ``worst`` and the
    adaptive ``ref_point`` are updated from the insertion onwards.
    """
    n = X.shape[0]
    m = F.shape[1]
    for c in range(m):
        if not np.isfinite(f_child[c]):
            return DELEGATE, 0
    if settings[_DUPLICATES] != 0.0 and (_contains_row(X, x_child) or _contains_row(F, f_child)):
        return DISCARDED, 0
    for p in range(n):
        for c in range(m):
            F_work[p, c] = F[p, c]
    for c in range(m):
        F_work[n, c] = f_child[c]
    if settings[_ADAPTIVE] != 0.0:
        # relative_reference_point: nadir + offset * (nadir - ideal), with a range of 1 when there is no spread.
        for c in range(m):
            top = F_work[0, c]
            bottom = F_work[0, c]
            for p in range(1, n + 1):
                value = F_work[p, c]
                if value > top:
                    top = value
                if value < bottom:
                    bottom = value
            span = top - bottom
            if not span > 0.0:
                span = 1.0
            ref_point[c] = top + settings[_OFFSET] * span
    child_pos = iwork[_CHILD_POS]
    for c in range(m):
        child_pos[c] = _upper_bound(F, orders[c], n, c, f_child[c])
    member = bwork[_MEMBER]
    feasible = True
    for p in range(n + 1):
        member[p] = True
        if settings[_CONSTRAINED] != 0.0:
            row = g_child if p == n else G[p]
            for c in range(row.shape[0]):
                if not row[c] <= 0.0:
                    member[p] = False
                    feasible = False
                    break

    if not feasible:
        # Feasibility-first survival: remove the (last) row with the largest violation.
        if settings[_EXACT_VIOLATION] == 0.0:
            return DELEGATE, 0
        violation = fwork[_VIOLATION]
        worst_cv = 0.0
        for p in range(n + 1):
            violation[p] = _violation(g_child if p == n else G[p])
            if violation[p] > worst_cv:
                worst_cv = violation[p]
        remove = n
        for p in range(n + 1):
            if violation[p] == worst_cv:
                remove = p
        if remove != n:
            _insert_child_ranks(F_work, ranks, ranks_work, member, iwork, bwork[_DISPLACED])
            commit_survivor(X, F, G, x_child, f_child, g_child, ranks, crowding, ranks_work, orders, iwork, bwork, fwork, remove)
        return remove, 0

    _insert_child_ranks(F_work, ranks, ranks_work, member, iwork, bwork[_DISPLACED])
    worst_rank = 0
    for p in range(n + 1):
        if ranks_work[p] > worst_rank:
            worst_rank = ranks_work[p]
    k = 0
    for p in range(n + 1):
        if ranks_work[p] == worst_rank:
            worst[k] = p
            k += 1
    if k == 1:
        remove = worst[0]
    else:
        if settings[_EXACT_2D] == 0.0:
            return CONTRIBUTIONS, k
        # The worst front in ascending f1 order: the stable objective order with the child spliced in.
        sorted_front = iwork[_TEMP]
        order = orders[0]
        size = 0
        for t in range(n + 1):
            if t == child_pos[0] and ranks_work[n] == worst_rank:
                sorted_front[size] = n
                size += 1
            if t < n and ranks_work[order[t]] == worst_rank:
                sorted_front[size] = order[t]
                size += 1
        contrib = fwork[_CONTRIB]
        if not _contributions_2d_sorted(F_work, sorted_front, k, ref_point, contrib):
            return CONTRIBUTIONS, k
        best = 0
        for pos in range(1, k):
            if contrib[worst[pos]] < contrib[worst[best]]:
                best = pos
        remove = worst[best]
    if remove != n:
        commit_survivor(X, F, G, x_child, f_child, g_child, ranks, crowding, ranks_work, orders, iwork, bwork, fwork, remove)
    return remove, k


class NumbaSMSEMOASurvival:
    """Steady-state SMS-EMOA survival with work buffers for one run."""

    name = "numba"

    def __init__(
        self,
        *,
        pop_size: int,
        n_obj: int,
        n_constraints: int,
        ref_offset: float,
        ref_adaptive: bool,
        eliminate_duplicates: bool,
        check_feasibility: bool,
    ) -> None:
        width = max(pop_size + 2, n_obj)
        self.F_work: FloatArray = np.empty((pop_size + 1, n_obj), dtype=np.float64)
        self.worst: IndexArray = np.empty(pop_size + 1, dtype=np.int64)
        self._ranks_work = np.empty(pop_size + 1, dtype=np.int64)
        self._orders = np.empty((n_obj, pop_size), dtype=np.int64)
        self._iwork = np.empty((12, width), dtype=np.int64)
        self._bwork = np.zeros((3, width), dtype=np.bool_)
        self._fwork = np.empty((2, width), dtype=np.float64)
        settings = np.zeros(6, dtype=np.float64)
        settings[_DUPLICATES] = float(bool(eliminate_duplicates))
        settings[_CONSTRAINED] = float(bool(check_feasibility))
        settings[_ADAPTIVE] = float(bool(ref_adaptive))
        settings[_EXACT_2D] = float(n_obj == 2 and front_contributions_2d_match_reference())
        settings[_EXACT_VIOLATION] = float(bool(check_feasibility) and violation_matches_reference(int(n_constraints)))
        settings[_OFFSET] = float(ref_offset)
        self._settings = settings

    def reset(self, F: FloatArray) -> None:
        for c in range(F.shape[1]):
            self._orders[c] = np.argsort(F[:, c], kind="stable")

    def insert(
        self,
        X: FloatArray,
        F: FloatArray,
        G: FloatArray,
        x_child: FloatArray,
        f_child: FloatArray,
        g_child: FloatArray,
        ranks: RankArray,
        crowding: FloatArray,
        ref_point: FloatArray,
    ) -> tuple[int, int]:
        status, size = survival_step(
            X,
            F,
            G,
            x_child,
            f_child,
            g_child,
            ranks,
            crowding,
            ref_point,
            self.F_work,
            self._ranks_work,
            self.worst,
            self._orders,
            self._iwork,
            self._bwork,
            self._fwork,
            self._settings,
        )
        return int(status), int(size)

    def commit(
        self,
        X: FloatArray,
        F: FloatArray,
        G: FloatArray,
        x_child: FloatArray,
        f_child: FloatArray,
        g_child: FloatArray,
        ranks: RankArray,
        crowding: FloatArray,
        remove: int,
    ) -> None:
        commit_survivor(
            X,
            F,
            G,
            x_child,
            f_child,
            g_child,
            ranks,
            crowding,
            self._ranks_work,
            self._orders,
            self._iwork,
            self._bwork,
            self._fwork,
            int(remove),
        )


__all__ = [
    "CONTRIBUTIONS",
    "DELEGATE",
    "DISCARDED",
    "NumbaSMSEMOASurvival",
    "commit_survivor",
    "survival_step",
]
