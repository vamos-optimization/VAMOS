"""SMS-EMOA helper functions.

This module contains utility functions for SMS-EMOA:
- Reference point initialization and management
- Cached selection metrics for steady-state parent selection
- Incremental survival selection using hypervolume contributions
- Population/offspring evaluation with constraints
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.components.hv_reference import relative_reference_point
from vamos.foundation.constraints.utils import compute_violation, is_feasible

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol

    from .state import SMSEMOAState

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
AnyArray = NDArray[np.generic]

__all__ = [
    "initialize_reference_point",
    "update_reference_point",
    "least_contributor",
    "fronts_from_ranks",
    "compute_front_crowding",
    "compute_crowding",
    "incremental_insert_fronts",
    "reindex_fronts",
    "ranks_from_fronts",
    "refresh_selection_cache",
    "survival_selection",
    "evaluate_population_with_constraints",
]


def _is_duplicate_row(matrix: AnyArray, row: AnyArray) -> bool:
    if matrix.size == 0:
        return False
    return bool(np.any(np.all(matrix == row, axis=1)))


def initialize_reference_point(
    F: FloatArray,
    ref_cfg: Mapping[str, object],
) -> tuple[FloatArray, float, bool]:
    """Initial reference point, offset and adaptivity for the hypervolume contributions.

    A user ``vector`` is the reference as given and stays fixed: ``adaptive`` does not
    apply to it, and points outside its box contribute 0. Otherwise the reference is
    ``nadir + offset * (nadir - ideal)`` of ``F`` (see :func:`relative_reference_point`),
    recomputed at every step when ``adaptive`` (the default).
    """
    offset_raw = ref_cfg.get("offset", 1.0)
    offset = float(offset_raw) if isinstance(offset_raw, (int, float, str)) else 1.0
    vector = ref_cfg.get("vector")

    if vector is not None:
        ref = np.array(vector, dtype=float)
        if ref.ndim != 1 or ref.shape[0] != F.shape[1]:
            raise ValueError("reference_point vector dimensionality mismatch.")
        if not bool(np.isfinite(ref).all()):
            raise ValueError("reference_point vector must contain finite values.")
        return ref, offset, False

    adaptive_raw = ref_cfg.get("adaptive", True)
    adaptive = True if adaptive_raw is None else bool(adaptive_raw)
    return relative_reference_point(F, offset), offset, adaptive


def update_reference_point(
    ref_point: FloatArray,
    F_new: FloatArray,
    ref_offset: float,
) -> FloatArray:
    """The adaptive reference of the population and the child: ``nadir + offset * range``."""
    del ref_point
    return relative_reference_point(F_new, ref_offset)


def least_contributor(kernel: KernelBackend, points: FloatArray, ref_point: FloatArray) -> int:
    """Row of ``points`` with the smallest hypervolume contribution (the first on ties).

    Points outside the reference box (a fixed user reference) contribute 0; the
    others are measured among themselves with the kernel's contribution hook.
    """
    inside = np.all(points <= ref_point, axis=1)
    if bool(inside.all()):
        contributions = kernel.hypervolume_contributions(points, ref_point)
    else:
        contributions = np.zeros(points.shape[0], dtype=float)
        if bool(inside.any()):
            contributions[inside] = kernel.hypervolume_contributions(points[inside], ref_point)
    return int(np.argmin(contributions))


def fronts_from_ranks(ranks: IntArray) -> list[list[int]]:
    if ranks.size == 0:
        return []
    max_rank = int(ranks.max(initial=-1))
    return [np.flatnonzero(ranks == rank).tolist() for rank in range(max_rank + 1)]


def compute_front_crowding(F: FloatArray, front: IntArray | list[int]) -> FloatArray:
    """Compute crowding distance for a single front."""
    front_arr = np.asarray(front, dtype=int)
    if front_arr.size == 0:
        return np.empty(0, dtype=float)
    if front_arr.size == 1:
        return np.array([np.inf], dtype=float)

    fvals = F[front_arr]
    n_obj = fvals.shape[1]
    crowding = np.zeros(front_arr.size, dtype=float)
    for obj in range(n_obj):
        order = np.argsort(fvals[:, obj], kind="mergesort")
        sorted_vals = fvals[order, obj]
        crowding[order[0]] = np.inf
        crowding[order[-1]] = np.inf
        span = sorted_vals[-1] - sorted_vals[0]
        if span <= 0.0:
            continue
        contrib = np.zeros_like(sorted_vals)
        contrib[1:-1] = (sorted_vals[2:] - sorted_vals[:-2]) / span
        crowding[order[1:-1]] += contrib[1:-1]
    return crowding


def compute_crowding(F: FloatArray, fronts: list[list[int]]) -> FloatArray:
    """Compute crowding distances for the provided fronts."""
    crowding = np.zeros(F.shape[0], dtype=float)
    for front in fronts:
        if not front:
            continue
        front_arr = np.asarray(front, dtype=int)
        crowding[front_arr] = compute_front_crowding(F, front_arr)
    return crowding


def incremental_insert_fronts(
    fronts: list[list[int]],
    ranks: IntArray,
    F: FloatArray,
    new_idx: int,
) -> list[int]:
    """Incrementally insert one solution into existing non-dominated fronts."""
    affected: list[int] = []
    f_new = F[new_idx]
    inserted = False

    for i, front in enumerate(fronts):
        if not front:
            continue
        front_arr = np.asarray(front, dtype=int)
        f_front = F[front_arr]
        dominates_new = np.all(f_front <= f_new, axis=1) & np.any(f_front < f_new, axis=1)
        if np.any(dominates_new):
            continue

        dominates_front = np.all(f_new <= f_front, axis=1) & np.any(f_new < f_front, axis=1)
        dominated = front_arr[dominates_front].tolist()

        if dominated:
            front = [idx for idx in front if idx not in dominated]
        front.append(new_idx)
        front.sort()
        fronts[i] = front
        ranks[new_idx] = i
        affected.append(i)

        displaced = dominated
        j = i + 1
        while displaced:
            if j >= len(fronts):
                fronts.append(sorted(displaced))
                for idx in displaced:
                    ranks[idx] = j
                affected.append(j)
                break
            front_j = fronts[j]
            front_j_arr = np.asarray(front_j, dtype=int)
            f_front_j = F[front_j_arr]
            dom_mask = np.zeros(front_j_arr.shape[0], dtype=bool)
            for idx in displaced:
                f_idx = F[idx]
                dom_mask |= np.all(f_idx <= f_front_j, axis=1) & np.any(f_idx < f_front_j, axis=1)
            dominated_j = front_j_arr[dom_mask].tolist()
            if dominated_j:
                front_j = [idx for idx in front_j if idx not in dominated_j]
            front_j.extend(displaced)
            front_j.sort()
            fronts[j] = front_j
            for idx in displaced:
                ranks[idx] = j
            affected.append(j)
            displaced = dominated_j
            j += 1

        inserted = True
        break

    if not inserted:
        fronts.append([new_idx])
        ranks[new_idx] = len(fronts) - 1
        affected.append(len(fronts) - 1)
    return affected


def reindex_fronts(fronts: list[list[int]], removed_idx: int) -> list[list[int]]:
    """Remove an index from fronts and shift higher indices down by one."""
    new_fronts: list[list[int]] = []
    for front in fronts:
        updated: list[int] = []
        for idx in front:
            if idx == removed_idx:
                continue
            updated.append(idx - 1 if idx > removed_idx else idx)
        if updated:
            new_fronts.append(updated)
    return new_fronts


def ranks_from_fronts(fronts: list[list[int]], size: int) -> IntArray:
    ranks = np.empty(size, dtype=int)
    for rank, front in enumerate(fronts):
        ranks[np.asarray(front, dtype=int)] = rank
    return ranks


def _constraints_active(G: FloatArray | None, constraint_mode: str) -> bool:
    return G is not None and constraint_mode != "none" and G.size > 0


def _build_selection_cache(
    F: FloatArray,
    G: FloatArray | None,
    constraint_mode: str,
    kernel: KernelBackend,
) -> tuple[IntArray, FloatArray, list[list[int]] | None]:
    if F.shape[0] == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float), []

    if not _constraints_active(G, constraint_mode):
        ranks, crowding = kernel.nsga2_ranking(F)
        return ranks.astype(int, copy=False), crowding, fronts_from_ranks(ranks)

    feasible = is_feasible(G, n=F.shape[0])
    if bool(np.all(feasible)):
        ranks, crowding = kernel.nsga2_ranking(F)
        return ranks.astype(int, copy=False), crowding, fronts_from_ranks(ranks)

    ranks = np.empty(F.shape[0], dtype=int)
    crowding = np.zeros(F.shape[0], dtype=float)
    next_rank = 0

    if bool(np.any(feasible)):
        feasible_idx = np.flatnonzero(feasible)
        feasible_ranks, feasible_crowding = kernel.nsga2_ranking(F[feasible_idx])
        ranks[feasible_idx] = feasible_ranks
        crowding[feasible_idx] = feasible_crowding
        next_rank = int(feasible_ranks.max(initial=-1)) + 1

    violation = compute_violation(G)
    infeasible_idx = np.flatnonzero(~feasible)
    if infeasible_idx.size:
        order = np.argsort(violation[infeasible_idx], kind="mergesort")
        sorted_idx = infeasible_idx[order]
        sorted_cv = violation[sorted_idx]
        start = 0
        while start < sorted_idx.size:
            end = start + 1
            while end < sorted_idx.size and sorted_cv[end] == sorted_cv[start]:
                end += 1
            ranks[sorted_idx[start:end]] = next_rank
            next_rank += 1
            start = end

    return ranks, crowding, None


def refresh_selection_cache(
    st: SMSEMOAState,
    kernel: KernelBackend,
) -> tuple[IntArray, FloatArray, list[list[int]] | None]:
    """Refresh cached ranks/crowding/fronts for parent selection and steady-state updates."""
    ranks, crowding, fronts = _build_selection_cache(st.F, st.G, st.constraint_mode, kernel)
    st.ranks = ranks
    st.crowding = crowding
    st.fronts = fronts
    return ranks, crowding, fronts


def _ensure_objective_buffer(st: SMSEMOAState, pop_n: int, n_obj: int) -> FloatArray:
    F_work = st._survival_F
    if F_work is None or F_work.shape != (pop_n + 1, n_obj):
        F_work = np.empty((pop_n + 1, n_obj), dtype=st.F.dtype)
        st._survival_F = F_work
    return F_work


def _ensure_constraint_buffer(
    st: SMSEMOAState,
    pop_n: int,
    n_constraints: int,
) -> FloatArray:
    if st.G is None:
        raise ValueError("Constraint buffer requested without state.G.")
    G_work = st._survival_G
    if G_work is None or G_work.shape != (pop_n + 1, n_constraints):
        G_work = np.empty((pop_n + 1, n_constraints), dtype=st.G.dtype)
        st._survival_G = G_work
    return G_work


def _write_child_survivor(
    st: SMSEMOAState,
    X_child: AnyArray,
    F_child: FloatArray,
    G_child: FloatArray | None,
    pop_n: int,
    remove_idx: int,
) -> None:
    if remove_idx < pop_n - 1:
        st.X[remove_idx : pop_n - 1] = st.X[remove_idx + 1 : pop_n]
        st.F[remove_idx : pop_n - 1] = st.F[remove_idx + 1 : pop_n]
        if st.G is not None:
            if G_child is None:
                raise ValueError("Constrained SMS-EMOA survival requires child constraint values.")
            st.G[remove_idx : pop_n - 1] = st.G[remove_idx + 1 : pop_n]
        if st.ids is not None and st.pending_offspring_ids is not None:
            st.ids[remove_idx : pop_n - 1] = st.ids[remove_idx + 1 : pop_n]

    st.X[pop_n - 1] = X_child[0]
    st.F[pop_n - 1] = F_child[0]
    if st.G is not None:
        if G_child is None:
            raise ValueError("Constrained SMS-EMOA survival requires child constraint values.")
        st.G[pop_n - 1] = G_child[0]
    if st.ids is not None and st.pending_offspring_ids is not None:
        st.ids[pop_n - 1] = st.pending_offspring_ids[0]


def survival_selection(
    st: SMSEMOAState,
    X_child: AnyArray,
    F_child: FloatArray,
    G_child: FloatArray | None,
    kernel: KernelBackend,
) -> None:
    """Perform steady-state SMS-EMOA survival selection."""
    pop_n = int(st.X.shape[0])
    if pop_n == 0:
        raise ValueError("Cannot perform survival selection with an empty population.")

    if X_child.shape[0] != 1 or F_child.shape[0] != 1:
        raise ValueError("survival_selection expects exactly one offspring (shape (1, ...)).")

    if st.eliminate_duplicates:
        child_x = X_child[0]
        child_f = F_child[0]
        if _is_duplicate_row(st.X, child_x) or _is_duplicate_row(st.F, child_f):
            return

    F_work = _ensure_objective_buffer(st, pop_n, st.F.shape[1])
    F_work[:pop_n] = st.F
    F_work[pop_n] = F_child[0]

    if st.ref_adaptive:
        st.ref_point = update_reference_point(st.ref_point, F_work, st.ref_offset)

    G_work: FloatArray | None = None
    constraints_active = _constraints_active(st.G, st.constraint_mode)
    if constraints_active:
        if G_child is None:
            raise ValueError("Constrained SMS-EMOA survival requires child constraint values.")
        assert st.G is not None
        G_work = _ensure_constraint_buffer(st, pop_n, st.G.shape[1])
        G_work[:pop_n] = st.G
        G_work[pop_n] = G_child[0]
        feasible = is_feasible(G_work, n=pop_n + 1)
        if not bool(np.all(feasible)):
            violations = compute_violation(G_work)
            worst_cv = float(violations.max(initial=0.0))
            worst_idx = np.flatnonzero(violations == worst_cv)
            remove_idx = int(worst_idx[-1])
            if remove_idx == pop_n:
                return
            _write_child_survivor(st, X_child, F_child, G_child, pop_n, remove_idx)
            refresh_selection_cache(st, kernel)
            return

    if st.fronts is None or st.ranks is None or st.crowding is None:
        refresh_selection_cache(st, kernel)

    if st.fronts is None or st.ranks is None:
        ranks, _ = kernel.nsga2_ranking(F_work)
        worst_rank = int(ranks.max(initial=0))
        worst_idx = np.flatnonzero(ranks == worst_rank)
        if worst_idx.size == 1:
            remove_idx = int(worst_idx[0])
        else:
            remove_idx = int(worst_idx[least_contributor(kernel, F_work[worst_idx], st.ref_point)])
        if remove_idx == pop_n:
            return
        _write_child_survivor(st, X_child, F_child, G_child, pop_n, remove_idx)
        refresh_selection_cache(st, kernel)
        return

    fronts = [list(front) for front in st.fronts]
    ranks = np.empty(pop_n + 1, dtype=int)
    ranks[:pop_n] = st.ranks
    ranks[pop_n] = -1
    incremental_insert_fronts(fronts, ranks, F_work, pop_n)

    worst_front = np.asarray(fronts[-1], dtype=int)
    if worst_front.size == 1:
        remove_idx = int(worst_front[0])
    else:
        remove_idx = int(worst_front[least_contributor(kernel, F_work[worst_front], st.ref_point)])

    if remove_idx == pop_n:
        return

    _write_child_survivor(st, X_child, F_child, G_child, pop_n, remove_idx)
    new_fronts = reindex_fronts(fronts, remove_idx)
    st.fronts = new_fronts
    st.ranks = ranks_from_fronts(new_fronts, pop_n)
    st.crowding = compute_crowding(st.F, new_fronts)


def evaluate_population_with_constraints(
    problem: ProblemProtocol,
    X: AnyArray,
) -> tuple[FloatArray, FloatArray | None]:
    """Evaluate population and compute constraints if present."""
    n_obj = problem.n_obj
    n_constraints = getattr(problem, "n_constraints", 0) or 0

    out: dict[str, FloatArray] = {"F": np.empty((X.shape[0], n_obj), dtype=np.float64)}
    if n_constraints > 0:
        out["G"] = np.empty((X.shape[0], n_constraints), dtype=np.float64)

    problem.evaluate(X, out)
    return out["F"], out.get("G")
