"""
Ask/tell operations for NSGA-II.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.termination import capped_offspring_size


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


from .helpers import (
    build_mating_pool,
    compute_crowding,
    compute_front_crowding,
    fronts_from_ranks,
    incremental_insert_fronts,
    select_nsga2,
)
from .state import compute_selection_metrics, track_offspring_genealogy, update_archives
from .steady_state import ask_steady_state, tell_steady_state

if TYPE_CHECKING:
    from .nsgaii import NSGAII
    from .state import NSGAIIState


def combine_ids(st: NSGAIIState) -> np.ndarray[Any, Any] | None:
    if not st.track_genealogy:
        return None
    current_ids = st.ids if st.ids is not None else np.array([], dtype=int)
    pending_ids = st.pending_offspring_ids if st.pending_offspring_ids is not None else np.array([], dtype=int)
    return np.asarray(np.concatenate([current_ids, pending_ids]), dtype=int)


def _coerce_parent_candidates(raw: Any, size: int) -> np.ndarray[Any, Any] | None:
    if raw is None:
        return None
    arr = np.asarray(raw)
    if arr.ndim != 1:
        return None
    if arr.dtype == bool:
        if arr.size != size:
            return None
        return np.flatnonzero(arr)
    try:
        idx = arr.astype(int, copy=False)
    except Exception:
        return None
    idx = idx[(idx >= 0) & (idx < size)]
    if idx.size == 0:
        return None
    unique_idx: np.ndarray[Any, Any] = np.unique(idx)
    return unique_idx


def _prefer_full_numba_steady_state_survival(st: NSGAIIState, kernel: Any) -> bool:
    """Prefer full Numba survival for bi-objective steady-state runs."""
    return getattr(kernel, "name", "") == "numba" and st.F.shape[1] == 2


def _build_incremental_survivor_state(
    fronts: list[list[int]],
    ranks: np.ndarray[Any, Any],
    crowding: np.ndarray[Any, Any],
    selected_idx: np.ndarray[Any, Any],
    new_F: np.ndarray[Any, Any],
) -> tuple[list[list[int]], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Rebuild cached fronts/ranks/crowding after selecting survivors from N+1 candidates.

    Only the frontier that lost the excluded individual needs fresh crowding
    distances; all earlier selected fronts can reuse the combined-population
    crowding values unchanged.
    """
    total = crowding.shape[0]
    new_size = selected_idx.shape[0]
    position = np.full(total, -1, dtype=int)
    position[selected_idx] = np.arange(new_size, dtype=int)

    new_fronts: list[list[int]] = []
    for front in fronts:
        mapped = [int(position[idx]) for idx in front if position[idx] >= 0]
        if mapped:
            mapped.sort()
            new_fronts.append(mapped)

    new_ranks = np.empty(new_size, dtype=int)
    for rank, front in enumerate(new_fronts):
        new_ranks[np.asarray(front, dtype=int)] = rank

    new_crowding = crowding[selected_idx].copy()
    removed_mask = np.ones(total, dtype=bool)
    removed_mask[selected_idx] = False
    removed = np.flatnonzero(removed_mask)
    if removed.size != 1:
        return new_fronts, new_ranks, compute_crowding(new_F, new_fronts)

    removed_rank = int(ranks[int(removed[0])])
    if removed_rank < len(new_fronts):
        changed_front = np.asarray(new_fronts[removed_rank], dtype=int)
        new_crowding[changed_front] = compute_front_crowding(new_F, changed_front)

    return new_fronts, new_ranks, new_crowding


def _prefilter_archive_candidates(
    st: NSGAIIState,
    kernel: Any,
    X: np.ndarray[Any, Any],
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any] | None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
    """Trim unconstrained archive updates to the combined nondominated set."""
    if (st.archive_manager is None and st.result_archive is None) or G is not None or F.shape[0] <= st.pop_size:
        return X, F, G

    try:
        ranks, _ = kernel.nsga2_ranking(F)
    except (ValueError, IndexError):
        _logger().debug("Failed to prefilter archive candidates; using full batch", exc_info=True)
        return X, F, G

    nd_mask = ranks == ranks.min(initial=0)
    if bool(np.all(nd_mask)):
        return X, F, G
    return X[nd_mask], F[nd_mask], None


def _update_steady_state_archives(
    st: NSGAIIState,
    kernel: Any,
    replaced: bool,
    previous: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
    X_off: np.ndarray[Any, Any],
    F_off: np.ndarray[Any, Any],
) -> None:
    """Archive update of the reference ``tell`` after a compiled steady-state step."""
    if not replaced:
        update_archives(st, kernel, X=st.X, F=st.F)
        return
    combined_X = np.vstack([previous[0], X_off])
    combined_F = np.vstack([previous[1], F_off])
    archive_X, archive_F, archive_G = _prefilter_archive_candidates(st, kernel, combined_X, combined_F, None)
    update_archives(st, kernel, X=archive_X, F=archive_F, G=archive_G)


def ask_nsgaii(algo: NSGAII) -> np.ndarray[Any, Any]:
    st = algo._st
    if st is None:
        raise RuntimeError("ask() called before initialization.")

    if st.steady_step is not None:
        X_steady = ask_steady_state(st, st.steady_step)
        if X_steady is not None:
            return X_steady

    if st.incremental_enabled and st.ranks is not None and st.crowding is not None and st.G is None:
        ranks, crowding = st.ranks, st.crowding
    else:
        ranks, crowding = compute_selection_metrics(
            algo.kernel,
            st.F,
            st.G,
            st.constraint_mode,
            precomputed=st.take_generation_ranking(),
        )
        if st.incremental_mode:
            st.ranks = ranks
            st.crowding = crowding
            st.fronts = fronts_from_ranks(ranks)
    request_size = capped_offspring_size(st.n_eval, st.max_evals, st.offspring_size, "NSGA-II")
    parents_per_group = st.variation.parents_per_group
    children_per_group = st.variation.children_per_group
    parent_count = int(np.ceil(request_size / children_per_group) * parents_per_group)

    candidate_indices: np.ndarray[Any, Any] | None = None
    filter_fn = st.parent_selection_filter
    if callable(filter_fn):
        selected_raw: Any | None = None
        try:
            selected_raw = filter_fn(st, ranks, crowding)
        except TypeError:
            try:
                selected_raw = filter_fn(st)
            except Exception:
                _logger().debug("parent_selection_filter(state) failed; using default selection", exc_info=True)
                selected_raw = None
        except Exception:
            _logger().debug("parent_selection_filter failed; using default selection", exc_info=True)
            selected_raw = None
        selected_idx = _coerce_parent_candidates(selected_raw, st.X.shape[0])
        if selected_idx is not None and selected_idx.size > 0:
            candidate_indices = selected_idx

    if st.non_breeding_indices.size > 0:
        if candidate_indices is None:
            candidate_indices = np.arange(st.X.shape[0], dtype=int)
        blocked = np.asarray(st.non_breeding_indices, dtype=int)
        blocked = blocked[(blocked >= 0) & (blocked < st.X.shape[0])]
        if blocked.size > 0:
            candidate_indices = candidate_indices[~np.isin(candidate_indices, blocked)]
            if candidate_indices.size == 0:
                candidate_indices = None

    mating_pairs = build_mating_pool(
        algo.kernel,
        ranks,
        crowding,
        st.pressure,
        st.rng,
        parent_count,
        parents_per_group,
        st.sel_method,
        candidate_indices=candidate_indices,
    )
    parent_idx = mating_pairs.reshape(-1)
    if st.immigration_manager is not None:
        st.immigration_manager.record_parent_indices(st.generation, parent_idx)
    X_parents = st.variation.gather_parents(st.X, parent_idx)
    X_off = st.variation.produce_offspring(X_parents, st.rng)

    if X_off.shape[0] > request_size:
        X_off = X_off[:request_size]
    st.pending_offspring = X_off

    track_offspring_genealogy(st, parent_idx, X_off.shape[0])
    return X_off


def tell_nsgaii(algo: NSGAII, eval_result: Any) -> bool:
    st = algo._st
    if st is None:
        raise RuntimeError("tell() called before initialization.")

    X_off = st.pending_offspring
    st.pending_offspring = None
    if X_off is None:
        raise ValueError("tell() called without a pending ask().")

    F_off = eval_result.F
    G_off = eval_result.G if st.constraint_mode != "none" else None
    assert st.hv_tracker is not None

    if st.steady_step is not None:
        previous = (st.X, st.F)
        replaced = tell_steady_state(st, st.steady_step, X_off, F_off, G_off)
        if replaced is not None:
            if st.archive_manager is not None or st.result_archive is not None:
                _update_steady_state_archives(st, algo.kernel, replaced, previous, X_off, F_off)
            return st.hv_tracker.enabled and st.hv_tracker.reached(st.hv_points_fn())
        if st.incremental_enabled and st.fronts is None and st.ranks is not None:
            # The compiled step keeps ranks only; the reference path below reads the fronts.
            st.fronts = fronts_from_ranks(st.ranks)

    combined_X = np.vstack([st.X, X_off])
    combined_F = np.vstack([st.F, F_off])
    combined_G = np.vstack([st.G, G_off]) if st.G is not None and G_off is not None else None
    combined_ids = combine_ids(st)
    used_incremental = False

    early_reject = False
    if st.incremental_mode and st.fronts is not None and st.G is None and G_off is None and F_off is not None:
        worst_front = st.fronts[-1] if st.fronts else []
        if worst_front:
            F_worst = st.F[np.asarray(worst_front, dtype=int)]
            F_off_arr = np.asarray(F_off, dtype=float)
            if F_off_arr.ndim == 1:
                F_off_arr = F_off_arr.reshape(1, -1)
            less_equal = F_worst[:, None, :] <= F_off_arr[None, :, :]
            strictly_less = F_worst[:, None, :] < F_off_arr[None, :, :]
            dominates = np.all(less_equal, axis=2) & np.any(strictly_less, axis=2)
            dominated_by_worst = np.any(dominates, axis=0)
            if dominated_by_worst.size and bool(np.all(dominated_by_worst)):
                early_reject = True

    use_incremental = (
        st.incremental_enabled
        and st.replacement_size == 1
        and st.G is None
        and G_off is None
        and X_off.shape[0] == 1
        and not _prefer_full_numba_steady_state_survival(st, algo.kernel)
    )
    if early_reject:
        new_X = st.X
        new_F = st.F
        new_G = st.G
        used_incremental = True
    elif use_incremental:
        if st.fronts is None or st.ranks is None or st.crowding is None:
            ranks, crowding = algo.kernel.nsga2_ranking(st.F)
            st.ranks = ranks
            st.crowding = crowding
            st.fronts = fronts_from_ranks(ranks)

        fronts = [list(front) for front in (st.fronts or [])]
        base_ranks = st.ranks if st.ranks is not None else np.empty(0, dtype=int)
        ranks = np.concatenate([base_ranks, np.array([-1], dtype=int)])
        incremental_insert_fronts(fronts, ranks, combined_F, combined_F.shape[0] - 1)
        crowding = compute_crowding(combined_F, fronts)

        selected_idx = select_nsga2(fronts, crowding, st.pop_size)
        new_X = combined_X[selected_idx]
        new_F = combined_F[selected_idx]
        new_G = None

        new_fronts, new_ranks, new_crowding = _build_incremental_survivor_state(fronts, ranks, crowding, selected_idx, new_F)

        st.fronts = new_fronts
        st.ranks = new_ranks
        st.crowding = new_crowding
        used_incremental = True
    elif st.G is None or G_off is None or st.constraint_mode == "none":
        new_X, new_F = algo.kernel.nsga2_survival(st.X, st.F, X_off, F_off, st.pop_size)
        new_G = None
    else:
        from .helpers import feasible_nsga2_survival

        new_X, new_F, new_G = feasible_nsga2_survival(algo.kernel, st.X, st.F, st.G, X_off, F_off, G_off, st.pop_size)

    if combined_ids is not None:
        from .helpers import match_ids

        st.ids = match_ids(new_X, combined_X, combined_ids)

    st.X, st.F, st.G = new_X, new_F, new_G
    st.n_eval += X_off.shape[0]
    st.pending_offspring_ids = None

    if st.incremental_enabled and not used_incremental:
        ranks, crowding = algo.kernel.nsga2_ranking(st.F)
        st.ranks = ranks
        st.crowding = crowding
        st.fronts = fronts_from_ranks(ranks)

    if early_reject:
        update_archives(st, algo.kernel, X=st.X, F=st.F)
    else:
        archive_X, archive_F, archive_G = _prefilter_archive_candidates(
            st,
            algo.kernel,
            combined_X,
            combined_F,
            combined_G,
        )
        update_archives(st, algo.kernel, X=archive_X, F=archive_F, G=archive_G)

    hv_reached = st.hv_tracker.enabled and st.hv_tracker.reached(st.hv_points_fn())

    return hv_reached


__all__ = ["ask_nsgaii", "tell_nsgaii", "combine_ids"]
