"""
Support functions for the NSGA-II evolutionary loop.
Separated to keep the main algorithm class focused on orchestration.
"""

from __future__ import annotations

from typing import Any, Literal, overload

import numpy as np

from vamos.engine.algorithm.components.matching import match_ids_by_tolerance
from vamos.engine.hooks.genealogy import (
    GenealogyTracker,
)
from vamos.engine.hooks.genealogy import (
    generation_contributions as genealogy_generation_contributions,
)
from vamos.engine.hooks.genealogy import (
    operator_success_stats as genealogy_operator_success_stats,
)
from vamos.foundation.constraints.utils import compute_violation, is_feasible
from vamos.foundation.kernel.backend import DecisionArray, FloatArray, IndexArray, NumericArray


def build_mating_pool(
    kernel: Any,
    ranks: NumericArray,
    crowding: FloatArray,
    pressure: int,
    rng: np.random.Generator,
    parent_count: int,
    group_size: int = 2,
    selection_method: str = "tournament",
    candidate_indices: IndexArray | None = None,
) -> IndexArray:
    if parent_count <= 0:
        raise ValueError("parent_count must be positive.")
    if parent_count % group_size != 0:
        raise ValueError("parent_count must be divisible by group_size.")
    assert ranks.shape == crowding.shape, "ranks and crowding must align"
    cand: IndexArray | None = None
    if candidate_indices is not None:
        cand = np.asarray(candidate_indices, dtype=int)
        cand = cand[(cand >= 0) & (cand < ranks.size)]
        cand = np.unique(cand)
        if cand.size == 0:
            cand = None

    if selection_method == "random":
        if cand is None:
            parent_indices = rng.integers(0, ranks.size, size=parent_count, dtype=int)
        else:
            parent_indices = rng.choice(cand, size=parent_count, replace=True)
    elif selection_method in ("boltzmann", "ranking", "sus"):
        # Build a composite fitness score from ranks and crowding.
        # Lower rank is better; within the same rank, higher crowding is better.
        cand_ranks = ranks if cand is None else ranks[cand]
        cand_crowd = crowding if cand is None else crowding[cand]
        max_rank = int(cand_ranks.max()) if cand_ranks.size else 0
        max_crowd = float(np.nanmax(cand_crowd[np.isfinite(cand_crowd)])) if np.any(np.isfinite(cand_crowd)) else 1.0
        # Normalise crowding to [0, 1] (inf -> 1.0)
        norm_crowd = np.where(np.isfinite(cand_crowd), cand_crowd / (max_crowd + 1e-30), 1.0)
        fitness = (max_rank - cand_ranks).astype(float) + norm_crowd
        local_items = list(range(cand_ranks.size))

        if selection_method == "boltzmann":
            from vamos.engine.algorithm.components.selection import BoltzmannSelection

            parent_local = BoltzmannSelection(temperature=max(float(pressure), 0.1), rng=rng)(
                local_items,
                parent_count,
                fitness=fitness,
            )
        elif selection_method == "ranking":
            from vamos.engine.algorithm.components.selection import RankingSelection

            sp = min(max(float(pressure) / 5.0, 1.0), 2.0)  # map pressure [2..10] -> sp [1..2]
            parent_local = RankingSelection(sp=sp, rng=rng)(
                local_items,
                parent_count,
                fitness=fitness,
            )
        else:  # sus
            from vamos.engine.algorithm.components.selection import SUSSelection

            parent_local = SUSSelection(rng=rng)(
                local_items,
                parent_count,
                fitness=fitness,
            )
        parent_local_arr = np.asarray(parent_local, dtype=int)
        parent_indices = parent_local_arr if cand is None else cand[parent_local_arr]
    else:
        if cand is None:
            parent_indices = kernel.tournament_selection(
                ranks,
                crowding,
                pressure,
                rng,
                n_parents=parent_count,
            )
        else:
            parent_local = kernel.tournament_selection(
                ranks[cand],
                crowding[cand],
                pressure,
                rng,
                n_parents=parent_count,
            )
            parent_indices = cand[np.asarray(parent_local, dtype=int)]
    if parent_indices.size != parent_count:
        raise ValueError("Selection operator returned an unexpected number of parents.")
    return parent_indices.reshape(parent_count // group_size, group_size)


@overload
def feasible_nsga2_survival(
    kernel: Any,
    X: DecisionArray,
    F: FloatArray,
    G: FloatArray | None,
    X_off: DecisionArray,
    F_off: FloatArray,
    G_off: FloatArray | None,
    pop_size: int,
    return_indices: Literal[False] = False,
) -> tuple[DecisionArray, FloatArray, FloatArray | None]: ...


@overload
def feasible_nsga2_survival(
    kernel: Any,
    X: DecisionArray,
    F: FloatArray,
    G: FloatArray | None,
    X_off: DecisionArray,
    F_off: FloatArray,
    G_off: FloatArray | None,
    pop_size: int,
    return_indices: Literal[True],
) -> tuple[DecisionArray, FloatArray, FloatArray | None, IndexArray]: ...


def feasible_nsga2_survival(
    kernel: Any,
    X: DecisionArray,
    F: FloatArray,
    G: FloatArray | None,
    X_off: DecisionArray,
    F_off: FloatArray,
    G_off: FloatArray | None,
    pop_size: int,
    return_indices: bool = False,
) -> tuple[DecisionArray, FloatArray, FloatArray | None] | tuple[DecisionArray, FloatArray, FloatArray | None, IndexArray]:
    """
    Feasibility rule:
      - Feasible dominate infeasible.
      - Among feasible: standard NSGA-II rank/crowding.
      - Among infeasible: lower sum violation wins.
    """
    X_comb = np.vstack([X, X_off])
    F_comb = np.vstack([F, F_off])
    G_comb = np.vstack([G, G_off]) if G is not None and G_off is not None else None
    if G_comb is None:
        if return_indices:
            X_sel, F_sel, sel = kernel.nsga2_survival(X, F, X_off, F_off, pop_size, return_indices=True)
            return X_sel, F_sel, None, sel
        X_sel, F_sel = kernel.nsga2_survival(X, F, X_off, F_off, pop_size)
        return X_sel, F_sel, None

    feas = is_feasible(G_comb)
    cv = compute_violation(G_comb)
    selected = []

    if feas.any():
        feas_idx = np.nonzero(feas)[0]
        ranks, crowd = kernel.nsga2_ranking(F_comb[feas_idx])
        # Order feasible by rank then crowding
        order = np.lexsort((-crowd, ranks))
        feas_ordered = feas_idx[order]
        selected.extend(feas_ordered.tolist())

    if len(selected) < pop_size:
        infeas_idx = np.nonzero(~feas)[0]
        if infeas_idx.size:
            order_infeas = infeas_idx[np.argsort(cv[infeas_idx])]
            selected.extend(order_infeas.tolist())

    selected = selected[:pop_size]
    if return_indices:
        return X_comb[selected], F_comb[selected], G_comb[selected], np.asarray(selected, dtype=int)
    return X_comb[selected], F_comb[selected], G_comb[selected]


def match_ids(new_X: DecisionArray, combined_X: DecisionArray, combined_ids: IndexArray) -> IndexArray:
    """Map surviving rows in new_X back to their ids from combined_X/combined_ids.

    Uses a close-enough row match (tolerance 1e-12) to handle floating-point
    drift introduced by genetic operators.  Falls back to -1 when no match is
    found.
    """
    return match_ids_by_tolerance(new_X, combined_X, combined_ids)


def operator_success_stats(tracker: GenealogyTracker, final_ids: list[int]) -> list[dict[str, object]]:
    return genealogy_operator_success_stats(tracker, final_ids)


def generation_contributions(tracker: GenealogyTracker, final_ids: list[int]) -> list[dict[str, object]]:
    return genealogy_generation_contributions(tracker, final_ids)


def fronts_from_ranks(ranks: IndexArray) -> list[list[int]]:
    if ranks.size == 0:
        return []
    max_rank = int(ranks.max(initial=-1))
    return [np.flatnonzero(ranks == r).tolist() for r in range(max_rank + 1)]


def compute_front_crowding(F: FloatArray, front: IndexArray | list[int]) -> FloatArray:
    """Compute crowding distance for a single front."""
    front_arr = np.asarray(front, dtype=int)
    if front_arr.size == 0:
        return np.empty(0, dtype=float)
    if front_arr.size <= 2:
        # One or two rows are both boundary points; an objective without spread is skipped below.
        return np.full(front_arr.size, np.inf, dtype=float)

    fvals = F[front_arr]
    n_obj = fvals.shape[1]
    d = np.zeros(front_arr.size, dtype=float)
    for m in range(n_obj):
        order = np.argsort(fvals[:, m], kind="mergesort")
        sorted_vals = fvals[order, m]
        span = sorted_vals[-1] - sorted_vals[0]
        if span <= 0.0:
            continue
        d[order[0]] = np.inf
        d[order[-1]] = np.inf
        contrib = np.zeros_like(sorted_vals)
        contrib[1:-1] = (sorted_vals[2:] - sorted_vals[:-2]) / span
        d[order[1:-1]] += contrib[1:-1]
    return d


def compute_crowding(F: FloatArray, fronts: list[list[int]]) -> FloatArray:
    """Compute crowding distance for the provided fronts."""
    N = F.shape[0]
    crowding = np.zeros(N, dtype=float)
    for front in fronts:
        if not front:
            continue
        front_arr = np.asarray(front, dtype=int)
        crowding[front_arr] = compute_front_crowding(F, front_arr)
    return crowding


def select_nsga2(fronts: list[list[int]], crowding: FloatArray, pop_size: int) -> IndexArray:
    """NSGA-II elitist selection based on fronts + crowding."""
    selected: list[int] = []
    for front in fronts:
        if not front:
            continue
        front_arr = np.asarray(front, dtype=int)
        if len(selected) + front_arr.size <= pop_size:
            selected.extend(front_arr.tolist())
        else:
            rem = pop_size - len(selected)
            order = np.argsort(crowding[front_arr])[::-1]
            selected.extend(front_arr[order[:rem]].tolist())
            break
    return np.asarray(selected, dtype=int)


def incremental_insert_fronts(
    fronts: list[list[int]],
    ranks: IndexArray,
    F: FloatArray,
    new_idx: int,
) -> list[int]:
    """
    Incrementally insert a single solution into existing non-dominated fronts.

    Assumes fronts are valid and indices are sorted. Returns affected front indices.
    """
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
    """Remove index from fronts and shift indices above it down by one."""
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


def ranks_from_fronts(fronts: list[list[int]], size: int) -> IndexArray:
    ranks = np.empty(size, dtype=int)
    for rank, front in enumerate(fronts):
        ranks[np.asarray(front, dtype=int)] = rank
    return ranks


__all__ = [
    "build_mating_pool",
    "feasible_nsga2_survival",
    "match_ids",
    "operator_success_stats",
    "generation_contributions",
    "fronts_from_ranks",
    "compute_front_crowding",
    "compute_crowding",
    "select_nsga2",
    "incremental_insert_fronts",
    "reindex_fronts",
    "ranks_from_fronts",
]
