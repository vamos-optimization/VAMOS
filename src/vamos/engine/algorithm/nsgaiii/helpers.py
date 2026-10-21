"""NSGA-III helper functions.

This module provides helper functions for NSGA-III including:
- Non-dominated sorting
- Reference point association
- Niching-based survival selection
- ASF (Achievement Scalarization Function) for extremes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.foundation.constraints.utils import compute_violation, is_feasible

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol


__all__ = [
    "fast_non_dominated_sort",
    "identify_extremes",
    "get_extreme_points",
    "get_nadir_point",
    "associate",
    "niche_selection",
    "nsgaiii_survival",
    "evaluate_population_with_constraints",
]


def _fronts_from_ranks(ranks: np.ndarray[Any, Any]) -> list[np.ndarray[Any, Any]]:
    """Rebuild front index arrays from a rank vector in stable index order."""
    if ranks.size == 0:
        return []

    rank_values = np.asarray(ranks, dtype=int)
    max_rank = int(rank_values.max(initial=-1))
    if max_rank < 0:
        return []

    counts = np.bincount(rank_values, minlength=max_rank + 1)
    offsets = np.empty(max_rank + 2, dtype=int)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    grouped = np.empty(rank_values.size, dtype=int)
    cursor = offsets[:-1].copy()
    for idx, rank in enumerate(rank_values):
        grouped[cursor[rank]] = idx
        cursor[rank] += 1

    return [grouped[offsets[rank] : offsets[rank + 1]].copy() for rank in range(max_rank + 1) if counts[rank] > 0]


def fast_non_dominated_sort(
    F: np.ndarray[Any, Any],
    kernel: KernelBackend | None = None,
) -> list[list[int]]:
    """Fast non-dominated sorting.

    Parameters
    ----------
    F : np.ndarray
        Objective values, shape (n, n_obj).

    Returns
    -------
    list[list[int]]
        List of fronts, where each front is a list of indices.
    """
    if kernel is not None:
        ranks, _ = kernel.nsga2_ranking(F)
        return [front.tolist() for front in _fronts_from_ranks(ranks)]

    n = F.shape[0]
    S: list[list[int]] = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    ranks = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                S[p].append(q)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in S[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    ranks[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()  # remove last empty front
    return fronts


def identify_extremes(shifted: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Identify extreme points using ASF (Achievement Scalarization Function)."""
    if shifted.size == 0:
        return np.array([], dtype=int)
    n_obj = shifted.shape[1]
    extremes = np.empty(n_obj, dtype=int)
    unit = np.eye(n_obj)
    for i in range(n_obj):
        weights = np.where(unit[i] == 0, 1e6, 1.0)
        asf = (shifted * weights).max(axis=1)
        extremes[i] = int(np.argmin(asf))
    return extremes


def get_extreme_points(
    F: np.ndarray[Any, Any],
    n_obj: int,
    ideal_point: np.ndarray[Any, Any],
    extreme_points: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
    """Identify extreme points using ASF, preserving previous extremes."""
    if F.size == 0:
        return np.empty((0, n_obj), dtype=float)
    base = F
    if extreme_points is not None and extreme_points.size:
        base = np.vstack([extreme_points, F])

    shifted = base - ideal_point
    shifted[shifted < 1e-3] = 0.0

    asf = np.eye(n_obj)
    asf[asf == 0] = 1e6
    vals = np.max(shifted * asf[:, None, :], axis=2)
    idx = np.argmin(vals, axis=1)
    return np.asarray(base[idx], dtype=float)


def get_nadir_point(
    extreme_points: np.ndarray[Any, Any],
    ideal_point: np.ndarray[Any, Any],
    worst_point: np.ndarray[Any, Any],
    worst_of_front: np.ndarray[Any, Any],
    worst_of_population: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Compute nadir point using extreme points (Deb & Jain 2014)."""
    try:
        M = extreme_points - ideal_point
        b = np.ones(extreme_points.shape[1])
        plane = np.linalg.solve(M, b)
        intercepts = 1.0 / plane
        nadir = ideal_point + intercepts
        if not np.allclose(M @ plane, b) or np.any(intercepts <= 1e-6) or np.any(nadir > worst_point):
            raise np.linalg.LinAlgError
    except Exception:
        nadir = worst_of_front

    mask = nadir - ideal_point <= 1e-6
    nadir = nadir.copy()
    nadir[mask] = worst_of_population[mask]
    return np.asarray(nadir)


def associate(
    normalized_F: np.ndarray[Any, Any],
    ref_dirs_norm: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Associate solutions with reference directions.

    Parameters
    ----------
    normalized_F : np.ndarray
        Normalized objective values, shape (n, n_obj).
    ref_dirs_norm : np.ndarray
        Normalized reference directions, shape (n_ref, n_obj).

    Returns
    -------
    tuple
        (associations, distances) where associations are ref direction indices
        and distances are perpendicular distances to the directions.
    """
    norms = np.linalg.norm(normalized_F, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1e-12)
    normalized_vectors = normalized_F / norms
    cosine = normalized_vectors @ ref_dirs_norm.T
    cosine = np.clip(cosine, -1.0, 1.0)
    associations = np.argmax(cosine, axis=1)
    cos_selected = cosine[np.arange(cosine.shape[0]), associations]
    distances = norms.flatten() * np.sqrt(1.0 - np.square(cos_selected))
    return associations, distances


def niche_selection(
    front: np.ndarray[Any, Any],
    n_remaining: int,
    niche_counts: np.ndarray[Any, Any],
    associations: np.ndarray[Any, Any],
    distances: np.ndarray[Any, Any],
    rng: np.random.Generator,
) -> np.ndarray[Any, Any]:
    """Perform niche-based selection from the critical front (NSGA-III)."""
    selected: list[int] = []
    if front.size == 0 or n_remaining <= 0:
        return np.empty(0, dtype=int)

    mask = np.ones(front.size, dtype=bool)
    while len(selected) < n_remaining:
        remaining = n_remaining - len(selected)
        active_pos = np.where(mask)[0]
        if active_pos.size == 0:
            break

        assoc_front = associations[front[active_pos]]
        unique_refs = np.unique(assoc_front)
        ref_counts = niche_counts[unique_refs]
        min_count = np.min(ref_counts)
        candidate_refs = unique_refs[ref_counts == min_count]
        rng.shuffle(candidate_refs)
        candidate_refs = candidate_refs[:remaining]

        for ref_choice in candidate_refs:
            candidates = active_pos[assoc_front == ref_choice]
            if candidates.size == 0:
                niche_counts[ref_choice] = np.inf
                continue
            rng.shuffle(candidates)
            if niche_counts[ref_choice] == 0:
                cand_dist = distances[front[candidates]]
                pick = candidates[int(np.argmin(cand_dist))]
            else:
                pick = candidates[0]
            mask[pick] = False
            selected.append(int(front[pick]))
            niche_counts[ref_choice] += 1
            if len(selected) >= n_remaining:
                break

    return np.asarray(selected, dtype=int)


def nsgaiii_survival(
    X: np.ndarray[Any, Any],
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any] | None,
    X_off: np.ndarray[Any, Any],
    F_off: np.ndarray[Any, Any],
    G_off: np.ndarray[Any, Any] | None,
    pop_size: int,
    ref_dirs_norm: np.ndarray[Any, Any],
    rng: np.random.Generator,
    ideal_point: np.ndarray[Any, Any],
    extreme_points: np.ndarray[Any, Any] | None,
    worst_point: np.ndarray[Any, Any],
    kernel: KernelBackend | None = None,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any] | None,
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any] | None,
    np.ndarray[Any, Any],
]:
    """Perform NSGA-III survival selection with niching.

    Parameters
    ----------
    X : np.ndarray
        Parent decision vectors.
    F : np.ndarray
        Parent objective values.
    G : np.ndarray or None
        Parent constraint values.
    X_off : np.ndarray
        Offspring decision vectors.
    F_off : np.ndarray
        Offspring objective values.
    G_off : np.ndarray or None
        Offspring constraint values.
    pop_size : int
        Target population size.
    ref_dirs_norm : np.ndarray
        Normalized reference directions.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    tuple
        (X_new, F_new, G_new, survivor_indices) where survivor_indices
        are indices into the combined [X, X_off] array.

    Notes
    -----
    With constraint values, feasible solutions come first (Jain and Deb, 2014):
    when more solutions are feasible than survive, reference-direction niching
    selects among the feasible ones only; otherwise every feasible solution
    survives and the least violated infeasible ones fill the population.
    """
    X_all = np.vstack([X, X_off])
    F_all = np.vstack([F, F_off])
    G_all = np.vstack([G, G_off]) if G is not None and G_off is not None else None

    candidates: np.ndarray[Any, Any] | None = None
    if G_all is not None:
        feasible_mask = is_feasible(G_all)
        feasible = np.flatnonzero(feasible_mask)
        if feasible.size <= pop_size:
            infeasible = np.flatnonzero(~feasible_mask)
            by_violation = infeasible[np.argsort(compute_violation(G_all)[infeasible], kind="stable")]
            survivor_arr = np.concatenate([feasible, by_violation[: pop_size - feasible.size]]).astype(int)
            return (
                X_all[survivor_arr],
                F_all[survivor_arr],
                G_all[survivor_arr],
                survivor_arr,
                ideal_point,
                extreme_points,
                worst_point,
            )
        candidates = feasible

    local, ideal_point, extreme_points, worst_point = _reference_direction_survival(
        F_all if candidates is None else F_all[candidates],
        pop_size,
        ref_dirs_norm,
        rng,
        ideal_point,
        extreme_points,
        worst_point,
        kernel,
    )
    survivor_arr = local if candidates is None else candidates[local]
    return (
        X_all[survivor_arr],
        F_all[survivor_arr],
        G_all[survivor_arr] if G_all is not None else None,
        survivor_arr,
        ideal_point,
        extreme_points,
        worst_point,
    )


def _reference_direction_survival(
    F_all: np.ndarray[Any, Any],
    pop_size: int,
    ref_dirs_norm: np.ndarray[Any, Any],
    rng: np.random.Generator,
    ideal_point: np.ndarray[Any, Any],
    extreme_points: np.ndarray[Any, Any] | None,
    worst_point: np.ndarray[Any, Any],
    kernel: KernelBackend | None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None, np.ndarray[Any, Any]]:
    """Select ``pop_size`` rows of ``F_all`` by non-dominated fronts and reference-direction niching."""
    if kernel is not None:
        ranks, _ = kernel.nsga2_ranking(F_all)
        fronts = _fronts_from_ranks(ranks)
    else:
        fronts = [np.asarray(front, dtype=int) for front in fast_non_dominated_sort(F_all)]
    survivor_indices: list[int] = []

    ideal_point = np.minimum(ideal_point, F_all.min(axis=0))
    worst_point = np.maximum(worst_point, F_all.max(axis=0))

    if fronts:
        front0 = fronts[0]
        nd_F = F_all[front0]
    else:
        nd_F = F_all

    extreme_points = get_extreme_points(nd_F, F_all.shape[1], ideal_point, extreme_points)
    worst_of_population = F_all.max(axis=0)
    worst_of_front = nd_F.max(axis=0) if nd_F.size else worst_of_population
    nadir_point = get_nadir_point(extreme_points, ideal_point, worst_point, worst_of_front, worst_of_population)

    denom = nadir_point - ideal_point
    denom = np.where(denom == 0, 1e-12, denom)
    normalized = (F_all - ideal_point) / denom

    associations, distances = associate(normalized, ref_dirs_norm)
    niche_counts = np.zeros(ref_dirs_norm.shape[0], dtype=int)

    last_front: np.ndarray[Any, Any] | None = None
    for front in fronts:
        if len(survivor_indices) + front.size <= pop_size:
            survivor_indices.extend(front.tolist())
            for idx in front:
                niche_counts[associations[int(idx)]] += 1
        else:
            last_front = front
            break

    if last_front is not None and len(survivor_indices) < pop_size:
        remaining = pop_size - len(survivor_indices)
        selected_idx = niche_selection(last_front, remaining, niche_counts, associations, distances, rng)
        survivor_indices.extend(selected_idx.tolist())

    return np.asarray(survivor_indices, dtype=int), ideal_point, extreme_points, worst_point


def evaluate_population_with_constraints(
    problem: ProblemProtocol,
    X: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
    """Evaluate population and compute constraints if present.

    Parameters
    ----------
    problem : ProblemProtocol
        Problem to evaluate.
    X : np.ndarray
        Decision vectors, shape (pop_size, n_var).

    Returns
    -------
    tuple
        (F, G) objective and constraint values.
    """
    n_obj = problem.n_obj
    n_con = getattr(problem, "n_con", 0) or 0

    out: dict[str, np.ndarray[Any, Any]] = {"F": np.empty((X.shape[0], n_obj), dtype=np.float64)}
    if n_con > 0:
        out["G"] = np.empty((X.shape[0], n_con), dtype=np.float64)

    # Some problems write into preallocated arrays (out["F"][:]=...),
    # while others replace out["F"] entirely (out["F"]=...). Always read back
    # from the out dict to support both.
    problem.evaluate(X, out)
    return out["F"], out.get("G")
