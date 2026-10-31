# algorithm/spea2/helpers.py
"""
Support functions for SPEA2.

This module contains the core SPEA2 selection functions:
- Dominance matrix computation
- SPEA2 fitness (strength + density)
- Environmental selection
- The SPEA2 truncation operator
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.constraints.utils import compute_violation, is_feasible

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import SelectionKernels


def dominance_matrix(
    F: np.ndarray[Any, Any], G: np.ndarray[Any, Any] | None, constraint_mode: str
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None, np.ndarray[Any, Any] | None]:
    """Compute dominance matrix with optional feasibility-aware handling.

    Parameters
    ----------
    F : np.ndarray[Any, Any]
        Objective values, shape (N, n_obj).
    G : np.ndarray[Any, Any] | None
        Constraint values, shape (N, n_con) or None.
    constraint_mode : str
        Constraint handling mode: "none" or "feasibility".

    Returns
    -------
    tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None, np.ndarray[Any, Any] | None]
        (dominance_matrix, feasibility_mask, constraint_violations)
    """
    n = F.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=bool), None, None

    feas: np.ndarray[Any, Any] | None = None
    cv: np.ndarray[Any, Any] | None = None

    if constraint_mode and constraint_mode != "none" and G is not None:
        cv = compute_violation(G)
        feas = is_feasible(G)

    dom = np.zeros((n, n), dtype=bool)
    if feas is not None and cv is not None:
        feas_col = feas[:, None]
        feasible_dom = feas_col & ~feas[None, :]
        infeasible_dom = (~feas_col) & (~feas[None, :]) & (cv[:, None] < cv[None, :])
        dom |= feasible_dom | infeasible_dom

        if feas.any():
            less_equal = F[:, None, :] <= F[None, :, :]
            strictly_less = F[:, None, :] < F[None, :, :]
            pareto = np.all(less_equal, axis=2) & np.any(strictly_less, axis=2)
            dom |= (feas_col & feas[None, :]) & pareto
    else:
        less_equal = F[:, None, :] <= F[None, :, :]
        strictly_less = F[:, None, :] < F[None, :, :]
        dom = np.all(less_equal, axis=2) & np.any(strictly_less, axis=2)

    np.fill_diagonal(dom, False)

    return dom, feas, cv


def spea2_fitness(
    F: np.ndarray[Any, Any], dom: np.ndarray[Any, Any], k: int | None = None
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Compute SPEA2 fitness and distance matrix.

    SPEA2 fitness consists of:
    1. Raw fitness: sum of strengths of all dominators
    2. Density: based on k-th nearest neighbor distance

    Parameters
    ----------
    F : np.ndarray[Any, Any]
        Objective values, shape (N, n_obj).
    dom : np.ndarray[Any, Any]
        Dominance matrix, shape (N, N).
    k : int | None
        k for k-th nearest neighbor distance. Defaults to sqrt(N).

    Returns
    -------
    tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
        (fitness, distance_matrix)
    """
    n = F.shape[0]
    if n == 0:
        return np.empty(0), np.empty((0, 0))

    if k is None:
        k = max(1, int(np.sqrt(n)))
    k = min(max(1, int(k)), n - 1) if n > 1 else 1

    raw_fitness = strength_raw_fitness(dom)
    dist = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)

    if n == 1:
        density = np.array([0.0], dtype=float)
    else:
        sigma_k = np.partition(dist, kth=k, axis=1)[:, k]
        density = 1.0 / (sigma_k + 2.0)

    return raw_fitness + density, dist


def strength_raw_fitness(dom: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Compute SPEA2 raw fitness (strength ranking) from dominance matrix."""
    n = dom.shape[0]
    if n == 0:
        return np.empty(0, dtype=float)
    strength = dom.sum(axis=1, dtype=np.int64)
    raw = (dom.T @ strength).astype(float, copy=False)
    return np.asarray(raw, dtype=float)


def knn_density(F: np.ndarray[Any, Any], k: int = 1) -> np.ndarray[Any, Any]:
    """Compute k-th nearest neighbor distance for each solution (higher is better)."""
    return _kth_neighbour_distance(np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2), k)


def _kth_neighbour_distance(dist: np.ndarray[Any, Any], k: int) -> np.ndarray[Any, Any]:
    n = dist.shape[0]
    if n == 0:
        return np.empty(0, dtype=float)
    if n == 1:
        return np.full(1, np.inf, dtype=float)
    k = min(max(int(k), 1), n - 1)
    # Each row's zero self-distance sorts first, so position k is the k-th neighbour.
    return np.asarray(np.partition(dist, kth=k, axis=1)[:, k], dtype=float)


def selection_fitness_density(
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any] | None,
    constraint_mode: str,
    k_neighbors: int | None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Compute the cached SPEA2 tournament metrics for a population/archive."""
    if F.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    dom, _, _ = dominance_matrix(F, G, constraint_mode)
    raw_fitness = strength_raw_fitness(dom)
    density = knn_density(F, int(k_neighbors) if k_neighbors is not None else 1)
    return raw_fitness, density


def truncate_lexicographic(
    dist_matrix: np.ndarray[Any, Any],
    keep: int,
    selection_kernels: SelectionKernels | None = None,
) -> np.ndarray[Any, Any]:
    """Apply the SPEA2 truncation operator and return the kept rows in ascending order.

    Each step removes the solution whose distances to the remaining ones, sorted
    in ascending order, are lexicographically smallest: the smallest
    nearest-neighbour distance, ties broken by the second nearest, then the
    third, and so on (the ``<=_d`` relation of Zitzler, Laumanns and Thiele,
    2001, as in jMetal's ``EnvironmentalSelection``). Rows with identical sorted
    distances remove the first one.
    """
    n = int(dist_matrix.shape[0])
    if n <= keep:
        return np.arange(n, dtype=int)
    if keep <= 0:
        return np.empty(0, dtype=int)

    dist = np.asarray(dist_matrix, dtype=float)
    if dist.ndim != 2 or dist.shape[1] != n:
        raise ValueError("dist_matrix must be a square 2D array.")

    if selection_kernels is not None and bool(np.isfinite(dist).all()):
        # Same candidates, lexicographic comparison and first-row ties as the loop below.
        return np.asarray(selection_kernels.lexicographic_truncation(dist, keep), dtype=int)

    active: NDArray[np.int_] = np.arange(n, dtype=int)
    while active.size > keep:
        sub = dist[np.ix_(active, active)]
        np.fill_diagonal(sub, np.inf)
        nearest = sub.min(axis=1)
        candidates = np.flatnonzero(nearest == nearest.min())
        if candidates.size > 1:
            # The self-distance sorts last in every row, so it never decides the order.
            rows = np.sort(sub[candidates], axis=1)
            candidates = candidates[np.lexsort(rows.T[::-1])]
        active = np.delete(active, int(candidates[0]))

    return active


def environmental_selection(
    X: np.ndarray[Any, Any],
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any] | None,
    archive_size: int,
    k_neighbors: int | None,
    constraint_mode: str,
    selection_kernels: SelectionKernels | None = None,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any] | None,
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
]:
    """SPEA2 environmental selection (Zitzler, Laumanns and Thiele, 2001, step 3).

    The non-dominated solutions (raw fitness 0) form the new archive. If they
    exceed ``archive_size``, the truncation operator removes the most crowded
    ones; if they fall short, the dominated solutions with the best fitness
    ``F = R + 1 / (sigma_k + 2)`` fill it, in that order.

    Parameters
    ----------
    X : np.ndarray[Any, Any]
        Decision variables, shape (N, n_var).
    F : np.ndarray[Any, Any]
        Objective values, shape (N, n_obj).
    G : np.ndarray[Any, Any] | None
        Constraint values, shape (N, n_con) or None.
    archive_size : int
        Target archive size.
    k_neighbors : int | None
        k of the k-th nearest-neighbour density (1 when ``None``).
    constraint_mode : str
        Constraint handling mode.
    selection_kernels : SelectionKernels | None
        Optional compiled truncation from the selected kernel backend.

    Returns
    -------
    tuple
        ``(X, F, G, raw_fitness, kth_distance)`` of the new archive. The raw
        fitness ``R`` and the k-th nearest-neighbour distance are those of each
        member in the whole input, which SPEA2's mating tournament uses.
    """
    n = F.shape[0]
    dist = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
    kth_distance = _kth_neighbour_distance(dist, int(k_neighbors) if k_neighbors is not None else 1)
    dom, _, _ = dominance_matrix(F, G, constraint_mode)
    raw_fitness = strength_raw_fitness(dom)

    nondominated = np.flatnonzero(raw_fitness == 0.0)
    if n <= archive_size:
        selected: NDArray[np.int_] = np.arange(n, dtype=int)
    elif nondominated.size > archive_size:
        keep = truncate_lexicographic(dist[np.ix_(nondominated, nondominated)], archive_size, selection_kernels)
        selected = nondominated[keep]
    else:
        dominated = np.flatnonzero(raw_fitness > 0.0)
        fitness = raw_fitness[dominated] + 1.0 / (kth_distance[dominated] + 2.0)
        fill = dominated[np.argsort(fitness, kind="stable")[: archive_size - nondominated.size]]
        selected = np.concatenate([nondominated, fill])

    return (
        X[selected],
        F[selected],
        G[selected] if G is not None else None,
        raw_fitness[selected],
        kth_distance[selected],
    )


def compute_selection_metrics(
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any] | None,
    constraint_mode: str,
    kernel: Any,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Compute selection metrics for mating selection.

    Parameters
    ----------
    F : np.ndarray[Any, Any]
        Objective values.
    G : np.ndarray[Any, Any] | None
        Constraint values.
    constraint_mode : str
        Constraint handling mode.
    kernel : KernelBackend
        Backend for ranking.

    Returns
    -------
    tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
        (ranks, crowding_distances)
    """
    if F.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    if constraint_mode and constraint_mode != "none" and G is not None:
        cv = compute_violation(G)
        feas = is_feasible(G)
        if feas.any():
            feas_idx = np.nonzero(feas)[0]
            ranks, crowd = kernel.nsga2_ranking(F[feas_idx])
            metrics_rank = np.full(F.shape[0], ranks.max(initial=0) + 1, dtype=int)
            metrics_crowd = np.zeros(F.shape[0], dtype=float)
            metrics_rank[feas_idx] = ranks
            metrics_crowd[feas_idx] = crowd
            metrics_crowd[~feas] = -cv[~feas]
            return metrics_rank, metrics_crowd
        ranks = np.zeros(F.shape[0], dtype=int)
        crowd = -cv
        return ranks, crowd

    ranks, crowd = kernel.nsga2_ranking(F)
    return np.asarray(ranks, dtype=int), np.asarray(crowd, dtype=float)


__all__ = [
    "dominance_matrix",
    "spea2_fitness",
    "strength_raw_fitness",
    "knn_density",
    "selection_fitness_density",
    "truncate_lexicographic",
    "environmental_selection",
    "compute_selection_metrics",
]
