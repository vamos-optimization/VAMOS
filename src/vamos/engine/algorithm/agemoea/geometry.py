"""Geometry helpers for AGE-MOEA survival selection."""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.kernel.backend import KernelBackend, SelectionKernels


def _point_to_line_distance(P: np.ndarray[Any, Any], A: np.ndarray[Any, Any], B: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    ba = B - A
    denom = np.dot(ba, ba)
    if denom == 0.0:
        return np.zeros(P.shape[0], dtype=float)
    pa = P - A
    t = (pa @ ba) / denom
    residual = pa - t[:, None] * ba
    return np.asarray(np.sum(residual * residual, axis=1), dtype=float)


def _find_corner_solutions(front: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    m, n = front.shape
    if m <= n:
        return np.arange(m)
    W = 1e-6 + np.eye(n)
    indexes = np.zeros(n, dtype=int)
    selected = np.zeros(m, dtype=bool)
    for i in range(n):
        dists = _point_to_line_distance(front, np.zeros(n), W[i, :])
        dists[selected] = np.inf
        idx = int(np.argmin(dists))
        indexes[i] = idx
        selected[idx] = True
    return indexes


def _normalize_front(front: np.ndarray[Any, Any], extreme: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    if len(extreme) != len(np.unique(extreme, axis=0)):
        normalization = np.max(front, axis=0)
        normalization[normalization == 0.0] = 1.0
        return front / normalization, normalization

    try:
        hyperplane = np.linalg.solve(front[extreme], np.ones(front.shape[1]))
        if np.any(~np.isfinite(hyperplane)) or np.any(hyperplane <= 0):
            normalization = np.max(front, axis=0)
        else:
            normalization = 1.0 / hyperplane
            if np.any(~np.isfinite(normalization)):
                normalization = np.max(front, axis=0)
    except np.linalg.LinAlgError:
        normalization = np.max(front, axis=0)

    normalization[normalization == 0.0] = 1.0
    return front / normalization, normalization


def _pairwise_distances(front: np.ndarray[Any, Any], p: float) -> np.ndarray[Any, Any]:
    diff = np.abs(front[:, None, :] - front[None, :, :])
    return np.asarray(np.sum(diff**p, axis=2) ** (1.0 / p), dtype=float)


def _minkowski_distances(A: np.ndarray[Any, Any], B: np.ndarray[Any, Any], p: float) -> np.ndarray[Any, Any]:
    diff = np.abs(A[:, None, :] - B[None, :, :])
    return np.asarray(np.sum(diff**p, axis=2) ** (1.0 / p), dtype=float)


def _compute_geometry(front: np.ndarray[Any, Any], extreme: np.ndarray[Any, Any], n_obj: int) -> float:
    d = _point_to_line_distance(front, np.zeros(n_obj), np.ones(n_obj))
    d[extreme] = np.inf
    index = int(np.argmin(d))
    mean_val = np.mean(front[index, :])
    if mean_val <= 0.0:
        return 1.0
    p = np.log(n_obj) / np.log(1.0 / mean_val)
    if np.isnan(p) or p <= 0.1:
        p = 1.0
    elif p > 20.0:
        p = 20.0
    return float(p)


def _survival_score(
    front: np.ndarray[Any, Any],
    ideal_point: np.ndarray[Any, Any],
    selection_kernels: SelectionKernels | None = None,
) -> tuple[np.ndarray[Any, Any], float, np.ndarray[Any, Any]]:
    front = np.round(front, 12, out=front.copy())
    m, n = front.shape
    crowd_dist = np.zeros(m, dtype=float)

    if m < n:
        p = 1.0
        normalization = np.max(front, axis=0)
        normalization[normalization == 0.0] = 1.0
        return crowd_dist, p, normalization

    front = front - ideal_point
    extreme = _find_corner_solutions(front)
    front, normalization = _normalize_front(front, extreme)

    crowd_dist[extreme] = np.inf
    selected = np.full(m, False)
    selected[extreme] = True

    p = _compute_geometry(front, extreme, n)
    nn = np.linalg.norm(front, ord=p, axis=1)
    nn[nn < 1e-8] = 1.0

    distances = _pairwise_distances(front, p)
    distances[distances < 1e-8] = 1e-8
    distances = distances / nn[:, None]

    remaining = np.flatnonzero(~selected)
    selected_idx = np.flatnonzero(selected)
    if remaining.size == 0:
        return crowd_dist, p, normalization

    D_init = distances[np.ix_(remaining, selected_idx)]
    if D_init.shape[1] > 1:
        nearest = np.partition(D_init, kth=1, axis=1)[:, :2]
        best1 = nearest[:, 0].copy()
        best2 = nearest[:, 1].copy()
        scores = best1 + best2
    else:
        best1 = D_init[:, 0].copy()
        best2 = np.zeros_like(best1)
        scores = best1.copy()

    selected_count = selected_idx.size
    if selection_kernels is not None:
        # Same greedy order, first-index ties and float accumulation as the loop below.
        selection_kernels.greedy_two_nearest_selection(distances, remaining, best1, best2, scores, selected_count, crowd_dist)
        return crowd_dist, p, normalization

    while remaining.size > 0:
        index = int(np.argmax(scores))
        best = int(remaining[index])
        d = float(scores[index])
        selected[best] = True
        crowd_dist[best] = d

        remaining = np.delete(remaining, index)
        best1 = np.delete(best1, index)
        best2 = np.delete(best2, index)
        scores = np.delete(scores, index)
        selected_count += 1
        if remaining.size == 0:
            break

        new_dist = distances[remaining, best]
        if selected_count == 2:
            lo = np.minimum(best1, new_dist)
            hi = np.maximum(best1, new_dist)
            best1 = lo
            best2 = hi
            scores = best1 + best2
            continue

        better_first = new_dist < best1
        best2 = np.where(better_first, best1, best2)
        best1 = np.where(better_first, new_dist, best1)
        better_second = (~better_first) & (new_dist < best2)
        best2 = np.where(better_second, new_dist, best2)
        scores = best1 + best2

    return crowd_dist, p, normalization


def age_survival(
    F: np.ndarray[Any, Any],
    n_survive: int,
    kernel: KernelBackend,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Select AGE-MOEA survivors from a combined parent-offspring front.

    Returns the survivor indices in ascending order and their survival scores,
    as PlatEMO's ``EnvironmentalSelection`` (Panichella's own code) defines
    them: the diversity/proximity score on the first front and the inverse
    Minkowski distance to the ideal point on every later front up to the
    truncated one, which is cut by that score. pymoo 0.6.2 leaves the truncated
    front unscored and cuts it by index. AGE-MOEA's mating tournament breaks
    rank ties with these scores.
    """
    n_survive = min(int(n_survive), F.shape[0])
    ranks, _ = kernel.nsga2_ranking(F)
    max_rank = int(ranks.max()) if ranks.size else 0

    fronts: list[np.ndarray[Any, Any]] = []
    ranked = 0
    last_rank = 0
    for r in range(max_rank + 1):
        front = np.where(ranks == r)[0]
        fronts.append(front)
        if ranked + front.size >= n_survive:
            last_rank = r
            break
        ranked += front.size

    selected = ranks < last_rank
    crowd_dist = np.zeros(F.shape[0], dtype=float)

    front0 = F[ranks == 0, :]
    ideal_point = np.min(front0, axis=0)
    crowd_dist[ranks == 0], p, normalization = _survival_score(front0, ideal_point, kernel.selection_kernels())

    for r in range(1, last_rank + 1):
        front_idx = fronts[r]
        if front_idx.size == 0:
            continue
        front = F[front_idx] / normalization
        dist = _minkowski_distances(front, ideal_point[None, :], p).squeeze()
        dist = np.where(dist < 1e-12, 1e-12, dist)
        crowd_dist[front_idx] = 1.0 / dist

    last = fronts[last_rank]
    if last.size > 0:
        order = np.argsort(crowd_dist[last])[::-1]
        remaining = n_survive - int(np.sum(selected))
        selected[last[order[:remaining]]] = True

    survivors = np.flatnonzero(selected)
    return survivors, crowd_dist[survivors]


__all__ = ["age_survival"]
