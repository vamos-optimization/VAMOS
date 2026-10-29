"""
Subset selection functions for choosing representative subsets from Pareto fronts.

All public functions follow the signature ``(F, k, ...) -> NDArray[np.int_]`` where *F*
is an ``(n, n_obj)`` objective matrix and the return value is a 1-D integer index
array of length ``min(k, n)`` into *F*.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.kernel.numpy_backend import _compute_crowding
from vamos.foundation.quality_indicators.hypervolume import hypervolume_contributions
from vamos.foundation.quality_indicators.pareto import pareto_filter

try:  # pragma: no cover - optional dependency
    import moocore as _moocore
except ImportError:  # pragma: no cover - optional dependency
    _moocore = None

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _single_front_crowding(F: NDArray[np.float64]) -> NDArray[np.float64]:
    """Crowding distance for a single nondominated front."""
    if F.shape[0] == 0:
        return np.empty(0, dtype=float)
    fronts = [list(range(F.shape[0]))]
    return _compute_crowding(F, fronts)


def _hv_contributions(F: NDArray[np.float64], ref: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the exact exclusive hypervolume contribution of each point.

    Objectives are minimized. Points that do not strictly dominate ``ref``
    and points dominated by another point contribute 0, and they do not change
    the contributions of the others, as in ``moocore.hv_contributions``.
    Duplicated points contribute 0. moocore computes the values when it is
    installed. Otherwise the non-dominated points inside the reference box go
    to the exact
    :func:`vamos.foundation.quality_indicators.hypervolume.hypervolume_contributions`.
    """
    F_arr = np.asarray(F, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    if F_arr.shape[0] == 0:
        return np.zeros(0, dtype=float)
    if _moocore is not None:
        return np.asarray(_moocore.hv_contributions(F_arr, ref=ref_arr), dtype=float)
    contributions = np.zeros(F_arr.shape[0], dtype=float)
    inside = np.flatnonzero(np.all(F_arr < ref_arr, axis=1))
    if inside.size == 0:
        return contributions
    _, front_local = pareto_filter(F_arr[inside], return_indices=True)
    front = inside[front_local]
    contributions[front] = hypervolume_contributions(F_arr[front], ref_arr)
    return contributions


# ---------------------------------------------------------------------------
# Crowding-distance selection (NSGA-II style)
# ---------------------------------------------------------------------------


def select_top_k_crowding(F: NDArray[np.float64], k: int) -> NDArray[np.int_]:
    """Select the top-*k* most spread solutions using crowding distance.

    Iteratively removes the solution with the lowest crowding distance
    until *k* solutions remain.  Boundary solutions (with ``inf`` crowding)
    are never removed, preserving the extremes of the front.

    If the front has ``<= k`` solutions, all indices are returned.

    Parameters
    ----------
    F:
        Objective matrix of shape ``(n, n_obj)``.  Assumed to be a single
        non-dominated front.
    k:
        Number of solutions to keep.

    Returns
    -------
    NDArray[np.int_]
        1-D integer array of indices into *F* for the selected solutions.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    n = F.shape[0]
    if n <= k:
        return np.arange(n, dtype=int)
    keep: NDArray[np.int_] = np.arange(n, dtype=int)
    while keep.size > k:
        crowd = _single_front_crowding(F[keep])
        worst_local = int(np.argmin(crowd))
        keep = np.delete(keep, worst_local)
    return keep


# ---------------------------------------------------------------------------
# Farthest-point (maximin) sampling
# ---------------------------------------------------------------------------


def select_top_k_farthest(F: NDArray[np.float64], k: int) -> NDArray[np.int_]:
    """Select the top-*k* most diverse solutions using farthest-point sampling.

    Greedy algorithm that iteratively picks the point maximising the minimum
    Euclidean distance to the already-selected set.  This produces a more
    uniformly spread subset than crowding distance, especially for 3+ objectives.

    If the front has ``<= k`` solutions, all indices are returned.

    Parameters
    ----------
    F:
        Objective matrix of shape ``(n, n_obj)``.
    k:
        Number of solutions to keep.

    Returns
    -------
    NDArray[np.int_]
        1-D integer array of indices into *F* for the selected solutions.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    n = F.shape[0]
    if n <= k:
        return np.arange(n, dtype=int)

    # Start from the point with the largest norm (an extreme point)
    selected = [int(np.argmax(np.linalg.norm(F, axis=1)))]
    min_dists = np.full(n, np.inf)

    for _ in range(k - 1):
        dists = np.linalg.norm(F - F[selected[-1]], axis=1)
        np.minimum(min_dists, dists, out=min_dists)
        min_dists[selected] = -1.0
        selected.append(int(np.argmax(min_dists)))

    return np.array(selected, dtype=int)


# ---------------------------------------------------------------------------
# K-nearest-neighbor density selection (SPEA2 style)
# ---------------------------------------------------------------------------


def select_top_k_knn(F: NDArray[np.float64], k: int, *, neighbors: int = 1) -> NDArray[np.int_]:
    """Select the top-*k* most diverse solutions via k-NN density truncation.

    Iteratively removes the solution with the smallest distance to its
    *neighbors*-th nearest neighbor (the most crowded point), preserving
    diversity.  This is the truncation procedure from SPEA2.

    Parameters
    ----------
    F:
        Objective matrix of shape ``(n, n_obj)``.
    k:
        Number of solutions to keep.
    neighbors:
        Which nearest-neighbor distance to use (1 = nearest, 2 = 2nd nearest,
        etc.).  Defaults to 1.

    Returns
    -------
    NDArray[np.int_]
        1-D integer array of indices into *F* for the selected solutions.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    n = F.shape[0]
    if n <= k:
        return np.arange(n, dtype=int)

    dist_matrix = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
    candidates = list(range(n))
    nn = max(1, int(neighbors))

    while len(candidates) > k:
        sub = dist_matrix[np.ix_(candidates, candidates)]
        np.fill_diagonal(sub, np.inf)
        kth = min(nn, len(candidates) - 1)
        nearest = np.partition(sub, kth, axis=1)[:, kth]
        remove_pos = int(np.argmin(nearest))
        del candidates[remove_pos]

    return np.asarray(candidates, dtype=int)


# ---------------------------------------------------------------------------
# Reference-direction association (NSGA-III style)
# ---------------------------------------------------------------------------


def select_top_k_reference_directions(F: NDArray[np.float64], k: int, *, divisions: int | None = None) -> NDArray[np.int_]:
    """Select the top-*k* solutions via reference-direction association.

    Generates *k* uniformly distributed reference directions on the simplex
    (Das-Dennis design), normalizes *F* to [0, 1], and associates each solution
    to its nearest reference direction by perpendicular distance.  Per direction,
    the closest solution is kept.  If fewer than *k* solutions are selected
    (because some directions have no nearby solutions), the remaining slots are
    filled from unselected solutions by smallest perpendicular distance to any
    direction.

    Particularly effective for many-objective problems (3+ objectives).

    Parameters
    ----------
    F:
        Objective matrix of shape ``(n, n_obj)``.
    k:
        Number of solutions to keep.
    divisions:
        Number of divisions for the Das-Dennis simplex lattice.  If ``None``,
        the minimum divisions yielding ``>= k`` reference directions are chosen
        automatically.

    Returns
    -------
    NDArray[np.int_]
        1-D integer array of indices into *F* for the selected solutions.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    n = F.shape[0]
    if n <= k:
        return np.arange(n, dtype=int)

    from vamos.engine.algorithm.components.utils import normalize_objectives
    from vamos.engine.algorithm.components.weight_vectors import (
        load_or_generate_weight_vectors,
    )

    F_norm = normalize_objectives(F)
    weights = load_or_generate_weight_vectors(k, F.shape[1], divisions=divisions)

    # Perpendicular distance from each solution to each reference direction
    # d_perp(x, w) = ||x - (x·w / w·w) * w||
    w_norm_sq = np.sum(weights**2, axis=1)  # (k_ref,)
    proj_coeff = (F_norm @ weights.T) / np.maximum(w_norm_sq, 1e-30)  # (n, k_ref)
    proj = proj_coeff[:, :, None] * weights[None, :, :]  # (n, k_ref, n_obj)
    d_perp = np.linalg.norm(F_norm[:, None, :] - proj, axis=2)  # (n, k_ref)

    # Associate each solution to nearest direction
    assoc = np.argmin(d_perp, axis=1)  # (n,) — direction index per solution

    selected: list[int] = []
    for w_idx in range(weights.shape[0]):
        members = np.flatnonzero(assoc == w_idx)
        if members.size == 0:
            continue
        best = members[int(np.argmin(d_perp[members, w_idx]))]
        selected.append(int(best))

    # Deduplicate (a solution could be nearest to multiple directions)
    selected = list(dict.fromkeys(selected))

    # Fill remaining slots from unselected solutions by smallest d_perp to any direction
    if len(selected) < k:
        remaining = np.setdiff1d(np.arange(n), selected)
        if remaining.size > 0:
            min_d = np.min(d_perp[remaining], axis=1)
            order = np.argsort(min_d)
            need = k - len(selected)
            selected.extend(remaining[order[:need]].tolist())

    # If we somehow got more than k (unlikely), truncate
    selected = selected[:k]

    return np.asarray(selected, dtype=int)


# ---------------------------------------------------------------------------
# K-means clustering selection
# ---------------------------------------------------------------------------


def select_top_k_kmeans(
    F: NDArray[np.float64],
    k: int,
    *,
    n_init: int = 10,
    max_iter: int = 300,
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.int_]:
    """Select the top-*k* solutions using k-means clustering.

    Normalizes *F*, runs k-means with *k* clusters, and picks the solution
    closest to each centroid.  Pure NumPy implementation (no scipy dependency).

    Parameters
    ----------
    F:
        Objective matrix of shape ``(n, n_obj)``.
    k:
        Number of solutions to keep (= number of clusters).
    n_init:
        Number of k-means restarts (best inertia wins).
    max_iter:
        Maximum iterations per restart.
    rng:
        Random seed or ``np.random.Generator`` for reproducibility.

    Returns
    -------
    NDArray[np.int_]
        1-D integer array of indices into *F* for the selected solutions.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    n = F.shape[0]
    if n <= k:
        return np.arange(n, dtype=int)

    from vamos.engine.algorithm.components.utils import normalize_objectives

    F_norm = normalize_objectives(F)

    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    best_labels: NDArray[np.int_] | None = None
    best_centroids: NDArray[np.float64] | None = None
    best_inertia = np.inf

    for _ in range(max(1, int(n_init))):
        # k-means++ initialization
        centroids = np.empty((k, F_norm.shape[1]), dtype=float)
        centroids[0] = F_norm[rng.integers(n)]
        for c in range(1, k):
            dists = np.min(
                np.linalg.norm(F_norm[:, None, :] - centroids[None, :c, :], axis=2),
                axis=1,
            )
            probs = dists**2
            total = probs.sum()
            if total > 0:
                probs /= total
            else:
                probs = np.ones(n) / n
            centroids[c] = F_norm[rng.choice(n, p=probs)]

        # Lloyd iterations
        for _ in range(max(1, int(max_iter))):
            dists_to_c = np.linalg.norm(F_norm[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists_to_c, axis=1)
            new_centroids = np.empty_like(centroids)
            for c in range(k):
                members = F_norm[labels == c]
                if members.size > 0:
                    new_centroids[c] = members.mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]
            if np.allclose(new_centroids, centroids):
                centroids = new_centroids
                break
            centroids = new_centroids

        inertia = sum(np.sum((F_norm[labels == c] - centroids[c]) ** 2) for c in range(k))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    assert best_labels is not None and best_centroids is not None

    # Pick the solution closest to each centroid
    selected: list[int] = []
    for c in range(k):
        members = np.flatnonzero(best_labels == c)
        if members.size == 0:
            # Empty cluster — pick globally closest unselected point
            remaining = np.setdiff1d(np.arange(n), selected)
            if remaining.size > 0:
                dists = np.linalg.norm(F_norm[remaining] - best_centroids[c], axis=1)
                selected.append(int(remaining[np.argmin(dists)]))
            continue
        dists = np.linalg.norm(F_norm[members] - best_centroids[c], axis=1)
        selected.append(int(members[np.argmin(dists)]))

    # Deduplicate (edge case: two centroids closest to same point)
    selected = list(dict.fromkeys(selected))

    # Fill if needed
    if len(selected) < k:
        remaining = np.setdiff1d(np.arange(n), selected)
        # Fill with farthest-point from remaining
        if remaining.size > 0:
            sub_idx = select_top_k_farthest(F_norm[remaining], k - len(selected))
            selected.extend(remaining[sub_idx].tolist())

    return np.asarray(selected[:k], dtype=int)


# ---------------------------------------------------------------------------
# Angle-based diversity selection
# ---------------------------------------------------------------------------


def select_top_k_angle(F: NDArray[np.float64], k: int) -> NDArray[np.int_]:
    """Select the top-*k* solutions by angle-based diversity.

    Iteratively removes the solution with the smallest minimum angle to any
    other remaining solution.  Angles are computed via cosine similarity on
    vectors from the ideal point, making the method scale-invariant.

    Particularly effective for many-objective problems where crowding distance
    becomes less discriminative.

    Parameters
    ----------
    F:
        Objective matrix of shape ``(n, n_obj)``.
    k:
        Number of solutions to keep.

    Returns
    -------
    NDArray[np.int_]
        1-D integer array of indices into *F* for the selected solutions.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    n = F.shape[0]
    if n <= k:
        return np.arange(n, dtype=int)

    # Shift to ideal point as origin
    ideal = F.min(axis=0)
    V = F - ideal
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    V_unit = V / norms

    # Cosine similarity matrix → angle matrix
    cos_sim = V_unit @ V_unit.T
    np.clip(cos_sim, -1.0, 1.0, out=cos_sim)
    angle_matrix = np.arccos(cos_sim)
    np.fill_diagonal(angle_matrix, np.inf)

    candidates = list(range(n))

    while len(candidates) > k:
        sub = angle_matrix[np.ix_(candidates, candidates)]
        min_angles = np.min(sub, axis=1)
        remove_pos = int(np.argmin(min_angles))
        del candidates[remove_pos]

    return np.asarray(candidates, dtype=int)


# ---------------------------------------------------------------------------
# Greedy hypervolume subset selection (inclusion-based)
# ---------------------------------------------------------------------------


def select_top_k_hv_greedy(
    F: NDArray[np.float64],
    k: int,
    *,
    ref_point: NDArray[np.float64] | list[float] | None = None,
    ref_offset: float = 1.0,
) -> NDArray[np.int_]:
    """Select the top-*k* solutions by greedy hypervolume inclusion.

    Iteratively adds the solution that maximizes the hypervolume contribution
    to the already-selected subset.  This is the *inclusion* counterpart to
    the iterative-removal strategy used by :class:`HypervolumeArchive`.

    Greedy inclusion provides a (1 - 1/e) approximation guarantee for the HV
    subset selection problem (Guerreiro & Fonseca, GECCO 2021).

    The contributions are exact with or without moocore. Without it they are
    computed in pure Python, which is fast with two objectives and slow with
    more: about 2 s for ``n = 100`` and ``k = 10`` with three objectives, and
    about 12 s with four.

    Parameters
    ----------
    F:
        Objective matrix of shape ``(n, n_obj)``.
    k:
        Number of solutions to keep.
    ref_point:
        Reference point for HV computation.  If ``None``, computed as
        ``max(F, axis=0) + ref_offset``.
    ref_offset:
        Offset added to the per-objective maximum when ``ref_point`` is ``None``.

    Returns
    -------
    NDArray[np.int_]
        1-D integer array of indices into *F* for the selected solutions.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    n = F.shape[0]
    if n <= k:
        return np.arange(n, dtype=int)

    if ref_point is not None:
        ref = np.asarray(ref_point, dtype=float)
    else:
        ref = np.max(F, axis=0) + ref_offset

    selected: list[int] = []
    remaining = set(range(n))

    for _ in range(k):
        best_idx = -1
        best_contrib = -np.inf

        for idx in remaining:
            trial = selected + [idx]
            contrib = _hv_contributions(F[trial], ref)
            # The contribution of the newly added point is the last element
            c = float(contrib[-1])
            if c > best_contrib:
                best_contrib = c
                best_idx = idx

        selected.append(best_idx)
        remaining.discard(best_idx)

    return np.asarray(selected, dtype=int)


__all__ = [
    "_single_front_crowding",
    "_hv_contributions",
    "select_top_k_crowding",
    "select_top_k_farthest",
    "select_top_k_knn",
    "select_top_k_reference_directions",
    "select_top_k_kmeans",
    "select_top_k_angle",
    "select_top_k_hv_greedy",
]
