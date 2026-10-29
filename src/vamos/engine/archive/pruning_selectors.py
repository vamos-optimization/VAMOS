from __future__ import annotations

import numpy as np
import numpy.typing as npt

from vamos.engine.reference_directions import FloatArray, das_dennis, largest_lattice_divisions

IndexArray = npt.NDArray[np.intp]


def _get_extreme_points(F: FloatArray, ideal: FloatArray) -> FloatArray:
    if F.size == 0:
        return np.empty((0, F.shape[1] if F.ndim == 2 else 0), dtype=float)
    shifted = F - ideal
    shifted[shifted < 1e-3] = 0.0
    n_obj = F.shape[1]
    asf = np.eye(n_obj)
    asf[asf == 0] = 1e6
    vals = np.max(shifted * asf[:, None, :], axis=2)
    idx = np.argmin(vals, axis=1)
    return np.asarray(F[idx], dtype=float)


def _get_nadir_point(extreme_points: FloatArray, ideal: FloatArray, worst: FloatArray) -> FloatArray:
    try:
        plane = np.linalg.solve(extreme_points - ideal, np.ones(extreme_points.shape[1], dtype=float))
        intercepts = 1.0 / plane
        nadir = ideal + intercepts
        if np.any(intercepts <= 1e-6) or np.any(nadir > worst):
            raise np.linalg.LinAlgError
    except Exception:
        nadir = worst

    mask = nadir - ideal <= 1e-6
    nadir = nadir.copy()
    nadir[mask] = worst[mask]
    return nadir


def _associate(normalized_F: FloatArray, ref_dirs_norm: FloatArray) -> tuple[IndexArray, FloatArray]:
    norms = np.linalg.norm(normalized_F, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1e-12)
    normalized_vectors = normalized_F / norms
    cosine = normalized_vectors @ ref_dirs_norm.T
    cosine = np.clip(cosine, -1.0, 1.0)
    associations = np.argmax(cosine, axis=1)
    cos_selected = cosine[np.arange(cosine.shape[0]), associations]
    distances = norms.flatten() * np.sqrt(1.0 - np.square(cos_selected))
    return associations, distances


def _closest_member_per_direction(
    associations: IndexArray,
    distances: FloatArray,
    n_directions: int,
    rng: np.random.Generator,
) -> list[int]:
    """Keep the member closest to each occupied direction, as NSGA-III niching does for empty niches."""
    selected: list[int] = []
    for direction in range(n_directions):
        members = np.flatnonzero(associations == direction)
        if members.size == 0:
            continue
        rng.shuffle(members)
        selected.append(int(members[int(np.argmin(distances[members]))]))
    return selected


def normalize_for_ref_dirs(F: FloatArray) -> FloatArray:
    """Normalize objective values for reference-direction association."""
    data = np.asarray(F, dtype=float)
    if data.ndim != 2:
        raise ValueError("F must be a 2D array.")
    if data.shape[0] == 0:
        return np.empty_like(data)

    ideal = data.min(axis=0)
    worst = data.max(axis=0)
    extreme_points = _get_extreme_points(data, ideal)
    nadir = _get_nadir_point(extreme_points, ideal, worst)

    denom = nadir - ideal
    fallback = worst - ideal
    denom = np.where(denom > 1e-12, denom, fallback)
    denom = np.where(denom > 1e-12, denom, 1.0)

    normalized = (data - ideal) / denom
    normalized[~np.isfinite(normalized)] = 0.0
    return np.asarray(normalized, dtype=float)


def _distances_to(normalized: FloatArray, index: int) -> FloatArray:
    return np.asarray(np.linalg.norm(normalized - normalized[index], axis=1), dtype=float)


def _farthest_pair(normalized: FloatArray) -> tuple[int, int]:
    """Return the first pair, in row-major order, at the largest distance.

    It scans one row at a time, so memory stays linear in the number of rows.
    """
    best_value = -np.inf
    pair = (0, 0)
    for row in range(int(normalized.shape[0])):
        dist = _distances_to(normalized, row)
        dist[row] = -np.inf
        column = int(np.argmax(dist))
        if dist[column] > best_value:
            best_value = float(dist[column])
            pair = (row, column)
    return pair


def _extend_by_maxmin(normalized: FloatArray, selected: list[int], target_size: int) -> list[int]:
    """Add the row farthest from the selected rows until ``target_size`` rows are selected.

    Ties go to the row with the smallest sum of normalized objectives. The
    distance of each row to its nearest selected row is updated incrementally,
    so memory stays linear in the number of rows.
    """
    if len(selected) >= target_size:
        return selected
    is_selected = np.zeros(int(normalized.shape[0]), dtype=bool)
    is_selected[np.asarray(selected, dtype=int)] = True
    nearest = np.full(int(normalized.shape[0]), np.inf)
    for index in selected:
        np.minimum(nearest, _distances_to(normalized, index), out=nearest)
    row_sums = np.sum(normalized, axis=1)

    while len(selected) < target_size:
        remaining = np.flatnonzero(~is_selected)
        if remaining.size == 0:
            break
        min_dist = nearest[remaining]
        best_value = float(np.max(min_dist))
        best_candidates = remaining[np.isclose(min_dist, best_value)]
        if best_candidates.size > 1:
            best_idx = int(best_candidates[np.argmin(row_sums[best_candidates])])
        else:
            best_idx = int(best_candidates[0])
        selected.append(best_idx)
        is_selected[best_idx] = True
        np.minimum(nearest, _distances_to(normalized, best_idx), out=nearest)
    return selected


def select_maxmin_subset(F: FloatArray, target_size: int) -> IndexArray:
    """Select a diverse subset by greedy max-min distance in normalized objective space."""
    data = np.asarray(F, dtype=float)
    n = int(data.shape[0])
    if target_size <= 0:
        return np.empty(0, dtype=np.intp)
    if n <= target_size:
        return np.arange(n, dtype=np.intp)
    if target_size == 1:
        normalized = normalize_for_ref_dirs(data)
        best = int(np.argmin(np.sum(normalized, axis=1)))
        return np.asarray([best], dtype=np.intp)

    normalized = normalize_for_ref_dirs(data)
    first, second = _farthest_pair(normalized)
    selected = _extend_by_maxmin(normalized, [first, second] if first != second else [first], target_size)
    return np.asarray(selected[:target_size], dtype=np.intp)


def select_ref_dirs_subset(
    F: FloatArray,
    target_size: int,
    rng: np.random.Generator,
) -> IndexArray:
    """Select a subset by NSGA-III-style reference-direction niching.

    The directions are the complete Das-Dennis lattice with the most directions
    that fit in ``target_size``, so the selection treats every objective alike.
    Each occupied direction keeps its closest member, and the remaining slots go
    to the rows farthest from the selected ones (greedy max-min). With fewer
    slots than objectives no lattice fits, and max-min selects the whole subset.
    ``rng`` breaks exact distance ties.
    """
    data = np.asarray(F, dtype=float)
    n = int(data.shape[0])
    if target_size <= 0:
        return np.empty(0, dtype=np.intp)
    if n <= target_size:
        return np.arange(n, dtype=np.intp)
    n_obj = int(data.shape[1])
    if n_obj < 2 or target_size < n_obj:
        return select_maxmin_subset(data, target_size)

    ref_dirs = das_dennis(n_obj, largest_lattice_divisions(target_size, n_obj))
    ref_dirs_norm = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)

    normalized = normalize_for_ref_dirs(data)
    associations, distances = _associate(normalized, ref_dirs_norm)
    selected = _closest_member_per_direction(associations, distances, ref_dirs_norm.shape[0], rng)
    selected = _extend_by_maxmin(normalized, selected, target_size)
    return np.asarray(selected[:target_size], dtype=np.intp)
