"""Memory and parity checks for the max-min and reference-direction archive pruning."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from vamos.engine.archive import pruning_selectors
from vamos.engine.reference_directions import das_dennis


def _sphere_front(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = np.abs(rng.normal(size=(n, 3)))
    return np.vstack([np.eye(3), points / np.linalg.norm(points, axis=1, keepdims=True)])


def _matrix_maxmin_fill(normalized: np.ndarray, selected: list[int], target_size: int) -> list[int]:
    """Greedy max-min fill over the full pairwise distance matrix, the rule the archive follows."""
    dist = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
    np.fill_diagonal(dist, -np.inf)
    selected = list(selected)
    while len(selected) < target_size:
        remaining = np.setdiff1d(np.arange(normalized.shape[0]), selected)
        if remaining.size == 0:
            break
        min_dist = dist[np.ix_(remaining, selected)].min(axis=1)
        best = remaining[np.isclose(min_dist, float(np.max(min_dist)))]
        selected.append(int(best[np.argmin(np.sum(normalized[best], axis=1))]) if best.size > 1 else int(best[0]))
    return selected


def _matrix_maxmin_subset(F: np.ndarray, target_size: int) -> list[int]:
    normalized = pruning_selectors.normalize_for_ref_dirs(F)
    dist = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
    np.fill_diagonal(dist, -np.inf)
    first, second = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return _matrix_maxmin_fill(normalized, [int(first), int(second)], target_size)


@pytest.mark.parametrize(("selector", "target_size"), [("ref_dirs", 21), ("ref_dirs", 30), ("ref_dirs", 2), ("maxmin", 20)])
def test_pruning_memory_grows_linearly_with_the_candidates(selector: str, target_size: int) -> None:
    # A pairwise distance matrix of 2503 three-objective rows needs a 150 MB temporary.
    F = _sphere_front(2500, seed=0)

    tracemalloc.start()
    try:
        if selector == "ref_dirs":
            selected = pruning_selectors.select_ref_dirs_subset(F, target_size, np.random.default_rng(0))
        else:
            selected = pruning_selectors.select_maxmin_subset(F, target_size)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert len(set(selected.tolist())) == target_size
    assert peak < 32 * 2**20, f"peak allocation {peak / 2**20:.0f} MiB"


@pytest.mark.parametrize("seed", range(6))
def test_maxmin_subset_matches_the_pairwise_matrix_greedy(seed: int) -> None:
    F = np.random.default_rng(seed).random((40, 3))

    for target_size in (2, 5, 13, 39):
        assert pruning_selectors.select_maxmin_subset(F, target_size).tolist() == _matrix_maxmin_subset(F, target_size)


def test_maxmin_fill_matches_the_pairwise_matrix_greedy_with_ties() -> None:
    lattice = das_dennis(3, 6)  # Equally spaced rows: many distance ties.

    for start in ([0], [0, 27], [5, 10, 20]):
        for target_size in (4, 9, 20, 28):
            expected = _matrix_maxmin_fill(lattice, start, target_size)
            assert pruning_selectors._extend_by_maxmin(lattice, list(start), target_size) == expected
