"""The one-offspring parent-sampling path must reproduce the batch path exactly."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vamos.engine.algorithm.moead.moead import MOEAD


def _reference_pairs(
    neighbors: np.ndarray, pop_size: int, active: np.ndarray, use_neighbors: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    # Verbatim batch path of MOEAD._sample_parent_pairs.
    parent_pairs = np.empty((active.shape[0], 2), dtype=int)
    neighbor_mask = np.asarray(use_neighbors, dtype=bool)
    if bool(np.any(neighbor_mask)):
        active_neighbors = neighbors[active[neighbor_mask]]
        local_count, local_size = active_neighbors.shape
        first_local = rng.integers(0, local_size, size=local_count, dtype=np.int64)
        second_local = rng.integers(0, local_size - 1, size=local_count, dtype=np.int64)
        second_local += second_local >= first_local
        rows = np.arange(local_count, dtype=np.int64)
        parent_pairs[neighbor_mask, 0] = active_neighbors[rows, first_local]
        parent_pairs[neighbor_mask, 1] = active_neighbors[rows, second_local]
    global_mask = ~neighbor_mask
    if bool(np.any(global_mask)):
        global_count = int(np.sum(global_mask))
        first_global = rng.integers(0, pop_size, size=global_count, dtype=np.int64)
        second_global = rng.integers(0, pop_size - 1, size=global_count, dtype=np.int64)
        second_global += second_global >= first_global
        parent_pairs[global_mask, 0] = first_global
        parent_pairs[global_mask, 1] = second_global
    return parent_pairs


@pytest.mark.parametrize("pop_size,neighbor_size", [(2, 2), (7, 3), (100, 20)])
def test_single_offspring_parent_sampling_matches_batch_reference(pop_size: int, neighbor_size: int) -> None:
    algorithm = MOEAD({"pop_size": pop_size})
    setup = np.random.default_rng(pop_size)
    neighbors = np.stack([setup.permutation(pop_size)[:neighbor_size] for _ in range(pop_size)])
    for seed in range(300):
        active = np.array([seed % pop_size])
        use_neighbors = np.array([seed % 3 != 0])
        state = SimpleNamespace(neighbors=neighbors, pop_size=pop_size, rng=np.random.default_rng(seed))
        reference_rng = np.random.default_rng(seed)

        pairs = algorithm._sample_parent_pairs(state, active, use_neighbors)  # type: ignore[arg-type]
        expected = _reference_pairs(neighbors, pop_size, active, use_neighbors, reference_rng)

        np.testing.assert_array_equal(pairs, expected)
        assert pairs.dtype == expected.dtype
        assert state.rng.bit_generator.state == reference_rng.bit_generator.state
