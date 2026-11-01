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


def _always_neighbourhood_config(batch_size: int) -> dict[str, object]:
    from vamos.engine.algorithm.config import MOEADConfig

    config = (
        MOEADConfig.builder()
        .pop_size(20)
        .batch_size(batch_size)
        .neighbor_size(5)
        .delta(1.0)
        .replace_limit(2)
        .crossover("de", cr=1.0, f=0.5)
        .mutation("pm", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .weight_vectors(divisions=19)
        .build()
    )
    return config.to_dict()


def test_neighbour_replacement_order_is_random(monkeypatch: pytest.MonkeyPatch) -> None:
    from vamos.engine.algorithm.moead import moead as moead_module
    from vamos.foundation.kernel.numpy_backend import NumPyKernel
    from vamos.foundation.problem.zdt1 import ZDT1Problem

    seen: list[tuple[int, np.ndarray, np.ndarray]] = []
    original = moead_module.update_neighborhood

    def spy(*, st, idx, candidate_order, **kwargs):  # type: ignore[no-untyped-def]
        seen.append((idx, np.array(candidate_order), np.array(st.neighbors[idx])))
        return original(st=st, idx=idx, candidate_order=candidate_order, **kwargs)

    monkeypatch.setattr(moead_module, "update_neighborhood", spy)
    MOEAD(_always_neighbourhood_config(1), kernel=NumPyKernel()).run(ZDT1Problem(n_var=6), termination=("max_evaluations", 120), seed=0)

    assert len(seen) == 100
    assert all(sorted(order.tolist()) == sorted(row.tolist()) for _, order, row in seen)
    assert any(order[0] != idx for idx, order, _ in seen)


def test_batch_neighbour_replacement_order_is_random(monkeypatch: pytest.MonkeyPatch) -> None:
    from vamos.engine.algorithm.moead import moead as moead_module
    from vamos.foundation.kernel.numpy_backend import NumPyKernel
    from vamos.foundation.problem.zdt1 import ZDT1Problem

    rows: list[tuple[np.ndarray, np.ndarray]] = []
    original = moead_module.update_neighborhood_batch

    def spy(*, st, candidate_orders, candidate_lengths, **kwargs):  # type: ignore[no-untyped-def]
        for order, length in zip(candidate_orders, candidate_lengths, strict=True):
            rows.append((np.array(order[:length]), st.neighbors))
        return original(st=st, candidate_orders=candidate_orders, candidate_lengths=candidate_lengths, **kwargs)

    monkeypatch.setattr(moead_module, "update_neighborhood_batch", spy)
    MOEAD(_always_neighbourhood_config(4), kernel=NumPyKernel()).run(ZDT1Problem(n_var=6), termination=("max_evaluations", 120), seed=0)

    assert rows
    as_sets = [sorted(order.tolist()) for order, _ in rows]
    neighbour_sets = {tuple(sorted(row.tolist())) for row in rows[0][1]}
    assert all(tuple(item) in neighbour_sets for item in as_sets)
    neighbour_rows = {tuple(row.tolist()) for row in rows[0][1]}
    assert any(tuple(order.tolist()) not in neighbour_rows for order, _ in rows)


def test_neighbourhoods_break_distance_ties_by_index() -> None:
    from vamos.engine.algorithm.moead.neighborhood import compute_neighbors

    # Dyadic weights make the distances exact: each interior vector has two equidistant neighbours at every
    # distance, and the default (unstable) sort orders such ties differently from one platform to another.
    steps = np.arange(65) / 64.0
    weights = np.column_stack([steps, 1.0 - steps])

    neighbors = compute_neighbors(weights, 20)

    np.testing.assert_array_equal(neighbors[32][:6], [32, 31, 33, 30, 34, 29])
    distances = np.abs(np.arange(65)[:, None] - neighbors)
    for row, dist in zip(neighbors, distances, strict=True):
        assert all((dist[i], row[i]) < (dist[i + 1], row[i + 1]) for i in range(len(row) - 1))
