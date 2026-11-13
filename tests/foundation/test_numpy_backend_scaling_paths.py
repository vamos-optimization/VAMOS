from __future__ import annotations

import numpy as np

import vamos.foundation.kernel.numpy_backend as numpy_backend


class _ChoiceForbiddenRNG:
    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def random(self, size: tuple[int, int]) -> np.ndarray:
        return self._rng.random(size)

    def choice(self, *args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("choice() should not be used in the chunked tournament path.")


def test_chunked_tournament_sampling_avoids_rng_choice(monkeypatch) -> None:
    monkeypatch.setattr(numpy_backend, "_TOURNAMENT_KEY_BUDGET", 8)
    rng = _ChoiceForbiddenRNG(seed=4)

    candidates = numpy_backend._sample_tournament_candidates(
        rng,
        n_candidates=9,
        n_parents=6,
        pressure=4,
    )

    assert candidates.shape == (6, 4)
    for row in candidates:
        assert len(set(row.tolist())) == 4


def test_blocked_non_dominated_sort_matches_dense_reference(monkeypatch) -> None:
    rng = np.random.default_rng(7)
    F = rng.integers(0, 9, size=(18, 3)).astype(float)

    dense_fronts, dense_ranks = numpy_backend._fast_non_dominated_sort_dense(F)
    monkeypatch.setattr(numpy_backend, "_DOMINANCE_TENSOR_BUDGET", 10)
    blocked_fronts, blocked_ranks = numpy_backend._fast_non_dominated_sort(F)

    assert blocked_fronts == dense_fronts
    np.testing.assert_array_equal(blocked_ranks, dense_ranks)


def _forbid_dominance_tensors(monkeypatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("the two-objective sort must not build the dominance tensor")

    monkeypatch.setattr(numpy_backend, "_fast_non_dominated_sort_dense", forbidden)
    monkeypatch.setattr(numpy_backend, "_fast_non_dominated_sort_blocked", forbidden)
    monkeypatch.setattr(numpy_backend, "_dominance_rows", forbidden)


def _two_objective_cases() -> list[np.ndarray]:
    rng = np.random.default_rng(11)
    continuous = rng.random((1000, 2))
    # Few distinct values: ties in each objective, duplicated rows and many fronts.
    ties = rng.integers(0, 12, size=(1000, 2)).astype(float)
    duplicates = np.repeat(rng.random((250, 2)), 4, axis=0)[rng.permutation(1000)]
    concave = np.column_stack([np.linspace(0.0, 1.0, 1000), 1.0 - np.linspace(0.0, 1.0, 1000) ** 2])
    infinite = continuous.copy()
    infinite[::97, 0] = np.inf
    infinite[5::89, 1] = -np.inf
    return [continuous, ties, duplicates, concave, infinite, ties[:1], ties[:2], np.repeat(ties[:1], 5, axis=0)]


def test_two_objective_sort_uses_sweep_and_matches_dense_ranks(monkeypatch) -> None:
    cases = _two_objective_cases()
    expected = [numpy_backend._fast_non_dominated_sort_dense(F) for F in cases]

    _forbid_dominance_tensors(monkeypatch)
    for F, (dense_fronts, dense_ranks) in zip(cases, expected):
        fronts, ranks = numpy_backend._fast_non_dominated_sort(F)
        assert fronts == dense_fronts
        np.testing.assert_array_equal(ranks, dense_ranks)
