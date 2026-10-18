"""Compiled survival/truncation loops must reproduce the NumPy reference loops exactly."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("numba")

import vamos
from vamos.engine.algorithm.agemoea.geometry import _survival_score
from vamos.engine.algorithm.components.archive import CrowdingDistanceArchive
from vamos.engine.algorithm.ibea.helpers import environmental_selection as ibea_environmental_selection
from vamos.engine.algorithm.spea2.helpers import truncate_by_kth_distance
from vamos.foundation.kernel import _numba_survival
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.kernel.numpy_backend import NumPyKernel

KERNELS = NumbaKernel().selection_kernels()


def _front(rng: np.random.Generator, trial: int, n_rows: int, n_obj: int) -> np.ndarray:
    kind = trial % 4
    if kind == 0:
        return np.round(rng.random((n_rows, n_obj)) * 4.0) / 4.0  # ties and duplicates
    if kind == 1:
        F = rng.random((n_rows, n_obj))
        return F / np.linalg.norm(F, axis=1, keepdims=True)
    if kind == 2:
        F = rng.random((n_rows, n_obj))
        F[:, 0] = 0.5  # degenerate objective
        return F
    return rng.random((n_rows, n_obj))


def _archive(kernels: Any) -> CrowdingDistanceArchive:
    return CrowdingDistanceArchive(10, 2, 3, np.float64, selection_kernels=kernels)


@pytest.mark.numba
def test_numpy_kernel_has_no_selection_kernels() -> None:
    assert NumPyKernel().selection_kernels() is None
    assert KERNELS is not None


@pytest.mark.numba
def test_crowding_truncation_matches_reference() -> None:
    rng = np.random.default_rng(1)
    for trial in range(800):
        n = int(rng.integers(2, 80))
        F = _front(rng, trial, n, int(rng.integers(1, 6)))
        target = int(rng.integers(1, n + 1))
        expected = _archive(None)._select_subset(F, target)
        actual = _archive(KERNELS)._select_subset(F, target)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype


@pytest.mark.numba
def test_crowding_truncation_keeps_reference_for_non_finite_values(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    class Recording:
        def iterative_crowding_truncation(self, F: np.ndarray, target_size: int) -> np.ndarray:
            calls.append(target_size)
            return np.arange(target_size)

    F = np.random.default_rng(2).random((12, 2))
    F[4, 1] = np.nan
    expected = _archive(None)._select_subset(F, 6)
    np.testing.assert_array_equal(_archive(Recording())._select_subset(F, 6), expected)
    assert calls == []


@pytest.mark.numba
def test_agemoea_survival_score_matches_reference() -> None:
    rng = np.random.default_rng(3)
    for trial in range(600):
        n_obj = int(rng.integers(1, 6))
        F = _front(rng, trial, int(rng.integers(1, 90)), n_obj)
        ideal = F.min(axis=0)
        # Degenerate fronts make the reference geometry divide by log(1); that warning is not under test.
        with np.errstate(divide="ignore", invalid="ignore"):
            crowd_ref, p_ref, norm_ref = _survival_score(F.copy(), ideal.copy())
            crowd_new, p_new, norm_new = _survival_score(F.copy(), ideal.copy(), KERNELS)
        np.testing.assert_array_equal(crowd_new, crowd_ref)
        assert p_new == p_ref
        np.testing.assert_array_equal(norm_new, norm_ref)


@pytest.mark.numba
def test_greedy_selection_handles_nan_distances_like_reference() -> None:
    rng = np.random.default_rng(4)
    for _ in range(200):
        m = int(rng.integers(3, 40))
        D = rng.random((m, m))
        D = (D + D.T) / 2.0
        D[int(rng.integers(0, m)), :] = np.nan
        selected = np.sort(rng.permutation(m)[: int(rng.integers(1, 4))])
        remaining = np.setdiff1d(np.arange(m), selected)
        init = D[np.ix_(remaining, selected)]
        if init.shape[1] > 1:
            nearest = np.partition(init, kth=1, axis=1)[:, :2]
            best1, best2 = nearest[:, 0].copy(), nearest[:, 1].copy()
            scores = best1 + best2
        else:
            best1, best2 = init[:, 0].copy(), np.zeros(remaining.size)
            scores = best1.copy()
        expected = np.zeros(m)
        _reference_greedy(D, remaining, best1, best2, scores, selected.size, expected)
        actual = np.zeros(m)
        KERNELS.greedy_two_nearest_selection(D, remaining, best1, best2, scores, selected.size, actual)
        np.testing.assert_array_equal(actual, expected)


def _reference_greedy(
    distances: np.ndarray,
    remaining: np.ndarray,
    best1: np.ndarray,
    best2: np.ndarray,
    scores: np.ndarray,
    selected_count: int,
    crowd_dist: np.ndarray,
) -> None:
    # Verbatim copy of the loop in vamos.engine.algorithm.agemoea.geometry._survival_score.
    while remaining.size > 0:
        index = int(np.argmax(scores))
        best = int(remaining[index])
        crowd_dist[best] = float(scores[index])
        remaining = np.delete(remaining, index)
        best1 = np.delete(best1, index)
        best2 = np.delete(best2, index)
        scores = np.delete(scores, index)
        selected_count += 1
        if remaining.size == 0:
            break
        new_dist = distances[remaining, best]
        if selected_count == 2:
            best1, best2 = np.minimum(best1, new_dist), np.maximum(best1, new_dist)
            scores = best1 + best2
            continue
        better_first = new_dist < best1
        best2 = np.where(better_first, best1, best2)
        best1 = np.where(better_first, new_dist, best1)
        better_second = (~better_first) & (new_dist < best2)
        best2 = np.where(better_second, new_dist, best2)
        scores = best1 + best2


@pytest.mark.numba
@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_spea2_kth_truncation_matches_reference_order(k: int) -> None:
    rng = np.random.default_rng(10 + k)
    for trial in range(300):
        n = int(rng.integers(2, 70))
        F = _front(rng, trial, n, int(rng.integers(1, 5)))
        dist = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
        keep = int(rng.integers(1, n + 1))
        expected = truncate_by_kth_distance(dist, keep, k)
        actual = truncate_by_kth_distance(dist, keep, k, KERNELS)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype


@pytest.mark.numba
@pytest.mark.parametrize("indicator", ["epsilon", "hypervolume"])
def test_ibea_removal_matches_reference(indicator: str) -> None:
    rng = np.random.default_rng(20)
    for trial in range(200):
        n = int(rng.integers(3, 60))
        F = _front(rng, trial, n, int(rng.integers(2, 5)))
        X = rng.random((n, 3))
        G = rng.normal(size=(n, 1)) if trial % 3 == 0 else None
        target = int(rng.integers(1, n + 1))
        expected = ibea_environmental_selection(X, F, G, target, indicator, 0.05)
        actual = ibea_environmental_selection(X, F, G, target, indicator, 0.05, KERNELS)
        for got, ref in zip(actual, expected, strict=True):
            if ref is None:
                assert got is None
            else:
                np.testing.assert_array_equal(got, ref)


@pytest.mark.numba
def test_selection_kernels_are_nopython() -> None:
    rng = np.random.default_rng(0)
    F = rng.random((8, 2))
    KERNELS.iterative_crowding_truncation(F, 4)
    dist = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
    KERNELS.kth_distance_truncation(dist, 4, 1)
    KERNELS.indicator_removal(rng.random(8), rng.random((8, 8)), 4)
    remaining = np.arange(2, 8)
    best = dist[2:, 0].copy()
    KERNELS.greedy_two_nearest_selection(dist, remaining, best, np.zeros(6), best.copy(), 1, np.zeros(8))
    for dispatcher in (
        _numba_survival.iterative_crowding_truncation,
        _numba_survival.kth_distance_truncation,
        _numba_survival.indicator_removal,
        _numba_survival.greedy_two_nearest_selection,
    ):
        assert dispatcher.signatures
        assert dispatcher.nopython_signatures


@pytest.mark.numba
@pytest.mark.parametrize(
    "algorithm,method",
    [
        ("agemoea", "greedy_two_nearest_selection"),
        ("spea2", "kth_distance_truncation"),
        ("ibea", "indicator_removal"),
        ("smpso", "iterative_crowding_truncation"),
    ],
)
def test_selection_kernels_run_only_for_the_numba_engine(monkeypatch: pytest.MonkeyPatch, algorithm: str, method: str) -> None:
    calls: list[str] = []
    original = getattr(KERNELS, method)

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(method)
        return original(*args, **kwargs)

    monkeypatch.setattr(KERNELS, method, counting)
    vamos.optimize("dtlz2", algorithm=algorithm, engine="numpy", seed=2, pop_size=24, max_evaluations=480, n_obj=3)
    assert calls == []
    vamos.optimize("dtlz2", algorithm=algorithm, engine="numba", seed=2, pop_size=24, max_evaluations=480, n_obj=3)
    assert calls
