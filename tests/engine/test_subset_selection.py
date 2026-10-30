"""Tests for subset selection functions in subset_selection.py."""

import numpy as np
import pytest

from vamos.engine.algorithm.components.subset_selection import (
    select_top_k_angle,
    select_top_k_crowding,
    select_top_k_farthest,
    select_top_k_hv_greedy,
    select_top_k_kmeans,
    select_top_k_knn,
    select_top_k_reference_directions,
)


def _tradeoff_front(n: int) -> np.ndarray:
    x = np.arange(n, dtype=float)
    y = (n - x).astype(float)
    return np.column_stack([x, y])


def _tradeoff_front_3d(n: int) -> np.ndarray:
    t = np.linspace(0, 1, n)
    f1 = np.cos(t * np.pi / 2)
    f2 = np.sin(t * np.pi / 2)
    f3 = 1.0 - f1 - f2 + 0.5
    return np.column_stack([f1, f2, f3])


# ---- Shared parametric tests across all functions --------------------------

ALL_SELECTORS = [
    ("crowding", select_top_k_crowding, {}),
    ("farthest", select_top_k_farthest, {}),
    ("knn", select_top_k_knn, {}),
    ("reference_directions", select_top_k_reference_directions, {}),
    ("kmeans", select_top_k_kmeans, {"rng": 42}),
    ("angle", select_top_k_angle, {}),
    ("hv_greedy", select_top_k_hv_greedy, {}),
]


@pytest.mark.parametrize("name,fn,kwargs", ALL_SELECTORS, ids=[s[0] for s in ALL_SELECTORS])
class TestCommonContract:
    def test_returns_k_indices(self, name, fn, kwargs):
        F = _tradeoff_front(20)
        idx = fn(F, 5, **kwargs)
        assert idx.shape == (5,)
        assert idx.dtype == int or np.issubdtype(idx.dtype, np.integer)
        assert len(set(idx.tolist())) == 5
        assert all(0 <= i < 20 for i in idx)

    def test_returns_all_when_k_ge_n(self, name, fn, kwargs):
        F = _tradeoff_front(3)
        idx = fn(F, 10, **kwargs)
        assert idx.shape == (3,)

    def test_rejects_non_positive_k(self, name, fn, kwargs):
        F = _tradeoff_front(5)
        with pytest.raises(ValueError, match="k must be a positive"):
            fn(F, 0, **kwargs)

    def test_returns_all_when_k_equals_n(self, name, fn, kwargs):
        F = _tradeoff_front(5)
        idx = fn(F, 5, **kwargs)
        assert idx.shape == (5,)


# ---- Function-specific tests ----------------------------------------------


class TestSelectTopKCrowding:
    def test_keeps_extremes(self):
        F = np.array([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0], [4.0, 5.5]])
        idx = select_top_k_crowding(F, 2)
        assert 0 in idx and 2 in idx


class TestSelectTopKFarthest:
    def test_selects_spread_points(self):
        F = np.array([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0], [4.9, 5.1]])
        idx = select_top_k_farthest(F, 3)
        assert len(idx) == 3
        # Should pick the two extremes and the middle, not the near-duplicate
        assert 0 in idx and 2 in idx


class TestSelectTopKKnn:
    def test_removes_crowded_point(self):
        # Two points very close, two far apart
        F = np.array([[0.0, 10.0], [10.0, 0.0], [5.0, 5.0], [5.1, 4.9]])
        idx = select_top_k_knn(F, 3)
        assert len(idx) == 3
        # Should remove one of the close pair
        assert not (3 in idx and 2 in idx)

    def test_neighbors_parameter(self):
        F = _tradeoff_front(20)
        idx1 = select_top_k_knn(F, 5, neighbors=1)
        idx2 = select_top_k_knn(F, 5, neighbors=3)
        assert idx1.shape == (5,)
        assert idx2.shape == (5,)


class TestSelectTopKReferenceDirections:
    def test_2d(self):
        F = _tradeoff_front(20)
        idx = select_top_k_reference_directions(F, 5)
        assert idx.shape == (5,)
        assert len(set(idx.tolist())) == 5

    def test_3d(self):
        F = _tradeoff_front_3d(30)
        idx = select_top_k_reference_directions(F, 10)
        assert idx.shape == (10,)
        assert len(set(idx.tolist())) == 10

    def test_with_divisions(self):
        F = _tradeoff_front(20)
        idx = select_top_k_reference_directions(F, 5, divisions=4)
        assert idx.shape == (5,)

    @pytest.mark.parametrize("perm", [(1, 2, 0), (2, 0, 1), (1, 0, 2)])
    def test_symmetric_between_objectives(self, perm):
        rng = np.random.default_rng(3)
        sphere = np.abs(rng.normal(size=(200, 3)))
        F = np.vstack([np.eye(3), sphere / np.linalg.norm(sphere, axis=1, keepdims=True)])

        idx = select_top_k_reference_directions(F, 20)
        idx_perm = select_top_k_reference_directions(F[:, list(perm)], 20)

        assert set(idx_perm.tolist()) == set(idx.tolist())
        assert {0, 1, 2} <= set(idx.tolist())


class TestSelectTopKKmeans:
    def test_deterministic_with_seed(self):
        F = _tradeoff_front(20)
        idx1 = select_top_k_kmeans(F, 5, rng=42)
        idx2 = select_top_k_kmeans(F, 5, rng=42)
        np.testing.assert_array_equal(idx1, idx2)

    def test_different_seeds_may_differ(self):
        F = _tradeoff_front(50)
        idx1 = select_top_k_kmeans(F, 10, rng=1)
        idx2 = select_top_k_kmeans(F, 10, rng=999)
        # Not guaranteed to differ, but very likely with 50 points
        # Just check they're valid
        assert idx1.shape == (10,)
        assert idx2.shape == (10,)


class TestSelectTopKAngle:
    def test_keeps_diverse_directions(self):
        # Points at very different angles from ideal
        F = np.array(
            [
                [0.0, 10.0],  # straight up
                [10.0, 0.0],  # straight right
                [5.0, 5.0],  # diagonal
                [4.8, 5.2],  # near-duplicate of diagonal
            ]
        )
        idx = select_top_k_angle(F, 3)
        assert len(idx) == 3
        # Should remove the near-duplicate direction
        assert not (2 in idx and 3 in idx)

    def test_3d(self):
        F = _tradeoff_front_3d(30)
        idx = select_top_k_angle(F, 10)
        assert idx.shape == (10,)
        assert len(set(idx.tolist())) == 10


class TestSelectTopKHvGreedy:
    def test_selects_hv_contributing_points(self):
        # Near-duplicate pair + two spread points: greedy should skip the duplicate
        F = np.array(
            [
                [0.0, 10.0],  # extreme
                [10.0, 0.0],  # extreme
                [5.0, 5.0],  # unique region
                [5.1, 4.9],  # near-duplicate of [5.0, 5.0]
            ]
        )
        idx = select_top_k_hv_greedy(F, 3)
        assert len(idx) == 3
        # Should not include both near-duplicates
        assert not (2 in idx and 3 in idx)

    def test_custom_ref_point(self):
        F = _tradeoff_front(10)
        idx = select_top_k_hv_greedy(F, 3, ref_point=[20.0, 20.0])
        assert idx.shape == (3,)
        assert len(set(idx.tolist())) == 3
