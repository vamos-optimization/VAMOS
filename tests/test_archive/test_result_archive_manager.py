from __future__ import annotations

import numpy as np
import pytest

import vamos.engine.algorithm.components.subset_selection as ss_mod
import vamos.foundation.quality_indicators.moocore_indicators as mi_mod
from vamos.engine.algorithm.components.archive import HypervolumeArchive, MaxMinArchive, ReferenceDirectionsArchive
from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.archive.factory import setup_result_archive


def _empty_X(n_rows: int) -> np.ndarray:
    return np.empty((n_rows, 0), dtype=float)


def _use_fake_moocore(monkeypatch: pytest.MonkeyPatch, fake: type) -> None:
    monkeypatch.setattr(ss_mod, "_moocore", fake)
    monkeypatch.setattr(mi_mod, "_moocore", fake)


def test_result_archive_size_cap_and_nondominated_filter():
    cfg = ExternalArchiveConfig(capacity=5, pruning="crowding")
    archive = setup_result_archive(cfg, n_var=0, n_obj=2, dtype=float)
    assert archive is not None

    F = np.array(
        [
            [1.0, 5.0],
            [2.0, 4.0],
            [3.0, 3.0],
            [4.0, 2.0],
            [5.0, 1.0],
            [6.0, 6.0],
            [2.5, 4.5],
        ]
    )
    _, archive_F = archive.update(_empty_X(F.shape[0]), F)
    assert archive_F.shape[0] <= 5


def test_external_archive_rejects_unknown_pruning_policy():
    with pytest.raises(ValueError, match="Unsupported pruning 'invalid-policy'"):
        ExternalArchiveConfig(capacity=10, pruning="invalid-policy")  # type: ignore[arg-type]


def test_knn_prune_policy_name_is_accepted():
    assert ExternalArchiveConfig(capacity=10, pruning="knn").pruning == "knn"


def test_knn_archive_pruning_keeps_extremes():
    # The two interior points crowd each other, so the SPEA2 truncation operator
    # removes one of them. Ranking by the second-nearest distance alone ties
    # (0, 1, 0) with both of them and removed that corner instead.
    cfg = ExternalArchiveConfig(capacity=4, pruning="knn")
    archive = setup_result_archive(cfg, n_var=0, n_obj=3, dtype=float)
    assert archive is not None
    F = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.1, 0.7, 0.2], [0.2, 0.7, 0.1]])

    _, archive_F = archive.update(_empty_X(F.shape[0]), F)

    kept = {tuple(row) for row in archive_F.tolist()}
    assert {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)} <= kept
    assert len(kept) == 4


def test_maxmin_and_ref_dirs_prune_policy_names_are_accepted():
    assert ExternalArchiveConfig(capacity=10, pruning="maxmin").pruning == "maxmin"
    assert ExternalArchiveConfig(capacity=10, pruning="ref_dirs").pruning == "ref_dirs"


def test_result_archive_maxmin_prunes_to_target_size():
    cfg = ExternalArchiveConfig(capacity=3, pruning="maxmin")
    archive = setup_result_archive(cfg, n_var=0, n_obj=2, dtype=float)
    assert isinstance(archive, MaxMinArchive)
    F = np.array(
        [
            [0.0, 10.0],
            [1.0, 9.0],
            [5.0, 5.0],
            [9.0, 1.0],
            [10.0, 0.0],
        ]
    )
    _, kept = archive.update(_empty_X(F.shape[0]), F)
    assert kept.shape[0] == 3
    assert np.any(np.all(kept == np.array([0.0, 10.0]), axis=1))
    assert np.any(np.all(kept == np.array([10.0, 0.0]), axis=1))


def test_result_archive_ref_dirs_prunes_to_target_size():
    cfg = ExternalArchiveConfig(capacity=3, pruning="ref_dirs")
    archive = setup_result_archive(cfg, n_var=0, n_obj=3, dtype=float)
    assert isinstance(archive, ReferenceDirectionsArchive)
    F = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
            [0.34, 0.34, 0.34],
            [0.60, 0.20, 0.20],
        ]
    )
    _, kept = archive.update(_empty_X(F.shape[0]), F)
    assert kept.shape[0] == 3
    assert np.any(np.all(kept == np.array([0.90, 0.05, 0.05]), axis=1))
    assert np.any(np.all(kept == np.array([0.05, 0.90, 0.05]), axis=1))
    assert np.any(np.all(kept == np.array([0.05, 0.05, 0.90]), axis=1))


@pytest.mark.parametrize("perm", [(1, 2, 0), (2, 0, 1), (1, 0, 2)])
def test_ref_dirs_archive_is_symmetric_between_objectives(perm: tuple[int, int, int]) -> None:
    rng = np.random.default_rng(3)
    sphere = np.abs(rng.normal(size=(200, 3)))
    F = np.vstack([np.eye(3), sphere / np.linalg.norm(sphere, axis=1, keepdims=True)])
    cfg = ExternalArchiveConfig(capacity=20, pruning="ref_dirs")

    original = setup_result_archive(cfg, n_var=0, n_obj=3, dtype=float)
    permuted = setup_result_archive(cfg, n_var=0, n_obj=3, dtype=float)
    assert original is not None and permuted is not None
    _, kept = original.update(_empty_X(F.shape[0]), F)
    _, kept_perm = permuted.update(_empty_X(F.shape[0]), F[:, list(perm)])

    assert kept.shape[0] == 20
    assert {tuple(row) for row in kept_perm} == {tuple(row) for row in kept[:, list(perm)]}
    for axis in np.eye(3):
        assert np.any(np.all(kept == axis, axis=1))


def test_result_archive_hv_uses_moocore_for_many_objective_pruning(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    class _FakeMooCore:
        @staticmethod
        def is_nondominated(F: np.ndarray) -> np.ndarray:
            return np.ones(F.shape[0], dtype=bool)

        @staticmethod
        def hv_contributions(F: np.ndarray, ref: np.ndarray) -> np.ndarray:
            calls.append((F.copy(), ref.copy()))
            return np.array([0.4, 0.1, 0.3, 0.2], dtype=float)

    _use_fake_moocore(monkeypatch, _FakeMooCore)

    cfg = ExternalArchiveConfig(capacity=3, pruning="hv", hv_ref_point=[2.0, 2.0, 2.0])
    archive = setup_result_archive(cfg, n_var=0, n_obj=3, dtype=float)
    assert isinstance(archive, HypervolumeArchive)
    F = np.array(
        [
            [0.9, 0.2, 0.2],
            [0.2, 0.9, 0.2],
            [0.2, 0.2, 0.9],
            [0.45, 0.45, 0.45],
        ]
    )

    _, kept = archive.update(_empty_X(F.shape[0]), F)

    assert len(calls) == 1
    assert kept.shape[0] == 3
    assert not np.any(np.all(kept == F[1], axis=1))


def test_result_archive_mc_hv_keeps_monte_carlo_path_even_with_moocore(monkeypatch: pytest.MonkeyPatch):
    class _FakeMooCore:
        @staticmethod
        def is_nondominated(F: np.ndarray) -> np.ndarray:
            return np.ones(F.shape[0], dtype=bool)

        @staticmethod
        def hv_contributions(F: np.ndarray, ref: np.ndarray) -> np.ndarray:
            raise AssertionError("mc_hv should not call moocore.hv_contributions")

    _use_fake_moocore(monkeypatch, _FakeMooCore)

    cfg = ExternalArchiveConfig(capacity=3, pruning="mc_hv", hv_ref_point=[2.0, 2.0, 2.0])
    archive = setup_result_archive(cfg, n_var=0, n_obj=3, dtype=float)
    assert isinstance(archive, HypervolumeArchive)
    monkeypatch.setattr(
        archive,
        "_select_subset",
        lambda F, target_size, G=None: np.array([0, 1, 3], dtype=int),
    )

    F = np.array(
        [
            [0.9, 0.2, 0.2],
            [0.2, 0.9, 0.2],
            [0.2, 0.2, 0.9],
            [0.45, 0.45, 0.45],
        ]
    )

    _, kept = archive.update(_empty_X(F.shape[0]), F)

    assert kept.shape[0] == 3
    assert not np.any(np.all(kept == F[2], axis=1))
