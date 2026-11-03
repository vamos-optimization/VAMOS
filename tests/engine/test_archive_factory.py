from __future__ import annotations

import numpy as np
import pytest

from vamos.engine.algorithm.components.archive import HypervolumeArchive, UnboundedArchive
from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.archive.factory import setup_archive
from vamos.foundation.kernel.numpy_backend import NumPyKernel


def test_setup_archive_honors_extended_external_archive_config():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]], dtype=float)
    F = np.array([[0.0, 2.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]], dtype=float)
    cfg = ExternalArchiveConfig(
        capacity=4,
        pruning="hv",
        hv_ref_point=[3.0, 3.0],
        truncate_size=3,
        deduplicate_in="both",
        objective_tolerance=1e-6,
        decision_tolerance=1e-6,
    )

    archive_X, archive_F, manager = setup_archive(NumPyKernel(), X, F, 2, 2, X.dtype, cfg)

    assert isinstance(manager, HypervolumeArchive)
    assert manager.truncate_size == 3
    assert np.array_equal(manager._fixed_ref, np.array([3.0, 3.0]))
    assert manager._deduplicate_in == "both"
    assert archive_X is not None and archive_F is not None
    assert archive_F.shape[0] == 3


def test_setup_archive_uses_shared_unbounded_archive_manager():
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    F = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    cfg = ExternalArchiveConfig(capacity=None, deduplicate_in="decision")

    archive_X, archive_F, manager = setup_archive(NumPyKernel(), X, F, 2, 2, X.dtype, cfg)

    assert isinstance(manager, UnboundedArchive)
    assert archive_X is not None and archive_F is not None
    assert archive_F.shape == (2, 2)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"capacity": True}, "capacity"),
        ({"capacity": 10.7}, "capacity"),
        ({"capacity": 0}, "capacity"),
        ({"capacity": 20, "truncate_size": True}, "truncate_size"),
        ({"rng_seed": 1.5}, "rng_seed"),
        ({"objective_tolerance": float("nan")}, "objective_tolerance"),
        ({"decision_tolerance": float("inf")}, "decision_tolerance"),
        ({"objective_tolerance": True}, "objective_tolerance"),
        ({"capacity": 20, "pruning": "hv", "hv_ref_point": [float("nan"), 1.0]}, "hv_ref_point"),
        ({"capacity": 20, "pruning": "hv", "hv_ref_point": []}, "hv_ref_point"),
        ({"capacity": 20, "pruning": "hv", "hv_ref_point": "1.1, 1.1"}, "hv_ref_point"),
        ({"deduplicate_in": "decisions"}, "deduplicate_in"),
    ],
)
def test_external_archive_config_rejects_invalid_values(kwargs, field):
    with pytest.raises((TypeError, ValueError), match=field):
        ExternalArchiveConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"capacity": 20, "pruning": "crowding", "hv_ref_point": [1.1, 1.1]}, "hv_ref_point"),
        ({"capacity": 20, "pruning": "maxmin", "rng_seed": 3}, "rng_seed"),
        ({"capacity": None, "pruning": "hv"}, "pruning"),
    ],
)
def test_external_archive_config_warns_about_fields_the_pruning_ignores(kwargs, field):
    with pytest.warns(UserWarning, match=field):
        ExternalArchiveConfig(**kwargs)


def test_external_archive_config_normalizes_numeric_fields():
    cfg = ExternalArchiveConfig(capacity=np.int64(20), pruning="hv", hv_ref_point=(np.float64(1.5), 2))

    assert cfg.capacity == 20 and type(cfg.capacity) is int
    assert cfg.hv_ref_point == [1.5, 2.0]
