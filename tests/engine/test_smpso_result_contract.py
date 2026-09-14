from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vamos.engine.algorithm.config import SMPSOConfig
from vamos.engine.algorithm.smpso import SMPSO
from vamos.experiment.artifacts.bundle import load_result_bundle, snapshot_result_arrays, write_result_bundle
from vamos.experiment.artifacts.bundle_safety import result_descriptor
from vamos.experiment.artifacts.models import LoadLimits
from vamos.experiment.artifacts.reader import _result_payload
from vamos.experiment.optimization_result import OptimizationResult
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _builder(pop_size: int = 10):
    return (
        SMPSOConfig.builder()
        .pop_size(pop_size)
        .archive_size(pop_size)
        .mutation("polynomial", prob="1/n", eta=20.0)
    )


def _run(config: SMPSOConfig):
    algorithm = SMPSO(config.to_dict(), kernel=NumPyKernel())
    result = algorithm.run(
        ZDT1Problem(n_var=6),
        termination=("max_evaluations", 20),
        seed=0,
    )
    return algorithm, result


def test_smpso_without_external_archive_preserves_leader_archive_result() -> None:
    algorithm, result = _run(_builder().build())

    assert algorithm.state is not None
    assert algorithm.state.result_archive is None
    assert result["archive"]["F"].shape[0] > 0
    np.testing.assert_allclose(result["F"], result["archive"]["F"])
    np.testing.assert_allclose(result["X"], result["archive"]["X"])


def test_smpso_external_archive_is_separate_and_drives_default_result() -> None:
    config = _builder().external_archive(capacity=3).build()
    assert config.result_mode == "non_dominated"

    algorithm, result = _run(config)

    assert algorithm.state is not None
    assert algorithm.state.result_archive is not None
    assert algorithm.state.result_archive is not algorithm.state.archive_manager
    assert result["archive"]["F"].shape[0] > 0
    assert result["archive"]["F"].shape[0] <= 3
    np.testing.assert_allclose(result["F"], result["archive"]["F"])
    np.testing.assert_allclose(result["X"], result["archive"]["X"])


def test_smpso_population_result_mode_wins_over_external_archive() -> None:
    config = _builder().external_archive(capacity=3).result_mode("population").build()

    _, result = _run(config)

    assert result["archive"]["F"].shape[0] > 0
    assert result["archive"]["F"].shape[0] <= 3
    np.testing.assert_allclose(result["F"], result["population"]["F"])
    np.testing.assert_allclose(result["X"], result["population"]["X"])


def test_smpso_configured_archive_round_trips_through_canonical_bundle(tmp_path: Path) -> None:
    config = _builder().external_archive(capacity=3).result_mode("population").build()
    _, raw_result = _run(config)
    limits = LoadLimits()
    arrays = snapshot_result_arrays(OptimizationResult(raw_result), limits=limits)
    bundle_path = tmp_path / "result.npz"

    write_result_bundle(bundle_path, arrays, limits=limits)
    loaded_arrays = load_result_bundle(
        bundle_path,
        descriptor=result_descriptor(),
        limits=limits,
        required_f=True,
        operation="test SMPSO result bundle",
    )
    payload = _result_payload(loaded_arrays, {})

    assert "archive" in payload
    np.testing.assert_allclose(payload["archive"]["F"], raw_result["archive"]["F"])
    np.testing.assert_allclose(payload["archive"]["X"], raw_result["archive"]["X"])
    np.testing.assert_allclose(payload["F"], raw_result["population"]["F"])
    np.testing.assert_allclose(payload["X"], raw_result["population"]["X"])


def test_smpso_rejects_invalid_raw_result_mode() -> None:
    raw = _builder().build().to_dict()
    raw["result_mode"] = "archive"

    algorithm = SMPSO(raw, kernel=NumPyKernel())
    with pytest.raises(ValueError, match="result_mode must be 'non_dominated' or 'population'"):
        algorithm.initialize(
            ZDT1Problem(n_var=6),
            termination=("max_evaluations", 20),
            seed=0,
        )
