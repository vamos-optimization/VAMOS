"""Where MOEA/D's weight vectors come from, as a run artifact and ``vamos tune`` record it."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.engine.algorithm.components import weight_vectors
from vamos.engine.algorithm.components.weight_vectors import load_or_generate_weight_vectors
from vamos.engine.algorithm.config import MOEADConfig
from vamos.engine.reference_directions import das_dennis
from vamos.resources import weight_path


def _resolved(pop_size: int, n_obj: int, config: MOEADConfig | None = None) -> dict[str, Any]:
    result = vamos.optimize(
        "dtlz2",
        algorithm="moead",
        algorithm_config=config,
        pop_size=None if config is not None else pop_size,
        n_obj=n_obj,
        max_evaluations=2 * pop_size,
        seed=1,
        engine="numpy",
    )
    return result.meta["run_artifact_resolved_spec"]


@pytest.mark.parametrize(
    ("pop_size", "n_obj", "expected"),
    [
        (91, 3, {"source": "das_dennis", "divisions": 12, "vectors": 91}),
        (136, 3, {"source": "das_dennis", "divisions": 15, "vectors": 136}),
        (100, 3, {"source": "packaged_file", "file": "W3D_100.dat", "vectors": 100}),
        # VAMOS also ships W3D_120.dat, but a complete lattice with that size comes first.
        (120, 3, {"source": "das_dennis", "divisions": 14, "vectors": 120}),
        (165, 4, {"source": "das_dennis", "divisions": 8, "vectors": 165}),
        (156, 8, {"source": "packaged_file", "file": "W8D_156.dat", "vectors": 156}),
        (50, 2, {"source": "uniform", "vectors": 50}),
    ],
)
def test_the_resolved_spec_records_the_population_and_the_weights(pop_size: int, n_obj: int, expected: dict[str, Any]) -> None:
    resolved = _resolved(pop_size, n_obj)

    assert resolved["population"]["initial_size"] == pop_size
    assert resolved["algorithm"]["resolution"]["weight_vectors"] == expected


def test_a_weight_file_is_recorded_by_its_name(tmp_path: Path) -> None:
    weights = das_dennis(3, 4)
    np.savetxt(tmp_path / "W3D_15.dat", weights)
    base = MOEADConfig.default(pop_size=15, n_var=12, n_obj=3)
    config = dataclasses.replace(base, weight_vectors={"path": str(tmp_path), "divisions": None})

    resolved = _resolved(15, 3, config)

    assert resolved["algorithm"]["resolution"]["weight_vectors"] == {"source": "file", "file": "W3D_15.dat", "vectors": 15}


@pytest.mark.parametrize(("pop_size", "n_obj", "divisions"), [(91, 3, None), (100, 3, None), (45, 3, 8), (210, 5, None), (30, 2, None)])
def test_the_description_names_the_weights_the_run_uses(pop_size: int, n_obj: int, divisions: int | None) -> None:
    described = weight_vectors.describe_weight_vectors(pop_size, n_obj, divisions=divisions)
    weights = load_or_generate_weight_vectors(pop_size, n_obj, divisions=divisions, mode="jmetalpy")

    assert described["vectors"] == len(weights)
    if described["source"] == "das_dennis":
        np.testing.assert_array_equal(weights, das_dennis(n_obj, int(described["divisions"])))
    elif described["source"] == "packaged_file":
        stored = np.loadtxt(weight_path(str(described["file"])))
        np.testing.assert_array_equal(weights, stored / stored.sum(axis=1, keepdims=True))
    else:
        assert described == {"source": "uniform", "vectors": pop_size}


def test_a_moead_run_with_packaged_weights_still_replays_exactly(tmp_path: Path) -> None:
    result = vamos.optimize("dtlz2", algorithm="moead", pop_size=100, n_obj=3, max_evaluations=200, seed=1, engine="numpy")
    source = vamos.save_result(result, tmp_path / "source")

    assert vamos.verify_run(source.root).effective_replayability == "exact"
    assert vamos.reproduce(source.root, output=tmp_path / "replay").exact is True


def test_tune_records_the_population_moead_ran(tmp_path: Path) -> None:
    from vamos.engine.tuning import Instance, Int, ParamSpace, TrialResult, TuningTask
    from vamos.experiment.cli._tune_post import persist_artifacts

    task = TuningTask(
        name="moead",
        param_space=ParamSpace(params={"pop_size": Int("pop_size", 20, 200)}),
        instances=[Instance(name="dtlz2", n_var=12, kwargs={})],
        seeds=[1],
        budget_per_run=300,
    )
    # 150 individuals have no weight vectors with three objectives: MOEA/D runs 136, a complete lattice.
    history = [TrialResult(0, {"pop_size": 150}, 0.5), TrialResult(1, {"pop_size": 91}, 0.4)]
    args = argparse.Namespace(
        backend="random",
        problem="dtlz2",
        algorithm="moead",
        n_var=12,
        n_obj=3,
        seed=1,
        aggregate_mode="mean",
        runtime_penalty=0.0,
        failure_score=0.0,
    )

    persist_artifacts(tmp_path, args, task, {"pop_size": 150}, history, 1.0, 1, available_backends_fn=dict)

    records = json.loads((tmp_path / "tuning_history.json").read_text(encoding="utf-8"))
    assert records[0]["details"]["effective"] == {"pop_size": 136}
    assert "effective" not in records[1]["details"]
    summary = json.loads((tmp_path / "tuning_summary.json").read_text(encoding="utf-8"))
    assert summary["best_config_effective"] == {"pop_size": 136}


def test_a_moead_run_without_the_record_is_not_verified_as_exact(tmp_path: Path) -> None:
    from vamos.experiment.artifacts.jsonio import canonical_json_bytes, manifest_self_hash, sha256_bytes, stored_json_bytes

    result = vamos.optimize("dtlz2", algorithm="moead", pop_size=91, n_obj=3, max_evaluations=182, seed=1, engine="numpy")
    source = vamos.save_result(result, tmp_path / "source")
    manifest = json.loads((source.root / "manifest.json").read_text(encoding="utf-8"))
    # A run stored before the weights were recorded.
    del manifest["resolved_spec"]["algorithm"]["resolution"]["weight_vectors"]
    manifest["task_id"] = "sha256:" + sha256_bytes(canonical_json_bytes(manifest["resolved_spec"]))
    manifest["integrity"]["manifest_sha256"] = manifest_self_hash(manifest)
    (source.root / "manifest.json").write_bytes(stored_json_bytes(manifest))

    assert vamos.verify_run(source.root).effective_replayability != "exact"


def test_assignment_adjustments_name_only_what_changes() -> None:
    from vamos.engine.tuning.racing.bridge_assignments import assignment_adjustments

    assert assignment_adjustments("moead", {"pop_size": 150, "n_obj": 3}) == {"pop_size": 136}
    assert assignment_adjustments("moead", {"pop_size": 150, "n_obj": 2}) == {}
    assert assignment_adjustments("moead", {"pop_size": 150}) == {}
    assert assignment_adjustments("moead", {"pop_size": 2, "n_obj": 3}) == {}
    assert assignment_adjustments("nsgaii", {"pop_size": 150, "n_obj": 3}) == {}
