from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.engine.algorithm.config.nsgaii import NSGAIIConfig
from vamos.experiment.artifacts import IncompleteRunError
from vamos.experiment.artifacts.jsonio import manifest_self_hash, sha256_file
from vamos.experiment.optimization_result import OptimizationResult


def _manual_meta(seed: int) -> dict[str, object]:
    source = json.loads(Path("docs/dev/run_artifact_examples/custom-manual/manifest.json").read_text(encoding="utf-8"))
    resolved = source["resolved_spec"]
    resolved["seed"] = seed
    return {
        "seed": seed,
        "run_artifact_requested_spec": source["requested_spec"],
        "run_artifact_resolved_spec": resolved,
    }


def test_minimal_builtin_nsgaii_round_trip_ra001_ra003(tmp_path: Path) -> None:
    result = vamos.optimize("zdt1", algorithm="nsgaii", pop_size=8, max_evaluations=16, engine="numpy", seed=7)

    stored = vamos.save_result(result, tmp_path / "run", labels={"purpose": "test"})
    loaded = vamos.load_result(tmp_path / "run")

    assert stored.status == "succeeded"
    assert loaded.manifest is stored.manifest or loaded.manifest == stored.manifest
    assert loaded.manifest is not None
    assert loaded.manifest["labels"] == {"purpose": "test"}
    assert np.array_equal(loaded.F, result.F)
    assert np.array_equal(loaded.X, result.X)
    assert loaded.F.dtype == result.F.dtype
    assert loaded.X.dtype == result.X.dtype
    manifest = stored.manifest
    assert manifest["integrity"]["manifest_sha256"] == manifest_self_hash(manifest)
    for descriptor in manifest.artifacts:
        artifact = stored.root / descriptor.path
        assert artifact.stat().st_size == descriptor.bytes
        assert sha256_file(artifact) == descriptor.sha256
    assert {item.name for item in (tmp_path / "run").iterdir()} == {
        "environment.json",
        "manifest.json",
        "result.npz",
    }


def test_rich_nsgaii_resolved_operators_round_trip_ra002(tmp_path: Path) -> None:
    config = (
        NSGAIIConfig.builder()
        .pop_size(8)
        .offspring_size(4)
        .crossover("sbx", prob=0.8, eta=15.0)
        .mutation("pm", prob="1/n", eta=30.0)
        .selection("tournament", size=3)
        .repair("clip")
        .result_mode("population")
        .build()
    )
    result = vamos.optimize(
        "zdt1",
        algorithm="nsgaii",
        max_evaluations=16,
        engine="numpy",
        seed=11,
        n_var=4,
        algorithm_config=config,
    )

    manifest = vamos.save_result(result, tmp_path / "rich").manifest
    operators = manifest.resolved_spec["operators"]
    algorithm = manifest.resolved_spec["algorithm"]

    assert operators["crossover"]["component_id"] == "vamos.operator:sbx@1"
    assert operators["crossover"]["config"] == {"prob": 0.8, "eta": 15.0}
    assert operators["mutation"]["component_id"] == "vamos.operator:polynomial@1"
    assert operators["mutation"]["config"]["prob"] == pytest.approx(0.25)
    assert operators["selection"]["config"]["size"] == 3
    assert operators["repair"]["component_id"] == "vamos.operator:clip@1"
    assert algorithm["config"]["offspring_size"] == 4
    assert algorithm["config"]["result_mode"] == "population"
    assert manifest["requested_spec"]["algorithms"]["nsgaii"]["mutation"][1]["prob"] == "1/n"


@pytest.mark.parametrize(
    ("problem", "dtype_kind"),
    [
        ("zdt1", "f"),
        ("int_alloc", "i"),
        ("bin_feat", "i"),
        ("tsp6", "i"),
        ("mixed_design", "f"),
    ],
)
def test_public_decision_encodings_round_trip(problem: str, dtype_kind: str, tmp_path: Path) -> None:
    result = vamos.optimize(problem, algorithm="nsgaii", pop_size=6, max_evaluations=6, engine="numpy", seed=2)
    loaded = vamos.load_result(vamos.save_result(result, tmp_path / problem).root)

    assert loaded.X is not None
    assert result.X is not None
    assert loaded.X.dtype.kind == dtype_kind
    assert loaded.X.dtype == result.X.dtype
    assert np.array_equal(loaded.X, result.X)


@pytest.mark.parametrize(
    ("f_dtype", "x_dtype"),
    [(np.dtype("<f4"), np.dtype(">i4")), (np.dtype(">f8"), np.dtype("u2"))],
)
def test_result_arrays_preserve_empty_shape_dtype_and_values_ra004(
    f_dtype: np.dtype[object],
    x_dtype: np.dtype[object],
    tmp_path: Path,
) -> None:
    result = OptimizationResult(
        {
            "F": np.empty((0, 2), dtype=f_dtype),
            "X": np.empty((0, 3), dtype=x_dtype),
            "evaluations": 0,
        },
        meta=_manual_meta(3),
    )

    loaded = vamos.load_result(vamos.save_result(result, tmp_path / "run").root)

    assert loaded.F is not None and loaded.F.shape == (0, 2) and loaded.F.dtype.str == f_dtype.str
    assert loaded.X is not None and loaded.X.shape == (0, 3) and loaded.X.dtype.str == x_dtype.str


def test_constraints_outcome_and_counters_round_trip_ra004_ra005(tmp_path: Path) -> None:
    f_array = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    x_array = np.array([[1, 2], [3, 4]], dtype=np.int16)
    g_array = np.array([[-1.0, 0.0], [0.25, -0.5]], dtype=np.float64)
    cv_array = np.maximum(g_array, 0.0).sum(axis=1)
    result = OptimizationResult(
        {
            "F": f_array,
            "X": x_array,
            "G": g_array,
            "CV": cv_array,
            "evaluations": 12,
            "generation": 3,
            "interrupted": False,
            "hv_reached": True,
        },
        meta=_manual_meta(5),
    )

    stored = vamos.save_result(result, tmp_path / "constraints")
    loaded = stored.result
    outcome = stored.manifest["outcome"]

    assert np.array_equal(loaded.data["G"], g_array)
    assert np.array_equal(loaded.data["CV"], cv_array)
    assert np.array_equal(loaded.data["CV"], np.maximum(loaded.data["G"], 0.0).sum(axis=1))
    assert outcome["evaluations"] == 12
    assert outcome["generations"] == 3
    assert outcome["n_solutions"] == 2
    assert outcome["n_objectives"] == 2
    assert outcome["n_variables"] == 2
    assert outcome["metrics"]["hv_reached"] is True


def test_auxiliary_population_archive_and_reference_arrays_round_trip(tmp_path: Path) -> None:
    result = OptimizationResult(
        {
            "F": np.array([[1.0, 2.0]]),
            "X": np.array([[0.25, 0.75]]),
            "population": {
                "F": np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32),
                "X": np.array([[0.25, 0.75], [0.75, 0.25]], dtype=np.float32),
                "G": np.array([[-1.0], [0.5]], dtype=np.float32),
                "CV": np.array([0.0, 0.5], dtype=np.float32),
            },
            "archive": {
                "F": np.array([[0.5, 0.5]], dtype=np.float64),
                "X": np.array([[0.5, 0.5]], dtype=np.float64),
            },
            "reference_directions": np.eye(2, dtype=np.float64),
        },
        meta=_manual_meta(6),
    )

    loaded = vamos.load_result(vamos.save_result(result, tmp_path / "auxiliary").root)

    for namespace in ("population", "archive"):
        for key, array in result.data[namespace].items():
            assert np.array_equal(loaded.data[namespace][key], array)
    assert np.array_equal(loaded.data["reference_directions"], result.data["reference_directions"])


def test_boolean_and_nonfinite_arrays_are_lossless_binary_data(tmp_path: Path) -> None:
    result = OptimizationResult(
        {
            "F": np.array([[np.nan, np.inf], [-np.inf, 0.0]], dtype=np.float64),
            "X": np.array([[True, False], [False, True]], dtype=np.bool_),
        },
        meta=_manual_meta(1),
    )

    loaded = vamos.load_result(vamos.save_result(result, tmp_path / "nonfinite").root)

    assert loaded.X is not None and loaded.X.dtype == np.dtype(np.bool_)
    assert np.array_equal(loaded.X, result.X)
    assert np.array_equal(loaded.F, result.F, equal_nan=True)


def test_requested_omissions_and_resolved_defaults_survive_ra002(tmp_path: Path) -> None:
    result = vamos.optimize("zdt1", pop_size=8, max_evaluations=8, seed=4)
    manifest = vamos.save_result(result, tmp_path / "defaults").manifest
    requested_defaults = manifest["requested_spec"]["defaults"]
    applied = {item["field"] for item in manifest.resolved_spec["defaults_applied"]}

    assert "algorithm" not in requested_defaults
    assert "engine" not in requested_defaults
    assert manifest.resolved_spec["algorithm"]["component_id"].startswith("vamos.algorithm:")
    assert "/algorithm" in applied
    assert "/engine" in applied


def test_approved_moead_and_failed_examples_load_ra022_ra024() -> None:
    examples = Path("docs/dev/run_artifact_examples")
    moead = vamos.load_run(examples / "moead-success")
    failed = vamos.load_run(examples / "failed-run")

    assert moead.result.data["reference_directions"].shape == (3, 3)
    assert moead.manifest.resolved_spec["algorithm"]["component_id"] == "vamos.algorithm:moead@1"
    assert failed.status == "failed"
    assert failed.manifest["failure"]["exception_type"] == "EvaluationError"
    with pytest.raises(IncompleteRunError):
        _ = failed.result


@pytest.mark.parametrize(
    ("name", "status"),
    [
        ("nsgaii-success", "succeeded"),
        ("moead-success", "succeeded"),
        ("custom-manual", "succeeded"),
        ("failed-run", "failed"),
    ],
)
def test_machine_readable_contract_examples_verify(name: str, status: str) -> None:
    examples = Path("docs/dev/run_artifact_examples")
    run = vamos.load_run(examples / name, verify="all")

    assert run.status == status
    if status == "succeeded":
        assert run.result.F is not None


def test_seed_none_is_resolved_before_execution_and_persisted_ra007(tmp_path: Path) -> None:
    result = vamos.optimize("zdt1", pop_size=4, max_evaluations=4, seed=None)
    requested = result.meta["run_artifact_requested_spec"]
    resolved = result.meta["run_artifact_resolved_spec"]

    assert requested["defaults"]["seed"] is None
    assert isinstance(resolved["seed"], int)
    assert result.meta["seed"] == resolved["seed"]
    stored = vamos.save_result(result, tmp_path / "generated-seed")
    assert stored.manifest.requested_spec["defaults"]["seed"] is None
    assert stored.manifest.resolved_spec["seed"] == resolved["seed"]


def test_seed_zero_remains_valid_and_is_not_replaced_ra006(tmp_path: Path) -> None:
    result = vamos.optimize("zdt1", pop_size=4, max_evaluations=4, seed=0)
    stored = vamos.save_result(result, tmp_path / "seed-zero")

    assert result.meta["seed"] == 0
    assert stored.manifest.requested_spec["defaults"]["seed"] == 0
    assert stored.manifest.resolved_spec["seed"] == 0


def test_run_directory_is_relocatable_and_contains_no_source_path_ra016(tmp_path: Path) -> None:
    result = vamos.optimize("zdt1", algorithm="nsgaii", pop_size=6, max_evaluations=6, seed=9)
    original = tmp_path / "source" / "run"
    destination = tmp_path / "unrelated" / "moved"
    vamos.save_result(result, original)
    destination.parent.mkdir()
    shutil.move(original, destination)

    loaded = vamos.load_result(destination, verify="all")
    stored_json = (destination / "manifest.json").read_text(encoding="utf-8") + (destination / "environment.json").read_text(
        encoding="utf-8"
    )

    assert np.array_equal(loaded.F, result.F)
    assert str(original) not in stored_json
    assert str(tmp_path) not in stored_json


def test_provenance_is_present_and_privacy_conscious(tmp_path: Path) -> None:
    result = vamos.optimize("zdt1", algorithm="nsgaii", pop_size=6, max_evaluations=6, seed=3)
    stored = vamos.save_result(result, tmp_path / "provenance")
    provenance = stored.manifest["provenance"]
    environment = stored.environment

    assert provenance["implementation"]["vamos_version"]
    assert provenance["source"]["dirty"] in {True, False, "unknown"}
    assert provenance["timestamps"] == stored.manifest["timestamps"]
    assert environment["python"]["version"]
    assert environment["platform"]["operating_system"]
    assert "hostname" not in environment
    assert set(environment["threads"]) <= {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }


def test_manifest_and_environment_are_recursively_immutable(tmp_path: Path) -> None:
    result = vamos.optimize("zdt1", algorithm="nsgaii", pop_size=6, max_evaluations=6, seed=9)
    stored = vamos.save_result(result, tmp_path / "immutable")

    with pytest.raises(TypeError):
        stored.manifest["status"] = "failed"  # type: ignore[index]
    with pytest.raises(TypeError):
        stored.environment["python"] = {}  # type: ignore[index]
    assert json.loads((stored.root / "manifest.json").read_text(encoding="utf-8"))["status"] == "succeeded"


def test_save_result_explains_windows_path_length_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vamos.experiment.artifacts import paths

    result = vamos.optimize("zdt1", algorithm="nsgaii", pop_size=6, max_evaluations=6, engine="numpy", seed=4)
    long_path = "C:\\" + "d" * 257
    assert len(long_path) == 260
    real_mkdir = Path.mkdir

    def mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
        if ".vamos-staging-" in self.name:
            raise FileNotFoundError(2, "The system cannot find the path specified", long_path)
        real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(paths, "_WINDOWS", True)
    with pytest.raises(FileNotFoundError) as caught:
        vamos.save_result(result, tmp_path / "run")

    message = str(caught.value)
    assert "The system cannot find the path specified" in message
    assert "260 characters" in message and "LongPathsEnabled" in message
    assert caught.value.filename == long_path
    assert not (tmp_path / "run").exists() and not list(tmp_path.glob(".run.vamos-save.lock"))

    monkeypatch.setattr(paths, "_WINDOWS", False)
    with pytest.raises(FileNotFoundError) as caught:
        vamos.save_result(result, tmp_path / "run")
    assert "LongPathsEnabled" not in str(caught.value)
