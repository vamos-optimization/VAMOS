"""Regenerate deterministic sanitized replay examples for the v1 contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vamos.experiment.artifacts.comparison import compare_array_collections
from vamos.experiment.artifacts.jsonio import sha256_file, stored_json_bytes
from vamos.experiment.artifacts.manifest import build_terminal_manifest
from vamos.experiment.artifacts.models import LoadLimits

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs" / "dev" / "run_artifact_examples"
SOURCE = EXAMPLES / "nsgaii-success"
SOURCE_RUN_ID = "11111111-1111-4111-8111-111111111111"
TIMESTAMPS = {"started_at": "2026-01-01T00:04:00Z", "completed_at": "2026-01-01T00:04:01Z"}
# The examples record a fixed VAMOS version, like their fixed IDs and timestamps (README.md there).
EXAMPLE_VERSION = "1.0.0"
BASE_EXAMPLES = ("nsgaii-success", "moead-success", "custom-manual", "failed-run")


def main() -> None:
    _normalize_base_examples()
    source_manifest = _json(SOURCE / "manifest.json")
    source_arrays = _arrays(SOURCE / "result.npz")
    _verification_examples(source_manifest)
    _successful_replay(source_manifest, source_arrays)
    _mismatch_replay(source_manifest, source_arrays)
    _failed_replay(source_manifest)


def _normalize_base_examples() -> None:
    """Keep the checked-in contract examples on the fixed example version."""
    for name in BASE_EXAMPLES:
        destination = EXAMPLES / name
        environment_path = destination / "environment.json"
        environment = _json(environment_path)
        environment["packages"]["vamos-optimization"] = EXAMPLE_VERSION
        _write_json(environment_path, environment)

        manifest_path = destination / "manifest.json"
        manifest = _json(manifest_path)
        implementation = manifest["provenance"]["implementation"]
        implementation["distribution"]["version"] = EXAMPLE_VERSION
        implementation["vamos_version"] = EXAMPLE_VERSION

        environment_artifact = next(artifact for artifact in manifest["artifacts"] if artifact["role"] == "environment")
        normalized_descriptor = _descriptor("environment", environment_path)
        environment_artifact.update(
            {
                "sha256": normalized_descriptor["sha256"],
                "bytes": normalized_descriptor["bytes"],
            }
        )
        manifest.pop("integrity", None)
        terminal = build_terminal_manifest(manifest, limits=LoadLimits())
        _write_json(manifest_path, terminal.as_dict())


def _verification_examples(source: dict[str, Any]) -> None:
    common = {
        "document_type": "vamos.verification-report",
        "version": "1",
        "root": "runs/nsgaii-success",
        "run_id": source["run_id"],
        "task_id": source["task_id"],
        "status": source["status"],
        "schema": "1.0.0",
        "artifact_integrity": "valid",
        "path_safety": "valid",
        "numerical_bundle_safety": "valid",
        "component_reconstructability": "reconstructable",
        "optimization_executed": False,
    }
    exact = {
        **common,
        "environment_compatibility": {"level": "exact", "findings": []},
        "effective_replayability": "exact",
        "reasons": [],
    }
    incompatible = {
        **common,
        "environment_compatibility": {
            "level": "compatible",
            "findings": [
                {
                    "field": "$.environment.python.version",
                    "stored": "3.12",
                    "current": "3.11",
                    "classification": "compatible",
                    "explanation": "Stored and current Python major/minor version differ.",
                    "blocks_exact": True,
                    "action": "Use Python 3.12.",
                }
            ],
        },
        "effective_replayability": "compatible",
        "reasons": [
            {
                "code": "environment_not_exact",
                "field": "$.environment.python.version",
                "message": "The current environment is not an exact match.",
                "action": "Use the recorded material environment.",
            }
        ],
    }
    _write_json(EXAMPLES / "verification-exact.json", exact)
    _write_json(EXAMPLES / "verification-incompatible.json", incompatible)


def _successful_replay(source: dict[str, Any], source_arrays: dict[str, np.ndarray]) -> None:
    destination = EXAMPLES / "replay-success"
    replay_arrays = {name: np.array(value, copy=True) for name, value in source_arrays.items()}
    comparisons = compare_array_collections(source_arrays, replay_arrays)
    manifest = _replay_manifest(
        source,
        run_id="55555555-5555-4555-8555-555555555555",
        name="Successful exact replay example",
        comparison_status="exact_match",
        comparisons=[item.as_dict() for item in comparisons],
        replayability="exact",
    )
    _write_run(destination, manifest, replay_arrays)


def _mismatch_replay(source: dict[str, Any], source_arrays: dict[str, np.ndarray]) -> None:
    destination = EXAMPLES / "replay-mismatch"
    replay_arrays = {name: np.array(value, copy=True) for name, value in source_arrays.items()}
    changed = replay_arrays["F"].view(np.uint64)
    changed.flat[0] ^= np.uint64(1)
    comparisons = compare_array_collections(source_arrays, replay_arrays)
    manifest = _replay_manifest(
        source,
        run_id="66666666-6666-4666-8666-666666666666",
        name="Completed replay mismatch example",
        comparison_status="mismatch",
        comparisons=[item.as_dict() for item in comparisons],
        replayability="unavailable",
    )
    _write_run(destination, manifest, replay_arrays)


def _failed_replay(source: dict[str, Any]) -> None:
    destination = EXAMPLES / "failed-replay"
    manifest = _replay_manifest(
        source,
        run_id="77777777-7777-4777-8777-777777777777",
        name="Failed replay attempt example",
        comparison_status="execution_failed",
        comparisons=[],
        replayability="unavailable",
    )
    manifest["status"] = "failed"
    manifest["outcome"] = {
        "evaluations": None,
        "generations": None,
        "runtime_ms": 1.0,
        "termination_reason": "replay_execution_error",
        "result_mode": "population",
        "interrupted": True,
        "usable_result": False,
        "n_solutions": None,
        "n_objectives": 2,
        "n_variables": 3,
        "metrics": {},
    }
    manifest["failure"] = {
        "phase": "optimization",
        "exception_type": "EvaluationError",
        "message": "Built-in objective evaluation failed.",
        "traceback": None,
        "optimization_executed": True,
    }
    _write_run(destination, manifest, None)


def _replay_manifest(
    source: dict[str, Any],
    *,
    run_id: str,
    name: str,
    comparison_status: str,
    comparisons: list[dict[str, Any]],
    replayability: str,
) -> dict[str, Any]:
    provenance = json.loads(json.dumps(source["provenance"]))
    provenance["timestamps"] = dict(TIMESTAMPS)
    provenance["entry_point"] = {
        "kind": "replay",
        "python": {"callable": "vamos.reproduce", "arguments_source": "resolved_spec"},
    }
    return {
        "document_type": "vamos.run-manifest",
        "schema_version": "1.0.0",
        "run_id": run_id,
        "task_id": source["task_id"],
        "status": "succeeded",
        "timestamps": dict(TIMESTAMPS),
        "requested_spec": source["requested_spec"],
        "resolved_spec": source["resolved_spec"],
        "provenance": provenance,
        "replayability": {
            "declared_level": replayability,
            "deterministic": True,
            "exact_requirements": ["bitwise_equal_result"] if replayability == "exact" else [],
            "reasons": [] if replayability == "exact" else [{"code": "example_non_exact", "message": "Example is not exact."}],
        },
        "outcome": dict(source["outcome"]),
        "name": name,
        "lineage": {
            "execution_kind": "replay",
            "source_run_id": SOURCE_RUN_ID,
            "source_manifest_sha256": source["integrity"]["manifest_sha256"],
            "root_run_id": SOURCE_RUN_ID,
            "depth": 1,
            "replay_plan_sha256": "b" * 64,
            "compatibility_level": "exact",
            "comparison": {"status": comparison_status, "arrays": comparisons},
        },
        "artifacts": [],
    }


def _write_run(destination: Path, manifest: dict[str, Any], arrays: dict[str, np.ndarray] | None) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    environment_path = destination / "environment.json"
    environment_path.write_bytes((SOURCE / "environment.json").read_bytes())
    artifacts = [_descriptor("environment", environment_path)]
    result_path = destination / "result.npz"
    if arrays is None:
        result_path.unlink(missing_ok=True)
    else:
        with result_path.open("wb") as handle:
            np.savez(handle, **arrays)
        result_descriptor = _descriptor("result_bundle", result_path)
        result_descriptor["array_contract"] = {
            name: {"dtype": value.dtype.str, "shape": list(value.shape)} for name, value in sorted(arrays.items())
        }
        artifacts.append(result_descriptor)
    manifest["artifacts"] = artifacts
    terminal = build_terminal_manifest(manifest, limits=LoadLimits())
    _write_json(destination / "manifest.json", terminal.as_dict())


def _descriptor(role: str, path: Path) -> dict[str, Any]:
    result = role == "result_bundle"
    return {
        "role": role,
        "path": path.name,
        "media_type": "application/vnd.vamos.result-bundle+npz" if result else "application/vnd.vamos.environment+json",
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "required_for": ["load", "inspect", "verify", "replay", "analysis"] if result else ["inspect", "verify", "replay"],
        "canonical": True,
    }


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {name: np.array(bundle[name], copy=True) for name in bundle.files}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(stored_json_bytes(value))


if __name__ == "__main__":
    main()
