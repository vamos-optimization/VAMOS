"""Bounded, data-only v1 run-artifact reader."""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vamos.experiment.optimization_result import OptimizationResult

from .bundle import load_result_bundle
from .errors import (
    ArtifactIntegrityError,
    ArtifactMissingError,
    ArtifactResourceLimitError,
    IncompleteRunError,
    ManifestValidationError,
    UnsupportedArtifactLayoutError,
    UnsupportedSchemaError,
)
from .jsonio import load_json_file, sha256_file
from .manifest import parse_run_manifest
from .models import ArtifactDescriptor, LoadLimits, RunManifest, StoredRun, VerifyMode, deep_freeze
from .paths import confined_artifact_path

_KNOWN_ARTIFACT_ROLES = (
    "environment",
    "result_bundle",
    "metrics",
    "events",
)


def read_run(path: str | Path, *, verify: VerifyMode, limits: LoadLimits) -> StoredRun:
    """Parse one run and return lazy data-only accessors."""
    root = _resolve_run_root(path)
    manifest_candidate = root / "manifest.json"
    if not manifest_candidate.is_file():
        _raise_unsupported_layout(root)
    manifest_path = confined_artifact_path(
        root,
        "manifest.json",
        role="manifest",
        operation="load run",
        must_exist=True,
    )
    manifest = parse_run_manifest(manifest_path, limits=limits, operation="load run")
    _validate_verify_mode(verify)
    if verify != "manifest":
        for descriptor in manifest.artifacts:
            if descriptor.role not in _KNOWN_ARTIFACT_ROLES:
                if verify == "required" and "load" in descriptor.required_for:
                    raise UnsupportedSchemaError(
                        operation="load run",
                        artifact_role=descriptor.role,
                        path=descriptor.path,
                        reason="uses an unknown artifact role required for loading",
                        expected="known v1 artifact role for every load requirement",
                        actual=descriptor.role,
                        action="Upgrade VAMOS to a reader that understands this role; the unknown artifact was not opened.",
                    )
                continue
            if verify == "all" or "load" in descriptor.required_for:
                verify_artifact(root, descriptor, limits=limits, operation="load run")
    return StoredRun(
        root=root,
        manifest=manifest,
        _result_loader=lambda: _read_result(root, manifest, verify=verify, limits=limits),
        _environment_loader=lambda: _read_environment(root, manifest, verify=verify, limits=limits),
    )


def verify_artifact(
    root: Path,
    descriptor: ArtifactDescriptor,
    *,
    limits: LoadLimits,
    operation: str,
) -> Path:
    """Confine and verify exact bytes for a known manifest descriptor."""
    configured_limit = limits.max_environment_bytes if descriptor.role == "environment" else limits.max_artifact_bytes
    if descriptor.bytes > configured_limit:
        raise ArtifactResourceLimitError(
            operation=operation,
            limit="max_environment_bytes" if descriptor.role == "environment" else "max_artifact_bytes",
            configured=configured_limit,
            observed=descriptor.bytes,
            artifact_role=descriptor.role,
            path=descriptor.path,
            action="Inspect the source and pass explicit trusted LoadLimits only if this declared size is expected.",
        )
    artifact_path = confined_artifact_path(
        root,
        descriptor.path,
        role=descriptor.role,
        operation=operation,
        must_exist=True,
    )
    if not artifact_path.is_file():
        raise ArtifactMissingError(
            operation=operation,
            artifact_role=descriptor.role,
            path=descriptor.path,
            reason="is missing or is not a regular file",
            expected={"sha256": descriptor.sha256, "bytes": descriptor.bytes},
            actual="missing",
            expected_sha256=descriptor.sha256,
            expected_bytes=descriptor.bytes,
            state="missing",
            action="Restore the exact referenced file from the original run.",
        )
    try:
        observed_bytes = artifact_path.stat().st_size
    except OSError as exc:
        raise ArtifactIntegrityError(
            operation=operation,
            artifact_role=descriptor.role,
            path=descriptor.path,
            reason="cannot be inspected",
            expected={"sha256": descriptor.sha256, "bytes": descriptor.bytes},
            actual=type(exc).__name__,
            expected_sha256=descriptor.sha256,
            expected_bytes=descriptor.bytes,
            state="unreadable",
            action="Restore the exact referenced file and its read permissions.",
        ) from exc
    if observed_bytes > configured_limit:
        raise ArtifactResourceLimitError(
            operation=operation,
            limit="max_environment_bytes" if descriptor.role == "environment" else "max_artifact_bytes",
            configured=configured_limit,
            observed=observed_bytes,
            artifact_role=descriptor.role,
            path=descriptor.path,
            action="Inspect the source and pass explicit trusted LoadLimits only if this file size is expected.",
        )
    if observed_bytes != descriptor.bytes:
        raise ArtifactIntegrityError(
            operation=operation,
            artifact_role=descriptor.role,
            path=descriptor.path,
            reason="has a byte length different from the manifest",
            expected=descriptor.bytes,
            actual=observed_bytes,
            expected_sha256=descriptor.sha256,
            expected_bytes=descriptor.bytes,
            actual_bytes=observed_bytes,
            state="length_mismatch",
            action="Restore the exact referenced file from the original run.",
        )
    try:
        actual_hash = sha256_file(artifact_path)
    except OSError as exc:
        raise ArtifactIntegrityError(
            operation=operation,
            artifact_role=descriptor.role,
            path=descriptor.path,
            reason="could not be read for integrity verification",
            expected=descriptor.sha256,
            actual=type(exc).__name__,
            expected_sha256=descriptor.sha256,
            expected_bytes=descriptor.bytes,
            actual_bytes=observed_bytes,
            state="unreadable",
            action="Restore the exact referenced file and its read permissions.",
        ) from exc
    if not hmac.compare_digest(actual_hash, descriptor.sha256):
        raise ArtifactIntegrityError(
            operation=operation,
            artifact_role=descriptor.role,
            path=descriptor.path,
            reason="has a SHA-256 hash different from the manifest",
            expected=descriptor.sha256,
            actual=actual_hash,
            expected_sha256=descriptor.sha256,
            actual_sha256=actual_hash,
            expected_bytes=descriptor.bytes,
            actual_bytes=observed_bytes,
            state="hash_mismatch",
            action="Restore the exact referenced file from the original run.",
        )
    return artifact_path


def _read_result(root: Path, manifest: RunManifest, *, verify: VerifyMode, limits: LoadLimits) -> OptimizationResult:
    descriptor = manifest.artifact("result_bundle")
    outcome = manifest.get("outcome")
    usable = outcome.get("usable_result") if isinstance(outcome, Mapping) else False
    if descriptor is None or manifest.status not in {"succeeded", "partial"} or usable is not True:
        raise IncompleteRunError(
            operation="load result",
            artifact_role="result_bundle",
            path=root,
            reason=f"run status {manifest.status!r} has no declared usable result",
            expected="a succeeded or usable partial run with canonical result_bundle",
            actual={"status": manifest.status, "usable_result": usable},
            action="Inspect load_run(path).manifest for failure evidence; failed runs do not contain a numerical result.",
        )
    if verify == "manifest":
        result_path = confined_artifact_path(root, descriptor.path, role=descriptor.role, operation="load result", must_exist=True)
        if not result_path.is_file():
            verify_artifact(root, descriptor, limits=limits, operation="load result")
    else:
        result_path = verify_artifact(root, descriptor, limits=limits, operation="load result")
    arrays = load_result_bundle(
        result_path,
        descriptor=descriptor,
        limits=limits,
        required_f=True,
        operation="load result",
    )
    _validate_outcome_arrays(manifest, arrays)
    payload = _result_payload(arrays, outcome)
    meta = _result_meta(manifest)
    return OptimizationResult(payload, meta=meta, manifest=manifest)


def _read_environment(root: Path, manifest: RunManifest, *, verify: VerifyMode, limits: LoadLimits) -> Mapping[str, Any]:
    descriptor = manifest.artifact("environment")
    if descriptor is None:
        raise IncompleteRunError(
            operation="load environment",
            artifact_role="environment",
            path=root,
            reason="has no environment descriptor",
            expected="manifest artifact role environment",
            actual="missing descriptor",
            action="Inspect the manifest provenance limitations or restore the original run.",
        )
    if verify == "manifest":
        environment_path = confined_artifact_path(
            root, descriptor.path, role=descriptor.role, operation="load environment", must_exist=True
        )
        if not environment_path.is_file():
            environment_path = verify_artifact(root, descriptor, limits=limits, operation="load environment")
    else:
        environment_path = verify_artifact(root, descriptor, limits=limits, operation="load environment")
    value = load_json_file(
        environment_path,
        operation="load environment",
        artifact_role="environment",
        max_bytes=limits.max_environment_bytes,
        max_depth=limits.max_json_depth,
    )
    if value.get("document_type") != "vamos.environment" or value.get("schema_version") != "1.0.0":
        raise ManifestValidationError(
            operation="load environment",
            artifact_role="environment",
            path=descriptor.path,
            reason="does not identify the supported v1 environment document",
            expected={"document_type": "vamos.environment", "schema_version": "1.0.0"},
            actual={"document_type": value.get("document_type"), "schema_version": value.get("schema_version")},
            action="Restore environment.json from the original run.",
        )
    frozen = deep_freeze(value)
    if not isinstance(frozen, Mapping):
        raise AssertionError("deep_freeze returned a non-mapping environment")
    return frozen


def _result_payload(arrays: Mapping[str, Any], outcome: object) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, value in arrays.items():
        if "/" not in name:
            payload[name] = value
            continue
        namespace, key = name.split("/", 1)
        nested = payload.setdefault(namespace, {})
        if isinstance(nested, dict):
            nested[key] = value
    if isinstance(outcome, Mapping):
        payload["evaluations"] = outcome.get("evaluations")
        payload["generation"] = outcome.get("generations")
        payload["interrupted"] = outcome.get("interrupted", False)
        payload["metrics"] = outcome.get("metrics", {})
    return payload


def _result_meta(manifest: RunManifest) -> dict[str, Any]:
    resolved = manifest.resolved_spec
    algorithm = resolved.get("algorithm")
    backend = resolved.get("backend")
    kernel = backend.get("kernel") if isinstance(backend, Mapping) else None
    defaults_applied = resolved.get("defaults_applied")
    default_sources = (
        {
            str(item["field"]).lstrip("/"): item["source"]
            for item in defaults_applied
            if isinstance(item, Mapping) and "field" in item and "source" in item
        }
        if isinstance(defaults_applied, tuple)
        else {}
    )
    outcome = manifest.get("outcome")
    timestamps = manifest.get("timestamps")
    run_artifact_timestamps = dict(timestamps) if isinstance(timestamps, Mapping) else {}
    if isinstance(outcome, Mapping):
        run_artifact_timestamps["runtime_ms"] = outcome.get("runtime_ms", 0.0)
    return {
        "algorithm": _component_name(algorithm),
        "engine": _component_name(kernel),
        "seed": resolved.get("seed"),
        "run_id": manifest.run_id,
        "task_id": manifest.task_id,
        "run_manifest": manifest,
        "run_artifact_requested_spec": dict(manifest.requested_spec),
        "run_artifact_resolved_spec": resolved.as_dict(),
        "run_artifact_timestamps": run_artifact_timestamps,
        # Marks a loaded result: saving it again records a manual copy, not this execution.
        "run_artifact_loaded_run_id": manifest.run_id,
        "default_sources": default_sources,
    }


def _component_name(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    resolution = value.get("resolution")
    if isinstance(resolution, Mapping) and isinstance(resolution.get("name"), str):
        return str(resolution["name"])
    component_id = value.get("component_id")
    if isinstance(component_id, str) and ":" in component_id:
        return component_id.split(":", 1)[1].split("@", 1)[0]
    return None


def _validate_outcome_arrays(manifest: RunManifest, arrays: Mapping[str, Any]) -> None:
    outcome = manifest.get("outcome")
    f_array = arrays.get("F")
    if not isinstance(outcome, Mapping) or f_array is None:
        return
    expected = {
        "n_solutions": int(f_array.shape[0]),
        "n_objectives": int(f_array.shape[1]),
    }
    x_array = arrays.get("X")
    if x_array is not None:
        expected["n_variables"] = int(x_array.shape[1])
    for field, derived in expected.items():
        if field not in outcome:
            continue
        if outcome.get(field) != derived:
            raise ManifestValidationError(
                operation="load result",
                field=f"$.outcome.{field}",
                path="manifest.json",
                reason="does not match the canonical result array shape",
                expected=derived,
                actual=outcome.get(field),
                action="Restore the original manifest and result bundle; cached counts never override arrays.",
            )


def _resolve_run_root(path: str | Path) -> Path:
    candidate = Path(path)
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise UnsupportedArtifactLayoutError(
            operation="load run",
            path=candidate,
            reason="does not exist or cannot be resolved",
            expected="existing v1 run directory containing manifest.json",
            actual=type(exc).__name__,
            action="Select the complete run directory.",
        ) from exc
    if not root.is_dir():
        raise UnsupportedArtifactLayoutError(
            operation="load run",
            path=root,
            reason="is not a directory",
            expected="v1 run directory containing manifest.json",
            actual="file",
            action="Select the directory that contains manifest.json.",
        )
    return root


def _raise_unsupported_layout(root: Path) -> None:
    raise UnsupportedArtifactLayoutError(
        operation="load run",
        path=root,
        reason="does not contain the canonical manifest.json",
        expected="v1 run directory with document_type vamos.run-manifest",
        actual="unsupported directory layout",
        action="This is a pre-1.0 development format; regenerate the run with the current VAMOS version.",
    )


def _validate_verify_mode(value: object) -> None:
    if value not in {"manifest", "required", "all"}:
        raise ValueError("verify must be 'manifest', 'required', or 'all'.")


__all__ = ["read_run", "verify_artifact"]
