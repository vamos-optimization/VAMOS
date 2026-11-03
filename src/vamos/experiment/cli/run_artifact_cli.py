"""CLI rendering for canonical inspection, verification, and replay."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from vamos.experiment.artifacts import (
    ArtifactIntegrityError,
    ArtifactResourceLimitError,
    DuplicateJSONKeyError,
    MalformedResultBundleError,
    ManifestValidationError,
    OutputCollisionError,
    ReplayExecutionError,
    ReplayUnavailableError,
    RunArtifactError,
    UnsafeArtifactPathError,
    UnsupportedArtifactLayoutError,
    UnsupportedSchemaError,
    VerificationRequirementError,
    inspect_run,
    reproduce,
    verify_run,
)


def run_results(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vamos results", description="Inspect or verify one canonical run directory.")
    commands = parser.add_subparsers(dest="results_command", required=True)
    inspect_parser = commands.add_parser("inspect", help="Read a concise manifest-only summary.")
    inspect_parser.add_argument("run_dir")
    inspect_parser.add_argument("--json", action="store_true", dest="json_output")
    verify_parser = commands.add_parser("verify", help="Fully verify integrity, compatibility, and replayability.")
    verify_parser.add_argument("run_dir")
    verify_parser.add_argument("--json", action="store_true", dest="json_output")
    verify_parser.add_argument(
        "--require-level",
        choices=("exact", "compatible", "best_effort", "manual", "unavailable"),
    )
    args = parser.parse_args(argv)
    if args.results_command == "inspect":
        _invoke(lambda: inspect_run(args.run_dir), json_output=args.json_output, renderer=_render_inspection)
        return
    _invoke(
        lambda: verify_run(args.run_dir, require_level=args.require_level),
        json_output=args.json_output,
        renderer=_render_verification,
    )


def run_reproduce(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vamos reproduce", description="Execute an exact same-environment built-in replay.")
    parser.add_argument("run_dir")
    parser.add_argument("--output")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    _invoke(
        lambda: reproduce(args.run_dir, output=args.output),
        json_output=args.json_output,
        renderer=_render_replay,
    )


def _invoke(operation: Callable[[], Any], *, json_output: bool, renderer: Callable[[Any], str]) -> None:
    try:
        result = operation()
    except RunArtifactError as exc:
        _emit_error(exc, json_output=json_output)
        raise SystemExit(run_artifact_exit_code(exc)) from None
    payload = result.as_dict() if hasattr(result, "as_dict") else result
    if json_output:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        print(renderer(result))


def _render_inspection(value: Mapping[str, Any]) -> str:
    arrays = value.get("arrays")
    array_lines = []
    if isinstance(arrays, list):
        array_lines = [f"  {item.get('role')}: {item.get('shape')} {item.get('dtype')}" for item in arrays if isinstance(item, Mapping)]
    lineage = value.get("lineage")
    lineage_line = "-"
    if isinstance(lineage, Mapping):
        lineage_line = f"source={lineage.get('source_run_id')} root={lineage.get('root_run_id')} depth={lineage.get('depth')}"
    termination = _mapping(value.get("termination"))
    timestamps = _mapping(value.get("timestamps"))
    lines = [
        f"Canonical run {value.get('run_id')}",
        f"Schema/status: {value.get('schema_version')} / {value.get('status')}",
        f"Task: {value.get('task_id')}",
        f"Execution: {value.get('execution_kind')}",
        f"Problem/algorithm/backend: {value.get('problem')} / {value.get('algorithm')} / {value.get('backend')}",
        f"Seed requested/resolved: {value.get('requested_seed')} / {value.get('resolved_seed')}",
        f"Population/budget/evaluations: {value.get('population_size')} / {value.get('evaluation_budget')} / {value.get('actual_evaluations')}",
        f"Termination: {termination.get('component')} ({termination.get('reason')})",
        f"Started/completed/duration ms: {timestamps.get('started_at')} / {timestamps.get('completed_at')} / {timestamps.get('duration_ms')}",
        f"Stored replayability: {value.get('replayability')}",
        f"Lineage: {lineage_line}",
        "Arrays (metadata only; values not loaded):",
        *(array_lines or ["  -"]),
        "Full artifact verification: not performed",
        f"Next: {value.get('recommended_next_command')}",
    ]
    return "\n".join(lines)


def _render_verification(report: Any) -> str:
    lines = [
        f"Verified canonical run {report.run_id}",
        f"Schema: {report.schema}",
        f"Artifact integrity: {report.artifact_integrity}",
        f"Path safety: {report.path_safety}",
        f"Numerical bundle safety: {report.numerical_bundle_safety}",
        f"Environment compatibility: {report.environment.level}",
        f"Component reconstructability: {report.component_reconstructability}",
        f"Effective replayability: {report.effective_replayability}",
        "Optimization executed: no",
    ]
    if report.reasons:
        lines.append("Blocking/limiting reasons:")
        lines.extend(f"  {reason.field}: {reason.message}" for reason in report.reasons)
    return "\n".join(lines)


def _render_replay(report: Any) -> str:
    lines = [
        f"Exact replay stored at: {report.output_root}",
        f"Source/replay run IDs: {report.source_run_id} / {report.replay_run_id}",
        f"Task: {report.task_id}",
        f"Replay plan SHA-256: {report.replay_plan_sha256}",
        f"Exact comparison: {'match' if report.exact else 'mismatch'}",
    ]
    lines.extend(
        f"  {item.role}: {'exact' if item.exact else item.mismatch} ({item.stored_sha256} / {item.replay_sha256})"
        for item in report.comparisons
    )
    return "\n".join(lines)


def _emit_error(exc: RunArtifactError, *, json_output: bool) -> None:
    if json_output:
        print(
            json.dumps(
                {"document_type": "vamos.run-command-error", "version": "1", "error": exc.as_dict()}, sort_keys=True, separators=(",", ":")
            )
        )
        return
    print(f"Error [{exc.category}]: {exc}", file=sys.stderr)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def run_artifact_exit_code(exc: RunArtifactError) -> int:
    """Exit code of a run-artifact error, shared by every command that reports one (8: output collision)."""
    if isinstance(exc, OutputCollisionError):
        return 8
    if isinstance(exc, ReplayExecutionError):
        return 7
    if isinstance(exc, ReplayUnavailableError):
        return 6
    if isinstance(exc, VerificationRequirementError):
        return 5
    if isinstance(
        exc,
        (ArtifactIntegrityError, ArtifactResourceLimitError, DuplicateJSONKeyError, MalformedResultBundleError, UnsafeArtifactPathError),
    ):
        return 3
    if isinstance(exc, (UnsupportedSchemaError, UnsupportedArtifactLayoutError, ManifestValidationError)):
        return 4
    return 3


__all__ = ["run_artifact_exit_code", "run_reproduce", "run_results"]
