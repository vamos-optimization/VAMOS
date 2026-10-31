"""Bounded loading and validation of task, attempt, and run references."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn, Protocol

from vamos.experiment.artifacts.errors import RunArtifactError
from vamos.experiment.artifacts.jsonio import sha256_bytes
from vamos.experiment.artifacts.models import LoadLimits, StoredRun
from vamos.experiment.artifacts.persistence import load_run
from vamos.experiment.artifacts.reader import verify_artifact

from .errors import (
    MalformedStudyError,
    ReferencedRunCorruptError,
    ReferencedRunMissingError,
    StudyError,
    StudyIntegrityError,
    UnsafeStudyPathError,
)
from .limits import StudyLoadLimits
from .models import AttemptRecord, AttemptReference, PlanTask, StudyManifest, TaskRecord
from .paths import confined_study_path
from .record_decoding import decode_attempt, decode_task
from .writing import is_writer_temporary

_ATTEMPT_FILE_PATTERN = r"([0-9a-f-]{36})\.json"
_RUN_ID_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
_KNOWN_RUN_ARTIFACT_ROLES = ("environment", "result_bundle", "metrics", "events")


class ReadDocument(Protocol):
    def __call__(self, relative: str, role: str, max_bytes: int) -> tuple[dict[str, Any], bytes]: ...


RunVerification = Literal["all", "metadata"]


class ObserveRunReference(Protocol):
    def __call__(
        self,
        attempt_id: str,
        reference: Mapping[str, Any],
        stored: StoredRun | None,
        error: StudyError | None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _LoadedAttempt:
    record: AttemptRecord
    path: str
    payload: bytes


def load_records(
    root: Path,
    manifest: StudyManifest,
    plan_tasks: tuple[PlanTask, ...],
    limits: StudyLoadLimits,
    read: ReadDocument,
    *,
    run_verification: RunVerification = "all",
    tolerate_run_errors: bool = False,
    observe_run_reference: ObserveRunReference | None = None,
) -> tuple[tuple[TaskRecord, ...], tuple[AttemptRecord, ...]]:
    """Load every planned checkpoint and bounded attempt document."""
    if not plan_tasks:
        _assert_no_task_entries(root)
        return (), ()
    tasks: list[TaskRecord] = []
    attempts: list[AttemptRecord] = []
    expected_dirs: set[str] = set()
    for plan_task in plan_tasks:
        digest = plan_task.task_id.removeprefix("sha256:")
        expected_dirs.add(digest)
        relative = f"tasks/{digest}/task.json"
        raw, _ = read(relative, "study_task", limits.max_task_bytes)
        task = decode_task(raw)
        _validate_task_identity(task, manifest, plan_task, relative)
        loaded = _load_attempts(
            root,
            task,
            limits,
            read,
            run_verification=run_verification,
            tolerate_run_errors=tolerate_run_errors,
            observe_run_reference=observe_run_reference,
        )
        _validate_attempt_references(task, loaded)
        tasks.append(task)
        attempts.extend(item.record for item in loaded)
    _assert_task_entries(root, expected_dirs)
    return tuple(tasks), tuple(attempts)


def validate_run_reference(
    root: Path,
    reference: Mapping[str, Any],
    *,
    expected_task_id: str,
    expected_status: str | None = None,
    verification: RunVerification = "all",
) -> StoredRun:
    """Validate an approved bounded reference and its complete canonical run."""
    run_id = reference.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch(_RUN_ID_PATTERN, run_id) is None:
        _unexpected("run_reference", "lowercase UUIDv4 run_id", run_id, reason="RUN_REFERENCE_MISMATCH")
    expected_path = f"runs/{run_id}/manifest.json"
    expected_contract = {
        "path": expected_path,
        "role": "run_manifest",
        "required_for": ("inspect", "verify", "resume"),
        "task_id": expected_task_id,
    }
    actual_contract = {
        "path": reference.get("path"),
        "role": reference.get("role"),
        "required_for": tuple(reference.get("required_for", ())),
        "task_id": reference.get("task_id"),
    }
    if actual_contract != expected_contract:
        _unexpected("run_reference", expected_contract, actual_contract, reason="RUN_REFERENCE_MISMATCH")
    if not os.path.lexists(root / expected_path):
        raise ReferencedRunMissingError(
            operation="load study",
            reason="REFERENCED_RUN_MISSING",
            document_role="run_manifest",
            path=expected_path,
            entity_id=run_id,
            expected="complete referenced canonical RunManifest",
            actual="missing manifest",
            action="Restore the exact referenced run; resume never replaces lost successful evidence.",
        )
    try:
        manifest_path = confined_study_path(root, expected_path, role="run_manifest", must_exist=True)
    except UnsafeStudyPathError as exc:
        raise _run_corrupt(expected_path, run_id, exc, verification=verification) from exc
    try:
        payload = manifest_path.read_bytes()
    except OSError as exc:
        raise MalformedStudyError(
            operation="load study",
            reason="UNREADABLE_DOCUMENT",
            document_role="run_manifest",
            path=expected_path,
            expected="readable canonical RunManifest",
            actual=type(exc).__name__,
            action="Restore the referenced canonical run directory.",
        ) from exc
    try:
        run_limits = LoadLimits()
        stored = load_run(
            manifest_path.parent,
            verify="all" if verification == "all" else "manifest",
            limits=run_limits,
        )
        if verification == "metadata":
            for descriptor in stored.manifest.artifacts:
                if descriptor.role not in _KNOWN_RUN_ARTIFACT_ROLES:
                    continue
                verify_artifact(
                    stored.root,
                    descriptor,
                    limits=run_limits,
                    operation="inspect study run reference",
                )
    except RunArtifactError as exc:
        raise _run_corrupt(expected_path, run_id, exc, verification=verification) from exc
    semantic = stored.manifest.get("integrity")
    semantic_hash = semantic.get("manifest_sha256") if isinstance(semantic, Mapping) else None
    observed = {
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "semantic_sha256": semantic_hash,
        "run_id": stored.manifest.run_id,
        "task_id": stored.manifest.task_id,
    }
    expected = {name: reference.get(name) for name in observed}
    if observed != expected:
        raise ReferencedRunCorruptError(
            operation="load study",
            reason="RUN_MANIFEST_HASH_MISMATCH",
            document_role="run_reference",
            path=expected_path,
            entity_id=run_id,
            expected=expected,
            actual=observed,
            action="Restore the exact referenced canonical run; resume never substitutes evidence.",
        )
    if stored.manifest.task_id != expected_task_id:
        raise ReferencedRunCorruptError(
            operation="load study",
            reason="RUN_TASK_ID_MISMATCH",
            document_role="run_reference",
            path=expected_path,
            entity_id=run_id,
            expected=expected_task_id,
            actual=stored.manifest.task_id,
            action="Restore the run belonging to this exact immutable task.",
        )
    if expected_status is not None and stored.status != expected_status:
        raise ReferencedRunCorruptError(
            operation="load study",
            reason="RUN_STATUS_MISMATCH",
            document_role="run_reference",
            path=expected_path,
            entity_id=run_id,
            expected=expected_status,
            actual=stored.status,
            action="Restore the terminal run matching the recorded attempt outcome.",
        )
    return stored


def _run_corrupt(
    path: str,
    run_id: str,
    exc: Exception,
    *,
    verification: RunVerification = "all",
) -> ReferencedRunCorruptError:
    expected = "fully verified" if verification == "all" else "metadata-verified"
    return ReferencedRunCorruptError(
        operation="load study",
        reason="REFERENCED_RUN_CORRUPT",
        document_role="run_manifest",
        path=path,
        entity_id=run_id,
        expected=f"{expected} referenced canonical RunManifest",
        actual=type(exc).__name__,
        action="Restore the exact referenced run; corrupt evidence is never converted to pending work.",
    )


def _load_attempts(
    root: Path,
    task: TaskRecord,
    limits: StudyLoadLimits,
    read: ReadDocument,
    *,
    run_verification: RunVerification,
    tolerate_run_errors: bool,
    observe_run_reference: ObserveRunReference | None,
) -> tuple[_LoadedAttempt, ...]:
    digest = task.task_id.removeprefix("sha256:")
    directory = f"tasks/{digest}/attempts"
    if not os.path.lexists(root / Path(directory)):
        return ()
    names = _entry_names(root, directory, "study_attempt")
    result: list[_LoadedAttempt] = []
    for name in sorted(names):
        match = re.fullmatch(_ATTEMPT_FILE_PATTERN, name)
        if match is None:
            _unexpected(directory, "lowercase UUID.json attempt files", name)
        relative = f"{directory}/{name}"
        raw, payload = read(relative, "study_attempt", limits.max_attempt_bytes)
        attempt = decode_attempt(raw)
        if attempt.attempt_id != match.group(1) or (attempt.study_id, attempt.task_id) != (task.study_id, task.task_id):
            _unexpected(
                "study_attempt",
                {"study_id": task.study_id, "task_id": task.task_id, "attempt_id": match.group(1)},
                {"study_id": attempt.study_id, "task_id": attempt.task_id, "attempt_id": attempt.attempt_id},
                reason="ATTEMPT_ID_MISMATCH",
            )
        if attempt.run_reference is not None:
            expected_status = attempt.status if attempt.status in {"succeeded", "failed"} else None
            _validate_observed_run(
                root,
                attempt,
                expected_status=expected_status,
                verification=run_verification,
                tolerate_errors=tolerate_run_errors,
                observer=observe_run_reference,
            )
        result.append(_LoadedAttempt(attempt, relative, payload))
    result.sort(key=lambda item: item.record.attempt_number)
    numbers = [item.record.attempt_number for item in result]
    if numbers != list(range(1, len(result) + 1)) or len({item.record.attempt_id for item in result}) != len(result):
        _unexpected("study_attempt", list(range(1, len(result) + 1)), numbers, reason="ATTEMPT_SEQUENCE_MISMATCH")
    return tuple(result)


def validate_observed_run(
    root: Path,
    attempt_id: str,
    reference: Mapping[str, Any],
    *,
    expected_task_id: str,
    expected_status: str | None,
    verification: RunVerification,
    tolerate_errors: bool,
    observer: ObserveRunReference | None,
) -> StoredRun | None:
    """Validate one run reference and optionally report a typed issue."""
    try:
        stored = validate_run_reference(
            root,
            reference,
            expected_task_id=expected_task_id,
            expected_status=expected_status,
            verification=verification,
        )
    except StudyError as exc:
        if observer is not None:
            observer(attempt_id, reference, None, exc)
        if not tolerate_errors:
            raise
        return None
    if observer is not None:
        observer(attempt_id, reference, stored, None)
    return stored


def _validate_observed_run(
    root: Path,
    attempt: AttemptRecord,
    *,
    expected_status: str | None,
    verification: RunVerification,
    tolerate_errors: bool,
    observer: ObserveRunReference | None,
) -> None:
    reference = attempt.run_reference
    if reference is None:
        return
    validate_observed_run(
        root,
        attempt.attempt_id,
        reference,
        expected_task_id=attempt.task_id,
        expected_status=expected_status,
        verification=verification,
        tolerate_errors=tolerate_errors,
        observer=observer,
    )


def _validate_task_identity(task: TaskRecord, manifest: StudyManifest, plan_task: PlanTask, relative: str) -> None:
    expected = (manifest.study_id, plan_task.task_id, plan_task.plan_index)
    actual = (task.study_id, task.task_id, task.plan_index)
    if actual != expected:
        raise StudyIntegrityError(
            operation="load study",
            reason="TASK_ID_MISMATCH",
            document_role="study_task",
            path=relative,
            expected=expected,
            actual=actual,
            action="Restore the task checkpoint matching the immutable plan.",
        )


def _validate_attempt_references(task: TaskRecord, attempts: tuple[_LoadedAttempt, ...]) -> None:
    by_id = {item.record.attempt_id: item for item in attempts}
    for index, reference in enumerate(task.attempts):
        loaded = by_id.get(reference.attempt_id)
        if loaded is None:
            _unexpected("study_task", "referenced attempt file", reference.attempt_id, reason="MISSING_ATTEMPT")
        assert loaded is not None
        expected = _attempt_reference(loaded)
        identity_matches = (
            reference.attempt_id,
            reference.attempt_number,
            reference.path,
            reference.role,
            reference.required_for,
        ) == (
            expected.attempt_id,
            expected.attempt_number,
            expected.path,
            expected.role,
            expected.required_for,
        )
        lagging_terminal_write = task.state == "running" and loaded.record.status in {"succeeded", "failed"}
        if not identity_matches or reference.attempt_number != index + 1 or (reference != expected and not lagging_terminal_write):
            _unexpected("study_task", expected, reference, reason="ATTEMPT_REFERENCE_MISMATCH")
    referenced = {item.attempt_id for item in task.attempts}
    unreferenced = [item.record for item in attempts if item.record.attempt_id not in referenced]
    if any(attempt.status not in {"created", "running", "succeeded", "failed"} for attempt in unreferenced):
        _unexpected("study_task", "checkpoint-lag-compatible attempts", unreferenced, reason="ATTEMPT_REFERENCE_MISMATCH")


def _attempt_reference(loaded: _LoadedAttempt) -> AttemptReference:
    attempt = loaded.record
    return AttemptReference(
        attempt_id=attempt.attempt_id,
        attempt_number=attempt.attempt_number,
        path=loaded.path,
        role="attempt",
        required_for=("inspect", "resume"),
        semantic_sha256=attempt.document_sha256,
        sha256=sha256_bytes(loaded.payload),
        bytes=len(loaded.payload),
    )


def _assert_no_task_entries(root: Path) -> None:
    if (root / "tasks").exists() and _entry_names(root, "tasks", "tasks"):
        _unexpected("tasks", "no task entries for an empty plan", "non-empty tasks directory")


def _assert_task_entries(root: Path, expected: set[str]) -> None:
    observed = _entry_names(root, "tasks", "tasks")
    if observed != expected:
        _unexpected("tasks", sorted(expected), sorted(observed))
    for digest in expected:
        entries = _entry_names(root, f"tasks/{digest}", "study_task")
        allowed = {"task.json"} if "attempts" not in entries else {"task.json", "attempts"}
        if entries != allowed:
            _unexpected(f"tasks/{digest}", sorted(allowed), sorted(entries))


def _entry_names(root: Path, relative: str, role: str) -> set[str]:
    directory = confined_study_path(root, relative, role=role, must_exist=True)
    try:
        if not directory.is_dir():
            raise OSError("not a directory")
        # A writer temporary is in flight or orphaned by a dead writer, never a record.
        return {entry.name for entry in directory.iterdir() if not is_writer_temporary(entry.name)}
    except OSError as exc:
        raise MalformedStudyError(
            operation="load study",
            reason="UNREADABLE_DOCUMENT_DIRECTORY",
            document_role=role,
            path=relative,
            expected="readable contained directory",
            actual=type(exc).__name__,
            action="Restore the canonical directory and its permissions.",
        ) from exc


def _unexpected(role: str, expected: object, actual: object, *, reason: str = "UNEXPECTED_LAYOUT") -> NoReturn:
    raise StudyIntegrityError(
        operation="load study",
        reason=reason,
        document_role=role,
        expected=expected,
        actual=actual,
        action="Restore the complete canonical study tree; do not infer state from unrelated files.",
    )


__all__ = [
    "ObserveRunReference",
    "ReadDocument",
    "RunVerification",
    "load_records",
    "validate_observed_run",
    "validate_run_reference",
]
