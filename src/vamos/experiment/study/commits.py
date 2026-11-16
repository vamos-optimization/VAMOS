"""Atomic event append and mutable checkpoint commits for study execution."""

from __future__ import annotations

import os
import signal
import threading
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any, cast

from vamos.experiment.artifacts.jsonio import sha256_bytes
from vamos.experiment.artifacts.models import StoredRun, deep_freeze

from .decoding import decode_manifest
from .documents import attempt_document, event_document, manifest_checkpoint_document, task_checkpoint_document
from .errors import StudyEventAppendError, StudyInfrastructureError
from .identity import new_uuid4
from .models import (
    AttemptRecord,
    AttemptReference,
    StudyCounts,
    StudyEvent,
    StudyManifest,
    StudyState,
    TaskRecord,
)
from .record_decoding import decode_attempt, decode_event, decode_task
from .writing import write_document_atomic

_SignalHandler = Callable[[int, FrameType | None], Any]
_HELD_SIGNALS = (signal.SIGINT, signal.SIGTERM)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class _HeldSignals:
    """Record SIGINT/SIGTERM while active, then run the original handlers."""

    def __init__(self) -> None:
        self.active = True
        self.received: list[int] = []
        self.originals: dict[int, _SignalHandler] = {}

    def handle(self, signum: int, frame: FrameType | None) -> None:
        if self.active:
            self.received.append(signum)
        else:  # A wrapper left installed by an interrupted release only forwards.
            self.originals[signum](signum, frame)

    def hold(self) -> None:
        for signum in _HELD_SIGNALS:
            current = signal.getsignal(signum)
            if callable(current):  # Default and ignored dispositions are not Python handlers.
                self.originals[signum] = current
                signal.signal(signum, self.handle)

    def release(self) -> None:
        self.active = False
        for signum, original in self.originals.items():
            signal.signal(signum, original)
        for signum in dict.fromkeys(self.received):
            self.originals[signum](signum, None)


@contextmanager
def commit_section() -> Iterator[None]:
    """Write one journal event and the checkpoints that record it without a stop in between.

    Ctrl+C and the Studio worker's stop reach the owning thread as Python signal
    handlers that raise. A handler that runs after an event is published and before
    its checkpoints are written would leave them behind the journal. Inside this
    section such handlers only record the signal; they run, once each, as soon as
    the section ends. Only the main thread receives Python signals.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    held = _HeldSignals()
    try:
        held.hold()
        yield
    finally:
        held.release()


def append_event(
    root: Path,
    previous: StudyEvent,
    *,
    event_type: str,
    entity_kind: str,
    entity_id: str,
    transition_from: str | None,
    transition_to: str,
    execution_id: str | None,
    timestamp: str | None = None,
    reason: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
) -> StudyEvent:
    """Append one immutable event without rewriting existing journal entries."""
    sequence = previous.sequence + 1
    relative = f"events/{sequence:020d}.json"
    target = root / relative
    if os.path.lexists(target):
        raise StudyEventAppendError(
            operation="append study event",
            reason="EVENT_APPEND_COLLISION",
            path=relative,
            expected="absent next event path",
            actual="occupied path",
            action="Stop execution and inspect the journal; no existing event is overwritten.",
        )
    document = event_document(
        sequence=sequence,
        event_id=new_uuid4(),
        event_type=event_type,
        entity_kind=entity_kind,
        entity_id=entity_id,
        transition_from=transition_from,
        transition_to=transition_to,
        execution_id=execution_id,
        timestamp=timestamp or now_utc(),
        reason=reason,
        payload=payload,
        previous_event_sha256=previous.file_sha256,
    )
    try:
        stored = write_document_atomic(target, document)
    except Exception as exc:
        raise StudyEventAppendError(
            operation="append study event",
            reason="EVENT_APPEND_FAILED",
            path=relative,
            expected="one atomically published immutable event",
            actual=type(exc).__name__,
            action="Stop execution and restore writable storage; the journal is never silently repaired.",
        ) from exc
    return decode_event(document, file_sha256=sha256_bytes(stored))


def checkpoint_attempt(root: Path, attempt: AttemptRecord) -> tuple[AttemptRecord, AttemptReference]:
    """Atomically write an attempt and return its exact task-level descriptor."""
    relative = _attempt_path(attempt)
    document = attempt_document(attempt)
    try:
        payload = write_document_atomic(root / relative, document)
    except Exception as exc:
        raise _checkpoint_failure("study_attempt", relative, exc) from exc
    decoded = decode_attempt(document)
    reference = AttemptReference(
        attempt_id=decoded.attempt_id,
        attempt_number=decoded.attempt_number,
        path=relative,
        role="attempt",
        required_for=("inspect", "resume"),
        semantic_sha256=decoded.document_sha256,
        sha256=sha256_bytes(payload),
        bytes=len(payload),
    )
    return decoded, reference


def checkpoint_task(root: Path, task: TaskRecord) -> TaskRecord:
    relative = f"tasks/{task.task_id.removeprefix('sha256:')}/task.json"
    document = task_checkpoint_document(task)
    try:
        write_document_atomic(root / relative, document)
    except Exception as exc:
        raise _checkpoint_failure("study_task", relative, exc) from exc
    return decode_task(document)


def checkpoint_manifest(
    root: Path,
    manifest: StudyManifest,
    *,
    state: StudyState,
    execution_id: str | None,
    tasks: tuple[TaskRecord, ...],
    event: StudyEvent,
) -> StudyManifest:
    updated = replace(
        manifest,
        state=state,
        updated_at=event.timestamp,
        execution_id=execution_id,
        counts=counts_for_tasks(tasks),
        checkpoint_sequence=event.sequence,
        checkpoint_event_sha256=event.file_sha256,
    )
    document = manifest_checkpoint_document(updated)
    try:
        write_document_atomic(root / "study-manifest.json", document)
    except Exception as exc:
        raise _checkpoint_failure("study_manifest", "study-manifest.json", exc) from exc
    return decode_manifest(document)


def run_reference(stored: StoredRun, *, study_root: Path) -> Mapping[str, Any]:
    manifest_path = stored.root / "manifest.json"
    payload = manifest_path.read_bytes()
    relative = manifest_path.relative_to(study_root).as_posix()
    integrity = stored.manifest.get("integrity")
    semantic = integrity.get("manifest_sha256") if isinstance(integrity, Mapping) else None
    if not isinstance(semantic, str):
        raise StudyInfrastructureError(
            operation="reference canonical run",
            reason="RUN_MANIFEST_HASH_MISSING",
            path=relative,
            expected="terminal manifest semantic hash",
            actual=semantic,
            action="Stop execution and inspect the unreferenced run directory.",
        )
    return cast(
        Mapping[str, Any],
        deep_freeze(
            {
                "path": relative,
                "role": "run_manifest",
                "required_for": ["inspect", "verify", "resume"],
                "semantic_sha256": semantic,
                "sha256": sha256_bytes(payload),
                "bytes": len(payload),
                "run_id": stored.manifest.run_id,
                "task_id": stored.manifest.task_id,
            }
        ),
    )


def counts_for_tasks(tasks: tuple[TaskRecord, ...]) -> StudyCounts:
    counts = Counter(task.state for task in tasks)
    return StudyCounts(
        tasks=len(tasks),
        pending=counts["pending"],
        running=counts["running"],
        succeeded=counts["succeeded"],
        failed=counts["failed"],
        interrupted=counts["interrupted"],
        cancelled=counts["cancelled"],
        skipped=counts["skipped"],
    )


def _attempt_path(attempt: AttemptRecord) -> str:
    digest = attempt.task_id.removeprefix("sha256:")
    return f"tasks/{digest}/attempts/{attempt.attempt_id}.json"


def _checkpoint_failure(role: str, path: str, exc: Exception) -> StudyInfrastructureError:
    return StudyInfrastructureError(
        operation="write study checkpoint",
        reason="CHECKPOINT_WRITE_FAILED",
        document_role=role,
        path=path,
        expected="atomically persisted canonical checkpoint",
        actual=type(exc).__name__,
        action="Stop execution and inspect the authoritative journal; loading never repairs checkpoints.",
    )


__all__ = [
    "append_event",
    "checkpoint_attempt",
    "checkpoint_manifest",
    "checkpoint_task",
    "commit_section",
    "counts_for_tasks",
    "now_utc",
    "run_reference",
]
