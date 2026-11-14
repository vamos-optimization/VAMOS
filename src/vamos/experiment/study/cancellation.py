"""Single-process durable cancellation for canonical studies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from vamos.experiment.artifacts.models import deep_freeze

from .commits import append_event, checkpoint_attempt, checkpoint_manifest, checkpoint_task, commit_section
from .errors import UnsupportedStudyExecutionStateError
from .identity import new_uuid4
from .loading import load_study
from .models import AttemptRecord, AttemptReference, Study, TaskRecord


def cancel_snapshot(snapshot: Study, active_roots: Mapping[Path, object], requests: dict[Path, None]) -> Study:
    """Cancel now when idle, or request cancellation from this process's runner."""
    root = snapshot.root.resolve(strict=True)
    current = load_study(root)
    _validate_identity(snapshot, current)
    if root in active_roots:
        requests[root] = None
        return current
    if current.status not in {"created", "paused"}:
        raise UnsupportedStudyExecutionStateError(
            operation="cancel study",
            reason="UNSUPPORTED_STUDY_CANCELLATION_STATE",
            study_id=current.study_id,
            current_state=current.status,
            expected_state="created, paused, or running in this process",
            expected="nonterminal cancellable study",
            actual=current.status,
            action="Inspect the study; completed, failed, and cancelled studies are terminal.",
        )
    return cancel_loaded(current, code="USER_CANCELLATION")


def cancel_loaded(study: Study, *, code: str) -> Study:
    """Commit terminal cancellation from a verified nonterminal snapshot."""
    if study.status not in {"created", "running", "paused"}:
        raise UnsupportedStudyExecutionStateError(
            operation="cancel study",
            reason="UNSUPPORTED_STUDY_CANCELLATION_STATE",
            study_id=study.study_id,
            current_state=study.status,
            expected_state="created, running, or paused",
            expected="nonterminal cancellable study",
            actual=study.status,
            action="Inspect the study; no cancellation transition was written.",
        )
    execution_id = study.manifest.execution_id or new_uuid4()
    reason = _cancellation_reason(code)
    tasks = list(study.tasks)
    event = study.events[-1]
    manifest = study.manifest
    attempts_by_task = {item.task_id: item for item in study.attempts if item.status in {"created", "running"}}
    for index, task in enumerate(tasks):
        attempt = attempts_by_task.get(task.task_id)
        if attempt is None:
            continue
        with commit_section():
            event, task = _cancel_attempt(study.root, event, task, attempt, execution_id, reason)
            tasks[index] = task
            manifest = checkpoint_manifest(
                study.root,
                manifest,
                state=study.status,
                execution_id=execution_id,
                tasks=tuple(tasks),
                event=event,
            )
    pending_ids = [task.task_id for task in tasks if task.state == "pending"]
    with commit_section():
        event = append_event(
            study.root,
            event,
            event_type="study_cancelled",
            entity_kind="study",
            entity_id=study.study_id,
            transition_from=study.status,
            transition_to="cancelled",
            execution_id=execution_id,
            reason=reason,
            payload={"cancelled_task_ids": pending_ids},
        )
        tasks = [
            checkpoint_task(study.root, replace(task, state="cancelled", reason=reason)) if task.state == "pending" else task
            for task in tasks
        ]
        checkpoint_manifest(
            study.root,
            manifest,
            state="cancelled",
            execution_id=execution_id,
            tasks=tuple(tasks),
            event=event,
        )
    return load_study(study.root)


def _cancel_attempt(
    root: Path,
    previous: Any,
    task: TaskRecord,
    attempt: AttemptRecord,
    execution_id: str,
    reason: Mapping[str, Any],
) -> tuple[Any, TaskRecord]:
    event = append_event(
        root,
        previous,
        event_type="attempt_cancelled",
        entity_kind="attempt",
        entity_id=attempt.attempt_id,
        transition_from=attempt.status,
        transition_to="cancelled",
        execution_id=execution_id,
        reason=reason,
    )
    timestamps = dict(attempt.timestamps)
    timestamps["completed_at"] = event.timestamp
    attempt = replace(attempt, status="cancelled", timestamps=deep_freeze(timestamps), failure=None, run_reference=None)
    attempt, reference = checkpoint_attempt(root, attempt)
    references = _replace_or_append_reference(task.attempts, reference)
    task = replace(
        task,
        state="cancelled",
        attempts=references,
        current_attempt_id=None,
        selected_success_attempt_id=None,
        retryability=replace(task.retryability, retryable=False, category="cancellation"),
        reason=reason,
    )
    return event, checkpoint_task(root, task)


def _replace_or_append_reference(references: tuple[AttemptReference, ...], replacement: AttemptReference) -> tuple[AttemptReference, ...]:
    if not any(item.attempt_id == replacement.attempt_id for item in references):
        return (*references, replacement)
    return tuple(replacement if item.attempt_id == replacement.attempt_id else item for item in references)


def _validate_identity(snapshot: Study, current: Study) -> None:
    if (snapshot.study_id, snapshot.plan_id) == (current.study_id, current.plan_id):
        return
    raise UnsupportedStudyExecutionStateError(
        operation="cancel study",
        reason="STUDY_IDENTITY_CHANGED",
        study_id=snapshot.study_id,
        current_state=current.status,
        expected_state=snapshot.status,
        expected={"study_id": snapshot.study_id, "plan_id": snapshot.plan_id},
        actual={"study_id": current.study_id, "plan_id": current.plan_id},
        action="Discard the stale handle and inspect the canonical study root.",
    )


def _cancellation_reason(code: str) -> Mapping[str, Any]:
    message = "Study cancellation was requested in the current process."
    if code == "PROCESS_INTERRUPTION":
        message = "Study execution received a graceful process interruption."
    return cast(
        Mapping[str, Any],
        deep_freeze(
            {
                "category": "cancellation",
                "code": code,
                "message": message,
                "retryable": False,
                "safe_action": "Load the study to inspect the durable cancelled state.",
            }
        ),
    )


__all__ = ["cancel_loaded", "cancel_snapshot"]
