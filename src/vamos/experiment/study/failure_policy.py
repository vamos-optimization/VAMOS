"""Durable study-level outcomes for task and infrastructure failures."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast

from vamos.experiment.artifacts.models import LoadLimits, deep_freeze

from .commits import append_event, checkpoint_manifest
from .errors import (
    StudyEventAppendError,
    StudyInfrastructureError,
    StudyRunPublicationError,
    StudyRunVerificationError,
)
from .execution_errors import finalization_error
from .limits import StudyLoadLimits
from .loading import Reloadable, reload_study
from .models import Study, StudyEvent, StudyManifest, StudyState, TaskRecord


class FailureExecutionState(Protocol):
    root: Path
    study_id: str
    execution_id: str
    manifest: StudyManifest
    tasks: list[TaskRecord]
    event: StudyEvent
    limits: StudyLoadLimits
    run_limits: LoadLimits


def pause_after_task_failure(state: FailureExecutionState, task: TaskRecord) -> Study:
    """Finish fail-fast by pausing after the task failure is fully durable."""
    reason = task.reason or _task_failure_fallback()
    failed_attempt = task.attempts[-1].attempt_id if task.attempts else None
    state.event = append_event(
        state.root,
        state.event,
        event_type="study_paused",
        entity_kind="study",
        entity_id=state.study_id,
        transition_from="running",
        transition_to="paused",
        execution_id=state.execution_id,
        reason=reason,
        payload={"failed_task_id": task.task_id, "failed_attempt_id": failed_attempt},
    )
    state.manifest = checkpoint_manifest(
        state.root,
        state.manifest,
        state="paused",
        execution_id=state.execution_id,
        tasks=tuple(state.tasks),
        event=state.event,
    )
    return reload_study(state)


def complete_running(
    state: FailureExecutionState,
    phase: Callable[[str], None],
    reload: Callable[[FailureExecutionState], Study],
) -> Study:
    """Finalize a fully traversed study according to its retained outcomes."""
    unfinished = [task for task in state.tasks if task.state in {"pending", "interrupted"}]
    if unfinished:
        return pause_with_runnable_work(state)
    failed_task_ids = [task.task_id for task in state.tasks if task.state == "failed"]
    completed_state: StudyState = "completed_with_failures" if failed_task_ids else "completed"
    event_type = "study_completed_with_failures" if failed_task_ids else "study_completed"
    reason = None
    payload = None
    if failed_task_ids:
        reason = {
            "category": "partial_completion",
            "code": "TASK_FAILURES_RETAINED",
            "message": "Study reached the end with durable task failures.",
            "retryable": False,
            "safe_action": "Inspect failed tasks and their verified failed runs.",
        }
        payload = {"failed_task_ids": failed_task_ids}
    try:
        phase("before_final_completed_event")
        state.event = append_event(
            state.root,
            state.event,
            event_type=event_type,
            entity_kind="study",
            entity_id=state.study_id,
            transition_from="running",
            transition_to=completed_state,
            execution_id=state.execution_id,
            reason=reason,
            payload=payload,
        )
        state.manifest = checkpoint_manifest(
            state.root,
            state.manifest,
            state=completed_state,
            execution_id=state.execution_id,
            tasks=tuple(state.tasks),
            event=state.event,
        )
    except StudyEventAppendError:
        raise
    except Exception as exc:
        raise finalization_error(state.study_id, "running", True, True, state.root, exc) from exc
    return reload(state)


def pause_with_runnable_work(state: FailureExecutionState) -> Study:
    """Pause an execution that deliberately leaves canonical work unclaimed."""
    pending = [task.task_id for task in state.tasks if task.state == "pending"]
    interrupted = [task.task_id for task in state.tasks if task.state == "interrupted"]
    reason = {
        "category": "recovery",
        "code": "RUNNABLE_WORK_REMAINS",
        "message": "The explicit operation ended with unfinished tasks still available.",
        "retryable": True,
        "safe_action": "Resume pending/interrupted work or explicitly retry failed work.",
    }
    state.event = append_event(
        state.root,
        state.event,
        event_type="study_paused",
        entity_kind="study",
        entity_id=state.study_id,
        transition_from="running",
        transition_to="paused",
        execution_id=state.execution_id,
        reason=reason,
        payload={"pending_task_ids": pending, "interrupted_task_ids": interrupted},
    )
    state.manifest = checkpoint_manifest(
        state.root,
        state.manifest,
        state="paused",
        execution_id=state.execution_id,
        tasks=tuple(state.tasks),
        event=state.event,
    )
    return reload_study(state)


def record_infrastructure_failure(source: Reloadable, error: StudyInfrastructureError) -> StudyInfrastructureError:
    """Record a trustworthy root failure when its authority remains writable."""
    if not _may_publish_failure(error):
        return error
    try:
        current = reload_study(source)
    except Exception:
        return error
    if current.status != "running":
        return error
    reason = _infrastructure_reason(error)
    try:
        event = append_event(
            current.root,
            current.events[-1],
            event_type="study_failed",
            entity_kind="study",
            entity_id=current.study_id,
            transition_from="running",
            transition_to="failed",
            execution_id=current.manifest.execution_id,
            reason=reason,
        )
        checkpoint_manifest(
            current.root,
            current.manifest,
            state="failed",
            execution_id=current.manifest.execution_id,
            tasks=current.tasks,
            event=event,
        )
    except Exception:
        return error
    return StudyInfrastructureError(
        operation="run study",
        reason="STUDY_INFRASTRUCTURE_FAILURE",
        study_id=current.study_id,
        task_id=error.task_id,
        attempt_id=error.attempt_id,
        current_state="failed",
        expected_state="completed",
        objective_evaluation_began=error.objective_evaluation_began,
        canonical_run_published=error.canonical_run_published,
        expected="trustworthy durable study execution",
        actual={"category": error.category, "reason": error.reason},
        action="Load the study to inspect the durable infrastructure failure; no task failure was fabricated.",
    )


def _may_publish_failure(error: StudyInfrastructureError) -> bool:
    if isinstance(error, (StudyEventAppendError, StudyRunPublicationError, StudyRunVerificationError)):
        return False
    return error.reason not in {
        "CHECKPOINT_WRITE_FAILED",
        "EVENT_APPEND_COLLISION",
        "EVENT_APPEND_FAILED",
        "RUN_PUBLICATION_FAILED",
        "RUN_PUBLICATION_INTERRUPTED",
        "RUN_VERIFICATION_FAILED",
    }


def _infrastructure_reason(error: StudyInfrastructureError) -> Mapping[str, Any]:
    return cast(
        Mapping[str, Any],
        deep_freeze(
            {
                "category": "infrastructure",
                "code": "STUDY_INFRASTRUCTURE_FAILURE",
                "message": "Study infrastructure failed; no scientific task outcome was inferred.",
                "retryable": False,
                "safe_action": "Restore trustworthy storage and inspect the durable study state.",
                "source_category": error.category,
                "source_reason": error.reason,
            }
        ),
    )


def _task_failure_fallback() -> Mapping[str, Any]:
    return cast(
        Mapping[str, Any],
        deep_freeze(
            {
                "category": "execution",
                "code": "TASK_EXECUTION_FAILED",
                "message": "A durable task failure paused fail-fast execution.",
                "retryable": False,
                "safe_action": "Inspect the failed task and its verified failed run.",
            }
        ),
    )


__all__ = [
    "FailureExecutionState",
    "complete_running",
    "pause_after_task_failure",
    "pause_with_runnable_work",
    "record_infrastructure_failure",
]
