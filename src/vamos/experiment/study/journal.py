"""Authoritative event-journal validation and data-only state derivation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn, cast

from vamos.experiment.artifacts.models import LoadLimits, deep_freeze

from .checkpoint_projection import EffectiveStudyState, project_effective_study, validate_pristine_created_state
from .errors import StudyIntegrityError
from .models import (
    AttemptRecord,
    AttemptState,
    StudyEvent,
    StudyManifest,
    StudyState,
    TaskRecord,
    TaskState,
)
from .record_loading import ObserveRunReference, RunVerification, validate_observed_run

_UUID4_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
_ATTEMPT_TERMINALS = {"succeeded", "failed", "interrupted", "cancelled"}
_TASK_TERMINALS = {"succeeded", "failed", "interrupted", "cancelled", "skipped"}


@dataclass(slots=True)
class _ReplayState:
    study_state: StudyState
    execution_id: str | None
    updated_at: str
    task_states: dict[str, TaskState]
    attempt_states: dict[str, AttemptState]
    attempts: dict[str, AttemptRecord]


def derive_effective_study(
    root: Path,
    manifest: StudyManifest,
    tasks: tuple[TaskRecord, ...],
    attempts: tuple[AttemptRecord, ...],
    events: tuple[StudyEvent, ...],
    *,
    run_verification: RunVerification = "all",
    run_limits: LoadLimits | None = None,
    tolerate_run_errors: bool = False,
    observe_run_reference: ObserveRunReference | None = None,
) -> EffectiveStudyState:
    """Return an immutable effective view while leaving lagging files untouched."""
    if not events:
        _integrity("EVENT_HASH_CHAIN_BROKEN", "events", "initial study_created event", "empty journal")
    state = _initial_state(manifest, tasks, attempts, events[0])
    if len(events) == 1:
        validate_pristine_created_state(manifest, tasks, attempts)
    checkpoint_state: _ReplayState | None = None
    for event in events[1:]:
        _apply_event(
            root,
            state,
            event,
            manifest.study_id,
            run_verification=run_verification,
            run_limits=run_limits,
            tolerate_run_errors=tolerate_run_errors,
            observe_run_reference=observe_run_reference,
        )
        if event.sequence == manifest.checkpoint_sequence:
            checkpoint_state = _copy_state(state)
    if manifest.checkpoint_sequence == 1:
        checkpoint_state = _copy_state(state)
    assert checkpoint_state is not None
    return project_effective_study(root, manifest, tasks, attempts, events, state, checkpoint_state)


def _initial_state(
    manifest: StudyManifest,
    tasks: tuple[TaskRecord, ...],
    attempts: tuple[AttemptRecord, ...],
    event: StudyEvent,
) -> _ReplayState:
    valid = (
        event.sequence == 1
        and event.event_type == "study_created"
        and event.entity_kind == "study"
        and event.entity_id == manifest.study_id
        and event.transition_from is None
        and event.transition_to == "created"
        and event.execution_id is None
        and event.timestamp == manifest.created_at
        and event.reason is None
        and not event.payload
    )
    if not valid:
        _integrity("INCONSISTENT_INITIAL_STATE", "events/00000000000000000001.json", "canonical study_created", event)
    return _ReplayState(
        study_state="created",
        execution_id=None,
        updated_at=event.timestamp,
        task_states={task.task_id: "pending" for task in tasks},
        attempt_states={attempt.attempt_id: "created" for attempt in attempts},
        attempts={attempt.attempt_id: attempt for attempt in attempts},
    )


def _apply_event(
    root: Path,
    state: _ReplayState,
    event: StudyEvent,
    study_id: str,
    *,
    run_verification: RunVerification,
    run_limits: LoadLimits | None,
    tolerate_run_errors: bool,
    observe_run_reference: ObserveRunReference | None,
) -> None:
    if event.event_type == "study_created":
        _integrity("INVALID_STATE_TRANSITION", "study_event", "study_created only at sequence 1", event.sequence)
    if event.entity_kind == "study":
        _apply_study_event(state, event, study_id)
    elif event.entity_kind == "task":
        _apply_task_event(state, event)
    elif event.entity_kind == "attempt":
        _apply_attempt_event(
            root,
            state,
            event,
            run_verification=run_verification,
            run_limits=run_limits,
            tolerate_run_errors=tolerate_run_errors,
            observe_run_reference=observe_run_reference,
        )
    else:
        _integrity("INVALID_EVENT_ENTITY", "study_event", "study, task, or attempt", event.entity_kind)
    state.updated_at = event.timestamp


def _apply_study_event(state: _ReplayState, event: StudyEvent, study_id: str) -> None:
    if event.entity_id != study_id:
        _integrity("STUDY_ID_MISMATCH", "study_event", study_id, event.entity_id)
    _validate_study_transition(state, event)
    _validate_study_execution_id(state, event)
    _validate_study_payload(state, event)
    state.study_state = cast(StudyState, event.transition_to)
    state.execution_id = event.execution_id


def _validate_study_transition(state: _ReplayState, event: StudyEvent) -> None:
    targets: dict[str, set[tuple[str, str]]] = {
        "execution_started": {("created", "running"), ("paused", "running"), ("completed_with_failures", "running")},
        "study_completed": {("created", "completed"), ("running", "completed")},
        "study_completed_with_failures": {("running", "completed_with_failures")},
        "study_paused": {("running", "paused")},
        "study_failed": {("running", "failed")},
        "study_cancelled": {("created", "cancelled"), ("running", "cancelled"), ("paused", "cancelled")},
    }
    transition = (state.study_state, event.transition_to)
    if event.event_type not in targets or transition not in targets[event.event_type] or event.transition_from != state.study_state:
        _integrity("INVALID_STATE_TRANSITION", "study_event", targets.get(event.event_type), transition)


def _validate_study_execution_id(state: _ReplayState, event: StudyEvent) -> None:
    if event.execution_id is None:
        _integrity("EXECUTION_ID_MISMATCH", "study_event", "UUIDv4 execution_id", None)
    if event.event_type != "execution_started" and state.execution_id is not None and event.execution_id != state.execution_id:
        _integrity("EXECUTION_ID_MISMATCH", "study_event", state.execution_id, event.execution_id)


def _validate_study_payload(state: _ReplayState, event: StudyEvent) -> None:
    if event.event_type == "execution_started":
        parent_id = event.payload.get("parent_execution_id")
        valid_initial = state.study_state == "created" and not event.payload
        valid_recovery = (
            state.study_state in {"paused", "completed_with_failures"}
            and set(event.payload) == {"parent_execution_id"}
            and isinstance(parent_id, str)
            and re.fullmatch(_UUID4_PATTERN, parent_id) is not None
            and parent_id == state.execution_id
        )
        if event.reason is not None or (not valid_initial and not valid_recovery):
            _integrity("INVALID_EVENT_PAYLOAD", "study_event", "exact prior execution identity on recovery", event)
    if event.event_type == "study_completed" and (event.reason is not None or event.payload):
        _integrity("INVALID_EVENT_PAYLOAD", "study_event", "no reason or payload", event)
    if event.event_type == "study_completed":
        if state.study_state == "created" and state.task_states:
            _integrity("INVALID_STATE_TRANSITION", "study_event", "empty created study", len(state.task_states))
        if state.study_state == "running" and any(value != "succeeded" for value in state.task_states.values()):
            _integrity("INVALID_STATE_TRANSITION", "study_event", "every task succeeded", dict(state.task_states))
    if event.event_type == "study_paused":
        failed_ids = sorted(task_id for task_id, value in state.task_states.items() if value == "failed")
        triggering_task = event.payload.get("failed_task_id")
        expected_attempt = _failed_attempt_id(state, triggering_task) if isinstance(triggering_task, str) else None
        failure_payload = (
            set(event.payload) == {"failed_task_id", "failed_attempt_id"}
            and triggering_task in failed_ids
            and event.payload.get("failed_attempt_id") == expected_attempt
        )
        pending = sorted(task_id for task_id, value in state.task_states.items() if value == "pending")
        interrupted = sorted(task_id for task_id, value in state.task_states.items() if value == "interrupted")
        recovery_payload = dict(event.payload) == {
            "pending_task_ids": tuple(pending),
            "interrupted_task_ids": tuple(interrupted),
        }
        if (not failure_payload and not recovery_payload) or not isinstance(event.reason, Mapping):
            _integrity(
                "INVALID_EVENT_PAYLOAD",
                "study_event",
                "one triggering failure or the exact unfinished recovery set",
                event,
            )
    if event.event_type == "study_completed_with_failures":
        failed_ids = sorted(task_id for task_id, value in state.task_states.items() if value == "failed")
        incomplete = {"pending", "running"}.intersection(state.task_states.values())
        expected_payload = {"failed_task_ids": tuple(failed_ids)}
        if incomplete or not failed_ids or dict(event.payload) != expected_payload or not isinstance(event.reason, Mapping):
            _integrity("INVALID_EVENT_PAYLOAD", "study_event", expected_payload, event)
    if event.event_type == "study_failed" and (not isinstance(event.reason, Mapping) or event.payload):
        _integrity("INVALID_EVENT_PAYLOAD", "study_event", "sanitized failure reason", event.reason)
    if event.event_type == "study_cancelled":
        pending_ids = sorted(task_id for task_id, value in state.task_states.items() if value == "pending")
        payload_ids = event.payload.get("cancelled_task_ids")
        if (
            set(event.payload) != {"cancelled_task_ids"}
            or not isinstance(payload_ids, tuple)
            or tuple(payload_ids) != tuple(pending_ids)
            or not isinstance(event.reason, Mapping)
            or "running" in state.task_states.values()
        ):
            _integrity("INVALID_EVENT_PAYLOAD", "study_event", {"cancelled_task_ids": pending_ids}, event)
        for task_id in pending_ids:
            state.task_states[task_id] = "cancelled"


def _apply_task_event(state: _ReplayState, event: StudyEvent) -> None:
    current = state.task_states.get(event.entity_id)
    if current is None:
        _integrity("TASK_ID_MISMATCH", "study_event", sorted(state.task_states), event.entity_id)
    _validate_execution_id(state, event)
    allowed = {
        "task_claimed": {("pending", "running"), ("failed", "running"), ("interrupted", "running")},
        "task_skipped": {("pending", "skipped")},
    }
    transition = (current, event.transition_to)
    if event.event_type not in allowed or transition not in allowed[event.event_type] or event.transition_from != current:
        _integrity("INVALID_STATE_TRANSITION", "study_event", allowed.get(event.event_type), transition)
    if event.event_type == "task_claimed":
        attempt_id = event.payload.get("attempt_id")
        run_id = event.payload.get("run_id")
        valid_payload = (
            set(event.payload) == {"attempt_id", "run_id"}
            and isinstance(attempt_id, str)
            and re.fullmatch(_UUID4_PATTERN, attempt_id) is not None
            and isinstance(run_id, str)
            and re.fullmatch(_UUID4_PATTERN, run_id) is not None
        )
        if event.reason is not None or not valid_payload or attempt_id == run_id:
            _integrity("INVALID_EVENT_PAYLOAD", "study_event", "distinct reserved attempt_id and run_id", event)
        candidates = (
            attempt
            for attempt_id, attempt in state.attempts.items()
            if state.attempt_states[attempt_id] == "created"
            and attempt.task_id == event.entity_id
            and attempt.execution_id == event.execution_id
            and attempt.attempt_id == event.payload.get("attempt_id")
        )
        if next(candidates, None) is None:
            _integrity("INVALID_STATE_TRANSITION", "study_event", "durable created attempt for claim", event.entity_id)
    state.task_states[event.entity_id] = cast(TaskState, event.transition_to)


def _apply_attempt_event(
    root: Path,
    state: _ReplayState,
    event: StudyEvent,
    *,
    run_verification: RunVerification,
    run_limits: LoadLimits | None,
    tolerate_run_errors: bool,
    observe_run_reference: ObserveRunReference | None,
) -> None:
    attempt, current = _attempt_context(state, event)
    target = _attempt_transition_target(current, event)
    if event.event_type == "attempt_started" and (event.reason is not None or event.payload):
        _integrity("INVALID_EVENT_PAYLOAD", "study_event", "attempt start without reason or payload", event)
    if target in _ATTEMPT_TERMINALS:
        attempt = _apply_terminal_attempt(
            root,
            state,
            event,
            attempt,
            target,
            run_verification=run_verification,
            run_limits=run_limits,
            tolerate_run_errors=tolerate_run_errors,
            observe_run_reference=observe_run_reference,
        )
    else:
        attempt = _apply_running_attempt(state, event, attempt)
    state.attempts[event.entity_id] = attempt
    state.attempt_states[event.entity_id] = target


def _attempt_context(state: _ReplayState, event: StudyEvent) -> tuple[AttemptRecord, AttemptState]:
    attempt = state.attempts.get(event.entity_id)
    current = state.attempt_states.get(event.entity_id)
    if attempt is None or current is None:
        _integrity("ATTEMPT_ID_MISMATCH", "study_event", "persisted attempt", event.entity_id)
    _validate_execution_id(state, event)
    if event.execution_id != attempt.execution_id:
        _integrity("EXECUTION_ID_MISMATCH", "study_event", attempt.execution_id, event.execution_id)
    return attempt, current


def _attempt_transition_target(current: AttemptState, event: StudyEvent) -> AttemptState:
    targets = {
        "attempt_started": "running",
        "attempt_succeeded": "succeeded",
        "attempt_failed": "failed",
        "attempt_interrupted": "interrupted",
        "attempt_cancelled": "cancelled",
    }
    target = targets.get(event.event_type)
    expected_from = "created" if event.event_type == "attempt_started" else "running"
    if event.event_type == "attempt_cancelled" and current == "created":
        expected_from = "created"
    if target is None or current != expected_from or event.transition_from != current or event.transition_to != target:
        _integrity(
            "INVALID_ATTEMPT_TRANSITION",
            "study_event",
            {"from": expected_from, "to": target},
            {"from": event.transition_from, "to": event.transition_to},
        )
    return cast(AttemptState, target)


def _apply_terminal_attempt(
    root: Path,
    state: _ReplayState,
    event: StudyEvent,
    attempt: AttemptRecord,
    target: AttemptState,
    *,
    run_verification: RunVerification,
    run_limits: LoadLimits | None,
    tolerate_run_errors: bool,
    observe_run_reference: ObserveRunReference | None,
) -> AttemptRecord:
    task_id = attempt.task_id
    payload = _terminal_payload(
        root,
        event,
        task_id,
        target,
        run_verification=run_verification,
        run_limits=run_limits,
        tolerate_run_errors=tolerate_run_errors,
        observe_run_reference=observe_run_reference,
    )
    timestamps = dict(attempt.timestamps)
    timestamps["started_at"] = timestamps.get("started_at") or event.timestamp
    timestamps["completed_at"] = event.timestamp
    terminal = replace(
        attempt,
        status=target,
        timestamps=deep_freeze(timestamps),
        failure=payload.get("failure"),
        run_reference=payload.get("run_reference"),
    )
    task_current = state.task_states.get(task_id)
    if task_current != "running" and not (target == "cancelled" and task_current == "pending"):
        _integrity("INVALID_STATE_TRANSITION", "study_event", "running task or pending cancellation", task_current)
    state.task_states[task_id] = cast(TaskState, target)
    return terminal


def _apply_running_attempt(state: _ReplayState, event: StudyEvent, attempt: AttemptRecord) -> AttemptRecord:
    task_id = attempt.task_id
    if state.task_states.get(task_id) != "running":
        _integrity("INVALID_STATE_TRANSITION", "study_event", "running task before attempt start", state.task_states.get(task_id))
    timestamps = dict(attempt.timestamps)
    timestamps["started_at"] = event.timestamp
    return replace(attempt, status="running", timestamps=deep_freeze(timestamps))


def _terminal_payload(
    root: Path,
    event: StudyEvent,
    task_id: str,
    status: str,
    *,
    run_verification: RunVerification,
    run_limits: LoadLimits | None,
    tolerate_run_errors: bool,
    observe_run_reference: ObserveRunReference | None,
) -> Mapping[str, Any]:
    payload = event.payload
    if status in {"interrupted", "cancelled"}:
        if payload or (status == "cancelled" and not isinstance(event.reason, Mapping)):
            _integrity("INVALID_TERMINAL_EVENT", "study_event", "bounded reason and empty payload", event)
        return cast(Mapping[str, Any], deep_freeze({"failure": None, "run_reference": None}))
    expected_fields = {"task_id", "run_reference"} | ({"failure"} if status == "failed" else set())
    if set(payload) != expected_fields or payload.get("task_id") != task_id:
        _integrity("INVALID_TERMINAL_EVENT", "study_event", sorted(expected_fields), dict(payload))
    reference = payload.get("run_reference")
    if not isinstance(reference, Mapping):
        _integrity("INVALID_TERMINAL_EVENT", "study_event", "run_reference object", reference)
    validate_observed_run(
        root,
        event.entity_id,
        reference,
        expected_task_id=task_id,
        expected_status=status,
        verification=run_verification,
        run_limits=run_limits,
        tolerate_errors=tolerate_run_errors,
        observer=observe_run_reference,
    )
    failure = payload.get("failure")
    if status == "failed" and not isinstance(failure, Mapping):
        _integrity("INVALID_TERMINAL_EVENT", "study_event", "sanitized failure object", failure)
    if status == "succeeded" and event.reason is not None:
        _integrity("INVALID_TERMINAL_EVENT", "study_event", "no success reason", event.reason)
    if status == "failed" and event.reason != failure:
        _integrity("INVALID_TERMINAL_EVENT", "study_event", "failure reason equal to payload failure", event.reason)
    return cast(Mapping[str, Any], deep_freeze(payload))


def _validate_execution_id(state: _ReplayState, event: StudyEvent) -> None:
    if state.execution_id is None or event.execution_id != state.execution_id:
        _integrity("EXECUTION_ID_MISMATCH", "study_event", state.execution_id, event.execution_id)


def _failed_attempt_id(state: _ReplayState, task_id: str) -> str | None:
    matches = [
        attempt.attempt_id
        for attempt_id, attempt in state.attempts.items()
        if attempt.task_id == task_id and state.attempt_states[attempt_id] == "failed"
    ]
    return matches[-1] if matches else None


def _copy_state(state: _ReplayState) -> _ReplayState:
    return _ReplayState(
        study_state=state.study_state,
        execution_id=state.execution_id,
        updated_at=state.updated_at,
        task_states=dict(state.task_states),
        attempt_states=dict(state.attempt_states),
        attempts=dict(state.attempts),
    )


def _integrity(
    reason: str,
    role: str,
    expected: object,
    actual: object,
    *,
    cause: Exception | None = None,
) -> NoReturn:
    error = StudyIntegrityError(
        operation="load study",
        reason=reason,
        document_role=role,
        expected=expected,
        actual=actual,
        action="Restore the contiguous authoritative journal and matching checkpoints.",
    )
    if cause is None:
        raise error
    raise error from cause


__all__ = ["EffectiveStudyState", "derive_effective_study"]
