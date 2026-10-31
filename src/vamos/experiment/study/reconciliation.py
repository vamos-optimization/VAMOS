"""Evidence-driven checkpoint and interrupted-attempt reconciliation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from vamos.experiment.artifacts.errors import RunArtifactError
from vamos.experiment.artifacts.jsonio import canonical_json_bytes, sha256_bytes
from vamos.experiment.artifacts.models import StoredRun, deep_freeze
from vamos.experiment.artifacts.persistence import load_run
from vamos.experiment.artifacts.verification import verify_run

from .commits import append_event, checkpoint_attempt, checkpoint_manifest, checkpoint_task, run_reference
from .documents import attempt_document, manifest_checkpoint_document, semantic_hash, task_checkpoint_document
from .errors import ReferencedRunCorruptError
from .failure_policy import complete_running, pause_after_task_failure, pause_with_runnable_work
from .loading import load_study
from .models import AttemptRecord, AttemptReference, PlanTask, Study, TaskRecord
from .serialization import stored_document_bytes
from .writing import remove_writer_temporaries


def reconcile_study(snapshot: Study) -> Study:
    """Write only proven checkpoint/outcome corrections; loading stays inert."""
    _remove_orphaned_temporaries(snapshot)
    current = _refresh_lagging_checkpoints(snapshot)
    if current.status != "running":
        return current
    active = [attempt for attempt in current.attempts if attempt.status in {"created", "running"}]
    if len(active) > 1:
        raise ReferencedRunCorruptError(
            operation="reconcile study",
            reason="CONTRADICTORY_ACTIVE_ATTEMPTS",
            study_id=current.study_id,
            expected="at most one active attempt in the sequential study",
            actual=[attempt.attempt_id for attempt in active],
            action="Restore the single authoritative sequential attempt history.",
        )
    trigger_task_id: str | None = None
    if active:
        current, attempt = _ensure_running_attempt(current, active[0])
        trigger_task_id = attempt.task_id
        _reconcile_attempt(current, attempt)
        current = _refresh_lagging_checkpoints(load_study(current.root))
    return _settle_reconciled_execution(current, trigger_task_id=trigger_task_id)


def _remove_orphaned_temporaries(study: Study) -> None:
    """Delete the temporaries a writer that died before its replace left behind.

    They never hold published state and loading already ignores them; the reconciling
    process is the single owner of the study, so none of them is in flight.
    """
    directories = [study.root, study.root / "events"]
    for plan_task in study.plan.tasks:
        task_directory = study.root / "tasks" / plan_task.task_id.removeprefix("sha256:")
        directories.extend((task_directory, task_directory / "attempts"))
    for directory in directories:
        remove_writer_temporaries(directory)


def _ensure_running_attempt(study: Study, attempt: AttemptRecord) -> tuple[Study, AttemptRecord]:
    """Finish only missing claim/start metadata before interrupting created work."""
    if attempt.status == "running":
        return study, attempt
    matching_claim = next(
        (
            event
            for event in study.events
            if event.event_type == "task_claimed"
            and event.execution_id == attempt.execution_id
            and event.payload.get("attempt_id") == attempt.attempt_id
        ),
        None,
    )
    event = study.events[-1]
    task = _task(study, attempt.task_id)
    if matching_claim is None:
        from .execution import _distinct_run_id

        event = append_event(
            study.root,
            event,
            event_type="task_claimed",
            entity_kind="task",
            entity_id=task.task_id,
            transition_from=task.state,
            transition_to="running",
            execution_id=attempt.execution_id,
            payload={"attempt_id": attempt.attempt_id, "run_id": _distinct_run_id(attempt.attempt_id)},
        )
    event = append_event(
        study.root,
        event,
        event_type="attempt_started",
        entity_kind="attempt",
        entity_id=attempt.attempt_id,
        transition_from="created",
        transition_to="running",
        execution_id=attempt.execution_id,
    )
    timestamps = dict(attempt.timestamps)
    timestamps["started_at"] = event.timestamp
    attempt, reference = checkpoint_attempt(
        study.root,
        replace(attempt, status="running", timestamps=deep_freeze(timestamps)),
    )
    task = checkpoint_task(
        study.root,
        replace(
            task,
            state="running",
            attempts=_replace_attempt_reference(task.attempts, reference),
            current_attempt_id=attempt.attempt_id,
            selected_success_attempt_id=None,
            claim_epoch=max(task.claim_epoch, attempt.attempt_number),
        ),
    )
    checkpoint_manifest(
        study.root,
        study.manifest,
        state="running",
        execution_id=attempt.execution_id,
        tasks=_replace_task(study.tasks, task),
        event=event,
    )
    loaded = load_study(study.root)
    return loaded, next(item for item in loaded.attempts if item.attempt_id == attempt.attempt_id)


def _reconcile_attempt(study: Study, attempt: AttemptRecord) -> None:
    run_id = _reserved_run_id(study, attempt)
    run_root = study.root / "runs" / run_id
    if not os.path.lexists(run_root):
        _commit_interrupted(study, attempt)
        return
    stored = _verified_expected_run(study, attempt, run_id, run_root)
    _commit_recovered_run(study, attempt, stored)


def _reserved_run_id(study: Study, attempt: AttemptRecord) -> str:
    matches = [
        event
        for event in study.events
        if event.event_type == "task_claimed"
        and event.execution_id == attempt.execution_id
        and event.entity_id == attempt.task_id
        and event.payload.get("attempt_id") == attempt.attempt_id
    ]
    run_id = matches[-1].payload.get("run_id") if len(matches) == 1 else None
    if not isinstance(run_id, str):
        raise ReferencedRunCorruptError(
            operation="reconcile study attempt",
            reason="EXPECTED_RUN_ID_MISSING",
            study_id=study.study_id,
            task_id=attempt.task_id,
            attempt_id=attempt.attempt_id,
            expected="one canonical claim event reserving a distinct run UUID",
            actual=len(matches),
            action="Restore the authoritative claim/start events; never infer an attempt from run directories.",
        )
    return run_id


def _verified_expected_run(study: Study, attempt: AttemptRecord, run_id: str, run_root: Path) -> StoredRun:
    try:
        verify_run(run_root)
        stored = load_run(run_root, verify="all")
    except RunArtifactError as exc:
        raise ReferencedRunCorruptError(
            operation="reconcile study attempt",
            reason="REFERENCED_RUN_CORRUPT",
            study_id=study.study_id,
            task_id=attempt.task_id,
            attempt_id=attempt.attempt_id,
            path=f"runs/{run_id}/manifest.json",
            expected="complete verified expected canonical run",
            actual=type(exc).__name__,
            action="Restore or remove the corrupt expected output before another attempt is considered.",
        ) from exc
    plan_task = _plan_task(study, attempt.task_id)
    valid = (
        stored.manifest.run_id == run_id
        and stored.manifest.task_id == attempt.task_id
        and stored.status in {"succeeded", "failed"}
        and canonical_json_bytes(stored.manifest.resolved_spec) == canonical_json_bytes(plan_task.resolved_run_spec)
    )
    if valid:
        return stored
    raise ReferencedRunCorruptError(
        operation="reconcile study attempt",
        reason="RUN_REFERENCE_MISMATCH",
        study_id=study.study_id,
        task_id=attempt.task_id,
        attempt_id=attempt.attempt_id,
        path=f"runs/{run_id}/manifest.json",
        expected={"run_id": run_id, "task_id": attempt.task_id, "status": ["succeeded", "failed"]},
        actual={"run_id": stored.manifest.run_id, "task_id": stored.manifest.task_id, "status": stored.status},
        action="Keep the ambiguous output unreferenced and restore the expected canonical run.",
    )


def _commit_interrupted(study: Study, attempt: AttemptRecord) -> None:
    reason = cast(
        Mapping[str, Any],
        deep_freeze(
            {
                "category": "interruption",
                "code": "STALE_ATTEMPT_INTERRUPTED",
                "message": "The prior single-process owner ended without a terminal canonical run.",
                "retryable": True,
                "safe_action": "Resume to create a fresh attempt from the persisted resolved task.",
            }
        ),
    )
    event = append_event(
        study.root,
        study.events[-1],
        event_type="attempt_interrupted",
        entity_kind="attempt",
        entity_id=attempt.attempt_id,
        transition_from="running",
        transition_to="interrupted",
        execution_id=attempt.execution_id,
        reason=reason,
    )
    timestamps = dict(attempt.timestamps)
    timestamps["completed_at"] = event.timestamp
    attempt, reference = checkpoint_attempt(
        study.root,
        replace(attempt, status="interrupted", timestamps=deep_freeze(timestamps)),
    )
    task = _task(study, attempt.task_id)
    task = checkpoint_task(
        study.root,
        replace(
            task,
            state="interrupted",
            attempts=_replace_attempt_reference(task.attempts, reference),
            current_attempt_id=None,
            selected_success_attempt_id=None,
            retryability=replace(
                task.retryability,
                retryable=task.retryability.attempts_remaining > 0,
                category="interruption",
            ),
            reason=reason,
        ),
    )
    checkpoint_manifest(
        study.root,
        study.manifest,
        state="running",
        execution_id=attempt.execution_id,
        tasks=_replace_task(study.tasks, task),
        event=event,
    )


def _commit_recovered_run(study: Study, attempt: AttemptRecord, stored: StoredRun) -> None:
    reference = run_reference(stored, study_root=study.root)
    failure = _recovered_failure(stored) if stored.status == "failed" else None
    event = append_event(
        study.root,
        study.events[-1],
        event_type="attempt_failed" if stored.status == "failed" else "attempt_succeeded",
        entity_kind="attempt",
        entity_id=attempt.attempt_id,
        transition_from="running",
        transition_to=cast(Any, stored.status),
        execution_id=attempt.execution_id,
        reason=failure,
        payload={"task_id": attempt.task_id, "run_reference": reference, **({"failure": failure} if failure else {})},
    )
    timestamps = dict(attempt.timestamps)
    timestamps["completed_at"] = event.timestamp
    attempt, attempt_reference = checkpoint_attempt(
        study.root,
        replace(
            attempt,
            status=cast(Any, stored.status),
            timestamps=deep_freeze(timestamps),
            failure=failure,
            run_reference=reference,
        ),
    )
    task = _task(study, attempt.task_id)
    retryable = bool(failure and failure["retryable"] and task.retryability.attempts_remaining > 0)
    task = checkpoint_task(
        study.root,
        replace(
            task,
            state=cast(Any, stored.status),
            attempts=_replace_attempt_reference(task.attempts, attempt_reference),
            current_attempt_id=None,
            selected_success_attempt_id=attempt.attempt_id if stored.status == "succeeded" else None,
            retryability=replace(
                task.retryability,
                retryable=retryable,
                category=str(failure["category"]) if failure is not None else None,
            ),
            reason=failure,
        ),
    )
    checkpoint_manifest(
        study.root,
        study.manifest,
        state="running",
        execution_id=attempt.execution_id,
        tasks=_replace_task(study.tasks, task),
        event=event,
    )


def _recovered_failure(stored: StoredRun) -> Mapping[str, Any]:
    source = stored.manifest.get("failure")
    retryable = isinstance(source, Mapping) and source.get("phase") == "optimization"
    return cast(
        Mapping[str, Any],
        deep_freeze(
            {
                "category": "execution" if retryable else "configuration",
                "code": "OBJECTIVE_EXECUTION_FAILED" if retryable else "RESOLVED_SPEC_RECONSTRUCTION_FAILED",
                "message": "Recovered the verified failed canonical run published by the prior attempt.",
                "retryable": retryable,
                "safe_action": (
                    "Correct the transient cause, then explicitly retry the task."
                    if retryable
                    else "Create a new study after correcting deterministic configuration."
                ),
            }
        ),
    )


def _settle_reconciled_execution(study: Study, *, trigger_task_id: str | None) -> Study:
    if study.status != "running" or any(task.state == "running" for task in study.tasks):
        return study
    state = _state_from_study(study)
    if all(task.state == "succeeded" for task in study.tasks):
        return complete_running(state, lambda _phase: None, load_study)
    failed = next(
        (task for task in study.tasks if task.state == "failed" and (trigger_task_id is None or task.task_id == trigger_task_id)),
        None,
    )
    if failed is not None and study.manifest.on_error == "fail_fast":
        return pause_after_task_failure(state, failed)
    if any(task.state in {"pending", "interrupted"} for task in study.tasks):
        return pause_with_runnable_work(state)
    return complete_running(state, lambda _phase: None, load_study)


def _refresh_lagging_checkpoints(study: Study) -> Study:
    references: dict[str, AttemptReference] = {}
    changed = False
    for attempt in study.attempts:
        relative = _attempt_path(attempt)
        expected = stored_document_bytes(attempt_document(attempt))
        actual = (study.root / relative).read_bytes()
        if actual != expected:
            _, reference = checkpoint_attempt(study.root, attempt)
            changed = True
        else:
            document = attempt_document(attempt)
            reference = AttemptReference(
                attempt_id=attempt.attempt_id,
                attempt_number=attempt.attempt_number,
                path=relative,
                role="attempt",
                required_for=("inspect", "resume"),
                semantic_sha256=semantic_hash(document),
                sha256=sha256_bytes(actual),
                bytes=len(actual),
            )
        references[attempt.attempt_id] = reference
    tasks: list[TaskRecord] = []
    for task in study.tasks:
        refs = tuple(
            references[item.attempt_id]
            for item in sorted(
                (attempt for attempt in study.attempts if attempt.task_id == task.task_id),
                key=lambda item: item.attempt_number,
            )
        )
        projected = replace(task, attempts=refs)
        relative = f"tasks/{task.task_id.removeprefix('sha256:')}/task.json"
        expected = stored_document_bytes(task_checkpoint_document(projected))
        if (study.root / relative).read_bytes() != expected:
            projected = checkpoint_task(study.root, projected)
            changed = True
        tasks.append(projected)
    projected_tasks = tuple(tasks)
    expected_manifest = stored_document_bytes(manifest_checkpoint_document(study.manifest))
    if changed or (study.root / "study-manifest.json").read_bytes() != expected_manifest:
        checkpoint_manifest(
            study.root,
            study.manifest,
            state=study.status,
            execution_id=study.manifest.execution_id,
            tasks=projected_tasks,
            event=study.events[-1],
        )
        changed = True
    return load_study(study.root) if changed else study


def _state_from_study(study: Study) -> Any:
    from .execution import _ExecutionState

    tasks = list(study.tasks)
    return _ExecutionState(
        study.root,
        study.study_id,
        cast(str, study.manifest.execution_id),
        study.manifest,
        tasks,
        {task.task_id: index for index, task in enumerate(tasks)},
        study.events[-1],
    )


def _task(study: Study, task_id: str) -> TaskRecord:
    return next(task for task in study.tasks if task.task_id == task_id)


def _plan_task(study: Study, task_id: str) -> PlanTask:
    return next(task for task in study.plan.tasks if task.task_id == task_id)


def _replace_task(tasks: tuple[TaskRecord, ...], replacement: TaskRecord) -> tuple[TaskRecord, ...]:
    return tuple(replacement if task.task_id == replacement.task_id else task for task in tasks)


def _replace_attempt_reference(references: tuple[AttemptReference, ...], replacement: AttemptReference) -> tuple[AttemptReference, ...]:
    if not any(reference.attempt_id == replacement.attempt_id for reference in references):
        return (*references, replacement)
    return tuple(replacement if reference.attempt_id == replacement.attempt_id else reference for reference in references)


def _attempt_path(attempt: AttemptRecord) -> str:
    return f"tasks/{attempt.task_id.removeprefix('sha256:')}/attempts/{attempt.attempt_id}.json"


__all__ = ["reconcile_study"]
