"""Durable single-process sequential execution for canonical studies."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from vamos.experiment.artifacts.compatibility import reuse_current_environment
from vamos.experiment.artifacts.models import deep_freeze
from vamos.experiment.artifacts.reconstruction import (
    ReconstructedRun,
    instantiate_reconstructed_problem,
    reconstruct_resolved_run,
)
from vamos.experiment.optimize import _OptimizeConfig, _run_config

from .cancellation import cancel_loaded, cancel_snapshot
from .commits import (
    append_event,
    checkpoint_attempt,
    checkpoint_manifest,
    checkpoint_task,
    now_utc,
    run_reference,
)
from .errors import (
    StudyError,
    StudyEventAppendError,
    StudyInfrastructureError,
    UnsupportedStudyExecutionStateError,
)
from .execution_errors import (
    active_run_published,
    enrich_execution_error,
    finalization_error,
    state_error,
    state_error_for_task,
)
from .failure_policy import complete_running, pause_after_task_failure, record_infrastructure_failure
from .identity import new_uuid4
from .loading import load_study
from .models import AttemptRecord, PlanTask, Study, StudyEvent, StudyManifest, TaskRecord
from .run_publication import attach_run_metadata, commit_task_failure, publish_success

_ACTIVE_ROOTS: dict[Path, None] = {}
_CANCELLATION_REQUESTS: dict[Path, None] = {}


@dataclass(slots=True)
class _ExecutionState:
    root: Path
    study_id: str
    execution_id: str
    manifest: StudyManifest
    tasks: list[TaskRecord]
    task_indexes: dict[str, int]
    event: StudyEvent
    active_task_id: str | None = None
    active_attempt_id: str | None = None
    active_run_id: str | None = None
    objective_evaluation_began: bool = False


def execute_study(snapshot: Study) -> Study:
    """Execute all pending tasks in canonical task-ID order and reload."""
    with reuse_current_environment():
        return _execute_study(snapshot)


def _execute_study(snapshot: Study) -> Study:
    root = snapshot.root.resolve(strict=True)
    state: _ExecutionState | None = None
    if root in _ACTIVE_ROOTS:
        raise state_error(snapshot, "REENTRANT_STUDY_EXECUTION", snapshot.status)
    _ACTIVE_ROOTS[root] = None
    try:
        current = load_study(root)
        _validate_snapshot(snapshot, current)
        if not current.tasks:
            return _complete_empty(current)
        state = _start_execution(current)
        for plan_task in current.plan.tasks:
            if _consume_cancellation(root):
                return cancel_loaded(load_study(root), code="USER_CANCELLATION")
            index = state.task_indexes[plan_task.task_id]
            if _run_task_attempt(state, index, plan_task):
                return load_study(root)
            if _consume_cancellation(root):
                return cancel_loaded(load_study(root), code="USER_CANCELLATION")
            task = state.tasks[index]
            if task.state == "failed" and state.manifest.on_error == "fail_fast":
                return pause_after_task_failure(state, task)
        return complete_running(state, _execution_phase, load_study)
    except KeyboardInterrupt as exc:
        try:
            return cancel_loaded(load_study(root), code="PROCESS_INTERRUPTION")
        except Exception as cancellation_error:
            raise StudyInfrastructureError(
                operation="cancel interrupted study execution",
                reason="CANCELLATION_PUBLICATION_FAILED",
                study_id=snapshot.study_id,
                task_id=state.active_task_id if state is not None else None,
                attempt_id=state.active_attempt_id if state is not None else None,
                current_state=state.manifest.state if state is not None else snapshot.status,
                expected_state="cancelled",
                objective_evaluation_began=state.objective_evaluation_began if state is not None else False,
                canonical_run_published=active_run_published(state),
                path=root,
                expected="durable cancellation without a fabricated outcome",
                actual=type(cancellation_error).__name__,
                action="Load the authoritative journal; cancellation could not be published safely.",
            ) from exc
    except StudyInfrastructureError as exc:
        enrich_execution_error(exc, snapshot, state)
        recorded = record_infrastructure_failure(root, exc)
        if recorded is exc:
            raise
        raise recorded from exc
    except StudyError as exc:
        enrich_execution_error(exc, snapshot, state)
        raise
    except Exception as exc:
        error = StudyInfrastructureError(
            operation="run study",
            reason="STUDY_EXECUTION_INTERRUPTED",
            study_id=snapshot.study_id,
            task_id=state.active_task_id if state is not None else None,
            attempt_id=state.active_attempt_id if state is not None else None,
            current_state=state.manifest.state if state is not None else snapshot.status,
            expected_state="completed",
            objective_evaluation_began=state.objective_evaluation_began if state is not None else False,
            canonical_run_published=active_run_published(state),
            path=root,
            expected="durable sequential execution or explicit terminal task failure",
            actual=type(exc).__name__,
            action="Load the study to inspect authoritative journal state.",
        )
        recorded = record_infrastructure_failure(root, error)
        raise recorded from exc
    finally:
        _ACTIVE_ROOTS.pop(root, None)
        _CANCELLATION_REQUESTS.pop(root, None)


def cancel_study(snapshot: Study) -> Study:
    """Cancel an idle study or request cancellation from this process's runner."""
    return cancel_snapshot(snapshot, _ACTIVE_ROOTS, _CANCELLATION_REQUESTS)


def _validate_snapshot(snapshot: Study, current: Study) -> None:
    if (snapshot.study_id, snapshot.plan_id) != (current.study_id, current.plan_id):
        raise UnsupportedStudyExecutionStateError(
            operation="run study",
            reason="STUDY_IDENTITY_CHANGED",
            study_id=snapshot.study_id,
            current_state=current.status,
            expected_state="created",
            path=current.root,
            expected={"study_id": snapshot.study_id, "plan_id": snapshot.plan_id},
            actual={"study_id": current.study_id, "plan_id": current.plan_id},
            action="Discard the stale handle and inspect the canonical study root.",
        )
    if current.status != "created":
        raise state_error(current, "UNSUPPORTED_STUDY_EXECUTION_STATE", current.status)
    if current.attempts or any(task.state != "pending" or task.attempts for task in current.tasks):
        raise UnsupportedStudyExecutionStateError(
            operation="run study",
            reason="NO_VALID_PENDING_TASK",
            study_id=current.study_id,
            current_state=current.status,
            expected_state="created with pristine pending tasks",
            expected="zero attempts and every task pending",
            actual={"attempts": len(current.attempts), "task_states": [task.state for task in current.tasks]},
            action="Do not retry or resume this root; create a new study until recovery support exists.",
        )


def _complete_empty(study: Study) -> Study:
    execution_id = new_uuid4()
    try:
        _execution_phase("before_final_completed_event")
        event = append_event(
            study.root,
            study.events[-1],
            event_type="study_completed",
            entity_kind="study",
            entity_id=study.study_id,
            transition_from="created",
            transition_to="completed",
            execution_id=execution_id,
        )
        checkpoint_manifest(
            study.root,
            study.manifest,
            state="completed",
            execution_id=execution_id,
            tasks=(),
            event=event,
        )
    except StudyEventAppendError:
        raise
    except Exception as exc:
        raise finalization_error(study.study_id, "created", False, False, study.root, exc) from exc
    return load_study(study.root)


def _start_execution(study: Study, *, parent_execution_id: str | None = None) -> _ExecutionState:
    execution_id = new_uuid4()
    event = append_event(
        study.root,
        study.events[-1],
        event_type="execution_started",
        entity_kind="study",
        entity_id=study.study_id,
        transition_from=study.status,
        transition_to="running",
        execution_id=execution_id,
        payload={"parent_execution_id": parent_execution_id} if parent_execution_id is not None else None,
    )
    manifest = checkpoint_manifest(
        study.root,
        study.manifest,
        state="running",
        execution_id=execution_id,
        tasks=study.tasks,
        event=event,
    )
    tasks = list(study.tasks)
    indexes = {task.task_id: index for index, task in enumerate(tasks)}
    return _ExecutionState(study.root, study.study_id, execution_id, manifest, tasks, indexes, event)


def _run_task_attempt(state: _ExecutionState, index: int, plan_task: PlanTask) -> bool:
    task = state.tasks[index]
    state.active_task_id = task.task_id
    state.active_attempt_id = None
    state.active_run_id = None
    state.objective_evaluation_began = False
    if task.state not in {"pending", "failed", "interrupted"}:
        raise state_error_for_task(state, task, "NO_VALID_RUNNABLE_TASK")
    _execution_phase("before_attempt_record_creation")
    attempt_id = new_uuid4()
    run_id = _distinct_run_id(attempt_id)
    state.active_attempt_id = attempt_id
    state.active_run_id = run_id
    created_at = now_utc()
    attempt = AttemptRecord(
        study_id=state.study_id,
        task_id=task.task_id,
        attempt_id=attempt_id,
        attempt_number=len(task.attempts) + 1,
        execution_id=state.execution_id,
        status="created",
        timestamps=deep_freeze({"created_at": created_at, "started_at": None, "completed_at": None}),
        lease_evidence=None,
        failure=None,
        run_reference=None,
        document_sha256="",
    )
    attempt, attempt_ref = checkpoint_attempt(state.root, attempt)
    _execution_phase("after_attempt_record_creation")
    state.event = append_event(
        state.root,
        state.event,
        event_type="task_claimed",
        entity_kind="task",
        entity_id=task.task_id,
        transition_from=task.state,
        transition_to="running",
        execution_id=state.execution_id,
        payload={"attempt_id": attempt_id, "run_id": run_id},
    )
    state.event = append_event(
        state.root,
        state.event,
        event_type="attempt_started",
        entity_kind="attempt",
        entity_id=attempt.attempt_id,
        transition_from="created",
        transition_to="running",
        execution_id=state.execution_id,
    )
    _execution_phase("after_running_event_publication")
    timestamps = dict(attempt.timestamps)
    timestamps["started_at"] = state.event.timestamp
    attempt = replace(attempt, status="running", timestamps=deep_freeze(timestamps))
    attempt, attempt_ref = checkpoint_attempt(state.root, attempt)
    task = replace(
        task,
        state="running",
        attempts=(*task.attempts, attempt_ref),
        current_attempt_id=attempt.attempt_id,
        selected_success_attempt_id=None,
        retryability=replace(
            task.retryability, retryable=False, category=None, attempts_remaining=task.retryability.attempts_remaining - 1
        ),
        claim_epoch=task.claim_epoch + 1,
    )
    task = checkpoint_task(state.root, task)
    state.tasks[index] = task
    _execution_phase("after_task_running_checkpoint")
    state.manifest = checkpoint_manifest(
        state.root,
        state.manifest,
        state="running",
        execution_id=state.execution_id,
        tasks=tuple(state.tasks),
        event=state.event,
    )
    return _execute_and_commit(state, index, task, attempt, plan_task, run_id)


def _execute_and_commit(
    state: _ExecutionState,
    index: int,
    task: TaskRecord,
    attempt: AttemptRecord,
    plan_task: PlanTask,
    run_id: str,
) -> bool:
    started_at = cast(str, attempt.timestamps["started_at"])
    objective_began = False
    try:
        reconstructed = reconstruct_resolved_run(plan_task.resolved_run_spec, root=state.root)
    except KeyboardInterrupt:
        cancel_loaded(load_study(state.root), code="PROCESS_INTERRUPTION")
        return True
    except Exception as exc:
        commit_task_failure(
            state,
            index,
            task,
            attempt,
            plan_task,
            run_id,
            exc,
            started_at=started_at,
            objective_began=objective_began,
        )
        return False
    _execution_phase("before_objective_evaluation")
    objective_began = True
    state.objective_evaluation_began = True
    started_monotonic = time.perf_counter()
    try:
        result = _execute_optimization(reconstructed, root=state.root)
    except KeyboardInterrupt:
        cancel_loaded(load_study(state.root), code="PROCESS_INTERRUPTION")
        return True
    except Exception as exc:
        commit_task_failure(
            state,
            index,
            task,
            attempt,
            plan_task,
            run_id,
            exc,
            started_at=started_at,
            objective_began=objective_began,
        )
        return False
    completed_at = now_utc()
    runtime_ms = (time.perf_counter() - started_monotonic) * 1000.0
    _execution_phase("after_optimization_result_exists")
    if _consume_cancellation(state.root):
        cancel_loaded(load_study(state.root), code="USER_CANCELLATION")
        return True
    attach_run_metadata(result, plan_task, run_id, started_at, completed_at, runtime_ms)
    stored = publish_success(state, task, plan_task, run_id, result, _execution_phase)
    del result
    reference = run_reference(stored, study_root=state.root)
    state.event = append_event(
        state.root,
        state.event,
        event_type="attempt_succeeded",
        entity_kind="attempt",
        entity_id=attempt.attempt_id,
        transition_from="running",
        transition_to="succeeded",
        execution_id=state.execution_id,
        payload={"task_id": task.task_id, "run_reference": reference},
    )
    _execution_phase("after_terminal_success_event")
    timestamps = dict(attempt.timestamps)
    timestamps["completed_at"] = state.event.timestamp
    attempt = replace(
        attempt,
        status="succeeded",
        timestamps=deep_freeze(timestamps),
        failure=None,
        run_reference=reference,
    )
    attempt, attempt_ref = checkpoint_attempt(state.root, attempt)
    _execution_phase("before_terminal_task_checkpoint")
    task = replace(
        task,
        state="succeeded",
        attempts=(*task.attempts[:-1], attempt_ref),
        current_attempt_id=None,
        selected_success_attempt_id=attempt.attempt_id,
        retryability=replace(task.retryability, retryable=False, category=None),
    )
    task = checkpoint_task(state.root, task)
    state.tasks[index] = task
    _execution_phase("before_terminal_study_checkpoint")
    state.manifest = checkpoint_manifest(
        state.root,
        state.manifest,
        state="running",
        execution_id=state.execution_id,
        tasks=tuple(state.tasks),
        event=state.event,
    )
    return False


def _execute_optimization(reconstructed: ReconstructedRun, *, root: Path) -> Any:
    problem = instantiate_reconstructed_problem(reconstructed, root=root)
    return _run_config(
        _OptimizeConfig(
            problem=problem,
            algorithm=reconstructed.algorithm,
            algorithm_config=reconstructed.algorithm_config,
            termination=reconstructed.termination,
            seed=reconstructed.seed,
            engine=reconstructed.engine,
            eval_strategy=reconstructed.eval_strategy,
        ),
        built_in_only=True,
    )


def _distinct_run_id(attempt_id: str) -> str:
    run_id = str(uuid.uuid4())
    while run_id == attempt_id:
        run_id = str(uuid.uuid4())
    return run_id


def _execution_phase(_phase: str) -> None:
    """Internal monkeypatch seam for deterministic failure-injection tests."""


def _consume_cancellation(root: Path) -> bool:
    if root not in _CANCELLATION_REQUESTS:
        return False
    _CANCELLATION_REQUESTS.pop(root, None)
    return True


__all__ = ["cancel_study", "execute_study"]
