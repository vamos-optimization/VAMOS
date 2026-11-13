"""Explicit single-owner reconciliation, resume, and bounded retry."""

from __future__ import annotations

from typing import Literal

from vamos.experiment.artifacts.compatibility import reuse_current_environment
from vamos.experiment.artifacts.reconstruction import reconstruct_resolved_run
from vamos.experiment.artifacts.verification import verify_run

from .cancellation import cancel_loaded
from .errors import (
    ResumeEnvironmentIncompatibilityError,
    RetryLimitError,
    RetryNotAllowedError,
    StudyError,
    StudyInfrastructureError,
    UnsupportedStudyExecutionStateError,
)
from .execution_errors import enrich_execution_error
from .failure_policy import complete_running, pause_after_task_failure, record_infrastructure_failure
from .loading import load_study
from .models import Study, TaskRecord
from .reconciliation import reconcile_study

_Operation = Literal["resume", "retry"]


def resume_study(snapshot: Study, *, retry_failed: bool = False) -> Study:
    """Reconcile first, then execute pending/interrupted and explicit failures."""
    with reuse_current_environment():
        return _operate(snapshot, operation="resume", include_failed=retry_failed, include_pending=True, include_interrupted=True)


def retry_study(snapshot: Study, *, failed_only: bool = True) -> Study:
    """Explicitly retry failed tasks, optionally including interrupted tasks."""
    with reuse_current_environment():
        return _operate(
            snapshot,
            operation="retry",
            include_failed=True,
            include_pending=False,
            include_interrupted=not failed_only,
        )


def _operate(
    snapshot: Study,
    *,
    operation: _Operation,
    include_failed: bool,
    include_pending: bool,
    include_interrupted: bool,
) -> Study:
    from . import execution

    root = snapshot.root.resolve(strict=True)
    state: execution._ExecutionState | None = None
    if root in execution._ACTIVE_ROOTS:
        raise _ownership_error(snapshot, operation)
    execution._ACTIVE_ROOTS[root] = None
    try:
        current = load_study(root)
        _validate_snapshot(snapshot, current, operation)
        current = reconcile_study(current)
        selected = _select_tasks(
            current,
            operation=operation,
            include_failed=include_failed,
            include_pending=include_pending,
            include_interrupted=include_interrupted,
        )
        if not selected:
            return load_study(root)
        _verify_resume_environment(current, selected, operation)
        parent_execution_id = current.manifest.execution_id
        state = execution._start_execution(current, parent_execution_id=parent_execution_id)
        for plan_task in current.plan.tasks:
            if plan_task.task_id not in selected:
                continue
            index = state.task_indexes[plan_task.task_id]
            if execution._run_task_attempt(state, index, plan_task):
                return load_study(root)
            task = state.tasks[index]
            if task.state == "failed" and state.manifest.on_error == "fail_fast":
                return pause_after_task_failure(state, task)
        return complete_running(state, execution._execution_phase, load_study)
    except KeyboardInterrupt as exc:
        try:
            return cancel_loaded(load_study(root), code="PROCESS_INTERRUPTION")
        except Exception as cancellation_error:
            raise StudyInfrastructureError(
                operation=f"cancel interrupted study {operation}",
                reason="CANCELLATION_PUBLICATION_FAILED",
                study_id=snapshot.study_id,
                current_state=state.manifest.state if state is not None else snapshot.status,
                expected_state="cancelled",
                path=root,
                expected="durable cancellation without a fabricated outcome",
                actual=type(cancellation_error).__name__,
                action="Load the authoritative journal; cancellation could not be published safely.",
            ) from exc
    except StudyInfrastructureError as exc:
        if state is None:
            raise
        enrich_execution_error(exc, snapshot, state)
        recorded = record_infrastructure_failure(root, exc)
        if recorded is exc:
            raise
        raise recorded from exc
    except StudyError:
        raise
    except Exception as exc:
        error = StudyInfrastructureError(
            operation=f"{operation} study",
            reason="STUDY_EXECUTION_INTERRUPTED",
            study_id=snapshot.study_id,
            current_state=state.manifest.state if state is not None else snapshot.status,
            expected_state="completed or paused",
            path=root,
            expected="durable explicit recovery operation",
            actual=type(exc).__name__,
            action="Load the study and inspect its authoritative event journal.",
        )
        if state is None:
            raise error from exc
        raise record_infrastructure_failure(root, error) from exc
    finally:
        execution._ACTIVE_ROOTS.pop(root, None)
        execution._CANCELLATION_REQUESTS.pop(root, None)


def _select_tasks(
    study: Study,
    *,
    operation: _Operation,
    include_failed: bool,
    include_pending: bool,
    include_interrupted: bool,
) -> set[str]:
    _validate_operation_state(study, operation)
    selected: list[TaskRecord] = []
    for task in study.tasks:
        if (
            (include_pending and task.state == "pending")
            or (include_failed and task.state == "failed")
            or (include_interrupted and task.state == "interrupted")
        ):
            selected.append(task)
    for task in selected:
        if task.retryability.attempts_remaining < 1:
            raise RetryLimitError(
                operation=f"{operation} study task",
                reason="RETRY_LIMIT_REACHED",
                study_id=study.study_id,
                task_id=task.task_id,
                current_state=task.state,
                expected_state="eligible below max_attempts_per_task",
                expected=f"fewer than {study.manifest.max_attempts_per_task} attempts",
                actual=len(task.attempts),
                action="Create a new study for any scientifically changed configuration.",
            )
        if task.state in {"failed", "interrupted"} and not task.retryability.retryable:
            raise RetryNotAllowedError(
                operation=f"{operation} study task",
                reason="NONRETRYABLE_FAILURE",
                study_id=study.study_id,
                task_id=task.task_id,
                current_state=task.state,
                expected_state="retryable failed or interrupted task",
                expected="contract-approved transient execution failure or interruption",
                actual=task.retryability.category,
                action="Create a new study after correcting deterministic configuration or integrity.",
            )
    if operation == "retry" and not selected:
        terminal = next((task for task in study.tasks if task.state in {"succeeded", "skipped", "cancelled"}), None)
        if terminal is not None:
            raise RetryNotAllowedError(
                operation="retry study task",
                reason="TERMINAL_TASK_NOT_RETRYABLE",
                study_id=study.study_id,
                task_id=terminal.task_id,
                current_state=terminal.state,
                expected_state="failed or interrupted",
                expected="retryable unsuccessful task",
                actual=terminal.state,
                action="Successful, skipped, and cancelled tasks have no force-retry operation in v1.",
            )
    return {task.task_id for task in selected}


def _verify_resume_environment(study: Study, selected: set[str], operation: _Operation) -> None:
    # The runs of one execution come from one process and share its environment, so one run per execution
    # stands for the others; load_study has already verified the integrity of every run.
    compared_executions: set[str] = set()
    for attempt in study.attempts:
        if attempt.run_reference is None or attempt.execution_id in compared_executions:
            continue
        run_id = attempt.run_reference.get("run_id")
        if not isinstance(run_id, str):
            continue
        compared_executions.add(attempt.execution_id)
        report = verify_run(study.root / "runs" / run_id)
        if report.environment.level != "exact":
            raise ResumeEnvironmentIncompatibilityError(
                operation=f"{operation} study",
                reason="RESUME_ENVIRONMENT_INCOMPATIBLE",
                study_id=study.study_id,
                task_id=attempt.task_id,
                current_state=study.status,
                expected_state="exact prior material environment",
                path=f"runs/{run_id}/environment.json",
                expected="exact material compatibility",
                actual=report.environment.level,
                action="Use the runtime and dependency build recorded by the prior canonical run.",
            )
    for plan_task in study.plan.tasks:
        if plan_task.task_id not in selected:
            continue
        try:
            reconstruct_resolved_run(plan_task.resolved_run_spec, root=study.root)
        except Exception as exc:
            raise ResumeEnvironmentIncompatibilityError(
                operation=f"{operation} study",
                reason="RESUME_ENVIRONMENT_INCOMPATIBLE",
                study_id=study.study_id,
                task_id=plan_task.task_id,
                current_state=study.status,
                expected_state="supported persisted built-in configuration",
                expected="exact reconstruction from the immutable resolved task",
                actual=type(exc).__name__,
                action="Restore the recorded supported environment; current defaults are never substituted.",
            ) from exc


def _validate_operation_state(study: Study, operation: _Operation) -> None:
    if operation == "resume" and study.status in {"paused", "completed", "completed_with_failures"}:
        return
    if operation == "retry" and study.status in {"paused", "completed", "completed_with_failures"}:
        return
    raise UnsupportedStudyExecutionStateError(
        operation=f"{operation} study",
        reason="INVALID_STATE_TRANSITION",
        study_id=study.study_id,
        current_state=study.status,
        expected_state="paused or completed_with_failures",
        expected="verified recoverable study",
        actual=study.status,
        action="Use run() only for a pristine created study; failed/cancelled studies are terminal.",
    )


def _validate_snapshot(snapshot: Study, current: Study, operation: _Operation) -> None:
    if (snapshot.study_id, snapshot.plan_id) == (current.study_id, current.plan_id):
        return
    raise UnsupportedStudyExecutionStateError(
        operation=f"{operation} study",
        reason="STUDY_IDENTITY_CHANGED",
        study_id=snapshot.study_id,
        current_state=current.status,
        expected_state=snapshot.status,
        expected={"study_id": snapshot.study_id, "plan_id": snapshot.plan_id},
        actual={"study_id": current.study_id, "plan_id": current.plan_id},
        action="Discard the stale handle and inspect the canonical study root.",
    )


def _ownership_error(study: Study, operation: _Operation) -> UnsupportedStudyExecutionStateError:
    return UnsupportedStudyExecutionStateError(
        operation=f"{operation} study",
        reason="ACTIVE_IN_PROCESS_OWNERSHIP",
        study_id=study.study_id,
        current_state=study.status,
        expected_state="no active local mutation",
        expected="one explicit recovery owner in this process",
        actual="root is already active",
        action="Wait for the current in-process operation; v1 makes no cross-process safety claim.",
    )


__all__ = ["resume_study", "retry_study"]
