"""Single read-only projection service for study reports and summaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from vamos.experiment.artifacts.models import RunManifest, StoredRun, deep_freeze

from .errors import StudyError
from .loading import load_study_projection
from .models import AttemptRecord, PlanTask, Study, TaskRecord
from .report_models import StudyAttemptReport, StudyIssue, StudyReport, StudySummary, StudySummaryRow


@dataclass(frozen=True, slots=True)
class StudyProjection:
    report: StudyReport
    summary: StudySummary


@dataclass(slots=True)
class _EvidenceCollector:
    runs: dict[str, StoredRun]
    issues: dict[tuple[str, str, str | None], StudyIssue]

    def observe(
        self,
        attempt_id: str,
        reference: Mapping[str, Any],
        stored: StoredRun | None,
        error: StudyError | None,
    ) -> None:
        if stored is not None:
            if not any(key[0] == attempt_id for key in self.issues):
                self.runs[attempt_id] = stored
            return
        if error is None:
            return
        self.runs.pop(attempt_id, None)
        run_id = _string(reference.get("run_id"))
        task_id = _string(reference.get("task_id")) or error.task_id
        key = (attempt_id, error.reason, error.path)
        self.issues[key] = StudyIssue(
            category=error.category,
            reason=error.reason,
            message=str(error),
            action=error.action,
            task_id=task_id,
            attempt_id=attempt_id,
            run_id=run_id,
            path=error.path,
        )


def project_study(snapshot: Study) -> StudyProjection:
    """Reload once and derive every supported read model from the same view."""
    collector = _EvidenceCollector({}, {})
    study = load_study_projection(snapshot.root, observe_run_reference=collector.observe)
    issues = tuple(collector.issues[key] for key in sorted(collector.issues, key=lambda item: (item[0], item[1], item[2] or "")))
    report = _report(study, collector.runs, issues)
    summary = _summary(study, collector.runs, issues)
    return StudyProjection(report=report, summary=summary)


def _report(study: Study, runs: Mapping[str, StoredRun], issues: tuple[StudyIssue, ...]) -> StudyReport:
    attempts = tuple(_attempt_report(item, runs.get(item.attempt_id)) for item in study.attempts)
    runnable_states = {"created", "running", "paused"}
    runnable = tuple(item.task_id for item in study.tasks if study.status in runnable_states and item.state in {"pending", "interrupted"})
    retry_states = {"paused", "completed", "completed_with_failures"}
    retryable = tuple(
        item.task_id for item in study.tasks if study.status in retry_states and item.state == "failed" and item.retryability.retryable
    )
    head = study.events[-1]
    return StudyReport(
        study_id=study.study_id,
        plan_id=study.plan_id,
        state=study.status,
        on_error=study.manifest.on_error,
        max_attempts_per_task=study.manifest.max_attempts_per_task,
        created_at=study.manifest.created_at,
        updated_at=study.manifest.updated_at,
        counts=cast(Mapping[str, int], deep_freeze(_counts(study))),
        attempts=attempts,
        total_attempt_count=len(attempts),
        verified_run_count=len({item.manifest.run_id for item in runs.values()}),
        event_head_sequence=head.sequence,
        event_head_sha256=head.file_sha256,
        checkpoint_sequence=study.stored_checkpoint_sequence,
        checkpoint_event_sha256=study.stored_checkpoint_event_sha256,
        journal_checkpoint_relation=_checkpoint_relation(study),
        reconciliation_required=study.reconciliation_required,
        runnable_task_ids=runnable,
        retryable_task_ids=retryable,
        runnable_work=bool(runnable),
        retryable_failed_work=bool(retryable),
        changed=False,
        issues=issues,
        next_actions=_next_actions(study, runnable, retryable, issues),
    )


def _summary(study: Study, runs: Mapping[str, StoredRun], issues: tuple[StudyIssue, ...]) -> StudySummary:
    tasks = {item.task_id: item for item in study.tasks}
    attempts = _attempts_by_task(study.attempts)
    rows = tuple(
        _summary_row(
            study,
            plan_task,
            tasks[plan_task.task_id],
            attempts.get(plan_task.task_id, ()),
            runs,
        )
        for plan_task in sorted(study.plan.tasks, key=lambda item: item.plan_index)
    )
    head = study.events[-1]
    return StudySummary(
        study_id=study.study_id,
        plan_id=study.plan_id,
        state=study.status,
        generated_at=study.manifest.updated_at,
        root_manifest_sha256=study.manifest.document_sha256,
        event_head_sequence=head.sequence,
        event_head_sha256=head.file_sha256,
        rows=rows,
        issues=issues,
    )


def _attempt_report(attempt: AttemptRecord, stored: StoredRun | None) -> StudyAttemptReport:
    reference = attempt.run_reference or {}
    timestamps = attempt.timestamps
    return StudyAttemptReport(
        task_id=attempt.task_id,
        attempt_id=attempt.attempt_id,
        attempt_number=attempt.attempt_number,
        status=attempt.status,
        started_at=_string(timestamps.get("started_at")),
        completed_at=_string(timestamps.get("completed_at")),
        run_id=_string(reference.get("run_id")),
        run_manifest_path=_string(reference.get("path")),
        run_status=stored.status if stored is not None else None,
        run_metadata_verified=stored is not None,
        failure=_frozen_mapping(attempt.failure),
    )


def _summary_row(
    study: Study,
    plan_task: PlanTask,
    task: TaskRecord,
    attempts: tuple[AttemptRecord, ...],
    runs: Mapping[str, StoredRun],
) -> StudySummaryRow:
    latest = attempts[-1] if attempts else None
    selected = next((item for item in attempts if item.attempt_id == task.selected_success_attempt_id), None)
    reported = selected or latest
    reference = reported.run_reference if reported is not None else None
    stored = runs.get(reported.attempt_id) if reported is not None else None
    manifest = stored.manifest if stored is not None else None
    resolved = plan_task.resolved_run_spec
    outcome = _mapping(manifest.get("outcome")) if manifest is not None else {}
    metrics = _frozen_mapping(outcome.get("metrics")) if manifest is not None else None
    failure = reported.failure if reported is not None else task.reason
    selected_reference = selected.run_reference if selected is not None else None
    return StudySummaryRow(
        study_id=study.study_id,
        plan_id=study.plan_id,
        task_id=task.task_id,
        plan_index=task.plan_index,
        state=task.state,
        problem_id=_component_id(resolved.get("problem")),
        algorithm_id=_component_id(resolved.get("algorithm")),
        backend_id=_component_id(_mapping(resolved.get("backend")).get("kernel")),
        seed=_integer(resolved.get("seed")),
        evaluation_budget=_budget(_mapping(resolved.get("termination"))),
        population_size=_integer(_mapping(resolved.get("population")).get("initial_size")),
        attempt_count=len(attempts),
        latest_attempt_id=latest.attempt_id if latest is not None else None,
        selected_attempt_id=task.selected_success_attempt_id,
        selected_run_id=_string(selected_reference.get("run_id")) if selected_reference is not None else None,
        evidence_run_id=_string(reference.get("run_id")) if reference is not None else None,
        run_manifest_path=_string(reference.get("path")) if reference is not None else None,
        run_metadata_available=manifest is not None,
        run_status=manifest.status if manifest is not None else None,
        evaluations=_integer(outcome.get("evaluations")),
        termination_reason=_string(outcome.get("termination_reason")),
        runtime_ms=_number(outcome.get("runtime_ms")),
        failure_category=_string(_mapping(failure).get("category")),
        failure_code=_string(_mapping(failure).get("code")),
        failure=_frozen_mapping(failure),
        retryable=task.retryability.retryable,
        metrics=metrics,
        run_manifest_sha256=_manifest_hash(manifest),
    )


def _counts(study: Study) -> dict[str, int]:
    value = study.manifest.counts
    return {
        "tasks": value.tasks,
        "pending": value.pending,
        "running": value.running,
        "succeeded": value.succeeded,
        "failed": value.failed,
        "interrupted": value.interrupted,
        "cancelled": value.cancelled,
        "skipped": value.skipped,
    }


def _checkpoint_relation(study: Study) -> str:
    if study.stored_checkpoint_sequence < study.events[-1].sequence:
        return "journal_ahead"
    if study.reconciliation_required:
        return "record_checkpoints_lag_journal"
    return "aligned"


def _next_actions(
    study: Study,
    runnable: tuple[str, ...],
    retryable: tuple[str, ...],
    issues: tuple[StudyIssue, ...],
) -> tuple[str, ...]:
    actions: list[str] = []
    if issues:
        actions.append("restore_referenced_run_evidence")
    # Resume reconciles a paused or completed study; a cancelled or failed one is terminal, and readers
    # already derive its state from the journal.
    if study.reconciliation_required and study.status not in {"cancelled", "failed"}:
        actions.append("resume_to_reconcile")
    if study.status == "running" and not study.reconciliation_required:
        actions.append("wait_for_current_owner_or_resume_after_interruption")
    if runnable:
        if study.status == "created":
            actions.append("run")
        elif study.status == "paused":
            actions.append("resume")
    if retryable:
        actions.append("retry_failed")
    if study.status == "failed":
        actions.append("create_new_study")
    if not actions:
        actions.append("summarize")
    return tuple(dict.fromkeys(actions))


def _attempts_by_task(attempts: tuple[AttemptRecord, ...]) -> dict[str, tuple[AttemptRecord, ...]]:
    grouped: dict[str, list[AttemptRecord]] = {}
    for attempt in attempts:
        grouped.setdefault(attempt.task_id, []).append(attempt)
    return {task_id: tuple(sorted(values, key=lambda item: item.attempt_number)) for task_id, values in grouped.items()}


def _manifest_hash(manifest: RunManifest | None) -> str | None:
    if manifest is None:
        return None
    integrity = _mapping(manifest.get("integrity"))
    return _string(integrity.get("manifest_sha256"))


def _component_id(value: object) -> str | None:
    return _string(_mapping(value).get("component_id"))


def _budget(termination: Mapping[str, Any]) -> int | None:
    config = _mapping(termination.get("config"))
    return _integer(config.get("hard_max_evaluations", config.get("max_evaluations")))


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _frozen_mapping(value: object) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return cast(Mapping[str, Any], deep_freeze(value))


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: object) -> float | int | None:
    return value if isinstance(value, (float, int)) and not isinstance(value, bool) else None


__all__ = ["StudyProjection", "project_study"]
