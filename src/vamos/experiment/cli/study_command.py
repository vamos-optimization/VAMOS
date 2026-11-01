"""Semantic command service and common envelope for the durable-study CLI."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from vamos.experiment.study.errors import StudyError, StudyInfrastructureError
from vamos.experiment.study.loading import load_study_projection
from vamos.experiment.study.projection import project_study
from vamos.experiment.study.report_models import StudyReport
from vamos.study_artifacts import create_study, load_study, plan_study

from .study_command_result import (
    COMMAND_DOCUMENT_TYPE,
    COMMAND_SCHEMA_VERSION,
    Operation,
    StudyCommandResult,
    map_exit_code,
)
from .study_spec_io import load_study_spec
from .study_summary_output import SummaryFormat, write_summary

SINGLE_OWNER_WARNING = "Concurrent mutation is unsupported: use one process and one mutation owner for each study."


@dataclass(frozen=True, slots=True)
class StudyCommandRequest:
    operation: Operation
    config: Path | None = None
    study_dir: Path | None = None
    output: Path | None = None
    json_output: bool = False
    retry_failed: bool = False
    failed_only: bool = True
    summary_format: SummaryFormat = "json"


def execute_study_command(request: StudyCommandRequest) -> StudyCommandResult:
    """Execute one CLI operation through canonical StudyManifest services."""
    try:
        return _COMMANDS[request.operation](request)
    except StudyError as exc:
        return _error_result(request.operation, exc)
    except KeyboardInterrupt:
        return _interrupted_result(request.operation)
    except Exception as exc:  # CLI boundary: raw tracebacks are opt-in elsewhere.
        error = StudyInfrastructureError(
            operation=f"{request.operation} study",
            reason="UNEXPECTED_INFRASTRUCTURE_FAILURE",
            expected="successful canonical study service operation",
            actual=type(exc).__name__,
            action="Inspect the study and environment, then retry the safe reported operation.",
        )
        return _error_result(request.operation, error)


def _plan(request: StudyCommandRequest) -> StudyCommandResult:
    config = _required(request.config, "config")
    report = plan_study(load_study_spec(config), output=request.output)
    payload = report.as_dict()
    next_actions = (
        ("vamos study create CONFIG --output STUDY_DIR",)
        if report.status == "ready"
        else ("choose an absent output path, then run vamos study plan again",)
    )
    payload["next_actions"] = list(next_actions)
    return StudyCommandResult(
        operation="plan",
        status=report.status,
        exit_code=map_exit_code("plan", report.status),
        study_id=None,
        plan_id=report.plan_id,
        changed=False,
        payload=payload,
        warnings=report.warnings,
        errors=tuple(item.as_dict() for item in report.errors),
        next_actions=next_actions,
    )


def _create(request: StudyCommandRequest) -> StudyCommandResult:
    config = _required(request.config, "config")
    output = _required(request.output, "output")
    study = create_study(load_study_spec(config), output=output)
    report = replace(study.inspect(), changed=True)
    return _report_result("create", report, execution_began=False, root=study.root, warnings=(SINGLE_OWNER_WARNING,))


def _run(request: StudyCommandRequest) -> StudyCommandResult:
    study = load_study(_required(request.study_dir, "study_dir"))
    before = study.inspect()
    completed = study.run()
    return _mutation_result("run", before, completed)


def _inspect(request: StudyCommandRequest) -> StudyCommandResult:
    snapshot = load_study_projection(_required(request.study_dir, "study_dir"))
    report = project_study(snapshot).report
    return _report_result("inspect", report, execution_began=False, root=snapshot.root)


def _resume(request: StudyCommandRequest) -> StudyCommandResult:
    study = load_study(_required(request.study_dir, "study_dir"))
    before = study.inspect()
    completed = study.resume(retry_failed=request.retry_failed)
    return _mutation_result("resume", before, completed)


def _retry(request: StudyCommandRequest) -> StudyCommandResult:
    study = load_study(_required(request.study_dir, "study_dir"))
    before = study.inspect()
    completed = study.retry(failed_only=request.failed_only)
    return _mutation_result("retry", before, completed)


def _summarize(request: StudyCommandRequest) -> StudyCommandResult:
    root = _required(request.study_dir, "study_dir")
    snapshot = load_study_projection(root)
    projection = project_study(snapshot)
    output: dict[str, object] = {"written": False, "derived": True, "format": request.summary_format, "path": None, "bytes": 0}
    if request.output is not None:
        _validate_summary_destination(snapshot.root, request.output)
        size = write_summary(projection.summary, request.output, output_format=request.summary_format)
        output.update({"written": True, "path": request.output, "bytes": size})
    payload = {"summary": projection.summary.as_dict(), "output": output, "canonical_state_changed": False}
    return StudyCommandResult(
        operation="summarize",
        status=projection.summary.state,
        exit_code=map_exit_code("summarize", projection.report),
        study_id=projection.summary.study_id,
        plan_id=projection.summary.plan_id,
        changed=False,
        payload=payload,
        warnings=(),
        errors=tuple(item.as_dict() for item in projection.summary.issues),
        next_actions=("vamos study inspect STUDY_DIR",),
    )


def _mutation_result(operation: Operation, before: StudyReport, completed: Any) -> StudyCommandResult:
    report = completed.inspect()
    changed = (before.event_head_sequence, before.event_head_sha256) != (report.event_head_sequence, report.event_head_sha256)
    execution_began = any(
        event.event_type == "execution_started" and event.sequence > before.event_head_sequence for event in completed.events
    )
    return _report_result(
        operation,
        replace(report, changed=changed),
        execution_began=execution_began,
        root=completed.root,
        warnings=(SINGLE_OWNER_WARNING,),
    )


def _report_result(
    operation: Operation,
    report: StudyReport,
    *,
    execution_began: bool,
    root: Path,
    warnings: tuple[str, ...] = (),
) -> StudyCommandResult:
    return StudyCommandResult(
        operation=operation,
        status=report.state,
        exit_code=map_exit_code(operation, report),
        study_id=report.study_id,
        plan_id=report.plan_id,
        changed=report.changed,
        payload={
            "report": report.as_dict(),
            "study_root": root,
            "execution_began": execution_began,
            "single_owner": operation in {"create", "run", "resume", "retry"},
        },
        warnings=warnings,
        errors=tuple(item.as_dict() for item in report.issues),
        next_actions=_command_actions(report.next_actions),
    )


def _error_result(operation: Operation, error: StudyError) -> StudyCommandResult:
    return StudyCommandResult(
        operation=operation,
        status="error",
        exit_code=map_exit_code(operation, error),
        study_id=error.study_id,
        plan_id=None,
        changed=error.published,
        payload={"execution_began": error.execution_occurred, "single_owner": operation in {"create", "run", "resume", "retry"}},
        warnings=(SINGLE_OWNER_WARNING,) if operation in {"create", "run", "resume", "retry"} else (),
        errors=(error.as_dict(),),
        next_actions=(error.action,),
    )


def _interrupted_result(operation: Operation) -> StudyCommandResult:
    return StudyCommandResult(
        operation=operation,
        status="interrupted",
        exit_code=8,
        study_id=None,
        plan_id=None,
        changed=False,
        payload={"execution_began": False, "single_owner": operation in {"create", "run", "resume", "retry"}},
        warnings=(SINGLE_OWNER_WARNING,) if operation in {"create", "run", "resume", "retry"} else (),
        errors=({"category": "interruption", "reason": "PROCESS_INTERRUPTED", "action": "Inspect the study before resuming."},),
        next_actions=("vamos study inspect STUDY_DIR",),
    )


def _command_actions(actions: tuple[str, ...]) -> tuple[str, ...]:
    mapping = {
        "run": "vamos study run STUDY_DIR",
        "resume": "vamos study resume STUDY_DIR",
        "retry_failed": "vamos study retry STUDY_DIR --failed",
        "summarize": "vamos study summarize STUDY_DIR",
        "resume_to_reconcile": "vamos study resume STUDY_DIR",
        "restore_referenced_run_evidence": "restore the exact referenced run evidence, then inspect again",
        "wait_for_current_owner_or_resume_after_interruption": "wait for the current owner; after it ends, inspect before resuming",
        "create_new_study": "vamos study create CONFIG --output STUDY_DIR",
    }
    return tuple(mapping.get(action, action) for action in actions)


def _validate_summary_destination(root: Path, destination: Path) -> None:
    resolved = destination.absolute()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return
    if not relative.parts or relative.parts[0] != "derived":
        from vamos.experiment.study.errors import UnsafeStudyPathError

        raise UnsafeStudyPathError(
            operation="write derived study summary",
            reason="UNSAFE_SUMMARY_DESTINATION",
            path=destination.name,
            expected="an explicit external path or a path below STUDY_DIR/derived",
            actual=relative.as_posix(),
            action="Choose an external destination or STUDY_DIR/derived/...; canonical study files are never summary outputs.",
        )


def _required(value: Path | None, name: str) -> Path:
    if value is None:
        raise AssertionError(f"parser omitted required {name}")
    return value


_COMMANDS = {
    "plan": _plan,
    "create": _create,
    "run": _run,
    "inspect": _inspect,
    "resume": _resume,
    "retry": _retry,
    "summarize": _summarize,
}


__all__ = [
    "COMMAND_DOCUMENT_TYPE",
    "COMMAND_SCHEMA_VERSION",
    "Operation",
    "SINGLE_OWNER_WARNING",
    "StudyCommandRequest",
    "StudyCommandResult",
    "execute_study_command",
    "map_exit_code",
]
