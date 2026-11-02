from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.experiment.optimization_result import OptimizationResult
from vamos.experiment.study.errors import RetryNotAllowedError, StudyInfrastructureError, StudyRunPublicationError
from vamos.experiment.study.models import StudyEvent


def _spec(*, policy: str = "fail_fast") -> vamos.StudySpec:
    return vamos.StudySpec(
        problems=["zdt1"],
        algorithms=["nsgaii"],
        seeds=[0, 1, 2],
        max_evaluations=8,
        pop_size=4,
        on_error=policy,
    )


def _result(reconstructed: Any) -> OptimizationResult:
    return OptimizationResult(
        {
            "F": np.full((1, reconstructed.n_obj), reconstructed.seed, dtype=np.float64),
            "X": np.full((1, reconstructed.n_var), reconstructed.seed, dtype=np.float64),
            "evaluations": 8,
            "generations": 2,
        }
    )


def _immutable_inputs(root: Path) -> tuple[bytes, bytes]:
    return root.joinpath("study-spec.json").read_bytes(), root.joinpath("plan.json").read_bytes()


def _verify_terminal_runs(study: Any) -> None:
    for attempt in study.attempts:
        if attempt.status not in {"succeeded", "failed"}:
            continue
        assert attempt.run_reference is not None
        run = vamos.load_run(study.root / "runs" / str(attempt.run_reference["run_id"]), verify="all")
        assert run.status == attempt.status


def test_sa_027_fail_fast_pauses_after_durable_second_task_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    created = vamos.create_study(_spec(), output=root)
    immutable = _immutable_inputs(root)
    calls: list[int] = []

    def execute(reconstructed: Any, *, root: Path) -> OptimizationResult:
        calls.append(reconstructed.seed)
        if len(calls) == 2:
            raise RuntimeError(f"private failure at {root}")
        return _result(reconstructed)

    monkeypatch.setattr(execution, "_execute_optimization", execute)
    paused = created.run()

    assert paused is not created
    assert created.status == "created"
    assert paused.status == "paused"
    assert len(calls) == 2
    assert paused.manifest.on_error == paused.spec.on_error == "fail_fast"
    assert paused.manifest.counts.succeeded == 1
    assert paused.manifest.counts.failed == 1
    assert paused.manifest.counts.pending == 1
    assert [task.state for task in paused.tasks] == ["succeeded", "failed", "pending"]
    assert [event.event_type for event in paused.events][-2:] == ["attempt_failed", "study_paused"]
    assert paused.events[-1].payload["failed_task_id"] == paused.tasks[1].task_id
    assert "private" not in repr(paused.tasks[1].reason)
    assert _immutable_inputs(root) == immutable
    _verify_terminal_runs(paused)


def test_failed_task_records_sanitized_cause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import vamos.experiment.study.execution as execution
    from vamos.foundation.eval.population import _validate_output_matrix

    def execute(reconstructed: Any, *, root: Path) -> OptimizationResult:
        # The objective writes non-finite values (as WFG1 with two variables did): retry would repeat the failure.
        _validate_output_matrix("F", np.full((4, 2), np.nan), expected_shape=(4, 2))
        raise AssertionError("unreachable")

    created = vamos.create_study(_spec(), output=tmp_path / "study")
    monkeypatch.setattr(execution, "_execute_optimization", execute)
    with caplog.at_level(logging.ERROR, logger="vamos.experiment.study"):
        paused = created.run()

    task = paused.tasks[0]
    assert task.state == "failed"
    reason = task.reason
    assert reason is not None
    assert reason["message"].startswith("EvaluationError: ") and "non-finite" in reason["message"]
    assert reason["retryable"] is False and task.retryability.retryable is False
    assert str(tmp_path) not in repr(reason)
    assert paused.attempts[0].failure == reason
    run = vamos.load_run(paused.root / "runs" / str(paused.attempts[0].run_reference["run_id"]), verify="all")
    assert "non-finite" in run.manifest["failure"]["message"]
    assert any(record.exc_info and "non-finite" in str(record.exc_info[1]) for record in caplog.records)
    with pytest.raises(RetryNotAllowedError) as caught:
        paused.retry()
    assert caught.value.reason == "NONRETRYABLE_FAILURE"


def test_failure_message_hides_absolute_paths_and_names_the_typed_field() -> None:
    from vamos.experiment.artifacts.errors import ComponentNotReconstructableError, sanitized_message

    windows = sanitized_message(OSError(r"cannot open D:\lab42\study\runs\x.json: denied"))
    posix = sanitized_message(RuntimeError("private failure at /srv/lab42/study, see https://example.org/a 1/n"))
    typed = sanitized_message(
        ComponentNotReconstructableError(
            operation="reconstruct",
            reason="is unsupported",
            expected="serial",
            actual="dask",
            action="Use a serial evaluation backend.",
            field="$.resolved_spec.backend.evaluation",
            path="/srv/lab42/study/runs/1",
        )
    )

    unc = sanitized_message(OSError(r"lost \\lab42-server\share\study"))

    assert "lab42" not in windows + posix + typed + unc
    assert unc == "lost <path>"
    assert windows == "cannot open <path> denied"
    assert posix == "private failure at <path> see https://example.org/a 1/n"
    assert "$.resolved_spec.backend.evaluation" in typed
    assert len(sanitized_message(ValueError("x" * 2000))) == 500


def test_sa_028_continue_runs_later_tasks_and_completes_with_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    created = vamos.create_study(_spec(policy="continue"), output=root)
    calls: list[int] = []

    def execute(reconstructed: Any, *, root: Path) -> OptimizationResult:
        calls.append(reconstructed.seed)
        if len(calls) == 2:
            raise ValueError("bounded scientific task failure")
        return _result(reconstructed)

    monkeypatch.setattr(execution, "_execute_optimization", execute)
    partial = created.run()

    assert partial.status == "completed_with_failures"
    assert len(calls) == 3
    assert partial.manifest.on_error == partial.spec.on_error == "continue"
    assert partial.manifest.counts.succeeded == 2
    assert partial.manifest.counts.failed == 1
    assert partial.manifest.counts.pending == 0
    assert [task.state for task in partial.tasks] == ["succeeded", "failed", "succeeded"]
    assert partial.events[-1].event_type == "study_completed_with_failures"
    assert tuple(partial.events[-1].payload["failed_task_ids"]) == (partial.tasks[1].task_id,)
    _verify_terminal_runs(partial)


@pytest.mark.parametrize("policy", ["fail_fast", "continue"])
def test_sa_029_writable_infrastructure_failure_stops_without_task_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / policy
    created = vamos.create_study(_spec(policy=policy), output=root)

    def inject(phase: str) -> None:
        if phase == "before_attempt_record_creation":
            raise OSError("scheduler metadata failure")

    monkeypatch.setattr(execution, "_execution_phase", inject)
    with pytest.raises(StudyInfrastructureError) as caught:
        created.run()

    failed = vamos.load_study(root)
    assert caught.value.reason == "STUDY_INFRASTRUCTURE_FAILURE"
    assert failed.status == "failed"
    assert failed.manifest.counts.failed == 0
    assert failed.manifest.counts.pending == 3
    assert failed.events[-1].event_type == "study_failed"
    assert failed.events[-1].reason is not None
    assert failed.events[-1].reason["category"] == "infrastructure"
    assert not failed.attempts
    assert not root.joinpath("runs").exists()


@pytest.mark.parametrize("policy", ["fail_fast", "continue"])
def test_sa_065_unpublishable_success_stays_running_and_never_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    import vamos.experiment.study.execution as execution
    import vamos.experiment.study.run_publication as publication

    root = tmp_path / policy
    created = vamos.create_study(_spec(policy=policy), output=root)
    calls = 0

    def execute(reconstructed: Any, *, root: Path) -> OptimizationResult:
        nonlocal calls
        calls += 1
        return _result(reconstructed)

    def fail_publication(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("storage unavailable")

    monkeypatch.setattr(execution, "_execute_optimization", execute)
    monkeypatch.setattr(publication, "save_result", fail_publication)
    with pytest.raises(StudyRunPublicationError):
        created.run()

    interrupted = vamos.load_study(root)
    assert calls == 1
    assert interrupted.status == "running"
    assert interrupted.manifest.counts.running == 1
    assert interrupted.manifest.counts.failed == 0
    assert interrupted.attempts[0].status == "running"
    assert interrupted.attempts[0].run_reference is None
    assert not any(event.event_type in {"attempt_failed", "study_failed"} for event in interrupted.events)


def test_sa_030_programmatic_cancellation_before_run_accounts_for_every_task(tmp_path: Path) -> None:
    root = tmp_path / "study"
    created = vamos.create_study(_spec(), output=root)
    immutable = _immutable_inputs(root)
    cancelled = created.cancel()

    assert created.status == "created"
    assert cancelled.status == "cancelled"
    assert cancelled.manifest.counts.cancelled == 3
    assert [task.state for task in cancelled.tasks] == ["cancelled"] * 3
    assert cancelled.attempts == ()
    assert cancelled.events[-1].event_type == "study_cancelled"
    assert tuple(cancelled.events[-1].payload["cancelled_task_ids"]) == tuple(task.task_id for task in cancelled.tasks)
    assert _immutable_inputs(root) == immutable
    assert not root.joinpath("runs").exists()


def test_sa_030_keyboard_interrupt_during_objective_cancels_active_and_unclaimed_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    created = vamos.create_study(_spec(), output=root)
    calls = 0

    def interrupt(_reconstructed: Any, *, root: Path) -> OptimizationResult:
        nonlocal calls
        calls += 1
        raise KeyboardInterrupt("secret interruption")

    monkeypatch.setattr(execution, "_execute_optimization", interrupt)
    cancelled = created.run()

    assert calls == 1
    assert cancelled.status == "cancelled"
    assert cancelled.manifest.counts.cancelled == 3
    assert cancelled.attempts[0].status == "cancelled"
    assert cancelled.attempts[0].run_reference is None
    assert [event.event_type for event in cancelled.events][-2:] == ["attempt_cancelled", "study_cancelled"]
    assert "secret" not in repr(cancelled.events[-2].reason)
    assert not root.joinpath("runs").exists()


def test_sa_030_keyboard_interrupt_between_tasks_starts_no_later_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    created = vamos.create_study(_spec(), output=root)
    claims = 0
    original_phase = execution._execution_phase

    def execute(reconstructed: Any, *, root: Path) -> OptimizationResult:
        return _result(reconstructed)

    def interrupt_before_second_claim(phase: str) -> None:
        nonlocal claims
        if phase == "before_attempt_record_creation":
            claims += 1
            if claims == 2:
                raise KeyboardInterrupt
        original_phase(phase)

    monkeypatch.setattr(execution, "_execute_optimization", execute)
    monkeypatch.setattr(execution, "_execution_phase", interrupt_before_second_claim)
    cancelled = created.run()

    assert cancelled.status == "cancelled"
    assert len(cancelled.attempts) == 1
    assert cancelled.attempts[0].status == "succeeded"
    assert cancelled.manifest.counts.succeeded == 1
    assert cancelled.manifest.counts.cancelled == 2
    assert not any(event.event_type == "task_claimed" for event in cancelled.events if event.entity_id != cancelled.tasks[0].task_id)


def test_same_process_programmatic_request_is_observed_after_objective_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    created = vamos.create_study(_spec(), output=root)
    observed: list[str] = []

    def execute(reconstructed: Any, *, root: Path) -> OptimizationResult:
        requested = vamos.load_study(root).cancel()
        observed.append(requested.status)
        return _result(reconstructed)

    monkeypatch.setattr(execution, "_execute_optimization", execute)
    cancelled = created.run()

    assert observed == ["running"]
    assert cancelled.status == "cancelled"
    assert cancelled.manifest.counts.cancelled == 3
    assert len(cancelled.attempts) == 1
    assert cancelled.attempts[0].status == "cancelled"
    assert not root.joinpath("runs").exists()


def test_paused_study_can_be_durably_abandoned_without_changing_failed_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    created = vamos.create_study(_spec(), output=root)
    calls = 0

    def execute(reconstructed: Any, *, root: Path) -> OptimizationResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("task failure")
        return _result(reconstructed)

    monkeypatch.setattr(execution, "_execute_optimization", execute)
    paused = created.run()
    cancelled = paused.cancel()

    assert paused.status == "paused"
    assert cancelled.status == "cancelled"
    assert [task.state for task in cancelled.tasks] == ["succeeded", "failed", "cancelled"]
    assert cancelled.manifest.counts.failed == 1
    assert cancelled.manifest.counts.cancelled == 1
    assert len(cancelled.attempts) == 2
    _verify_terminal_runs(cancelled)


def test_sa_031_forced_process_termination_performs_no_cancellation_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    created = vamos.create_study(_spec(), output=root)

    def terminate(_reconstructed: Any, *, root: Path) -> OptimizationResult:
        raise SystemExit(9)

    monkeypatch.setattr(execution, "_execute_optimization", terminate)
    with pytest.raises(SystemExit, match="9"):
        created.run()

    interrupted = vamos.load_study(root)
    assert interrupted.status == "running"
    assert interrupted.tasks[0].state == "running"
    assert interrupted.attempts[0].status == "running"
    assert interrupted.attempts[0].run_reference is None
    assert not any(event.event_type in {"attempt_cancelled", "study_cancelled"} for event in interrupted.events)
    assert not root.joinpath("runs").exists()


def test_fail_fast_pause_payload_selects_triggering_failure_when_prior_failures_exist() -> None:
    import vamos.experiment.study.journal as journal

    state = SimpleNamespace(
        task_states={"task-a": "failed", "task-b": "failed", "task-c": "pending"},
        attempt_states={"attempt-a": "failed", "attempt-b": "failed"},
        attempts={
            "attempt-a": SimpleNamespace(task_id="task-a", attempt_id="attempt-a"),
            "attempt-b": SimpleNamespace(task_id="task-b", attempt_id="attempt-b"),
        },
    )
    event = StudyEvent(
        sequence=12,
        event_id="11111111-1111-4111-8111-111111111111",
        event_type="study_paused",
        entity_kind="study",
        entity_id="22222222-2222-4222-8222-222222222222",
        transition_from="running",
        transition_to="paused",
        execution_id="33333333-3333-4333-8333-333333333333",
        timestamp="2026-08-31T00:00:00Z",
        reason={"category": "execution"},
        payload={"failed_task_id": "task-b", "failed_attempt_id": "attempt-b"},
        previous_event_sha256="0" * 64,
        document_sha256="1" * 64,
    )

    journal._validate_study_payload(state, event)
