from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.experiment.optimization_result import OptimizationResult
from vamos.experiment.study.errors import (
    ResumeEnvironmentIncompatibilityError,
    RetryLimitError,
    RetryNotAllowedError,
    StudyEventAppendError,
    StudyRunVerificationError,
    UnsupportedStudyExecutionStateError,
)


def _spec(
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    policy: str = "fail_fast",
    max_attempts: int = 3,
) -> vamos.StudySpec:
    return vamos.StudySpec(
        problems=["zdt1"],
        algorithms=["nsgaii"],
        seeds=seeds,
        max_evaluations=8,
        pop_size=4,
        on_error=policy,
        max_attempts_per_task=max_attempts,
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


def _tree(root: Path) -> dict[str, bytes | None]:
    return {path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None for path in sorted(root.rglob("*"))}


def test_sa_039_043_045_049_050_resume_pending_then_explicit_retry_preserves_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    created = vamos.create_study(_spec(), output=root)
    calls: list[int] = []

    def fail_second_once(reconstructed: Any, *, root: Path) -> OptimizationResult:
        calls.append(reconstructed.seed)
        if len(calls) == 2:
            raise RuntimeError("transient objective failure")
        return _result(reconstructed)

    monkeypatch.setattr(execution, "_execute_optimization", fail_second_once)
    paused = created.run()
    failed_task = next(task for task in paused.tasks if task.state == "failed")
    failed_attempt = next(attempt for attempt in paused.attempts if attempt.task_id == failed_task.task_id)
    failed_bytes = (root / failed_task.attempts[0].path).read_bytes()
    succeeded_ids = {task.task_id for task in paused.tasks if task.state == "succeeded"}

    resumed = paused.resume()

    assert resumed.status == "completed_with_failures"
    assert next(task for task in resumed.tasks if task.task_id == failed_task.task_id).state == "failed"
    assert {task.task_id for task in resumed.tasks if task.state == "succeeded"}.issuperset(succeeded_ids)
    assert len([attempt for attempt in resumed.attempts if attempt.task_id == failed_task.task_id]) == 1
    assert (root / failed_task.attempts[0].path).read_bytes() == failed_bytes

    completed = resumed.retry(failed_only=True)

    retried = next(task for task in completed.tasks if task.task_id == failed_task.task_id)
    lineage = [attempt for attempt in completed.attempts if attempt.task_id == failed_task.task_id]
    assert completed.status == "completed"
    assert [attempt.attempt_number for attempt in lineage] == [1, 2]
    assert lineage[0].attempt_id == failed_attempt.attempt_id
    assert lineage[0].status == "failed"
    assert lineage[1].status == "succeeded"
    assert lineage[0].attempt_id != lineage[1].attempt_id
    assert lineage[1].run_reference is not None
    assert lineage[1].run_reference["run_id"] not in {lineage[0].attempt_id, lineage[1].attempt_id}
    assert retried.selected_success_attempt_id == lineage[1].attempt_id
    assert (root / failed_task.attempts[0].path).read_bytes() == failed_bytes


def test_sa_036_044_051_055_interrupted_attempt_gets_fresh_attempt_and_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    created = vamos.create_study(_spec(seeds=(0,)), output=tmp_path / "study")

    def terminate(_reconstructed: Any, *, root: Path) -> OptimizationResult:
        raise SystemExit(9)

    monkeypatch.setattr(execution, "_execute_optimization", terminate)
    with pytest.raises(SystemExit):
        created.run()
    original = vamos.load_study(created.root).attempts[0]
    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))

    completed = vamos.load_study(created.root).resume()

    assert completed.status == "completed"
    assert [attempt.status for attempt in completed.attempts] == ["interrupted", "succeeded"]
    assert [attempt.attempt_number for attempt in completed.attempts] == [1, 2]
    assert completed.attempts[0].attempt_id == original.attempt_id
    assert len({attempt.attempt_id for attempt in completed.attempts}) == 2
    assert completed.attempts[1].run_reference is not None
    assert completed.attempts[1].run_reference["run_id"] not in {attempt.attempt_id for attempt in completed.attempts}
    assert "attempt_interrupted" in [event.event_type for event in completed.events]


def test_sa_037_reconciles_verified_run_published_before_terminal_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution
    import vamos.experiment.study.run_publication as publication

    root = tmp_path / "study"
    created = vamos.create_study(_spec(seeds=(0,)), output=root)
    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))
    original_verify = publication.verify_run

    def interrupt_verification(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("verification interruption")

    monkeypatch.setattr(publication, "verify_run", interrupt_verification)
    # The interrupted verification surfaces as the study's typed verification error.
    with pytest.raises(StudyRunVerificationError):
        created.run()
    interrupted = vamos.load_study(root)
    expected_run_id = next(event for event in interrupted.events if event.event_type == "task_claimed").payload["run_id"]
    monkeypatch.setattr(publication, "verify_run", original_verify)

    recovered = interrupted.resume()

    assert recovered.status == "completed"
    assert len(recovered.attempts) == 1
    assert recovered.attempts[0].status == "succeeded"
    assert recovered.attempts[0].run_reference is not None
    assert recovered.attempts[0].run_reference["run_id"] == expected_run_id
    assert sum(event.event_type == "attempt_succeeded" for event in recovered.events) == 1


def test_failed_run_recovery_does_not_retry_until_explicit_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution
    import vamos.experiment.study.run_publication as publication

    root = tmp_path / "study"
    created = vamos.create_study(_spec(seeds=(0,)), output=root)
    monkeypatch.setattr(execution, "_execute_optimization", lambda _reconstructed, *, root: (_ for _ in ()).throw(RuntimeError("once")))
    original_append = publication.append_event

    def interrupt_failure_event(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("event_type") == "attempt_failed":
            raise StudyEventAppendError(
                operation="append study event",
                reason="EVENT_APPEND_FAILED",
                expected="terminal failure event",
                actual="injected interruption",
                action="Reconcile the published failed run.",
            )
        return original_append(*args, **kwargs)

    monkeypatch.setattr(publication, "append_event", interrupt_failure_event)
    with pytest.raises(StudyEventAppendError):
        created.run()
    monkeypatch.setattr(publication, "append_event", original_append)
    interrupted = vamos.load_study(root)

    reconciled = interrupted.resume()

    assert reconciled.status == "paused"
    assert len(reconciled.attempts) == 1
    assert reconciled.attempts[0].status == "failed"
    assert reconciled.tasks[0].retryability.retryable is True


def test_sa_046_no_runnable_resume_is_fresh_and_byte_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    created = vamos.create_study(_spec(seeds=(0,)), output=tmp_path / "study")
    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))
    completed = created.run()
    before = _tree(completed.root)

    unchanged = completed.resume()

    assert unchanged is not completed
    assert unchanged.status == "completed"
    assert unchanged.events[-1].file_sha256 == completed.events[-1].file_sha256
    assert _tree(completed.root) == before


def test_sa_052_retry_limit_and_sa_053_nonretryable_refuse_before_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    limited = vamos.create_study(_spec(seeds=(0,), max_attempts=1), output=tmp_path / "limited")
    monkeypatch.setattr(execution, "_execute_optimization", lambda _reconstructed, *, root: (_ for _ in ()).throw(RuntimeError("once")))
    failed = limited.run()
    before = _tree(failed.root)
    with pytest.raises(RetryLimitError, match="RETRY_LIMIT_REACHED"):
        failed.retry()
    assert _tree(failed.root) == before

    configured = vamos.create_study(_spec(seeds=(0,)), output=tmp_path / "configuration")
    original_reconstruct = execution.reconstruct_resolved_run
    monkeypatch.setattr(execution, "reconstruct_resolved_run", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad config")))
    nonretryable = configured.run()
    monkeypatch.setattr(execution, "reconstruct_resolved_run", original_reconstruct)
    before = _tree(nonretryable.root)
    with pytest.raises(RetryNotAllowedError, match="NONRETRYABLE_FAILURE"):
        nonretryable.retry()
    assert _tree(nonretryable.root) == before


def test_sa_047_environment_drift_refuses_before_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution
    import vamos.experiment.study.recovery as recovery

    created = vamos.create_study(_spec(), output=tmp_path / "study")
    calls = 0

    def fail_second(reconstructed: Any, *, root: Path) -> OptimizationResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("transient")
        return _result(reconstructed)

    monkeypatch.setattr(execution, "_execute_optimization", fail_second)
    paused = created.run()
    before = _tree(paused.root)
    monkeypatch.setattr(recovery, "compare_current_environment", lambda _manifest, _environment: SimpleNamespace(level="compatible"))

    with pytest.raises(ResumeEnvironmentIncompatibilityError, match="RESUME_ENVIRONMENT_INCOMPATIBLE"):
        paused.resume()
    assert _tree(paused.root) == before


def _paused_after_two_executions(root: Path, monkeypatch: pytest.MonkeyPatch) -> vamos.Study:
    """Five tasks: the third and fourth objective calls fail, so two executions publish runs and the study pauses."""
    import vamos.experiment.study.execution as execution

    calls = 0

    def fail_third_and_fourth(reconstructed: Any, *, root: Path) -> OptimizationResult:
        nonlocal calls
        calls += 1
        if calls in (3, 4):
            raise RuntimeError("transient")
        return _result(reconstructed)

    monkeypatch.setattr(execution, "_execute_optimization", fail_third_and_fourth)
    return vamos.create_study(_spec(seeds=(0, 1, 2, 3, 4)), output=root).run().resume()


def test_resume_compares_every_earlier_run_with_one_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.artifacts.compatibility as compatibility
    import vamos.experiment.study.recovery as recovery

    paused = _paused_after_two_executions(tmp_path / "study", monkeypatch)
    referenced = {str(attempt.run_reference["run_id"]) for attempt in paused.attempts if attempt.run_reference is not None}
    assert len(referenced) >= 3
    compared: list[str] = []
    captures: list[str] = []
    real_compare = recovery.compare_current_environment
    real_capture = compatibility.capture_runtime_evidence

    def counting_compare(manifest: Any, environment: Any) -> Any:
        compared.append(manifest.run_id)
        return real_compare(manifest, environment)

    def counting_capture(*, backend: str) -> Any:
        captures.append(backend)
        return real_capture(backend=backend)

    monkeypatch.setattr(recovery, "compare_current_environment", counting_compare)
    monkeypatch.setattr(compatibility, "capture_runtime_evidence", counting_capture)

    resumed = paused.resume()

    assert resumed.status == "completed_with_failures"
    assert sorted(compared) == sorted(referenced)
    assert len(captures) == 1


def test_resume_refuses_drift_in_the_backend_of_any_earlier_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.artifacts.compatibility as compatibility
    import vamos.experiment.study.execution as execution

    calls = 0

    def fail_third(reconstructed: Any, *, root: Path) -> OptimizationResult:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("transient")
        return _result(reconstructed)

    monkeypatch.setattr(execution, "_execute_optimization", fail_third)
    paused = vamos.create_study(_spec(seeds=(0, 1, 2, 3)), output=tmp_path / "study").run()
    assert paused.status == "paused" and any(task.state == "pending" for task in paused.tasks)
    referenced = [attempt for attempt in paused.attempts if attempt.run_reference is not None]
    assert len(referenced) >= 2 and len({attempt.execution_id for attempt in referenced}) == 1
    second_run = vamos.load_run(paused.root / "runs" / str(referenced[1].run_reference["run_id"]))
    second_seed = second_run.manifest.resolved_spec["seed"]
    # One execution, two backends: the second run used another backend, whose package changed since.
    real_backend_name = compatibility._backend_name
    real_capture = compatibility.capture_runtime_evidence

    def backend_name(resolved: Any) -> str:
        return "other-backend" if resolved.get("seed") == second_seed else real_backend_name(resolved)

    def capture(*, backend: str) -> Any:
        evidence, environment = real_capture(backend="numpy")
        if backend == "other-backend":
            environment["backend"]["package"] = {**environment["backend"]["package"], "version": "0.0"}
        return evidence, environment

    monkeypatch.setattr(compatibility, "_backend_name", backend_name)
    monkeypatch.setattr(compatibility, "capture_runtime_evidence", capture)
    before = _tree(paused.root)

    with pytest.raises(ResumeEnvironmentIncompatibilityError, match="RESUME_ENVIRONMENT_INCOMPATIBLE"):
        paused.resume()
    assert _tree(paused.root) == before


def test_study_operations_capture_the_current_environment_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.artifacts.compatibility as compatibility
    import vamos.experiment.study.execution as execution

    captures: list[str] = []
    real_capture = compatibility.capture_runtime_evidence

    def counting_capture(*, backend: str) -> Any:
        captures.append(backend)
        return real_capture(backend=backend)

    monkeypatch.setattr(compatibility, "capture_runtime_evidence", counting_capture)
    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))
    completed = vamos.create_study(_spec(), output=tmp_path / "study").run()
    assert completed.status == "completed"
    assert len(captures) == 1

    reference = completed.attempts[0].run_reference
    assert reference is not None
    run_root = (completed.root / str(reference["path"])).parent
    vamos.verify_run(run_root)
    vamos.verify_run(run_root)
    assert len(captures) == 3


def test_relocation_and_same_process_reentry_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    original = tmp_path / "original"
    moved = tmp_path / "elsewhere" / "moved"
    created = vamos.create_study(_spec(), output=original)
    calls = 0

    def fail_second(reconstructed: Any, *, root: Path) -> OptimizationResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("transient")
        return _result(reconstructed)

    monkeypatch.setattr(execution, "_execute_optimization", fail_second)
    paused = created.run()
    moved.parent.mkdir()
    shutil.move(paused.root, moved)
    relocated = vamos.load_study(moved)
    resumed = relocated.resume()
    assert resumed.root == moved.resolve()
    assert resumed.status == "completed_with_failures"

    retried = resumed.retry()
    retried_task = next(task for task in retried.tasks if len(task.attempts) == 2)
    retried_attempt = next(attempt for attempt in retried.attempts if attempt.attempt_id == retried_task.attempts[1].attempt_id)
    assert retried.status == "completed"
    assert [attempt.attempt_number for attempt in retried_task.attempts] == [1, 2]
    assert retried_attempt.run_reference is not None
    assert (moved / "runs" / str(retried_attempt.run_reference["run_id"])).is_dir()

    execution._ACTIVE_ROOTS[retried.root] = None
    try:
        with pytest.raises(UnsupportedStudyExecutionStateError, match="ACTIVE_IN_PROCESS_OWNERSHIP"):
            retried.retry()
    finally:
        execution._ACTIVE_ROOTS.pop(retried.root, None)


def test_orphan_run_never_supplies_attempt_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    created = vamos.create_study(_spec(seeds=(0,)), output=tmp_path / "study")
    monkeypatch.setattr(execution, "_execute_optimization", lambda _reconstructed, *, root: (_ for _ in ()).throw(SystemExit(9)))
    with pytest.raises(SystemExit):
        created.run()
    interrupted = vamos.load_study(created.root)
    claim = next(event for event in interrupted.events if event.event_type == "task_claimed")
    expected = str(claim.payload["run_id"])
    orphan = created.root / "runs" / "11111111-1111-4111-8111-111111111111"
    orphan.mkdir(parents=True)
    (orphan / "manifest.json").write_text("not a canonical run", encoding="utf-8")
    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))

    completed = interrupted.resume()

    assert completed.status == "completed"
    assert completed.attempts[0].status == "interrupted"
    assert completed.attempts[1].status == "succeeded"
    assert not (created.root / "runs" / expected).exists()
    assert orphan.exists()


def test_sa_054_succeeded_task_has_no_force_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    created = vamos.create_study(_spec(seeds=(0,)), output=tmp_path / "study")
    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))
    completed = created.run()
    before = _tree(completed.root)
    with pytest.raises(RetryNotAllowedError, match="TERMINAL_TASK_NOT_RETRYABLE"):
        completed.retry()
    assert _tree(completed.root) == before
