from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.experiment.artifacts.jsonio import canonical_json_bytes, sha256_bytes
from vamos.experiment.optimization_result import OptimizationResult
from vamos.experiment.study.errors import (
    PlanMismatchError,
    ReferencedRunCorruptError,
    ReferencedRunMissingError,
    ResumeEnvironmentIncompatibilityError,
    StudyEventAppendError,
    StudyInfrastructureError,
    StudyRunVerificationError,
)
from vamos.experiment.study.serialization import seal_document, stored_document_bytes


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


def _raw(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_reconcile_refreshes_lagging_checkpoints_once_and_is_then_byte_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution
    from vamos.experiment.study.reconciliation import reconcile_study

    root = tmp_path / "study"
    created = vamos.create_study(_spec(seeds=(0,)), output=root)
    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))

    lagging_manifest: bytes | None = None

    def interrupt(phase: str) -> None:
        nonlocal lagging_manifest
        if phase == "after_terminal_success_event":
            lagging_manifest = (root / "study-manifest.json").read_bytes()
            raise OSError("leave entity checkpoints behind the journal")

    monkeypatch.setattr(execution, "_execution_phase", interrupt)
    with pytest.raises(StudyInfrastructureError):
        created.run()
    assert lagging_manifest is not None
    (root / "study-manifest.json").write_bytes(lagging_manifest)
    effective = vamos.load_study(root)
    task_path = next(root.glob("tasks/*/task.json"))
    attempt_path = next(root.glob("tasks/*/attempts/*.json"))
    before_task = task_path.read_bytes()
    before_attempt = attempt_path.read_bytes()
    before_manifest = (root / "study-manifest.json").read_bytes()

    reconciled = reconcile_study(effective)

    assert reconciled.tasks[0].state == "succeeded"
    assert reconciled.attempts[0].status == "succeeded"
    assert task_path.read_bytes() != before_task
    assert attempt_path.read_bytes() != before_attempt
    assert (root / "study-manifest.json").read_bytes() != before_manifest
    assert _raw(task_path)["state"] == "succeeded"
    assert _raw(attempt_path)["status"] == "succeeded"
    manifest = _raw(root / "study-manifest.json")
    task = _raw(task_path)
    assert manifest["counts"]["succeeded"] == 1
    assert manifest["checkpoint"]["sequence"] == len(reconciled.events)
    assert task["attempts"][0]["sha256"] == sha256_bytes(attempt_path.read_bytes())
    stable = _tree(root)
    again = reconcile_study(reconciled)
    assert again is reconciled
    assert _tree(root) == stable


def test_created_attempt_before_claim_is_consumed_once_then_retry_uses_attempt_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    created = vamos.create_study(_spec(seeds=(0,)), output=tmp_path / "study")

    def terminate(phase: str) -> None:
        if phase == "after_attempt_record_creation":
            raise SystemExit(17)

    monkeypatch.setattr(execution, "_execution_phase", terminate)
    with pytest.raises(SystemExit):
        created.run()
    orphan_created = vamos.load_study(created.root).attempts[0]
    assert orphan_created.status == "created"
    assert orphan_created.attempt_number == 1

    monkeypatch.setattr(execution, "_execution_phase", lambda _phase: None)
    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))
    completed = vamos.load_study(created.root).resume()

    assert completed.status == "completed"
    assert [attempt.attempt_number for attempt in completed.attempts] == [1, 2]
    assert [attempt.status for attempt in completed.attempts] == ["interrupted", "succeeded"]
    assert completed.attempts[0].attempt_id == orphan_created.attempt_id
    claims = [event for event in completed.events if event.event_type == "task_claimed"]
    assert [event.payload["attempt_id"] for event in claims] == [attempt.attempt_id for attempt in completed.attempts]
    assert all(uuid.UUID(str(event.payload["run_id"])).version == 4 for event in claims)
    assert len({str(event.payload["run_id"]) for event in claims}) == 2
    assert completed.tasks[0].retryability.attempts_remaining == 1


def test_corrupt_expected_published_run_refuses_reconciliation_without_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution
    import vamos.experiment.study.run_publication as publication

    root = tmp_path / "study"
    created = vamos.create_study(_spec(seeds=(0,)), output=root)
    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))
    original_verify = publication.verify_run
    monkeypatch.setattr(publication, "verify_run", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("stop")))
    # The interrupted verification surfaces as the study's typed verification error.
    with pytest.raises(StudyRunVerificationError):
        created.run()
    interrupted = vamos.load_study(root)
    run_id = str(next(event for event in interrupted.events if event.event_type == "task_claimed").payload["run_id"])
    monkeypatch.setattr(publication, "verify_run", original_verify)
    (root / "runs" / run_id / "manifest.json").write_bytes(b"not canonical json\n")
    before = _tree(root)

    with pytest.raises(ReferencedRunCorruptError, match="REFERENCED_RUN_CORRUPT"):
        interrupted.resume()
    assert _tree(root) == before


def test_missing_and_corrupt_selected_runs_have_distinct_typed_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    source = tmp_path / "source"
    created = vamos.create_study(_spec(seeds=(0,)), output=source)
    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))
    completed = created.run()
    run_id = str(completed.attempts[0].run_reference["run_id"])

    missing_root = tmp_path / "missing"
    shutil.copytree(source, missing_root)
    missing = vamos.load_study(missing_root)
    (missing_root / "runs" / run_id / "manifest.json").unlink()
    missing_before = _tree(missing_root)
    with pytest.raises(ReferencedRunMissingError, match="REFERENCED_RUN_MISSING"):
        missing.resume()
    assert _tree(missing_root) == missing_before

    corrupt_root = tmp_path / "corrupt"
    shutil.copytree(source, corrupt_root)
    corrupt = vamos.load_study(corrupt_root)
    (corrupt_root / "runs" / run_id / "manifest.json").write_bytes(b"corrupt\n")
    corrupt_before = _tree(corrupt_root)
    with pytest.raises(ReferencedRunCorruptError, match="REFERENCED_RUN_CORRUPT"):
        corrupt.resume()
    assert _tree(corrupt_root) == corrupt_before


def test_resume_rejects_plan_mismatch_before_any_recovery_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    created = vamos.create_study(_spec(seeds=(0, 1)), output=tmp_path / "study")
    monkeypatch.setattr(
        execution,
        "_execute_optimization",
        lambda _reconstructed, *, root: (_ for _ in ()).throw(RuntimeError("pause")),
    )
    paused = created.run()
    plan_path = paused.root / "plan.json"
    raw = _raw(plan_path)
    raw["tasks"] = []
    raw["task_count"] = 0
    plan_path.write_bytes(stored_document_bytes(seal_document(raw)))
    before = _tree(paused.root)

    with pytest.raises(PlanMismatchError, match="PLAN_MISMATCH"):
        paused.resume()
    assert _tree(paused.root) == before


def test_resume_retry_failed_is_explicit_and_continue_retry_never_reruns_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    direct = vamos.create_study(_spec(seeds=(0,)), output=tmp_path / "direct")
    monkeypatch.setattr(
        execution,
        "_execute_optimization",
        lambda _reconstructed, *, root: (_ for _ in ()).throw(RuntimeError("once")),
    )
    paused = direct.run()
    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))
    assert paused.resume(retry_failed=True).status == "completed"

    observed: list[int] = []
    failed_once = False

    def fail_seed_one_once(reconstructed: Any, *, root: Path) -> OptimizationResult:
        nonlocal failed_once
        observed.append(reconstructed.seed)
        if reconstructed.seed == 1 and not failed_once:
            failed_once = True
            raise RuntimeError("retryable")
        return _result(reconstructed)

    continued = vamos.create_study(_spec(policy="continue"), output=tmp_path / "continued")
    monkeypatch.setattr(execution, "_execute_optimization", fail_seed_one_once)
    partial = continued.run()
    retained = {
        task.task_id: (task.attempts[0].attempt_id, (partial.root / task.attempts[0].path).read_bytes())
        for task in partial.tasks
        if task.state == "succeeded"
    }
    observed.clear()
    completed = partial.retry()

    assert completed.status == "completed"
    assert observed == [1]
    for task_id, (attempt_id, payload) in retained.items():
        task = next(item for item in completed.tasks if item.task_id == task_id)
        assert len(task.attempts) == 1
        assert task.attempts[0].attempt_id == attempt_id
        assert (completed.root / task.attempts[0].path).read_bytes() == payload


def test_recovered_failure_pause_names_the_trigger_with_an_older_failure_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution
    import vamos.experiment.study.run_publication as publication

    root = tmp_path / "study"
    created = vamos.create_study(_spec(), output=root)
    call_count = 0

    def fail_second(reconstructed: Any, *, root: Path) -> OptimizationResult:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("older failure")
        return _result(reconstructed)

    monkeypatch.setattr(execution, "_execute_optimization", fail_second)
    paused = created.run()
    older_failed_id = next(task.task_id for task in paused.tasks if task.state == "failed")
    trigger_id = next(task.task_id for task in paused.tasks if task.state == "pending")
    monkeypatch.setattr(
        execution,
        "_execute_optimization",
        lambda _reconstructed, *, root: (_ for _ in ()).throw(RuntimeError("recovered failure")),
    )
    original_append = publication.append_event

    def stop_terminal_event(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("event_type") == "attempt_failed":
            raise StudyEventAppendError(
                operation="append study event",
                reason="EVENT_APPEND_FAILED",
                expected="terminal event",
                actual="injected stop",
                action="Reconcile the exact published run.",
            )
        return original_append(*args, **kwargs)

    monkeypatch.setattr(publication, "append_event", stop_terminal_event)
    with pytest.raises(StudyEventAppendError):
        paused.resume()
    running = vamos.load_study(root)
    trigger_attempt = next(attempt for attempt in running.attempts if attempt.task_id == trigger_id and attempt.status == "running")
    monkeypatch.setattr(publication, "append_event", original_append)

    reconciled = running.resume()

    assert older_failed_id != trigger_id
    assert reconciled.status == "paused"
    pause = reconciled.events[-1]
    assert pause.event_type == "study_paused"
    assert pause.payload == {"failed_task_id": trigger_id, "failed_attempt_id": trigger_attempt.attempt_id}


def test_persisted_resolved_defaults_are_reused_after_current_defaults_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution
    import vamos.experiment.study.planning as planning
    import vamos.experiment.study.recovery as recovery

    created = vamos.create_study(
        vamos.StudySpec(problems=["zdt1"], algorithms=["auto"], seeds=[0, 1, 2], max_attempts_per_task=3),
        output=tmp_path / "study",
    )
    calls = 0

    def fail_second(reconstructed: Any, *, root: Path) -> OptimizationResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("pause")
        return _result(reconstructed)

    monkeypatch.setattr(execution, "_execute_optimization", fail_second)
    paused = created.run()
    expected = {
        canonical_json_bytes(task.resolved_run_spec)
        for task in paused.plan.tasks
        if task.task_id in {t.task_id for t in paused.tasks if t.state == "pending"}
    }
    original_reconstruct = recovery.reconstruct_resolved_run
    observed: list[bytes] = []

    def reconstruct(persisted: Any, *, root: Path) -> Any:
        observed.append(canonical_json_bytes(persisted))
        return original_reconstruct(persisted, root=root)

    monkeypatch.setattr(planning, "_compute_pop_size", lambda *_args: 999)
    monkeypatch.setattr(planning, "_select_algorithm", lambda *_args: "spea2")
    monkeypatch.setattr(planning, "resolve_spec", lambda *_args: (_ for _ in ()).throw(AssertionError("no replanning")))
    monkeypatch.setattr(recovery, "reconstruct_resolved_run", reconstruct)
    monkeypatch.setattr(execution, "reconstruct_resolved_run", reconstruct)
    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))

    resumed = paused.resume()

    assert resumed.status == "completed_with_failures"
    assert set(observed) == expected
    assert len(observed) == 2 * len(expected)


def test_unavailable_persisted_component_refuses_before_claim_with_action(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution
    import vamos.experiment.study.recovery as recovery

    created = vamos.create_study(_spec(seeds=(0, 1)), output=tmp_path / "study")
    calls = 0

    def fail_second(reconstructed: Any, *, root: Path) -> OptimizationResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("leave failure")
        return _result(reconstructed)

    monkeypatch.setattr(execution, "_execute_optimization", fail_second)
    partial = created.run().resume()
    before = _tree(partial.root)
    monkeypatch.setattr(
        recovery,
        "reconstruct_resolved_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LookupError("backend unavailable")),
    )

    with pytest.raises(ResumeEnvironmentIncompatibilityError) as caught:
        partial.retry()
    assert caught.value.reason == "RESUME_ENVIRONMENT_INCOMPATIBLE"
    assert "Restore" in caught.value.action
    assert _tree(partial.root) == before
