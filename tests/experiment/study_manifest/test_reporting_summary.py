from __future__ import annotations

import json
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.experiment.optimization_result import OptimizationResult
from vamos.experiment.study.errors import StudyInfrastructureError, StudyRunPublicationError
from vamos.experiment.study.reconciliation import reconcile_study


def _spec(*, seeds: tuple[int, ...] = (3,), policy: str = "fail_fast") -> vamos.StudySpec:
    return vamos.StudySpec(
        problems=["zdt1"],
        algorithms=["nsgaii"],
        seeds=seeds,
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
            "metrics": {"score": float(reconstructed.seed)},
        }
    )


def _tree(root: Path) -> dict[str, bytes | None]:
    return {item.relative_to(root).as_posix(): item.read_bytes() if item.is_file() else None for item in sorted(root.rglob("*"))}


def _complete(root: Path, monkeypatch: pytest.MonkeyPatch, *, seeds: tuple[int, ...] = (3,)) -> vamos.Study:
    import vamos.experiment.study.execution as execution

    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))
    return vamos.create_study(_spec(seeds=seeds), output=root).run()


def test_sa_067_created_projection_is_deterministic_immutable_relocatable_and_write_free(tmp_path: Path) -> None:
    root = tmp_path / "source" / "study"
    created = vamos.create_study(_spec(seeds=(7, 1)), output=root)
    moved = tmp_path / "relocated" / "renamed-with-misleading-components"
    moved.parent.mkdir()
    shutil.move(root, moved)
    relocated = vamos.load_study(moved)
    (moved / "runs" / "not-canonical-evidence").mkdir(parents=True)
    before = _tree(moved)

    first_report = relocated.inspect()
    first_summary = relocated.summarize()
    second_report = relocated.inspect()
    second_summary = relocated.summarize()

    assert _tree(moved) == before
    assert first_report == second_report
    assert first_summary == second_summary
    assert first_report.changed is False
    assert first_report.runnable_work is True
    assert first_report.retryable_failed_work is False
    assert first_report.total_attempt_count == 0
    assert first_report.counts == {
        "tasks": 2,
        "pending": 2,
        "running": 0,
        "succeeded": 0,
        "failed": 0,
        "interrupted": 0,
        "cancelled": 0,
        "skipped": 0,
    }
    assert first_report.next_actions == ("run",)
    assert [row.plan_index for row in first_summary.rows] == [0, 1]
    ordered_tasks = sorted(relocated.plan.tasks, key=lambda task: task.plan_index)
    assert [row.seed for row in first_summary.rows] == [int(task.resolved_run_spec["seed"]) for task in ordered_tasks]
    assert all(row.problem_id and "zdt1" in row.problem_id for row in first_summary.rows)
    assert all(row.run_metadata_available is False and row.metrics is None for row in first_summary.rows)
    assert json.loads(json.dumps(first_report.as_dict()))["study_id"] == created.study_id
    assert json.loads(json.dumps(first_summary.as_dict()))["plan_id"] == created.plan_id
    with pytest.raises(FrozenInstanceError):
        first_report.changed = True  # type: ignore[misc]
    with pytest.raises(TypeError):
        first_report.counts["tasks"] = 0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        first_summary.rows[0].state = "failed"  # type: ignore[misc]


def test_inspect_reloads_current_state_without_objective_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    snapshot = vamos.create_study(_spec(), output=tmp_path / "study")
    cancelled = vamos.load_study(snapshot.root).cancel()

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("inspection must not evaluate an objective")

    monkeypatch.setattr(execution, "_execute_optimization", forbidden)
    before = _tree(snapshot.root)
    report = snapshot.inspect()

    assert _tree(snapshot.root) == before
    assert snapshot.status == "created"
    assert cancelled.status == report.state == "cancelled"
    assert report.counts["cancelled"] == 1


def test_sa_073_summary_uses_verified_manifest_metadata_without_materializing_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vamos.experiment.artifacts.reader as reader

    completed = _complete(tmp_path / "study", monkeypatch, seeds=(0, 2))
    before = _tree(completed.root)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("F/X materialization is forbidden during study projection")

    monkeypatch.setattr(reader, "load_result_bundle", forbidden)
    report = completed.inspect()
    summary = completed.summarize()

    assert _tree(completed.root) == before
    assert report.state == "completed"
    assert report.verified_run_count == 2
    assert report.issues == ()
    assert report.total_attempt_count == 2
    assert len(summary.rows) == 2
    tasks_by_id = {item.task_id: item for item in completed.tasks}
    attempts_by_id = {item.attempt_id: item for item in completed.attempts}
    for row in summary.rows:
        task = tasks_by_id[row.task_id]
        selected_attempt_id = row.selected_attempt_id
        assert selected_attempt_id is not None
        attempt = attempts_by_id[selected_attempt_id]
        reference = attempt.run_reference
        assert reference is not None
        assert row.task_id == task.task_id
        assert row.selected_attempt_id == attempt.attempt_id
        assert row.selected_run_id == reference["run_id"]
        assert row.evidence_run_id == reference["run_id"]
        assert row.run_manifest_path == reference["path"]
        assert row.run_status == "succeeded"
        assert row.evaluations == 8
        assert row.termination_reason is not None
        assert row.runtime_ms is not None
        assert row.run_metadata_available is True
        assert row.metrics is not None
        assert row.run_manifest_sha256 == reference["semantic_sha256"]


@pytest.mark.parametrize("mutation", ["missing_manifest", "corrupt_result"])
def test_inspect_reports_missing_or_corrupt_run_evidence_without_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    completed = _complete(tmp_path / mutation, monkeypatch)
    reference = completed.attempts[0].run_reference
    assert reference is not None
    run_root = completed.root / "runs" / str(reference["run_id"])
    if mutation == "missing_manifest":
        (run_root / "manifest.json").unlink()
    else:
        result_path = run_root / "result.npz"
        result_path.write_bytes(result_path.read_bytes() + b"corrupt")
    before = _tree(completed.root)

    report = completed.inspect()
    summary = completed.summarize()

    assert _tree(completed.root) == before
    assert report.changed is False
    assert report.verified_run_count == 0
    assert len(report.issues) == 1
    assert report.issues[0].attempt_id == completed.attempts[0].attempt_id
    assert report.issues[0].reason in {"REFERENCED_RUN_MISSING", "REFERENCED_RUN_CORRUPT"}
    assert report.next_actions[0] == "restore_referenced_run_evidence"
    assert summary.rows[0].run_metadata_available is False
    assert summary.rows[0].metrics is None
    assert len(summary.issues) == 1


def test_report_distinguishes_every_supported_study_and_task_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution
    import vamos.experiment.study.run_publication as publication

    original_save_result = publication.save_result
    created = vamos.create_study(_spec(), output=tmp_path / "created")
    cancelled = vamos.create_study(_spec(), output=tmp_path / "cancelled").cancel()

    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))
    completed = vamos.create_study(_spec(), output=tmp_path / "completed").run()

    def task_failure(_reconstructed: Any, *, root: Path) -> OptimizationResult:
        raise RuntimeError("bounded task failure")

    monkeypatch.setattr(execution, "_execute_optimization", task_failure)
    paused = vamos.create_study(_spec(), output=tmp_path / "paused").run()
    partial = vamos.create_study(_spec(policy="continue"), output=tmp_path / "partial").run()

    original_phase = execution._execution_phase

    def infrastructure_failure(phase: str) -> None:
        if phase == "before_attempt_record_creation":
            raise OSError("infrastructure")

    monkeypatch.setattr(execution, "_execution_phase", infrastructure_failure)
    failed_snapshot = vamos.create_study(_spec(), output=tmp_path / "failed")
    with pytest.raises(StudyInfrastructureError):
        failed_snapshot.run()
    failed = vamos.load_study(failed_snapshot.root)
    monkeypatch.setattr(execution, "_execution_phase", original_phase)

    def publication_failure(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("storage")

    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))
    monkeypatch.setattr(publication, "save_result", publication_failure)
    running_snapshot = vamos.create_study(_spec(), output=tmp_path / "running")
    with pytest.raises(StudyRunPublicationError):
        running_snapshot.run()
    running = vamos.load_study(running_snapshot.root)

    interrupted_snapshot = vamos.create_study(_spec(), output=tmp_path / "interrupted")

    def process_exit(_reconstructed: Any, *, root: Path) -> OptimizationResult:
        raise SystemExit(9)

    monkeypatch.setattr(publication, "save_result", original_save_result)
    monkeypatch.setattr(execution, "_execute_optimization", process_exit)
    with pytest.raises(SystemExit):
        interrupted_snapshot.run()
    interrupted = reconcile_study(vamos.load_study(interrupted_snapshot.root))

    states = {
        "created": created.inspect(),
        "cancelled": cancelled.inspect(),
        "completed": completed.inspect(),
        "paused": paused.inspect(),
        "completed_with_failures": partial.inspect(),
        "failed": failed.inspect(),
        "running": running.inspect(),
    }
    assert set(states) == {report.state for report in states.values()}
    assert states["created"].runnable_work is True
    assert states["paused"].retryable_failed_work is True
    assert states["completed_with_failures"].retryable_failed_work is True
    assert states["failed"].next_actions == ("create_new_study",)
    assert "wait_for_current_owner_or_resume_after_interruption" in states["running"].next_actions
    interrupted_report = interrupted.inspect()
    assert interrupted_report.state == "paused"
    assert interrupted_report.counts["interrupted"] == 1
    assert interrupted_report.runnable_work is True


def test_journal_newer_than_checkpoint_is_reported_without_reconciliation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    created = vamos.create_study(_spec(), output=tmp_path / "study")

    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))

    def crash_after_event(phase: str) -> None:
        if phase == "after_terminal_success_event":
            raise SystemExit(7)

    monkeypatch.setattr(execution, "_execution_phase", crash_after_event)
    with pytest.raises(SystemExit):
        created.run()
    current = vamos.load_study(created.root)
    before = _tree(created.root)

    report = current.inspect()

    assert _tree(created.root) == before
    assert report.reconciliation_required is True
    assert report.journal_checkpoint_relation == "journal_ahead"
    assert report.changed is False
    assert report.next_actions[0] == "resume_to_reconcile"


@pytest.mark.parametrize("mutation", ["missing_manifest", "corrupt_result"])
def test_inspect_reports_corrupt_run_instead_of_failing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    from vamos.experiment.study.errors import StudyError

    completed = _complete(tmp_path / mutation, monkeypatch)
    reference = completed.attempts[0].run_reference
    assert reference is not None
    run_root = completed.root / "runs" / str(reference["run_id"])
    if mutation == "missing_manifest":
        (run_root / "manifest.json").unlink()
    else:
        result_path = run_root / "result.npz"
        result_path.write_bytes(result_path.read_bytes() + b"corrupt")
    with pytest.raises(StudyError):
        vamos.load_study(completed.root)
    before = _tree(completed.root)

    loaded = vamos.load_study(completed.root, verify="metadata")
    report = loaded.inspect()
    summary = loaded.summarize()

    assert _tree(completed.root) == before
    assert loaded.status == "completed"
    assert len(report.issues) == 1
    assert report.issues[0].reason in {"REFERENCED_RUN_MISSING", "REFERENCED_RUN_CORRUPT"}
    assert summary.rows[0].run_metadata_available is False
    # Running work still verifies every run.
    with pytest.raises(StudyError):
        loaded.resume()
    with pytest.raises(ValueError, match="verify must be"):
        vamos.load_study(completed.root, verify="none")  # type: ignore[arg-type]
