from __future__ import annotations

import collections
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.experiment.artifacts.jsonio import sha256_bytes
from vamos.experiment.optimization_result import OptimizationResult
from vamos.experiment.study.errors import (
    StudyCheckpointError,
    StudyError,
    StudyInfrastructureError,
    StudyIntegrityError,
    StudyRunPublicationError,
    StudyRunVerificationError,
)
from vamos.experiment.study.serialization import seal_document, stored_document_bytes

_PHASES = (
    "before_attempt_record_creation",
    "after_attempt_record_creation",
    "after_running_event_publication",
    "after_task_running_checkpoint",
    "before_objective_evaluation",
    "after_optimization_result_exists",
    "after_canonical_run_publication",
    "after_run_verification",
    "after_terminal_success_event",
    "before_terminal_task_checkpoint",
    "before_terminal_study_checkpoint",
    "before_final_completed_event",
)


def _spec(*, seeds: tuple[int, ...] = (0,)) -> vamos.StudySpec:
    return vamos.StudySpec(
        problems=["zdt1"],
        algorithms=["nsgaii"],
        seeds=seeds,
        max_evaluations=8,
        pop_size=4,
    )


def _result(reconstructed: Any) -> OptimizationResult:
    return OptimizationResult(
        {
            "F": np.zeros((1, reconstructed.n_obj), dtype=np.float64),
            "X": np.zeros((1, reconstructed.n_var), dtype=np.float64),
            "evaluations": 8,
            "generations": 2,
        }
    )


def _snapshot(root: Path) -> dict[str, bytes | None]:
    return {path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None for path in sorted(root.rglob("*"))}


def _raw(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rewrite(path: Path, update: dict[str, Any]) -> None:
    value = _raw(path)
    value.update(update)
    path.write_bytes(stored_document_bytes(seal_document(value)))


def _run_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> vamos.Study:
    import vamos.experiment.study.execution as execution

    tmp_path.mkdir(parents=True, exist_ok=True)

    def execute(reconstructed: Any, *, root: Path) -> OptimizationResult:
        return _result(reconstructed)

    monkeypatch.setattr(execution, "_execute_optimization", execute)
    return vamos.create_study(_spec(), output=tmp_path / "study").run()


@pytest.mark.parametrize("phase", _PHASES)
def test_all_twelve_crash_boundaries_remain_explicit_and_never_fabricate_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    keep = tmp_path / "keep.txt"
    keep.write_text("unrelated", encoding="utf-8")
    created = vamos.create_study(_spec(), output=root)
    source_documents = (root.joinpath("study-spec.json").read_bytes(), root.joinpath("plan.json").read_bytes())

    def execute(reconstructed: Any, *, root: Path) -> OptimizationResult:
        return _result(reconstructed)

    def inject(observed: str) -> None:
        if observed == phase:
            raise OSError(f"injected {phase}")

    monkeypatch.setattr(execution, "_execute_optimization", execute)
    monkeypatch.setattr(execution, "_execution_phase", inject)
    with pytest.raises(StudyError):
        created.run()

    interrupted = vamos.load_study(root)
    before_reload = _snapshot(root)
    reloaded = vamos.load_study(root)
    assert _snapshot(root) == before_reload
    assert reloaded.status == interrupted.status
    assert keep.read_text(encoding="utf-8") == "unrelated"
    assert (root.joinpath("study-spec.json").read_bytes(), root.joinpath("plan.json").read_bytes()) == source_documents
    assert interrupted.status != "completed"
    for attempt in interrupted.attempts:
        if attempt.status == "succeeded":
            assert attempt.run_reference is not None
            run = root / "runs" / str(attempt.run_reference["run_id"])
            assert vamos.load_run(run, verify="all").status == "succeeded"
        elif attempt.run_reference is not None:
            run = root / "runs" / str(attempt.run_reference["run_id"])
            assert vamos.load_run(run, verify="all").status == attempt.status
    success_events = [event for event in interrupted.events if event.event_type == "attempt_succeeded"]
    assert len(success_events) == sum(attempt.status == "succeeded" for attempt in interrupted.attempts)


def test_valid_success_and_infrastructure_events_override_stale_entity_checkpoints_without_loading_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    created = vamos.create_study(_spec(), output=root)

    def execute(reconstructed: Any, *, root: Path) -> OptimizationResult:
        return _result(reconstructed)

    def inject(observed: str) -> None:
        if observed == "after_terminal_success_event":
            raise OSError("stale terminal checkpoints")

    monkeypatch.setattr(execution, "_execute_optimization", execute)
    monkeypatch.setattr(execution, "_execution_phase", inject)
    with pytest.raises(StudyInfrastructureError):
        created.run()

    task_checkpoint = _raw(next(root.glob("tasks/*/task.json")))
    attempt_checkpoint = _raw(next(root.glob("tasks/*/attempts/*.json")))
    manifest_checkpoint = _raw(root / "study-manifest.json")
    assert task_checkpoint["state"] == "running"
    assert attempt_checkpoint["status"] == "running"
    assert manifest_checkpoint["state"] == "failed"
    assert manifest_checkpoint["checkpoint"]["sequence"] == len(list((root / "events").iterdir()))

    before = _snapshot(root)
    effective = vamos.load_study(root)
    assert _snapshot(root) == before
    assert effective.status == "failed"
    assert effective.tasks[0].state == "succeeded"
    assert effective.attempts[0].status == "succeeded"
    assert effective.attempts[0].run_reference is not None
    attempt_reference = effective.tasks[0].attempts[0]
    checkpoint_payload = (root / attempt_reference.path).read_bytes()
    checkpoint_document = _raw(root / attempt_reference.path)
    assert attempt_reference.bytes == len(checkpoint_payload)
    assert attempt_reference.sha256 == sha256_bytes(checkpoint_payload)
    assert attempt_reference.semantic_sha256 == checkpoint_document["integrity"]["document_sha256"]


def test_sa_064_reconstruction_failure_records_failed_run_without_objective(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    created = vamos.create_study(_spec(), output=root)

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("untrusted reconstruction details C:\\private")

    monkeypatch.setattr(execution, "reconstruct_resolved_run", fail)
    returned = created.run()

    failed = vamos.load_study(root)
    assert returned.status == "paused"
    assert failed.status == "paused"
    assert failed.attempts[0].status == "failed"
    assert failed.attempts[0].failure is not None
    assert "private" not in repr(dict(failed.attempts[0].failure))
    reference = failed.attempts[0].run_reference
    assert reference is not None
    assert vamos.load_run(root / "runs" / str(reference["run_id"]), verify="all").status == "failed"
    assert not any(event.event_type == "attempt_succeeded" for event in failed.events)


def test_sa_065_publication_interruption_cannot_fabricate_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution
    import vamos.experiment.study.run_publication as publication

    root = tmp_path / "study"
    created = vamos.create_study(_spec(), output=root)

    def execute(reconstructed: Any, *, root: Path) -> OptimizationResult:
        return _result(reconstructed)

    def fail_publication(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("publication unavailable")

    monkeypatch.setattr(execution, "_execute_optimization", execute)
    monkeypatch.setattr(publication, "save_result", fail_publication)
    with pytest.raises(StudyRunPublicationError) as caught:
        created.run()

    interrupted = vamos.load_study(root)
    assert caught.value.canonical_run_published is False
    assert interrupted.status == "running"
    assert interrupted.tasks[0].state == "running"
    assert interrupted.attempts[0].status == "running"
    assert interrupted.attempts[0].run_reference is None
    assert not any(event.event_type == "attempt_succeeded" for event in interrupted.events)
    assert not root.joinpath("runs").exists()


def test_verification_interruption_leaves_only_an_unreferenced_orphan_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution
    import vamos.experiment.study.run_publication as publication

    root = tmp_path / "study"
    created = vamos.create_study(_spec(), output=root)

    def execute(reconstructed: Any, *, root: Path) -> OptimizationResult:
        return _result(reconstructed)

    def fail_verification(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("verification unavailable")

    monkeypatch.setattr(execution, "_execute_optimization", execute)
    monkeypatch.setattr(publication, "verify_run", fail_verification)
    with pytest.raises(StudyRunVerificationError):
        created.run()

    interrupted = vamos.load_study(root)
    assert interrupted.attempts[0].run_reference is None
    assert not any(event.event_type == "attempt_succeeded" for event in interrupted.events)
    orphan_runs = list(root.glob("runs/*"))
    assert len(orphan_runs) == 1
    assert vamos.load_run(orphan_runs[0], verify="all").status == "succeeded"


def test_checkpoint_ahead_of_journal_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    completed = _run_mocked(tmp_path, monkeypatch)
    path = completed.root / "study-manifest.json"
    raw = _raw(path)
    checkpoint = raw["checkpoint"]
    assert isinstance(checkpoint, dict)
    checkpoint["sequence"] = len(completed.events) + 1
    path.write_bytes(stored_document_bytes(seal_document(raw)))

    with pytest.raises(StudyCheckpointError) as caught:
        vamos.load_study(completed.root)
    assert caught.value.reason == "CHECKPOINT_AHEAD_OF_JOURNAL"


def test_checkpoint_hash_fork_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    completed = _run_mocked(tmp_path, monkeypatch)
    path = completed.root / "study-manifest.json"
    raw = _raw(path)
    checkpoint = raw["checkpoint"]
    assert isinstance(checkpoint, dict)
    checkpoint["event_sha256"] = "0" * 64
    path.write_bytes(stored_document_bytes(seal_document(raw)))

    with pytest.raises(StudyCheckpointError) as caught:
        vamos.load_study(completed.root)
    assert caught.value.reason == "CHECKPOINT_JOURNAL_INCONSISTENCY"


def test_gapped_broken_and_duplicate_event_chains_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gap = _run_mocked(tmp_path / "gap", monkeypatch)
    gap.root.joinpath("events", "00000000000000000002.json").unlink()
    with pytest.raises(StudyIntegrityError, match="EVENT_HASH_CHAIN_BROKEN"):
        vamos.load_study(gap.root)

    monkeypatch.undo()
    broken = _run_mocked(tmp_path / "broken", monkeypatch)
    event_two = broken.root / "events" / "00000000000000000002.json"
    _rewrite(event_two, {"previous_event_sha256": "0" * 64})
    with pytest.raises(StudyIntegrityError, match="EVENT_HASH_CHAIN_BROKEN"):
        vamos.load_study(broken.root)

    monkeypatch.undo()
    duplicate = _run_mocked(tmp_path / "duplicate", monkeypatch)
    first = _raw(duplicate.root / "events" / "00000000000000000001.json")
    second_path = duplicate.root / "events" / "00000000000000000002.json"
    _rewrite(second_path, {"event_id": first["event_id"]})
    with pytest.raises(StudyIntegrityError, match="EVENT_HASH_CHAIN_BROKEN"):
        vamos.load_study(duplicate.root)


def _crash_after_run_publication(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave a two-task study ``running`` with an unfinished attempt, as a killed process does."""
    import vamos.experiment.study.execution as execution

    created = vamos.create_study(_spec(seeds=(0, 1)), output=root)

    def execute(reconstructed: Any, *, root: Path) -> OptimizationResult:
        return _result(reconstructed)

    def die(observed: str) -> None:
        # SystemExit escapes the runner's handlers, so nothing is recorded after it.
        if observed == "after_canonical_run_publication":
            raise SystemExit("injected process death")

    monkeypatch.setattr(execution, "_execute_optimization", execute)
    monkeypatch.setattr(execution, "_execution_phase", die)
    with pytest.raises(SystemExit):
        created.run()
    monkeypatch.setattr(execution, "_execution_phase", lambda observed: None)


@pytest.mark.parametrize("location", ["events", "task", "attempts", "root"])
def test_orphan_writer_temporary_does_not_block_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, location: str) -> None:
    root = tmp_path / "study"
    _crash_after_run_publication(root, monkeypatch)
    digest = vamos.load_study(root).attempts[0].task_id.removeprefix("sha256:")
    directory = {
        "events": root / "events",
        "task": root / "tasks" / digest,
        "attempts": root / "tasks" / digest / "attempts",
        "root": root,
    }[location]
    # The process died between creating its temporary and replacing the destination.
    orphan = directory / ".tmp-0123abcd"
    orphan.write_bytes(b'{"partial":')

    loaded = vamos.load_study(root)

    assert loaded.status == "running"
    assert orphan.exists(), "loading must stay write-free"
    resumed = loaded.resume()
    assert resumed.status == "completed"
    assert [task.state for task in resumed.tasks] == ["succeeded", "succeeded"]
    assert not orphan.exists(), "reconciliation removes the writer's orphaned temporaries"


def test_reader_names_outside_the_writer_pattern_still_break_the_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "study"
    _crash_after_run_publication(root, monkeypatch)
    digest = vamos.load_study(root).attempts[0].task_id.removeprefix("sha256:")
    (root / "tasks" / digest / ".tmp-not-hex!").write_bytes(b"")

    with pytest.raises(StudyIntegrityError, match="UNEXPECTED_LAYOUT"):
        vamos.load_study(root)


def test_atomic_replace_retries_sharing_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vamos.experiment.study import writing

    real_replace = os.replace
    calls: list[str] = []

    def busy_then_free(source: Any, target: Any) -> None:
        calls.append(os.fspath(target))
        if len(calls) <= 2:
            # What Windows raises while another process has the destination open.
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        real_replace(source, target)

    monkeypatch.setattr(writing, "_RETRY_SHARING_VIOLATIONS", True, raising=False)
    monkeypatch.setattr(os, "replace", busy_then_free)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    target = tmp_path / "study-manifest.json"

    writing.write_bytes_atomic(target, b"published")

    assert target.read_bytes() == b"published"
    assert len(calls) == 3
    assert [path.name for path in tmp_path.iterdir()] == ["study-manifest.json"]


def test_atomic_replace_gives_up_after_bounded_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vamos.experiment.study import writing

    calls: list[str] = []

    def always_busy(source: Any, target: Any) -> None:
        calls.append(os.fspath(target))
        raise PermissionError(13, "busy")

    monkeypatch.setattr(writing, "_RETRY_SHARING_VIOLATIONS", True, raising=False)
    monkeypatch.setattr(os, "replace", always_busy)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    with pytest.raises(PermissionError):
        writing.write_bytes_atomic(tmp_path / "task.json", b"x")
    assert 1 < len(calls) <= 50
    assert list(tmp_path.iterdir()) == []

    calls.clear()
    monkeypatch.setattr(writing, "_RETRY_SHARING_VIOLATIONS", False)
    with pytest.raises(PermissionError):
        writing.write_bytes_atomic(tmp_path / "task.json", b"x")
    assert len(calls) == 1, "outside Windows a PermissionError is a real permission problem"


def test_loading_reads_each_study_document_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _run_mocked(tmp_path, monkeypatch).root
    real_read_bytes = Path.read_bytes
    reads: collections.Counter[str] = collections.Counter()

    def counting_read_bytes(self: Path) -> bytes:
        reads[self.resolve().relative_to(root.resolve()).as_posix()] += 1
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    vamos.load_study(root)

    study_documents = {name: count for name, count in reads.items() if not name.startswith("runs/")}
    assert "study-manifest.json" in study_documents
    # One read per document: a second read could see a newer replacement than the one parsed.
    assert set(study_documents.values()) == {1}, study_documents


def test_document_read_retries_sharing_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vamos.experiment.study import writing

    root = _run_mocked(tmp_path, monkeypatch).root
    real_read_bytes = Path.read_bytes
    refused: list[str] = []

    def replaced_meanwhile(self: Path) -> bytes:
        # Windows refuses to open a document for a moment while its owner replaces it.
        if self.name == "study-manifest.json" and len(refused) < 2:
            refused.append(self.name)
            raise PermissionError(13, "Access is denied")
        return real_read_bytes(self)

    monkeypatch.setattr(writing, "_RETRY_SHARING_VIOLATIONS", True)
    monkeypatch.setattr(Path, "read_bytes", replaced_meanwhile)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    assert vamos.load_study(root).status == "completed"
    assert len(refused) == 2
