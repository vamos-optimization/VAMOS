from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import vamos
from vamos.experiment.study.creation import publish_study
from vamos.experiment.study.decoding import decode_spec
from vamos.experiment.study.errors import (
    MalformedStudyError,
    PlanMismatchError,
    StudyInfrastructureError,
    StudyOutputCollisionError,
    UnsupportedStudySchemaError,
)
from vamos.experiment.study.paths import validate_study_relative_path
from vamos.experiment.study.planning import resolve_spec
from vamos.experiment.study.serialization import seal_document, stored_document_bytes
from vamos.study_artifacts import Study


def _spec(*, empty: bool = False) -> vamos.StudySpec:
    return vamos.StudySpec(
        problems=[] if empty else ["zdt1"],
        algorithms=[] if empty else ["nsgaii"],
        seeds=[] if empty else [0],
        max_evaluations=24,
        pop_size=8,
    )


def _tree_bytes(root: Path) -> dict[str, bytes | None]:
    return {path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None for path in sorted(root.rglob("*"))}


def _load_raw(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_sa_002_empty_study_has_manifest_spec_plan_event_and_no_task_tree(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    study = vamos.create_study(_spec(empty=True), output=root)

    assert study.status == "created"
    assert study.manifest.counts.tasks == 0
    assert {path.relative_to(root).as_posix() for path in root.rglob("*")} == {
        "events",
        "events/00000000000000000001.json",
        "plan.json",
        "study-manifest.json",
        "study-spec.json",
    }


def test_sa_008_modified_plan_is_rejected_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "study"
    vamos.create_study(_spec(), output=root)
    plan_path = root / "plan.json"
    raw = _load_raw(plan_path)
    tasks = raw["tasks"]
    assert isinstance(tasks, list)
    tasks.clear()
    raw["task_count"] = 0
    plan_path.write_bytes(stored_document_bytes(seal_document(raw)))
    before = _tree_bytes(root)

    with pytest.raises(PlanMismatchError, match="PLAN_MISMATCH"):
        vamos.load_study(root)
    assert _tree_bytes(root) == before


def test_sa_010_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "study"
    vamos.create_study(_spec(), output=root)
    (root / "study-manifest.json").write_bytes(b'{"document_type":"a","document_type":"b"}\n')

    with pytest.raises(MalformedStudyError) as caught:
        vamos.load_study(root)
    assert caught.value.reason == "DUPLICATE_JSON_KEY"


def test_sa_011_unknown_document_field_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "study"
    vamos.create_study(_spec(), output=root)
    raw = _load_raw(root / "study-spec.json")
    raw["surprise"] = True

    with pytest.raises(MalformedStudyError) as caught:
        decode_spec(raw)
    assert caught.value.reason == "UNKNOWN_FIELD"


def test_sa_013_future_schema_is_rejected_actionably(tmp_path: Path) -> None:
    root = tmp_path / "study"
    vamos.create_study(_spec(), output=root)
    manifest_path = root / "study-manifest.json"
    raw = _load_raw(manifest_path)
    raw["schema_version"] = "2.0.0"
    manifest_path.write_bytes(stored_document_bytes(seal_document(raw)))

    with pytest.raises(UnsupportedStudySchemaError) as caught:
        vamos.load_study(root)
    assert caught.value.reason == "UNSUPPORTED_SCHEMA"
    assert "Regenerate" in caught.value.action


def test_sa_014_uppercase_uuid_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "study"
    study = vamos.create_study(_spec(), output=root)
    manifest_path = root / "study-manifest.json"
    raw = _load_raw(manifest_path)
    raw["study_id"] = study.study_id.upper()
    manifest_path.write_bytes(stored_document_bytes(seal_document(raw)))

    with pytest.raises(MalformedStudyError) as caught:
        vamos.load_study(root)
    assert caught.value.reason == "INVALID_IDENTITY"


@pytest.mark.parametrize(
    "unsafe",
    ["/absolute", "C:/drive", "//server/share", "https://example.test/a", "../escape", "a/../b", "a\\b", "a//b", "a%2fb", "a\x00b", ""],
)
def test_sa_015_unsafe_path_classes_are_rejected(unsafe: str) -> None:
    with pytest.raises(Exception) as caught:
        validate_study_relative_path(unsafe, role="test")
    assert getattr(caught.value, "reason", None) == "UNSAFE_PATH"


@pytest.mark.parametrize("kind", ["empty_directory", "valid_study", "unrelated", "partial", "file"])
def test_sa_016_every_existing_destination_collides_without_mutation(tmp_path: Path, kind: str) -> None:
    root = tmp_path / kind
    if kind == "valid_study":
        vamos.create_study(_spec(), output=root)
    elif kind == "file":
        root.write_text("occupied", encoding="utf-8")
    else:
        root.mkdir()
        if kind == "unrelated":
            (root / "keep.txt").write_text("keep", encoding="utf-8")
        elif kind == "partial":
            (root / "study-spec.json").write_text("partial", encoding="utf-8")
    before = root.read_bytes() if root.is_file() else _tree_bytes(root)

    with pytest.raises(StudyOutputCollisionError) as caught:
        vamos.create_study(_spec(), output=root)

    after = root.read_bytes() if root.is_file() else _tree_bytes(root)
    assert caught.value.reason == "OUTPUT_COLLISION"
    assert caught.value.execution_occurred is False
    assert after == before


@pytest.mark.parametrize(
    "phase",
    [
        "staging_created",
        "spec_written",
        "plan_written",
        "tasks_written",
        "event_written",
        "manifest_written",
        "documents_written",
        "staging_verified",
        "before_publish",
    ],
)
def test_sa_017_failure_before_rename_leaves_no_destination_or_staging(tmp_path: Path, phase: str) -> None:
    destination = tmp_path / "study"
    spec = _spec()
    plan = resolve_spec(spec)

    def fail_at(observed: str) -> None:
        if observed == phase:
            raise OSError(f"injected at {phase}")

    with pytest.raises(StudyInfrastructureError) as caught:
        publish_study(spec, plan=plan, destination=destination, phase_hook=fail_at)

    assert caught.value.reason == "ATOMIC_PUBLICATION_FAILED"
    assert caught.value.published is False
    assert not destination.exists()
    assert list(tmp_path.glob(".study.vamos-study-staging-*")) == []


def test_create_study_reports_publication_failure_cause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vamos.experiment.artifacts import paths

    spec = _spec()
    plan = resolve_spec(spec)
    long_path = "C:\\" + "d" * 257

    def fail_at(observed: str) -> None:
        if observed == "tasks_written":
            raise FileNotFoundError(2, "The system cannot find the path specified", long_path)

    monkeypatch.setattr(paths, "_WINDOWS", False)
    with pytest.raises(StudyInfrastructureError) as caught:
        publish_study(spec, plan=plan, destination=tmp_path / "study", phase_hook=fail_at)
    assert caught.value.reason == "ATOMIC_PUBLICATION_FAILED"
    assert "The system cannot find the path specified" in caught.value.actual
    assert repr(long_path) in caught.value.actual
    assert "LongPathsEnabled" not in caught.value.actual

    monkeypatch.setattr(paths, "_WINDOWS", True)
    with pytest.raises(StudyInfrastructureError) as caught:
        publish_study(spec, plan=plan, destination=tmp_path / "study", phase_hook=fail_at)
    assert caught.value.reason == "ATOMIC_PUBLICATION_FAILED"
    assert "260 characters" in caught.value.actual and "LongPathsEnabled" in caught.value.actual
    assert not (tmp_path / "study").exists()


def test_sa_018_relocated_study_retains_all_identities(tmp_path: Path) -> None:
    original = tmp_path / "original"
    moved = tmp_path / "elsewhere" / "moved"
    created = vamos.create_study(_spec(), output=original)
    moved.parent.mkdir()
    shutil.move(original, moved)

    loaded = vamos.load_study(moved)
    assert loaded.study_id == created.study_id
    assert loaded.plan_id == created.plan_id
    assert [task.task_id for task in loaded.tasks] == [task.task_id for task in created.tasks]


def test_sa_019_nonempty_creation_is_planned_pending_and_has_no_attempt_or_run(tmp_path: Path) -> None:
    root = tmp_path / "study"
    study = vamos.create_study(_spec(), output=root)

    assert study.status == "created"
    assert [task.state for task in study.tasks] == ["pending"]
    assert all(not task.attempts for task in study.tasks)
    paths = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    assert not any("attempt" in path or path.startswith("runs") for path in paths)


def test_sa_020_load_is_byte_identical_and_reports_unchanged_state(tmp_path: Path) -> None:
    root = tmp_path / "study"
    created = vamos.create_study(_spec(), output=root)
    before = _tree_bytes(root)

    loaded = vamos.load_study(root)

    assert _tree_bytes(root) == before
    assert loaded.study_id == created.study_id
    assert loaded.status == "created"


def test_sa_069_top_level_factories_return_public_study_handles(tmp_path: Path) -> None:
    assert vamos.StudySpec is not None
    root = tmp_path / "study"
    created = vamos.create_study(_spec(), output=root)
    loaded = vamos.load_study(root)

    assert isinstance(created, Study)
    assert isinstance(loaded, Study)
    assert created.study_id == loaded.study_id
    assert created.plan_id == loaded.plan_id
    assert hasattr(created, "run")
    assert hasattr(created, "resume")
    assert hasattr(created, "retry")


def test_same_plan_creates_distinct_study_ids(tmp_path: Path) -> None:
    first = vamos.create_study(_spec(), output=tmp_path / "first")
    second = vamos.create_study(_spec(), output=tmp_path / "second")
    assert first.study_id != second.study_id
    assert first.plan_id == second.plan_id
    assert [task.task_id for task in first.tasks] == [task.task_id for task in second.tasks]
