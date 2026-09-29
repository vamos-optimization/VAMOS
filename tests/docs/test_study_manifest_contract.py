from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "dev"
EXAMPLES = DOCS / "study_manifest_examples"
ACCEPTANCE = DOCS / "study_manifest_acceptance_tests.md"
PLAN_ACCEPTANCE = DOCS / "study_plan_acceptance_tests.md"
CONTRACT = DOCS / "study_manifest_contract.md"

EXPECTED_EXAMPLES = {
    "01-empty-created.json": (True, None, "created"),
    "02-running.json": (True, None, "running"),
    "03-succeeded.json": (True, None, "completed"),
    "04-completed-with-failures.json": (True, None, "completed_with_failures"),
    "05-fail-fast-paused.json": (True, None, "paused"),
    "06-interrupted-attempt.json": (True, None, "paused"),
    "07-retried-task.json": (True, None, "completed"),
    "08-relocated.json": (True, None, "completed"),
    "09-invalid-transition.json": (False, "INVALID_STATE_TRANSITION", "completed"),
    "10-corrupt-run-reference.json": (False, "RUN_MANIFEST_HASH_MISMATCH", "completed"),
}

ATTEMPT_STATES = {"created", "running", "succeeded", "failed", "interrupted", "cancelled"}
TASK_STATES = {"pending", "running", "succeeded", "failed", "interrupted", "cancelled", "skipped"}
STUDY_STATES = {"created", "running", "paused", "completed", "completed_with_failures", "failed", "cancelled"}
STUDY_TRANSITIONS = {
    (None, "created"),
    ("created", "running"),
    ("created", "completed"),
    ("created", "cancelled"),
    ("running", "paused"),
    ("running", "completed"),
    ("running", "completed_with_failures"),
    ("running", "failed"),
    ("running", "cancelled"),
    ("paused", "running"),
    ("paused", "cancelled"),
    ("completed_with_failures", "running"),
}


class DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(key)
        result[key] = value
    return result


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    assert isinstance(value, dict)
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _assert_confined(path_text: str) -> None:
    path = PurePosixPath(path_text)
    assert path_text
    assert "\\" not in path_text
    assert ":" not in path_text
    assert "\x00" not in path_text
    assert not path.is_absolute()
    assert all(part not in {"", ".", ".."} for part in path.parts)


def _file_map(fixture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = fixture["canonical_files"]
    assert isinstance(entries, list)
    result = {entry["path"]: entry for entry in entries}
    assert len(result) == len(entries)
    return result


def _validate_inventory(fixture: dict[str, Any]) -> list[tuple[str, str]]:
    files = _file_map(fixture)
    mismatches: list[tuple[str, str]] = []
    for path, entry in files.items():
        _assert_confined(path)
        document = entry["document"]
        raw = _canonical(document)
        assert entry["bytes"] == len(raw)
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
        assert document["schema_version"] == "1.0.0"
        integrity = document["integrity"]
        assert len(integrity) == 1
        projection = {key: value for key, value in document.items() if key != "integrity"}
        assert next(iter(integrity.values())) == _sha256(projection)

        for candidate in _walk(document):
            descriptor_fields = {"path", "bytes", "sha256", "semantic_sha256"}
            if not descriptor_fields.issubset(candidate):
                continue
            target_path = candidate["path"]
            _assert_confined(target_path)
            assert target_path in files
            target = files[target_path]
            if candidate["bytes"] != target["bytes"]:
                mismatches.append((path, "bytes"))
            if candidate["sha256"] != target["sha256"]:
                mismatches.append((path, "sha256"))
            target_integrity = next(iter(target["document"]["integrity"].values()))
            if candidate["semantic_sha256"] != target_integrity:
                mismatches.append((path, "semantic_sha256"))
            if "run_id" in candidate:
                assert candidate["run_id"] == target["document"]["run_id"]
                assert candidate["task_id"] == target["document"]["task_id"]
    return mismatches


def _validate_plan(files: dict[str, dict[str, Any]]) -> None:
    plan = files["plan.json"]["document"]
    tasks = plan["tasks"]
    assert tasks == sorted(tasks, key=lambda item: item["task_id"])
    assert plan["task_count"] == len(tasks)
    projection = {
        "document_type": "vamos.resolved-study-plan",
        "schema_version": "1.0.0",
        "tasks": tasks,
    }
    assert plan["plan_id"] == "sha256:" + _sha256(projection)
    for task in tasks:
        resolved_hash = _sha256(task["resolved_run_spec"])
        assert task["task_id"] == "sha256:" + resolved_hash
        assert task["task_spec_sha256"] == resolved_hash


def _validate_events(files: dict[str, dict[str, Any]]) -> None:
    entries = sorted(
        (entry for entry in files.values() if entry["document"]["document_type"] == "vamos.study-event"),
        key=lambda entry: entry["document"]["sequence"],
    )
    previous = None
    for sequence, entry in enumerate(entries, 1):
        event = entry["document"]
        assert event["sequence"] == sequence
        assert entry["path"] == f"events/{sequence:020d}.json"
        assert event["previous_event_sha256"] == previous
        previous = entry["sha256"]
        if event["entity"]["kind"] == "study":
            transition = (event["transition"]["from"], event["transition"]["to"])
            assert transition in STUDY_TRANSITIONS
    manifest = files["study-manifest.json"]["document"]
    assert manifest["checkpoint"] == {"event_sha256": previous, "sequence": len(entries)}


def _validate_states(files: dict[str, dict[str, Any]]) -> None:
    documents = [entry["document"] for entry in files.values()]
    manifest = next(doc for doc in documents if doc["document_type"] == "vamos.study-manifest")
    tasks = [doc for doc in documents if doc["document_type"] == "vamos.study-task"]
    attempts = [doc for doc in documents if doc["document_type"] == "vamos.study-attempt"]
    assert manifest["state"] in STUDY_STATES
    assert all(task["state"] in TASK_STATES for task in tasks)
    assert all(attempt["status"] in ATTEMPT_STATES for attempt in attempts)
    assert manifest["counts"]["tasks"] == len(tasks)
    for state in TASK_STATES:
        assert manifest["counts"][state] == sum(task["state"] == state for task in tasks)
    assert manifest["plan_id"] == files["plan.json"]["document"]["plan_id"]
    assert manifest["study_id"] == files["study-spec.json"]["document"]["study_id"]


@pytest.mark.parametrize("filename", sorted(EXPECTED_EXAMPLES))
def test_machine_readable_example(filename: str) -> None:
    expected_valid, expected_error, expected_state = EXPECTED_EXAMPLES[filename]
    fixture = _load(EXAMPLES / filename)
    assert fixture["fixture_type"] == "vamos.study-contract-example"
    assert fixture["fixture_version"] == "1.0.0"
    assert fixture["expected"] == {
        "error": expected_error,
        "study_state": expected_state,
        "valid": expected_valid,
    }
    files = _file_map(fixture)
    assert {"study-manifest.json", "study-spec.json", "plan.json"}.issubset(files)
    _validate_plan(files)
    _validate_events(files)
    _validate_states(files)
    mismatches = _validate_inventory(fixture)
    if expected_error == "RUN_MANIFEST_HASH_MISMATCH":
        assert mismatches == [
            (
                "tasks/5833698650ca110e3d2cd184e00be47980be653b9dc50ca8e6d094a1b13ceebf/attempts/33333333-3333-4333-8333-333333333331.json",
                "sha256",
            )
        ]
    else:
        assert mismatches == []
    if expected_error == "INVALID_STATE_TRANSITION":
        operation = fixture["operation"]
        assert operation["from"] == expected_state
        assert (operation["from"], operation["requested_to"]) not in STUDY_TRANSITIONS
    else:
        assert fixture["operation"] is None


def test_example_set_is_exact_and_duplicate_keys_are_rejected() -> None:
    assert {path.name for path in EXAMPLES.glob("*.json")} == set(EXPECTED_EXAMPLES)
    with pytest.raises(DuplicateKeyError, match="state"):
        json.loads('{"state":"created","state":"running"}', object_pairs_hook=_reject_duplicate_keys)


def test_acceptance_inventory_is_complete_and_fully_populated() -> None:
    rows = [line for line in ACCEPTANCE.read_text(encoding="utf-8").splitlines() if re.match(r"^\| SA-\d{3} \|", line)]
    expected_ids = [f"SA-{number:03d}" for number in range(1, 75)]
    cells = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows]
    assert [row[0] for row in cells] == expected_ids
    assert all(len(row) == 12 for row in cells)
    assert all(all(field for field in row) for row in cells)


@pytest.mark.parametrize(
    "document",
    [
        CONTRACT,
        ACCEPTANCE,
        PLAN_ACCEPTANCE,
        DOCS / "studies.md",
        DOCS / "adr" / "0008-durable-study-manifest-contract.md",
        EXAMPLES / "README.md",
    ],
)
def test_new_document_relative_links_resolve(document: Path) -> None:
    text = document.read_text(encoding="utf-8")
    links = re.findall(r"]\(([^)#]+)(?:#[^)]+)?\)", text, flags=re.MULTILINE)
    for link in links:
        if "://" in link or link.startswith("mailto:"):
            continue
        assert (document.parent / link).resolve().exists(), f"broken link in {document}: {link}"


def test_contract_freezes_one_schema_and_first_slice() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    assert "`vamos.study-manifest`" in text
    assert 'All use `schema_version="1.0.0"`' in text
    assert "No design choice remains open for Goal 1" in text
    assert "V1 has no append, delete, patch, extension, or" in text
    assert "no CSV-to-study import" in text
    assert "SA-001 through SA-074" in text
    assert "`task_id` is exactly the canonical RunManifest task ID" in text
    assert "V1 performs no implicit or automatic retry" in text
    assert "Succeeded tasks are never rerun" in text


def test_planning_and_recovery_share_one_persisted_plan_boundary() -> None:
    contract = CONTRACT.read_text(encoding="utf-8")
    guide = " ".join((DOCS / "studies.md").read_text(encoding="utf-8").split())
    assert "`plan_study` and `create_study` use the same canonical resolver" in contract
    assert "`Study.resume(retry_failed=False)` explicitly reconciles first" in contract
    assert "No operation after creation consults current defaults to change the plan." in contract
    assert "planning neither creates nor reserves the path" in guide
    assert "persisted built-in configuration is reconstructed" in guide


def test_adr_records_and_navigation_publish_the_contract() -> None:
    adr_index = (DOCS / "adr" / "index.md").read_text(encoding="utf-8")
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    contract = (DOCS / "study_manifest_contract.md").read_text(encoding="utf-8")
    assert adr_index.count("0008-durable-study-manifest-contract.md") == 1
    assert nav.count("dev/study_manifest_contract.md") == 1
    assert nav.count("dev/study_manifest_acceptance_tests.md") == 1
    assert nav.count("dev/study_plan_acceptance_tests.md") == 1
    assert "dev/adr/0008-durable-study-manifest-contract.md" not in nav
    assert "`docs/dev/adr/0008-durable-study-manifest-contract.md`" in contract
