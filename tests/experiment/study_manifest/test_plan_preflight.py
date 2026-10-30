from __future__ import annotations

import json
import re
from dataclasses import FrozenInstanceError
from importlib import metadata as importlib_metadata
from pathlib import Path

import pytest

import vamos
from vamos.experiment.study.errors import (
    DuplicateStudyTaskError,
    InvalidStudySpecError,
    StudyOutputCollisionError,
    UnresolvedStudyTaskError,
)
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _spec(**changes: object) -> vamos.StudySpec:
    values: dict[str, object] = {
        "problems": ["zdt1"],
        "algorithms": ["nsgaii"],
        "seeds": [0, 1],
        "max_evaluations": 24,
        "pop_size": 8,
    }
    values.update(changes)
    return vamos.StudySpec(**values)  # type: ignore[arg-type]


def _tree(root: Path) -> dict[str, bytes | None]:
    if not root.exists():
        return {}
    return {path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None for path in sorted(root.rglob("*"))}


def test_pl_001_minimal_report_is_public_immutable_and_resolved() -> None:
    report = vamos.plan_study(_spec(seeds=[7]))

    assert isinstance(report, vamos.StudyPlanReport)
    assert report.status == "ready"
    assert report.valid is True
    assert report.task_count == 1
    assert report.task_ids == tuple(task.task_id for task in report.plan.tasks)
    assert report.problem_ids == ("vamos.problem:zdt1@1",)
    assert report.algorithm_ids == ("vamos.algorithm:nsgaii@1",)
    assert report.reconstructable is True
    assert report.duplicate_tasks is False
    with pytest.raises(FrozenInstanceError):
        report.valid = False  # type: ignore[misc]


def test_pl_002_empty_plan_is_ready_and_zero_budget() -> None:
    report = vamos.plan_study(_spec(problems=[], algorithms=[], seeds=[]))

    assert report.status == "ready"
    assert report.task_count == 0
    assert report.task_ids == ()
    assert report.total_evaluation_budget == 0
    assert report.problem_ids == report.algorithm_ids == report.backend_ids == ()


def test_pl_003_004_repeated_and_reordered_specs_have_equal_identity() -> None:
    first = vamos.plan_study(_spec(problems=["zdt1", "zdt2"], seeds=[0, 1]))
    repeated = vamos.plan_study(_spec(problems=["zdt1", "zdt2"], seeds=[0, 1]))
    reordered = vamos.plan_study(_spec(problems=["zdt2", "zdt1"], seeds=[1, 0]))

    assert first.plan_id == repeated.plan_id == reordered.plan_id
    assert first.task_ids == repeated.task_ids == reordered.task_ids


def test_pl_005_006_007_exact_budget_seed_and_backend_summaries() -> None:
    report = vamos.plan_study(_spec(problems=["zdt1", "zdt2"], seeds=[3, 5], max_evaluations=40))

    assert report.task_count == 4
    assert report.total_evaluation_budget == 160
    assert report.seeds == (3, 5)
    assert report.population_sizes == (8,)
    assert report.backend_ids == ("vamos.evaluation:serial@1", "vamos.kernel:numpy@1")
    assert report.termination_categories == ("vamos.termination:max_evaluations@1",)


def test_pl_008_nsgaiii_reference_direction_cardinality_is_preflighted() -> None:
    valid = vamos.plan_study(_spec(problems=["dtlz2"], algorithms=["nsgaiii"], seeds=[0], n_obj=3, pop_size=None, max_evaluations=100))
    assert valid.population_sizes == (91,)

    with pytest.raises(UnresolvedStudyTaskError) as caught:
        vamos.plan_study(_spec(problems=["dtlz2"], algorithms=["nsgaiii"], seeds=[0], n_obj=3, pop_size=8))
    assert caught.value.reason == "REFERENCE_DIRECTION_POPULATION_MISMATCH"
    assert caught.value.execution_occurred is False

    with pytest.raises(UnresolvedStudyTaskError) as rvea:
        vamos.plan_study(_spec(problems=["dtlz2"], algorithms=["rvea"], seeds=[0], n_obj=3, pop_size=8))
    assert rvea.value.reason == "REFERENCE_DIRECTION_POPULATION_MISMATCH"


def test_pl_008_external_reference_direction_path_is_not_followed(tmp_path: Path) -> None:
    external = tmp_path / "directions.csv"
    external.write_text("not read", encoding="utf-8")
    before = external.read_bytes()

    with pytest.raises(UnresolvedStudyTaskError) as caught:
        vamos.plan_study(
            _spec(
                problems=["dtlz2"],
                algorithms=["nsgaiii"],
                seeds=[0],
                n_obj=3,
                pop_size=91,
                max_evaluations=100,
                algorithm_configs={"nsgaiii": {"reference_directions": {"path": str(external), "divisions": 12}}},
            )
        )
    assert caught.value.reason == "UNSAFE_REFERENCE_DIRECTIONS"
    assert external.read_bytes() == before


def test_pl_009_010_011_012_invalid_inputs_are_typed_and_side_effect_free(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(DuplicateStudyTaskError) as duplicate:
        vamos.plan_study(_spec(seeds=[0, 0]))
    assert duplicate.value.reason == "DUPLICATE_CANONICAL_TASK"

    with pytest.raises(UnresolvedStudyTaskError) as component:
        vamos.plan_study(_spec(problems=["not-a-problem"], seeds=[0]))
    assert component.value.field == "$.matrix[0]"

    with pytest.raises(UnresolvedStudyTaskError) as budget:
        vamos.plan_study(_spec(seeds=[0], max_evaluations=7, pop_size=8))
    assert budget.value.reason == "INVALID_EVALUATION_BUDGET"

    with pytest.raises(UnresolvedStudyTaskError) as backend:
        vamos.plan_study(_spec(seeds=[0], engine="not-a-backend"))
    assert backend.value.execution_occurred is False

    installed_version = importlib_metadata.version

    def version_without_numba(package: str) -> str:
        if package == "numba":
            raise importlib_metadata.PackageNotFoundError(package)
        return installed_version(package)

    monkeypatch.setattr(importlib_metadata, "version", version_without_numba)
    with pytest.raises(UnresolvedStudyTaskError) as unavailable:
        vamos.plan_study(_spec(seeds=[0], engine="numba"))
    assert unavailable.value.reason == "BACKEND_UNAVAILABLE"
    assert unavailable.value.execution_occurred is False


@pytest.mark.parametrize("eval_strategy", ["multiprocessing", "dask"])
def test_pl_022_plan_blocks_non_serial_eval_strategy(tmp_path: Path, eval_strategy: str) -> None:
    spec = _spec(seeds=[0], eval_strategy=eval_strategy)
    output = tmp_path / "study"

    with pytest.raises(UnresolvedStudyTaskError) as planned:
        vamos.plan_study(spec, output=output)
    assert planned.value.reason == "UNSUPPORTED_EVALUATION_BACKEND"
    assert planned.value.field == "$.matrix[0].backend.evaluation"
    assert planned.value.execution_occurred is False

    with pytest.raises(UnresolvedStudyTaskError) as created:
        vamos.create_study(spec, output=output)
    assert created.value.reason == "UNSUPPORTED_EVALUATION_BACKEND"
    assert not output.exists()


def test_pl_022_plan_and_create_reject_negative_seeds(tmp_path: Path) -> None:
    spec = _spec(seeds=[0, -1])
    output = tmp_path / "study"

    with pytest.raises(InvalidStudySpecError) as planned:
        vamos.plan_study(spec, output=output)
    assert planned.value.field == "seeds[1]"

    with pytest.raises(InvalidStudySpecError):
        vamos.create_study(spec, output=output)
    assert not output.exists()


@pytest.mark.parametrize("eval_strategy", ["serial", "multiprocessing", "dask"])
def test_pl_022_every_plannable_eval_strategy_runs(tmp_path: Path, eval_strategy: str) -> None:
    spec = _spec(seeds=[0, 1], eval_strategy=eval_strategy)
    try:
        report = vamos.plan_study(spec)
    except UnresolvedStudyTaskError as exc:
        assert exc.reason == "UNSUPPORTED_EVALUATION_BACKEND"
        return

    assert report.status == "ready"
    completed = vamos.create_study(spec, output=tmp_path / "study").run()
    assert completed.status == "completed"
    assert [task.state for task in completed.tasks] == ["succeeded", "succeeded"]


def test_pl_013_absent_output_is_advisory_and_not_created(tmp_path: Path) -> None:
    output = tmp_path / "missing-parent" / "study"
    report = vamos.plan_study(_spec(seeds=[0]), output=output)

    assert report.status == "ready"
    assert report.output.status == "available"
    assert report.output.available is True
    assert report.output.collision is False
    assert report.output.advisory and "does not reserve" in report.output.advisory
    assert not output.parent.exists()


def test_pl_014_all_existing_output_classes_collide_like_create(tmp_path: Path) -> None:
    spec = _spec(seeds=[0])
    file_path = tmp_path / "file"
    file_path.write_text("occupied", encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("keep", encoding="utf-8")
    canonical = tmp_path / "canonical"
    vamos.create_study(spec, output=canonical)
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "study-manifest.json").write_text("{}", encoding="utf-8")

    expected = {
        file_path: "existing_file",
        empty: "empty_directory",
        unrelated: "unrelated_directory",
        canonical: "canonical_study",
        invalid: "invalid_study_directory",
    }
    for path, status in expected.items():
        before = path.read_bytes() if path.is_file() else _tree(path)
        report = vamos.plan_study(spec, output=path)
        assert report.plan_id
        assert report.status == "blocked"
        assert report.output.status == status
        assert report.output.collision is True
        assert report.errors[0].reason == "OUTPUT_COLLISION"
        with pytest.raises(StudyOutputCollisionError):
            vamos.create_study(spec, output=path)
        after = path.read_bytes() if path.is_file() else _tree(path)
        assert after == before


def test_pl_015_016_planning_writes_nothing_and_never_evaluates_objective(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def forbidden_evaluate(self: ZDT1Problem, x: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("objective evaluation is forbidden during planning")

    monkeypatch.setattr(ZDT1Problem, "evaluate", forbidden_evaluate)
    before = _tree(tmp_path)
    report = vamos.plan_study(_spec(seeds=[0]), output=tmp_path / "future" / "study")

    assert report.status == "ready"
    assert calls == 0
    assert _tree(tmp_path) == before
    payload = report.as_dict()
    assert payload["execution_occurred"] is False
    assert payload["filesystem_write_occurred"] is False


def test_pl_020_plan_and_create_are_exactly_equal(tmp_path: Path) -> None:
    spec = _spec(problems=["zdt2", "zdt1"], seeds=[5, 3], max_evaluations=40)
    report = vamos.plan_study(spec, output=tmp_path / "study")
    created = vamos.create_study(spec, output=tmp_path / "study")

    assert report.plan == created.plan
    assert report.plan_id == created.plan_id
    assert report.task_ids == tuple(task.task_id for task in created.plan.tasks)
    for planned, published in zip(report.plan.tasks, created.plan.tasks, strict=True):
        assert planned == published
        assert planned.resolved_run_spec["seed"] == published.resolved_run_spec["seed"]


def test_pl_acceptance_inventory_is_contiguous_and_populated() -> None:
    path = Path(__file__).resolve().parents[3] / "docs" / "dev" / "study_plan_acceptance_tests.md"
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if re.match(r"^\| PL-\d{3} \|", line)]
    cells = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows]
    assert [row[0] for row in cells] == [f"PL-{number:03d}" for number in range(1, 23)]
    assert all(len(row) == 5 and all(row) for row in cells)


def test_report_json_payload_is_detached_and_complete() -> None:
    report = vamos.plan_study(_spec(seeds=[0]))
    payload = report.as_dict()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    assert "document_type" not in payload
    assert "schema_version" not in payload
    assert payload["status"] == "ready"
    assert payload["errors"] == []
    assert payload["next_actions"]
    assert "runtime_estimate" not in encoded
    assert "memory_estimate" not in encoded
    assert "disk_size_estimate" not in encoded
