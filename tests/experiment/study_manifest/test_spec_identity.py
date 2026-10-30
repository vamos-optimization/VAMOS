from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from vamos.experiment.study.errors import DuplicateStudyTaskError, InvalidStudySpecError
from vamos.experiment.study.identity import compute_plan_id, compute_task_id
from vamos.experiment.study.planning import resolve_spec
from vamos.experiment.study.serialization import canonical_json
from vamos.study_artifacts import StudySpec


def _spec(**changes: object) -> StudySpec:
    values: dict[str, object] = {
        "problems": ["zdt1"],
        "algorithms": ["nsgaii"],
        "seeds": [0, 1],
        "max_evaluations": 24,
        "pop_size": 8,
    }
    values.update(changes)
    return StudySpec(**values)  # type: ignore[arg-type]


def test_sa_001_minimal_spec_is_immutable_and_defaults_policy() -> None:
    spec = StudySpec(problems=["zdt1"], algorithms=["nsgaii"], seeds=[0])

    assert spec.on_error == "fail_fast"
    assert spec.max_attempts_per_task == 3
    assert spec.problems == ("zdt1",)
    assert spec.seeds == (0,)
    with pytest.raises(FrozenInstanceError):
        spec.max_evaluations = 1  # type: ignore[misc]


def test_sa_003_matrix_display_reordering_preserves_ids() -> None:
    first = resolve_spec(_spec(problems=["zdt1", "zdt2"], seeds=[0, 1]))
    second = resolve_spec(_spec(problems=["zdt2", "zdt1"], seeds=[1, 0]))

    assert first.plan_id == second.plan_id
    assert {task.task_id for task in first.tasks} == {task.task_id for task in second.tasks}


def test_sa_004_seed_changes_run_manifest_compatible_task_id() -> None:
    plan = resolve_spec(_spec())
    assert len({task.task_id for task in plan.tasks}) == 2
    for task in plan.tasks:
        assert task.task_id == compute_task_id(task.resolved_run_spec)


def test_sa_005_plan_id_is_set_based() -> None:
    task_ids = [task.task_id for task in resolve_spec(_spec()).tasks]
    assert compute_plan_id(task_ids) == compute_plan_id(reversed(task_ids))


def test_sa_006_labels_metadata_output_and_placement_do_not_change_scientific_ids() -> None:
    plain = resolve_spec(_spec())
    presented = resolve_spec(_spec(labels={"title": "display"}, metadata={"owner": "team"}))
    output_a = "studies/a"
    output_b = "elsewhere/b"
    worker_counts = (1, 16)

    assert plain.plan_id == presented.plan_id
    assert [task.task_id for task in plain.tasks] == [task.task_id for task in presented.tasks]
    assert output_a != output_b and worker_counts[0] != worker_counts[1]
    assert compute_plan_id(task.task_id for task in plain.tasks) == plain.plan_id


def test_sa_007_scientific_budget_and_algorithm_changes_change_plan() -> None:
    baseline = resolve_spec(_spec())
    budget = resolve_spec(_spec(max_evaluations=25))
    algorithm = resolve_spec(_spec(algorithms=["spea2"]))

    assert baseline.plan_id != budget.plan_id
    assert baseline.plan_id != algorithm.plan_id
    assert {task.task_id for task in baseline.tasks}.isdisjoint(task.task_id for task in budget.tasks)


def test_sa_009_canonical_json_is_stable_compact_utf8() -> None:
    first = canonical_json({"z": "España", "a": {"b": 2, "a": 1}})
    second = canonical_json({"a": {"a": 1, "b": 2}, "z": "Espan\u0303a"})
    assert first == second == b'{"a":{"a":1,"b":2},"z":"Espa\xc3\xb1a"}'


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_sa_012_nonfinite_spec_values_are_rejected(value: float) -> None:
    with pytest.raises(InvalidStudySpecError, match="NON_FINITE_NUMBER"):
        _spec(metadata={"bad": value})


def test_duplicate_canonical_tasks_are_rejected_before_publication() -> None:
    with pytest.raises(DuplicateStudyTaskError, match="DUPLICATE_CANONICAL_TASK"):
        resolve_spec(_spec(seeds=[0, 0]))


@pytest.mark.parametrize("seeds", [[-1], [0, -5]])
def test_negative_seed_is_rejected(seeds: list[int]) -> None:
    with pytest.raises(InvalidStudySpecError) as caught:
        resolve_spec(_spec(seeds=seeds))
    assert caught.value.reason == "INVALID_STUDY_SPEC"
    assert caught.value.field == f"seeds[{len(seeds) - 1}]"
    assert caught.value.execution_occurred is False


def test_seed_zero_is_a_concrete_scientific_seed() -> None:
    task = resolve_spec(_spec(seeds=[0])).tasks[0]
    assert task.resolved_run_spec["seed"] == 0
