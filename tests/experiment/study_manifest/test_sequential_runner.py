from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.experiment.artifacts.jsonio import canonical_json_bytes
from vamos.experiment.optimization_result import OptimizationResult
from vamos.experiment.study.errors import UnsupportedStudyExecutionStateError
from vamos.experiment.study.identity import compute_task_id


def _spec(*, seeds: tuple[int, ...] = (0,), empty: bool = False) -> vamos.StudySpec:
    return vamos.StudySpec(
        problems=[] if empty else ["zdt1"],
        algorithms=[] if empty else ["nsgaii"],
        seeds=[] if empty else seeds,
        max_evaluations=8,
        pop_size=4,
    )


def _mock_result(n_var: int, n_obj: int, seed: int) -> OptimizationResult:
    return OptimizationResult(
        {
            "F": np.full((1, n_obj), float(seed), dtype=np.float64),
            "X": np.full((1, n_var), seed, dtype=np.float64),
            "evaluations": 8,
            "generations": 2,
        }
    )


def _immutable_inputs(root: Path) -> tuple[bytes, bytes]:
    return (root.joinpath("study-spec.json").read_bytes(), root.joinpath("plan.json").read_bytes())


def test_sa_021_to_sa_026_sequential_run_is_durable_ordered_and_fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    created = vamos.create_study(_spec(seeds=(9, 0, 4)), output=root)
    expected_order = [task.task_id for task in created.plan.tasks]
    expected_seeds = [int(task.resolved_run_spec["seed"]) for task in created.plan.tasks]
    observed: list[tuple[str, int]] = []

    def execute(reconstructed: Any, *, root: Path) -> OptimizationResult:
        durable = vamos.load_study(root)
        running = [task for task in durable.tasks if task.state == "running"]
        assert durable.status == "running"
        assert len(running) == 1
        assert len(durable.attempts) == len(observed) + 1
        assert durable.attempts[-1].status == "running"
        assert running[0].current_attempt_id == durable.attempts[-1].attempt_id
        task_id = compute_task_id(reconstructed.resolved_spec)
        observed.append((task_id, reconstructed.seed))
        return _mock_result(reconstructed.n_var, reconstructed.n_obj, reconstructed.seed)

    monkeypatch.setattr(execution, "_execute_optimization", execute)
    completed = created.run()

    assert completed is not created
    assert created.status == "created"
    assert completed.status == "completed"
    assert completed.study_id == created.study_id
    assert completed.plan_id == created.plan_id
    assert observed == list(zip(expected_order, expected_seeds, strict=True))
    assert completed.manifest.counts.succeeded == 3
    assert completed.manifest.counts.pending == 0
    assert [task.state for task in completed.tasks] == ["succeeded"] * 3
    assert len(completed.attempts) == 3
    assert len({attempt.attempt_id for attempt in completed.attempts}) == 3
    assert [event.sequence for event in completed.events] == list(range(1, len(completed.events) + 1))
    assert all(
        event.previous_event_sha256 == completed.events[index - 1].file_sha256 for index, event in enumerate(completed.events[1:], start=1)
    )


def test_sa_061_to_sa_063_attempt_links_exact_fully_verified_canonical_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    created = vamos.create_study(_spec(seeds=(0, 1)), output=root)

    def execute(reconstructed: Any, *, root: Path) -> OptimizationResult:
        return _mock_result(reconstructed.n_var, reconstructed.n_obj, reconstructed.seed)

    monkeypatch.setattr(execution, "_execute_optimization", execute)
    completed = created.run()

    for task, attempt in zip(completed.tasks, completed.attempts, strict=True):
        reference = attempt.run_reference
        assert reference is not None
        run_id = reference["run_id"]
        assert run_id != attempt.attempt_id
        assert reference["task_id"] == task.task_id
        assert reference["path"] == f"runs/{run_id}/manifest.json"
        assert not Path(str(reference["path"])).is_absolute()
        run_root = root / "runs" / str(run_id)
        assert {item.name for item in run_root.iterdir()} == {"environment.json", "manifest.json", "result.npz"}
        stored = vamos.load_run(run_root, verify="all")
        result = vamos.load_result(run_root)
        plan_task = next(item for item in completed.plan.tasks if item.task_id == task.task_id)
        assert stored.manifest.task_id == task.task_id
        assert canonical_json_bytes(stored.manifest.resolved_spec) == canonical_json_bytes(plan_task.resolved_run_spec)
        assert result.F is not None and result.X is not None
        assert task.selected_success_attempt_id == attempt.attempt_id
        assert task.current_attempt_id is None
        assert len(task.attempts) == 1
        assert task.attempts[0].attempt_id == attempt.attempt_id
        assert attempt.failure is None
        assert attempt.lease_evidence is None


def test_sa_024_success_event_is_appended_only_after_full_run_verification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution
    import vamos.experiment.study.run_publication as publication

    root = tmp_path / "study"
    created = vamos.create_study(_spec(), output=root)
    verified: set[str] = set()
    fully_loaded: set[str] = set()
    original_verify = publication.verify_run
    original_load = publication.load_run
    original_append = execution.append_event

    def execute(reconstructed: Any, *, root: Path) -> OptimizationResult:
        return _mock_result(reconstructed.n_var, reconstructed.n_obj, reconstructed.seed)

    def verify(path: str | Path, *args: Any, **kwargs: Any) -> Any:
        result = original_verify(path, *args, **kwargs)
        verified.add(Path(path).name)
        return result

    def load(path: str | Path, *args: Any, **kwargs: Any) -> Any:
        assert kwargs.get("verify") == "all"
        result = original_load(path, *args, **kwargs)
        fully_loaded.add(Path(path).name)
        return result

    def append(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("event_type") == "attempt_succeeded":
            payload = kwargs["payload"]
            assert isinstance(payload, Mapping)
            reference = payload["run_reference"]
            assert isinstance(reference, Mapping)
            assert reference["run_id"] in verified
            assert reference["run_id"] in fully_loaded
        return original_append(*args, **kwargs)

    monkeypatch.setattr(execution, "_execute_optimization", execute)
    monkeypatch.setattr(publication, "verify_run", verify)
    monkeypatch.setattr(publication, "load_run", load)
    monkeypatch.setattr(execution, "append_event", append)

    completed = created.run()
    assert completed.status == "completed"
    assert [event.event_type for event in completed.events].index("attempt_succeeded") > 0


def test_real_builtin_execution_uses_persisted_seed_and_exact_budget(tmp_path: Path) -> None:
    completed = vamos.create_study(_spec(seeds=(0,)), output=tmp_path / "study").run()
    reference = completed.attempts[0].run_reference
    assert reference is not None
    stored = vamos.load_run(completed.root / "runs" / str(reference["run_id"]), verify="all")
    assert stored.manifest.resolved_spec["seed"] == 0
    assert stored.manifest["outcome"]["evaluations"] == 8


def test_execution_passes_the_exact_persisted_configuration_to_the_internal_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    created = vamos.create_study(_spec(seeds=(0,)), output=tmp_path / "study")
    persisted = created.plan.tasks[0].resolved_run_spec
    captured: list[Any] = []

    def run_config(config: Any, *, built_in_only: bool) -> OptimizationResult:
        assert built_in_only is True
        captured.append(config)
        return _mock_result(config.problem.n_var, config.problem.n_obj, config.seed)

    monkeypatch.setattr(execution, "_run_config", run_config)
    assert created.run().status == "completed"
    assert len(captured) == 1
    config = captured[0]
    assert config.seed == persisted["seed"] == 0
    assert config.termination == ("max_evaluations", 8)
    assert config.eval_strategy == "serial"
    assert config.algorithm == "nsgaii"
    rebuilt = config.algorithm_config.to_dict()
    for name, value in persisted["algorithm"]["config"].items():
        assert rebuilt[name] == value
    assert rebuilt["crossover"] == ("sbx", dict(persisted["operators"]["crossover"]["config"]))
    assert rebuilt["mutation"] == ("pm", dict(persisted["operators"]["mutation"]["config"]))
    assert rebuilt["selection"] == ("tournament", dict(persisted["operators"]["selection"]["config"]))
    assert rebuilt["repair"] == ("clip", dict(persisted["operators"]["repair"]["config"]))
    assert config.engine == persisted["backend"]["kernel"]["resolution"]["name"]


def test_empty_study_completes_without_execution_attempt_or_run(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    created = vamos.create_study(_spec(empty=True), output=root)
    completed = created.run()

    assert created.status == "created"
    assert completed.status == "completed"
    assert completed.plan_id == created.plan_id
    assert completed.tasks == ()
    assert completed.attempts == ()
    assert [event.event_type for event in completed.events] == ["study_created", "study_completed"]
    assert not root.joinpath("runs").exists()


def test_completed_study_and_same_process_reentry_are_rejected_actionably(tmp_path: Path) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "empty"
    completed = vamos.create_study(_spec(empty=True), output=root).run()
    with pytest.raises(UnsupportedStudyExecutionStateError) as completed_error:
        completed.run()
    assert completed_error.value.study_id == completed.study_id
    assert completed_error.value.current_state == "completed"
    assert completed_error.value.expected_state == "created"
    assert completed_error.value.execution_occurred is False
    assert "resume" in completed_error.value.action.lower()

    created = vamos.create_study(_spec(), output=tmp_path / "other")
    execution._ACTIVE_ROOTS[created.root.resolve()] = None
    try:
        with pytest.raises(UnsupportedStudyExecutionStateError) as reentrant:
            created.run()
        assert reentrant.value.reason == "REENTRANT_STUDY_EXECUTION"
    finally:
        execution._ACTIVE_ROOTS.pop(created.root.resolve(), None)


def test_objective_failure_is_durable_sanitized_and_stops_later_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    created = vamos.create_study(_spec(seeds=(0, 1)), output=root)
    immutable = _immutable_inputs(root)
    calls = 0

    def fail(_reconstructed: Any, *, root: Path) -> OptimizationResult:
        nonlocal calls
        calls += 1
        raise RuntimeError(f"secret-token at {root}\\private\ntrace")

    monkeypatch.setattr(execution, "_execute_optimization", fail)
    returned = created.run()

    failed = vamos.load_study(root)
    assert calls == 1
    assert returned.status == "paused"
    assert failed.status == "paused"
    assert failed.manifest.counts.failed == 1
    assert failed.manifest.counts.pending == 1
    assert [task.state for task in failed.tasks].count("failed") == 1
    assert [task.state for task in failed.tasks].count("pending") == 1
    assert len(failed.attempts) == 1
    attempt = failed.attempts[0]
    assert attempt.status == "failed"
    assert attempt.failure is not None
    serialized = repr(dict(attempt.failure))
    assert "secret-token" not in serialized
    assert str(root) not in serialized
    assert "traceback" not in serialized.lower()
    assert failed.events[-1].event_type == "study_paused"
    reference = attempt.run_reference
    assert reference is not None
    stored = vamos.load_run(root / "runs" / str(reference["run_id"]), verify="all")
    assert stored.status == "failed"
    assert stored.manifest["failure"] is not None
    assert stored.manifest["failure"]["traceback"] is None
    assert _immutable_inputs(root) == immutable


def test_execution_does_not_call_public_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.artifacts.replay as replay
    import vamos.experiment.study.execution as execution

    created = vamos.create_study(_spec(), output=tmp_path / "study")

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("execution shortcut invoked")

    def execute(reconstructed: Any, *, root: Path) -> OptimizationResult:
        return _mock_result(reconstructed.n_var, reconstructed.n_obj, reconstructed.seed)

    monkeypatch.setattr(replay, "reproduce", forbidden)
    monkeypatch.setattr(execution, "_execute_optimization", execute)
    assert created.run().status == "completed"


def test_hundred_task_mocked_smoke_selects_by_constant_time_index_without_per_task_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vamos.experiment.study.execution as execution

    created = vamos.create_study(_spec(seeds=tuple(range(100))), output=tmp_path / "study")
    selected: list[tuple[int, str]] = []
    load_calls = 0

    def load(_source: Any) -> vamos.Study:
        nonlocal load_calls
        load_calls += 1
        return created

    def start(study: vamos.Study) -> Any:
        tasks = list(study.tasks)
        indexes = {task.task_id: index for index, task in enumerate(tasks)}
        return execution._ExecutionState(
            study.root,
            study.study_id,
            "22222222-2222-4222-8222-222222222222",
            study.manifest,
            tasks,
            indexes,
            study.events[-1],
        )

    def run_task(_state: Any, index: int, plan_task: Any) -> None:
        selected.append((index, plan_task.task_id))

    def append(_root: Path, previous: Any, **_kwargs: Any) -> Any:
        return previous

    def checkpoint(_root: Path, manifest: Any, **_kwargs: Any) -> Any:
        return manifest

    def complete(state: Any, _phase: Any, _reload: Any) -> vamos.Study:
        return load(state)

    monkeypatch.setattr(execution, "reload_study", load)
    monkeypatch.setattr(execution, "_start_execution", start)
    monkeypatch.setattr(execution, "_run_task_attempt", run_task)
    monkeypatch.setattr(execution, "append_event", append)
    monkeypatch.setattr(execution, "checkpoint_manifest", checkpoint)
    monkeypatch.setattr(execution, "complete_running", complete)

    created.run()
    assert load_calls == 2
    assert selected == [(index, task.task_id) for index, task in enumerate(created.plan.tasks)]
