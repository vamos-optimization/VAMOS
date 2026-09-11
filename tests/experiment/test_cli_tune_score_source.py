from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vamos.algorithms import NSGAIIConfig
import vamos.experiment.cli._tune_runtime as tune_runtime
from vamos.experiment.cli._tune_runtime import BUILDERS, build_task
from vamos.experiment.cli._tune_scoring import (
    _ensure_constrained_tuning_supported,
    _force_population_result_mode,
    _population_front_for_scoring,
    _score_population_result,
)


class _FakeResult:
    def __init__(self) -> None:
        population_f = np.array(
            [
                [1.0, 1.0],
                [0.5, 2.0],
                [2.0, 0.5],
                [0.2, 0.2],
            ]
        )
        self.F = population_f.copy()
        self.data = {
            # Population result mode guarantees this top-level F/G pair is
            # row-aligned when the population payload does not embed G itself.
            "F": population_f.copy(),
            "G": np.array([[0.0], [0.0], [0.0], [1.0]]),
            "population": {"F": population_f.copy()},
            "_tuning_n_constraints": 1,
        }


def test_population_front_for_scoring_filters_infeasible_population_rows() -> None:
    front = _population_front_for_scoring(_FakeResult())

    expected = np.array(
        [
            [1.0, 1.0],
            [0.5, 2.0],
            [2.0, 0.5],
        ]
    )
    assert front.shape == expected.shape
    assert {tuple(row) for row in front.tolist()} == {tuple(row) for row in expected.tolist()}


def test_population_front_for_scoring_rejects_misaligned_constraint_fallback() -> None:
    result = _FakeResult()
    result.data["F"] = np.array([[0.0, 0.0]])

    with pytest.raises(RuntimeError, match="top-level F to align with population F"):
        _population_front_for_scoring(result)


def test_population_embedded_constraints_do_not_require_top_level_population_result() -> None:
    result = _FakeResult()
    population_g = result.data.pop("G")
    result.data["population"]["G"] = population_g
    # Model archive-backed top-level result semantics such as SMPSO leaders.
    result.data["F"] = np.array([[0.0, 0.0]])

    front = _population_front_for_scoring(result)

    expected = np.array(
        [
            [1.0, 1.0],
            [0.5, 2.0],
            [2.0, 0.5],
        ]
    )
    assert front.shape == expected.shape
    assert {tuple(row) for row in front.tolist()} == {tuple(row) for row in expected.tolist()}


def test_successful_empty_feasible_front_scores_zero_not_failure_score() -> None:
    result = _FakeResult()
    result.data["G"] = np.ones((4, 1))

    score = _score_population_result(result, [10.0, 10.0], runtime_penalty=0.0, failure_score=-7.0)

    assert score == 0.0


def test_failed_tuning_run_uses_failure_score() -> None:
    result = _FakeResult()
    result.data["_tuning_failed"] = True

    score = _score_population_result(result, [10.0, 10.0], runtime_penalty=0.0, failure_score=-7.0)

    assert score == -7.0


def test_cli_tuning_forces_population_result_mode() -> None:
    cfg = NSGAIIConfig.default(pop_size=20, n_var=10)
    forced = _force_population_result_mode(cfg)

    assert forced.to_dict()["result_mode"] == "population"
    assert cfg.to_dict()["result_mode"] != "population"


@pytest.mark.parametrize("algorithm_name", ["agemoea", "rvea"])
def test_constrained_cli_tuning_rejects_algorithms_without_aligned_population_constraints(algorithm_name: str) -> None:
    with pytest.raises(RuntimeError, match="does not yet support constrained"):
        _ensure_constrained_tuning_supported(algorithm_name, 1)


@pytest.mark.parametrize("algorithm_name", ["agemoea", "rvea"])
def test_unconstrained_cli_tuning_keeps_agemoea_and_rvea_available(algorithm_name: str) -> None:
    _ensure_constrained_tuning_supported(algorithm_name, 0)


def test_constrained_cli_tuning_keeps_supported_algorithm_available() -> None:
    _ensure_constrained_tuning_supported("nsgaii", 1)


def test_constrained_cli_preflight_happens_before_backend_dispatch(monkeypatch) -> None:
    class _Selection:
        def instantiate(self):
            return SimpleNamespace(n_constraints=1)

    monkeypatch.setattr(tune_runtime, "make_problem_selection", lambda *_args, **_kwargs: _Selection())
    task = SimpleNamespace(instances=[SimpleNamespace(name="fake", n_var=2, kwargs={})])
    args = SimpleNamespace(algorithm="agemoea", n_obj=2)

    with pytest.raises(RuntimeError, match="does not yet support constrained"):
        tune_runtime._preflight_constraint_support(args, task)


def test_cli_tuning_task_excludes_external_archive_controls() -> None:
    algo_space = BUILDERS["nsgaii"]()
    param_space = algo_space.to_param_space()
    assert "use_external_archive" in param_space.params

    args = SimpleNamespace(
        instances="",
        problem="zdt1",
        algorithm="nsgaii",
        backend="random",
        seed=1,
        n_seeds=2,
        n_var=10,
        aggregate_mode="mean",
    )
    task = build_task(args, param_space, budget_per_run=100)

    assert "use_external_archive" not in task.param_space.params
    assert "archive_unbounded" not in task.param_space.params
    assert "archive_prune_policy" not in task.param_space.params
    sample = task.param_space.sample(np.random.default_rng(7))
    assert not {"use_external_archive", "archive_unbounded", "archive_prune_policy"} & sample.keys()
