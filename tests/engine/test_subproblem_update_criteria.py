"""MOEA/D's subproblem update criteria, against jMetal's own (tests/engine/data/subproblem_update_jmetal_decisions.json)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vamos import optimize
from vamos.algorithms import MOEADConfig
from vamos.engine.algorithm.moead.update_criteria import (
    ImprovedEpsilonCriterion,
    SubproblemUpdateCriterion,
    ViolationThresholdCriterion,
    make_subproblem_update_criterion,
    normalize_subproblem_update_criterion,
)
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.problems import LIRCMOP

FIXTURE = Path(__file__).resolve().parent / "data" / "subproblem_update_jmetal_decisions.json"
CASES: list[dict[str, Any]] = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
IMPROVED_EPSILON = ("improved_epsilon", {"tao": 0.1, "feasibility_ratio_threshold": 0.95, "decay_fraction": 0.8})
CRITERIA: list[Any] = [IMPROVED_EPSILON, ("violation_threshold", {})]


def _populations(case: dict[str, Any]) -> list[np.ndarray]:
    population = np.asarray(case["population"], dtype=float)
    result = []
    for step in case["steps"]:
        result.append(population)
        population = population.copy()
        population[step["replaced"]] = step["child"]
    return result


def _criterion(case: dict[str, Any]) -> SubproblemUpdateCriterion:
    params = case["params"]
    if case["criterion"] == "improved_epsilon":
        return ImprovedEpsilonCriterion(params["tao"], params["feasibility_ratio_threshold"], int(params["tc"]))
    return ViolationThresholdCriterion()


def _run(criterion: object, engine: str = "numpy", *, seed: int = 4, evaluations: int = 2000) -> Any:
    config = dataclasses.replace(MOEADConfig.default(pop_size=30, n_var=10), subproblem_update_criterion=criterion)
    return optimize(LIRCMOP(2, n_var=10), algorithm="moead", algorithm_config=config, max_evaluations=evaluations, seed=seed, engine=engine)


# ---- Decisions of jMetal's criteria ------------------------------------------


def test_the_fixture_covers_both_criteria_and_their_states() -> None:
    assert {case["criterion"] for case in CASES} == {"improved_epsilon", "violation_threshold"}
    decisions = [value for case in CASES for step in case["steps"] for value in step["replaces"]]
    assert any(decisions) and not all(decisions)
    epsilons = {step["state"] for case in CASES if case["criterion"] == "improved_epsilon" for step in case["steps"]}
    assert 0.0 in epsilons and len(epsilons) > 4


@pytest.mark.parametrize("case", CASES, ids=lambda case: f"{case['criterion']}-{case['n']}x{case['k']}")
def test_decisions_match_jmetal(case: dict[str, Any]) -> None:
    criterion = _criterion(case)
    for number, (step, population) in enumerate(zip(case["steps"], _populations(case))):
        child = np.asarray(step["child"], dtype=float)
        criterion.update(population, child)
        rows = np.asarray(step["rows"], dtype=int)
        replaces = criterion.replaces(child, np.asarray(step["child_values"]), population[rows], np.asarray(step["values"]))

        assert replaces.tolist() == step["replaces"], f"step {number}"
        state = criterion.state()
        value = state["epsilon"] if case["criterion"] == "improved_epsilon" else state["threshold"]
        assert value == pytest.approx(step["state"], rel=1e-12, abs=0.0), f"step {number}"


@pytest.mark.parametrize(("decay_fraction", "budget", "pop_size", "tc"), [(0.8, 10000, 100, 80), (0.25, 1000, 100, 3), (0.1, 50, 100, 1)])
def test_tc_rounds_the_decay_generation_as_evolver(decay_fraction: float, budget: int, pop_size: int, tc: int) -> None:
    """Java's Math.round takes 2.5 to 3; with fewer evaluations than the population, one generation."""
    criterion = make_subproblem_update_criterion(
        ("improved_epsilon", {"tao": 0.1, "feasibility_ratio_threshold": 0.9, "decay_fraction": decay_fraction}), budget, pop_size
    )

    assert isinstance(criterion, ImprovedEpsilonCriterion)
    assert criterion.tc == tc


def test_feasibility_rules_keeps_moead_own_replacement() -> None:
    assert make_subproblem_update_criterion(None, 1000, 10) is None
    assert make_subproblem_update_criterion("feasibility_rules", 1000, 10) is None
    assert isinstance(make_subproblem_update_criterion("violation_threshold", 1000, 10), ViolationThresholdCriterion)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("tabu", "Unknown subproblem update criterion"),
        ("improved_epsilon", "missing"),
        (("violation_threshold", {"tao": 0.1}), "unknown"),
        (("improved_epsilon", {"tao": 1.0, "feasibility_ratio_threshold": 0.9, "decay_fraction": 0.5}), r"tao must lie in \(0.0, 1.0\)"),
        (("improved_epsilon", {"tao": 0.1, "feasibility_ratio_threshold": 0.9, "decay_fraction": 0.0}), "decay_fraction"),
        (3, "criterion name or"),
    ],
)
def test_invalid_criteria_are_rejected(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_subproblem_update_criterion(value)


# ---- MOEA/D configuration ----------------------------------------------------


def test_the_criterion_is_the_last_optional_field_and_round_trips_through_json() -> None:
    fields = dataclasses.fields(MOEADConfig)
    assert fields[-1].name == "subproblem_update_criterion"
    assert fields[-1].default is None

    builder = MOEADConfig.builder()
    assert builder.subproblem_update_criterion(IMPROVED_EPSILON[0], **IMPROVED_EPSILON[1]) is builder
    config = dataclasses.replace(MOEADConfig.default(pop_size=30, n_var=10), subproblem_update_criterion=IMPROVED_EPSILON)

    restored = MOEADConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    assert restored == config
    assert restored.subproblem_update_criterion == IMPROVED_EPSILON


def test_a_criterion_needs_the_feasibility_mode() -> None:
    config = dataclasses.replace(MOEADConfig.default(pop_size=30, n_var=10), constraint_mode="none")

    with pytest.raises(ValueError, match="needs constraint_mode 'feasibility'"):
        dataclasses.replace(config, subproblem_update_criterion="violation_threshold")
    assert dataclasses.replace(config, subproblem_update_criterion="feasibility_rules").subproblem_update_criterion == (
        "feasibility_rules",
        {},
    )


# ---- MOEA/D runs ----------------------------------------------------------------


def test_explicit_feasibility_rules_reproduce_the_default() -> None:
    default, explicit = _run(None), _run("feasibility_rules")

    assert np.array_equal(default.X, explicit.X)
    assert np.array_equal(default.F, explicit.F)


@pytest.mark.parametrize("criterion", CRITERIA, ids=lambda criterion: criterion[0])
def test_each_criterion_runs_reproducibly_and_changes_the_search(criterion: Any) -> None:
    first, second = _run(criterion), _run(criterion)

    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.F, second.F)
    assert not np.array_equal(first.X, _run(None).X)


@pytest.mark.parametrize("criterion", CRITERIA, ids=lambda criterion: criterion[0])
def test_engines_give_the_same_run(criterion: Any) -> None:
    pytest.importorskip("numba")
    reference, compiled = _run(criterion, "numpy"), _run(criterion, "numba")

    assert np.array_equal(reference.X, compiled.X)
    assert np.array_equal(reference.F, compiled.F)


class _StopAtGeneration:
    """A live callback that asks the run to stop once it reaches ``generation``."""

    def __init__(self, generation: int) -> None:
        self.generation = generation
        self.reached = False

    def on_start(self, ctx: object = None) -> None:
        return None

    def on_generation(self, generation: int, F: object = None, X: object = None, stats: object = None) -> None:
        self.reached = generation >= self.generation

    def on_end(self, final_F: object = None, final_stats: object = None) -> None:
        return None

    def should_stop(self) -> bool:
        return self.reached


@pytest.mark.parametrize("criterion", CRITERIA, ids=lambda criterion: criterion[0])
def test_a_run_resumed_from_the_returned_checkpoint_keeps_the_criterion_state(criterion: Any) -> None:
    from vamos.engine.algorithm.moead import MOEAD

    def run(**kwargs: Any) -> dict[str, Any]:
        config = dataclasses.replace(MOEADConfig.default(pop_size=30, n_var=10), subproblem_update_criterion=criterion)
        algorithm = MOEAD(config.to_dict(), kernel=resolve_kernel("numpy"))
        return dict(algorithm.run(LIRCMOP(2, n_var=10), ("max_evaluations", 2400), seed=9, **kwargs))

    whole = run()
    stopped = run(live_viz=_StopAtGeneration(25))
    checkpoint = stopped["checkpoint"]
    assert checkpoint["n_eval"] < 2400
    assert "subproblem_update_criterion" in checkpoint["extra"]

    resumed = run(checkpoint=checkpoint)

    assert np.array_equal(whole["X"], resumed["X"])
    assert np.array_equal(whole["F"], resumed["F"])


# ---- Tuning space ---------------------------------------------------------------


def test_the_moead_constrained_space_tunes_the_criterion() -> None:
    from vamos.engine.tuning.racing.bridge import build_moead_constrained_config_space
    from vamos.engine.tuning.racing.bridge_assignments import config_from_assignment

    space = build_moead_constrained_config_space()
    criterion = next(parameter for parameter in space.flatten() if parameter.name == "subproblem_update_criterion")
    names = {parameter.name for parameter in space.flatten({"subproblem_update_criterion": "improved_epsilon"})}
    assignment = space.sample(np.random.default_rng(0))
    assignment.update(
        subproblem_update_criterion="improved_epsilon",
        improved_epsilon_tao=0.2,
        improved_epsilon_feasibility_ratio_threshold=0.7,
        improved_epsilon_decay_fraction=0.5,
    )

    assert list(criterion.choices) == ["feasibility_rules", "improved_epsilon", "violation_threshold"]
    assert {"improved_epsilon_tao", "improved_epsilon_feasibility_ratio_threshold", "improved_epsilon_decay_fraction"} <= names
    assert config_from_assignment("moead_constrained", assignment).subproblem_update_criterion == (
        "improved_epsilon",
        {"tao": 0.2, "feasibility_ratio_threshold": 0.7, "decay_fraction": 0.5},
    )
