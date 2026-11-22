"""The constraint-handling techniques that jMetal does not have: static and dynamic penalties, and the violation as an objective.

jMetal has no implementation of them to compare with, so these tests check ranks of small populations computed by
hand, and the generation counter of the dynamic penalty. ``test_constraint_handling_techniques`` and
``test_constraint_handling_algorithms`` run them in every algorithm that takes them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vamos.engine.algorithm.components.constrained_ranking import pareto_ranks, violation
from vamos.engine.algorithm.components.constraint_handling import (
    CONSTRAINT_HANDLING_TECHNIQUES,
    JMETAL_TECHNIQUES,
    PAIRWISE_TECHNIQUES,
    make_constraint_handling,
    normalize_constraint_handling,
    run_technique,
)
from vamos.engine.algorithm.components.constraint_penalties import DynamicPenalty, StaticPenalty, ViolationObjective, penalized_objectives
from vamos.engine.tuning.grammar import builtin_parameter_space
from vamos.engine.tuning.racing.bridge_assignments import config_from_assignment
from vamos.engine.tuning.racing.parameter_names import to_evolver
from vamos.foundation.kernel.registry import resolve_kernel

# Every technique that jMetal does not have; test_constraint_handling_rankings checks ToR and ShiP, and
# test_constraint_handling_orders IDEA and ATM.
NEW_TECHNIQUES = (
    "static_penalty",
    "dynamic_penalty",
    "violation_objective",
    "two_rankings",
    "shift_penalty",
    "infeasibility_driven",
    "adaptive_tradeoff",
)
ENGINES = ["numpy", "numba"]
DYNAMIC = {"constant": 0.5, "alpha": 2.0, "beta": 2.0}


def _kernel(engine: str) -> Any:
    if engine == "numba":
        pytest.importorskip("numba")
    return resolve_kernel(engine)


def _ranks(technique: Any, F: list[list[float]], G: list[list[float]], engine: str = "numpy") -> list[int]:
    ranks = technique.ranks(_kernel(engine), np.array(F, dtype=float), np.array(G, dtype=float), np.random.default_rng(0))
    return [int(rank) for rank in ranks]


def test_the_techniques_follow_those_of_jmetal() -> None:
    assert CONSTRAINT_HANDLING_TECHNIQUES == (*JMETAL_TECHNIQUES, *NEW_TECHNIQUES)
    assert not set(NEW_TECHNIQUES) & set(PAIRWISE_TECHNIQUES)


# ---- static_penalty ------------------------------------------------------------


# Three feasible rows that do not dominate one another, and an infeasible one at (0.2, 0.2) with violation 1.
F4 = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.2]]
G4 = [[-1.0], [-1.0], [-1.0], [1.0]]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_large_coefficient_puts_the_violation_first(engine: str) -> None:
    # With r = 1 the infeasible row moves to (1.2, 1.2), which (0.5, 0.5) dominates.
    assert _ranks(StaticPenalty(1.0), F4, G4, engine) == [0, 0, 0, 1]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_small_coefficient_lets_good_objectives_win(engine: str) -> None:
    # With r = 0.1 it moves to (0.3, 0.3), which dominates (0.5, 0.5).
    assert _ranks(StaticPenalty(0.1), F4, G4, engine) == [0, 0, 1, 0]


def test_the_static_penalty_adds_the_overall_violation_to_every_objective() -> None:
    F = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    G = np.array([[0.5, -2.0, 1.5], [-1.0, 0.0, -0.1]])

    # The overall violations are 0.5 + 1.5 = 2 and 0.
    assert violation(G).tolist() == [2.0, 0.0]
    assert penalized_objectives(F, 10.0 * violation(G)).tolist() == [[21.0, 22.0, 23.0], [0.0, 0.0, 0.0]]


@pytest.mark.parametrize("technique", [StaticPenalty(3.0), DynamicPenalty(**DYNAMIC), ViolationObjective()])
def test_a_feasible_population_ranks_by_pareto_dominance(technique: Any) -> None:
    kernel = resolve_kernel("numpy")
    rng = np.random.default_rng(5)
    F = rng.random((30, 3))
    G = -rng.random((30, 2))
    if isinstance(technique, DynamicPenalty):
        technique.bind_run(10, 1000)
        technique.update(kernel, F, G, 0.5)

    assert technique.ranks(kernel, F, G, rng).tolist() == pareto_ranks(kernel, F).tolist()


@pytest.mark.parametrize("technique", [StaticPenalty(2.0), DynamicPenalty(**DYNAMIC), ViolationObjective()])
def test_equal_rows_share_a_front(technique: Any) -> None:
    F = [[0.3, 0.3], [0.3, 0.3], [0.0, 1.0], [1.0, 0.0]]
    G = [[0.5], [0.5], [-1.0], [-1.0]]
    if isinstance(technique, DynamicPenalty):
        technique.bind_run(10, 100)
        technique.update(resolve_kernel("numpy"), np.array(F), np.array(G), 0.2)

    ranks = _ranks(technique, F, G)

    assert ranks[0] == ranks[1]


# ---- dynamic_penalty ------------------------------------------------------------


def test_the_generation_counts_the_evaluations_in_populations() -> None:
    kernel = resolve_kernel("numpy")
    technique = DynamicPenalty(**DYNAMIC)
    technique.bind_run(20, 1000)
    F, G = np.zeros((2, 2)), np.zeros((2, 1))

    # Joines and Houck's generation number starts at 1: the initial population, before any update, and
    # a generational algorithm's first replacement, which ranks it with its offspring.
    assert technique.generation == 1.0
    technique.update(kernel, F, G, 20 / 1000)
    assert technique.generation == pytest.approx(1.0)
    technique.update(kernel, F, G, 0.0)
    assert technique.generation == 1.0
    technique.update(kernel, F, G, 40 / 1000)  # the second replacement
    assert technique.generation == pytest.approx(2.0)
    technique.update(kernel, F, G, 500 / 1000)
    assert technique.generation == pytest.approx(25.0)
    # A steady-state replacement, one offspring later, adds a twentieth of a generation.
    technique.update(kernel, F, G, 501 / 1000)
    assert technique.generation == pytest.approx(25.05)
    assert technique.weight == pytest.approx((0.5 * 25.05) ** 2)


@pytest.mark.parametrize("engine", ENGINES)
def test_the_penalty_grows_with_the_generation(engine: str) -> None:
    kernel = _kernel(engine)
    technique = DynamicPenalty(**DYNAMIC)
    technique.bind_run(10, 100)
    F = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.2], [0.5, 0.5]])
    G = np.array([[-1.0], [-1.0], [0.5], [-1.0]])

    # Before any update k = 1: (0.5 * 1)^2 * 0.5^2 = 0.0625 moves (0.2, 0.2) to (0.2625, 0.2625), which
    # still dominates (0.5, 0.5).
    assert _ranks(technique, F.tolist(), G.tolist(), engine) == [0, 0, 0, 1]
    technique.update(kernel, F, G, 0.1)  # k = 1 again
    assert _ranks(technique, F.tolist(), G.tolist(), engine) == [0, 0, 0, 1]
    # k = 3: (0.5 * 3)^2 * 0.25 = 0.5625 moves it to (0.7625, 0.7625), which (0.5, 0.5) dominates.
    technique.update(kernel, F, G, 0.3)
    assert _ranks(technique, F.tolist(), G.tolist(), engine) == [0, 0, 1, 0]


@pytest.mark.parametrize(("beta", "expected"), [(1.0, [0, 1]), (2.0, [1, 0])])
def test_beta_raises_each_violation(beta: float, expected: list[int]) -> None:
    kernel = resolve_kernel("numpy")
    # A feasible row at (0.6, 0.6), and one at (0, 0) that violates two constraints by 0.7 each.
    F = np.array([[0.6, 0.6], [0.0, 0.0]])
    G = np.array([[-1.0, -1.0], [0.7, 0.7]])
    technique = DynamicPenalty(constant=0.5, alpha=1.0, beta=beta)
    technique.bind_run(10, 100)
    technique.update(kernel, F, G, 0.1)  # k = 1, so the weight is 0.5

    # beta = 1: 0.5 * 1.4 = 0.7 > 0.6, so (0.6, 0.6) dominates; beta = 2: 0.5 * 0.98 = 0.49 < 0.6.
    assert technique.ranks(kernel, F, G, np.random.default_rng(0)).tolist() == expected


def test_the_dynamic_penalty_needs_the_run() -> None:
    technique = DynamicPenalty(**DYNAMIC)
    kernel = resolve_kernel("numpy")

    with pytest.raises(ValueError, match="bind_run"):
        technique.update(kernel, np.zeros((2, 2)), np.ones((2, 1)), 0.5)
    technique.bind_run(10, 100)
    with pytest.raises(ValueError, match="needs its progress"):
        technique.update(kernel, np.zeros((2, 2)), np.ones((2, 1)), float("nan"))
    with pytest.raises(ValueError, match="positive population and budget"):
        technique.bind_run(0, 100)


def test_the_generation_is_the_state_of_a_checkpoint() -> None:
    technique = DynamicPenalty(**DYNAMIC)
    technique.bind_run(10, 100)
    technique.update(resolve_kernel("numpy"), np.zeros((2, 2)), np.ones((2, 1)), 0.37)

    restored = DynamicPenalty(**DYNAMIC)
    restored.restore(technique.state())

    assert technique.state() == {"generation": pytest.approx(3.7)}
    assert restored.generation == technique.generation


def test_run_technique_gives_the_population_and_budget() -> None:
    technique = run_technique(("dynamic_penalty", DYNAMIC), "feasibility", population_size=20, max_evaluations=1000)

    assert isinstance(technique, DynamicPenalty)
    assert (technique.population_size, technique.max_evaluations) == (20, 1000)
    assert isinstance(
        run_technique(("static_penalty", {"coefficient": 1.0}), "feasibility", population_size=20, max_evaluations=1000), StaticPenalty
    )


# ---- violation_objective --------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_the_violation_is_one_more_objective(engine: str) -> None:
    technique = ViolationObjective()
    # (f1, f2, CV): (0, 0, 2), (1, 1, 1) and (2, 2, 0) trade the violation off against the objectives;
    # (3, 3, 0) is dominated by (2, 2, 0), and (0, 0, 3) by (0, 0, 2).
    F = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [0.0, 0.0]]
    G = [[2.0, -1.0], [0.5, 0.5], [-1.0, 0.0], [-2.0, -2.0], [1.0, 2.0]]

    assert _ranks(technique, F, G, engine) == [0, 0, 0, 1, 1]


# ---- Validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (("static_penalty", {"coefficient": 0.0}), "coefficient must lie in"),
        (("static_penalty", {}), "missing: \\['coefficient'\\]"),
        (("dynamic_penalty", {"constant": 0.5, "alpha": 2.0}), "missing: \\['beta'\\]"),
        (("dynamic_penalty", {**DYNAMIC, "alpha": -1.0}), "alpha must lie in"),
        (("dynamic_penalty", {**DYNAMIC, "constant": float("nan")}), "constant must lie in"),
        (("violation_objective", {"coefficient": 1.0}), "unknown: \\['coefficient'\\]"),
    ],
)
def test_invalid_parameters_are_rejected(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_constraint_handling(value)


def test_make_constraint_handling_builds_each_technique() -> None:
    assert isinstance(make_constraint_handling(("static_penalty", {"coefficient": 5.0})), StaticPenalty)
    assert isinstance(make_constraint_handling(("dynamic_penalty", DYNAMIC)), DynamicPenalty)
    assert isinstance(make_constraint_handling("violation_objective"), ViolationObjective)


# ---- Tuning spaces and Evolver ------------------------------------------------------


@pytest.mark.parametrize("space_name", ["nsgaii_constrained", "nsgaiii_constrained", "smsemoa_constrained", "agemoea_constrained"])
def test_the_constrained_spaces_offer_the_new_techniques(space_name: str) -> None:
    space = builtin_parameter_space(space_name)
    technique = space.parameter("constraint_handling")
    # ToR, IDEA and ATM are defined for NSGA-II only (test_constraint_handling_rankings and _orders).
    nsgaii_only = ("two_rankings", "infeasibility_driven", "adaptive_tradeoff")
    offered = NEW_TECHNIQUES if space_name == "nsgaii_constrained" else tuple(name for name in NEW_TECHNIQUES if name not in nsgaii_only)
    assert technique.values == (*JMETAL_TECHNIQUES, *offered)
    assert technique.default == "cdp"
    assert [child.name for child in technique.production("static_penalty")] == ["static_penalty_coefficient"]
    assert [child.name for child in technique.production("dynamic_penalty")] == [
        "dynamic_penalty_constant",
        "dynamic_penalty_alpha",
        "dynamic_penalty_beta",
    ]
    # Dam and Truong's options (test_constraint_handling_violation_variants); feasible_first is NSGA-II's only.
    options = ["violation_objective_norm", "violation_objective_ratio"]
    options += ["violation_objective_feasible_first"] if space_name == "nsgaii_constrained" else []
    assert [child.name for child in technique.production("violation_objective")] == options

    defaults = dict(space.default_values())
    static = {**defaults, "constraint_handling": "static_penalty", "static_penalty_coefficient": 250.0}
    dynamic = {
        **defaults,
        "constraint_handling": "dynamic_penalty",
        "dynamic_penalty_constant": 0.05,
        "dynamic_penalty_alpha": 1.0,
        "dynamic_penalty_beta": 1.5,
    }

    assert config_from_assignment(space_name, static).constraint_handling == ("static_penalty", {"coefficient": 250.0})
    assert config_from_assignment(space_name, dynamic).constraint_handling == (
        "dynamic_penalty",
        {"constant": 0.05, "alpha": 1.0, "beta": 1.5},
    )
    objective = {**defaults, "constraint_handling": "violation_objective"}
    assert config_from_assignment(space_name, objective).constraint_handling == ("violation_objective", {})


@pytest.mark.parametrize("space_name", ["rvea_constrained", "smpso_constrained"])
def test_rvea_and_smpso_spaces_leave_them_out(space_name: str) -> None:
    assert builtin_parameter_space(space_name).parameter("constraint_handling").values == PAIRWISE_TECHNIQUES


@pytest.mark.parametrize("technique", NEW_TECHNIQUES)
def test_evolver_has_no_equivalent(technique: str) -> None:
    with pytest.raises(ValueError, match="no Evolver equivalent"):
        to_evolver({"constraint_handling": technique})


@pytest.mark.parametrize("technique", JMETAL_TECHNIQUES)
def test_the_techniques_of_jmetal_still_pass_to_evolver(technique: str) -> None:
    assert to_evolver({"constraint_handling": technique})["constraint_handling"] == technique


def test_the_reference_lists_every_technique() -> None:
    reference = (Path(__file__).resolve().parents[2] / "docs" / "reference" / "constraints.md").read_text(encoding="utf-8")
    for technique in CONSTRAINT_HANDLING_TECHNIQUES:
        assert f"| `{technique}`" in reference, technique
