"""ToP (Liu and Wang, 2019): parity with jMetal's port, the two phases, exact budgets and the configuration."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.algorithms import ToPConfig
from vamos.engine.algorithm.catalog import initial_evaluations
from vamos.engine.algorithm.registry import get_builtin_algorithm_names, resolve_algorithm
from vamos.engine.algorithm.top import (
    ToP,
    binomial_crossover,
    constrained_front_numbers,
    constraint_violation,
    crowding_distances,
    current_to_rand_1,
    environmental_selection,
    feasible_ratio,
    index_of_lowest_mean_objective,
    mean_objectives,
    offspring_replaces_parent,
    phase_one_continues,
    rand_1,
    rand_to_best_1,
    top_tournament,
    tournament_winners,
    updated_delta,
)
from vamos.engine.tuning import build_top_config_space, config_from_assignment
from vamos.foundation.eval.backends import SerialEvalBackend
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.kernel.operator_primitives import available_variation_kernels, polynomial_mutation_population
from vamos.foundation.problem.registry import make_problem_selection

FIXTURE = Path(__file__).parent / "data" / "top_jmetal_reference.json"
_PAYLOAD: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
SELECTION_CASES: list[dict[str, Any]] = _PAYLOAD["selection"]
DE_CASES: list[dict[str, Any]] = _PAYLOAD["differential_evolution"]
PM_CASES: list[dict[str, Any]] = _PAYLOAD["polynomial_mutation"]
PHASE_ONE_CASES: list[dict[str, Any]] = _PAYLOAD["phase_one"]
PHASE_RULE_CASES: list[dict[str, Any]] = _PAYLOAD["phase_rule"]
_OPERATORS = {"current_to_rand_1": current_to_rand_1, "rand_to_best_1": rand_to_best_1, "rand_1": rand_1}


def _decoded(values: Any) -> np.ndarray:
    """The fixture's floats, with ``"inf"``, ``"-inf"`` and ``"nan"`` for the non-finite ones."""
    return np.asarray([float(v) for v in np.ravel(np.asarray(values, dtype=object))], dtype=float).reshape(np.shape(values))


class _ShiftedFrontProblem:
    """Two objectives whose unconstrained front (x1 = 0) is infeasible: the constraint asks for x1 >= 0.5."""

    n_var = 4
    n_obj = 2
    n_constraints = 1
    xl = np.zeros(4)
    xu = np.ones(4)
    encoding = "real"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        g = 1.0 + X[:, 1] + X[:, 2:].sum(axis=1)
        out["F"] = np.column_stack([X[:, 0], g * (1.0 - np.sqrt(X[:, 0] / g))])
        out["G"] = (0.5 - X[:, 1])[:, None]


class _InfeasibleProblem:
    """Two variables, f = (x1, x2), and a constraint violated everywhere, so Pf is always zero (jMetal's ToPTest)."""

    n_var = 2
    n_obj = 2
    n_constraints = 1
    xl = np.zeros(2)
    xu = np.ones(2)
    encoding = "real"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        out["F"] = X.copy()
        out["G"] = (1.0 + X[:, :1]).copy()


class _LinearFrontProblem:
    """f = (x1, 1 - x1) without constraints: the normalized objectives of every solution add up to one, so delta is about zero."""

    n_var = 2
    n_obj = 2
    xl = np.zeros(2)
    xu = np.ones(2)
    encoding = "real"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        out["F"] = np.column_stack([X[:, 0], 1.0 - X[:, 0]])


def _config(pop_size: int = 10, **fields: Any) -> dict[str, Any]:
    config = ToPConfig.builder().pop_size(pop_size).crossover("de", cr=1.0, f=0.5).mutation("pm", prob="1/n", eta=20.0).build().to_dict()
    config.update(fields)
    return config


# ---------------------------------------------------------------------------
# Parity with jMetal's ToPSelection, ToPDifferentialEvolution, PolynomialMutation and ToP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", SELECTION_CASES, ids=[case["label"] for case in SELECTION_CASES])
def test_selection_matches_jmetal(case: dict[str, Any]) -> None:
    F = np.asarray(case["F"], dtype=float)
    violation = np.asarray(case["violation"], dtype=float)

    fronts = constrained_front_numbers(F, violation)
    distances = crowding_distances(F, fronts)

    assert fronts.tolist() == case["fronts"]
    np.testing.assert_allclose(distances, _decoded(case["crowding"]), rtol=0.0, atol=1e-12, equal_nan=True)
    for size, expected in zip(case["sizes"], case["selected"], strict=True):
        assert environmental_selection(F, violation, size).tolist() == expected, size
    assert tournament_winners(fronts, distances, np.asarray(case["draws"])).tolist() == case["winners"]


@pytest.mark.parametrize("case", DE_CASES, ids=[case["label"] for case in DE_CASES])
def test_trial_vectors_match_jmetal_on_fixed_draws(case: dict[str, Any]) -> None:
    """The mutant, the binomial crossover without j_rand (a draw equal to CR does not cross) and the clamping."""
    vectors = np.asarray(case["vectors"], dtype=float)
    lower, upper = np.asarray(case["lower"]), np.asarray(case["upper"])
    columns = [vectors[:, v] for v in range(vectors.shape[1])]
    mutant = _OPERATORS[case["operator"]](*columns, case["f"])
    # The base vector: x_i for DE/current-to-rand/1, x_r1 for DE/rand-to-best/1/bin, x_1 for DE/rand/1/bin.
    base = columns[1] if case["operator"] == "rand_to_best_1" else columns[0]

    trials = np.clip(binomial_crossover(base, mutant, np.asarray(case["draws"]), case["cr"]), lower, upper)

    assert case["draws_used"] == trials.size
    np.testing.assert_allclose(trials, case["trials"], rtol=0.0, atol=1e-12)


def _variation_kernels() -> list[Any]:
    kernels: list[Any] = [pytest.param(None, id="numpy")]
    compiled = available_variation_kernels()
    return kernels if compiled is None else [*kernels, pytest.param(compiled, id="numba")]


@pytest.mark.parametrize("kernels", _variation_kernels())
@pytest.mark.parametrize("case", PM_CASES, ids=[case["label"] for case in PM_CASES])
def test_polynomial_mutation_matches_jmetal_on_fixed_draws(case: dict[str, Any], kernels: Any) -> None:
    """VAMOS's polynomial mutation is jMetal's: a draw not greater than the probability mutates, the same formula, the clamping.

    jMetal draws the perturbation only for the variables it mutates, and none for a variable whose
    bounds are equal; VAMOS draws two full grids. The fixture's sequence interleaves the grids in
    jMetal's order.
    """
    X = np.asarray(case["X"], dtype=float)

    mutated = polynomial_mutation_population(
        X,
        rng=np.random.default_rng(0),
        lower=np.asarray(case["lower"]),
        upper=np.asarray(case["upper"]),
        prob_mutation=case["prob"],
        eta=case["eta"],
        rnd_mask=np.asarray(case["mask_draws"]),
        rnd_delta=np.asarray(case["delta_draws"]),
        kernels=kernels,
    )

    assert case["draws_used"] == len(case["sequence"])
    np.testing.assert_allclose(mutated, case["mutated"], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("case", PHASE_ONE_CASES, ids=[case["label"] for case in PHASE_ONE_CASES])
def test_phase_one_matches_jmetal(case: dict[str, Any]) -> None:
    """Violations, mean objectives, the best solution, Pf, delta and the one-to-one replacement."""
    F, F_off = np.asarray(case["F"], dtype=float), np.asarray(case["offspring_F"], dtype=float)
    # jMetal's negative constraint values are violated; VAMOS's positive ones.
    G, G_off = -np.asarray(case["constraints"], dtype=float), -np.asarray(case["offspring_constraints"], dtype=float)
    violation, offspring_violation = constraint_violation(G, F.shape[0]), constraint_violation(G_off, F.shape[0])

    np.testing.assert_allclose(violation, case["violation"], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(offspring_violation, case["offspring_violation"], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(mean_objectives(F), case["mean"], rtol=0.0, atol=1e-12)
    assert index_of_lowest_mean_objective(F) == case["best"]
    assert feasible_ratio(violation) == case["feasible_ratio"]
    np.testing.assert_allclose(
        updated_delta(F, violation, case["delta"]), _decoded(case["delta_before"]), rtol=0.0, atol=1e-12, equal_nan=True
    )

    replaces = offspring_replaces_parent(offspring_violation, mean_objectives(F_off), violation, mean_objectives(F))
    assert replaces.tolist() == case["replaces"] == case["kept_offspring"]
    next_F = np.where(replaces[:, None], F_off, F)
    next_violation = np.where(replaces, offspring_violation, violation)
    assert feasible_ratio(next_violation) == case["next_feasible_ratio"]
    np.testing.assert_allclose(
        updated_delta(next_F, next_violation, case["delta"]), _decoded(case["next_delta"]), rtol=0.0, atol=1e-12, equal_nan=True
    )


def test_phase_rule_matches_jmetal() -> None:
    for case in PHASE_RULE_CASES:
        delta = float(case["delta"])
        decision = phase_one_continues(delta, case["feasible_ratio"], case["evaluations"], case["max_evaluations"])
        assert decision == case["phase_one"], case


def test_fixture_exercises_the_edge_cases() -> None:
    """NaN and infinite distances, fronts cut by the truncation, equal violations, a NaN delta and a kept delta."""
    seen = set()
    for case in SELECTION_CASES:
        distances = _decoded(case["crowding"])
        fronts = np.asarray(case["fronts"])
        violation = np.asarray(case["violation"])
        if np.isnan(distances).any():
            seen.add("nan distance")
        if np.isinf(distances).any() and np.isfinite(distances).any():
            seen.add("finite and infinite distances")
        infeasible = violation[violation > 0.0]
        if infeasible.size > np.unique(infeasible).size:
            seen.add("equal violations")
        counts = np.cumsum(np.bincount(fronts))
        if any(0 < size < fronts.size and size not in counts for size in case["sizes"]):
            seen.add("truncated front")
    for case in PHASE_ONE_CASES:
        delta = float(case["delta_before"]) if isinstance(case["delta_before"], float) else math.nan
        if math.isnan(delta):
            seen.add("nan delta")
        if delta == case["delta"]:
            seen.add("kept delta")
    assert seen == {"nan distance", "finite and infinite distances", "equal violations", "truncated front", "nan delta", "kept delta"}


# ---------------------------------------------------------------------------
# Selection, phase 1 and the switch
# ---------------------------------------------------------------------------


def test_constrained_sorting_puts_infeasible_behind_and_groups_equal_violations() -> None:
    F = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 1.0], [0.0, 0.0], [5.0, 5.0], [0.1, 0.1]])
    violation = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.5])

    # Two feasible fronts, then the violations 0.5 and 1.0: (5, 5) and (0, 0) share a front.
    assert constrained_front_numbers(F, violation).tolist() == [0, 1, 0, 3, 3, 2]


# Four non-dominated rows with a flat first objective, and an infeasible one.
_FLAT_F = np.array([[0.5, 0.1, 0.9], [0.5, 0.4, 0.6], [0.5, 0.9, 0.1], [0.5, 0.6, 0.4], [0.6, 0.2, 0.2]])
_FLAT_VIOLATION = np.array([0.0, 0.0, 0.0, 0.0, 1.0])


def test_crowding_gives_nan_to_a_flat_objective_and_keeps_extremes_infinite() -> None:
    fronts = constrained_front_numbers(_FLAT_F, _FLAT_VIOLATION)

    distances = crowding_distances(_FLAT_F, fronts)

    # Objective 1 is flat: its first and last rows in row order (0 and 3) are infinite and the inner ones (1 and 2)
    # NaN; objective 2 sorts the rows as 0, 1, 3, 2, which makes row 2 infinite again, and objective 3 as 2, 3, 1, 0.
    assert fronts.tolist() == [0, 0, 0, 0, 1]
    np.testing.assert_array_equal(distances, [np.inf, np.nan, np.inf, np.inf, np.inf])


def test_truncation_prefers_nan_then_larger_distances_and_the_tournament_takes_nan_as_worst() -> None:
    fronts = constrained_front_numbers(_FLAT_F, _FLAT_VIOLATION)
    distances = crowding_distances(_FLAT_F, fronts)

    # Row 1 has a NaN distance and rows 0, 2 and 3 infinite ones: NaN first, then the first rows.
    assert environmental_selection(_FLAT_F, _FLAT_VIOLATION, 1).tolist() == [1]
    assert environmental_selection(_FLAT_F, _FLAT_VIOLATION, 2).tolist() == [0, 1]
    assert environmental_selection(_FLAT_F, _FLAT_VIOLATION, 3).tolist() == [0, 1, 2]
    assert environmental_selection(_FLAT_F, _FLAT_VIOLATION, 5).tolist() == [0, 1, 2, 3, 4]
    pairs = np.array([[1, 0], [0, 1], [1, 3], [3, 2], [4, 1], [2, 2]])
    assert tournament_winners(fronts, distances, pairs).tolist() == [0, 0, 3, 2, 1, 2]


def test_tournament_draws_pairs_with_replacement() -> None:
    fronts = np.array([0, 1, 0, 2, 1, 0, 3])
    distances = np.array([np.inf, 0.5, 0.2, np.nan, np.inf, 0.2, 1.0])
    draws = np.random.default_rng(3).integers(0, 7, size=(40, 2))

    winners = top_tournament(fronts, distances, 40, np.random.default_rng(3))

    np.testing.assert_array_equal(winners, tournament_winners(fronts, distances, draws))


def test_replacement_follows_the_feasibility_rule() -> None:
    """jMetal's ToPTest cases: (offspring violation, mean) against (parent violation, mean)."""
    cases = [
        (0.1, 9.0, 0.2, 1.0, True),
        (0.0, 9.0, 0.2, 1.0, True),
        (0.3, 1.0, 0.2, 9.0, False),
        (0.2, 9.0, 0.0, 1.0, False),
        (0.0, 1.0, 0.0, 2.0, True),
        (0.0, 2.0, 0.0, 1.0, False),
        (0.0, 1.0, 0.0, 1.0, False),
        (0.5, 1.0, 0.5, 2.0, True),
        (0.5, 2.0, 0.5, 1.0, False),
    ]
    columns = np.asarray([case[:4] for case in cases]).T

    replaces = offspring_replaces_parent(columns[0], columns[1], columns[2], columns[3])

    assert replaces.tolist() == [case[4] for case in cases]


def test_delta_keeps_its_value_below_three_feasible_and_is_nan_for_a_flat_objective() -> None:
    F = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 4.0], [4.0, 2.0], [2.0, 2.0], [4.0, 4.0], [-10.0, -10.0]])
    violation = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])

    # Normalized with (0, 0) and (4, 4), the sums are 0, 0.5, 1.5, 1.5, 1 and 2: the second minus the first.
    assert updated_delta(F, violation, 1.0) == pytest.approx(0.5, abs=1e-12)
    assert updated_delta(F[:3], np.array([0.0, 0.0, 0.2]), 0.7) == 0.7
    assert math.isnan(updated_delta(np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]), np.zeros(3), 1.0))
    # A NaN delta fails its test: only a low Pf keeps phase 1.
    assert not phase_one_continues(math.nan, 1.0, 100, 1000)
    assert phase_one_continues(math.nan, 0.0, 100, 1000)


def test_ask_creates_phase_one_trial_vectors_from_the_documented_draws() -> None:
    """The draws replayed: the operator choice, r1 r2 r3, the crossover, then the mutation (clamping first)."""
    problem = make_problem_selection("mw3", n_var=6).instantiate()
    algorithm = ToP(_config(12), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 1000), seed=5)
    state = algorithm.state
    assert state is not None and state.variation is not None and algorithm.next_phase() == 1
    rng = copy.deepcopy(state.rng)
    X = state.X
    best = X[index_of_lowest_mean_objective(state.F)]
    choice = (rng.random(12) < 0.5)[:, None]
    drawn = rng.integers(0, 12, size=(12, 3))
    r1, r2, r3 = X[drawn[:, 0]], X[drawn[:, 1]], X[drawn[:, 2]]
    mutant = np.where(choice, current_to_rand_1(X, r1, r2, r3, 0.5), rand_to_best_1(best, r1, r2, r3, 0.5))
    trial = np.clip(binomial_crossover(np.where(choice, X, r1), mutant, rng.random(X.shape), 1.0), problem.xl, problem.xu)
    expected = state.variation.mutate(trial, rng)

    np.testing.assert_array_equal(algorithm.ask(), expected)


def test_ask_creates_phase_two_offspring_from_the_documented_draws() -> None:
    problem = _LinearFrontProblem()
    algorithm = ToP(_config(10), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 1000), seed=2)
    algorithm.tell(SerialEvalBackend().evaluate(algorithm.ask(), problem))
    state = algorithm.state
    assert state is not None and state.variation is not None and algorithm.next_phase() == 2
    rng = copy.deepcopy(state.rng)
    fronts = constrained_front_numbers(state.F, np.zeros(10))
    parents = top_tournament(fronts, crowding_distances(state.F, fronts), 10, rng)
    others = rng.integers(0, 10, size=(10, 2))
    base = state.X[parents]
    trial = np.clip(
        binomial_crossover(base, rand_1(base, state.X[others[:, 0]], state.X[others[:, 1]], 0.5), rng.random(base.shape), 1.0), 0.0, 1.0
    )
    expected = state.variation.mutate(trial, rng)

    np.testing.assert_array_equal(algorithm.ask(), expected)
    assert state.phase_two_start == 20


# ---------------------------------------------------------------------------
# The algorithm: phases, budget and results
# ---------------------------------------------------------------------------


def test_without_feasible_solutions_phase_two_starts_after_ninety_percent() -> None:
    """jMetal's ToPTest: phase 1 runs at 10, 20, ..., 180 evaluations (180 <= 0.9 * 200)."""
    result = ToP(_config(10), kernel=NumPyKernel()).run(_InfeasibleProblem(), ("max_evaluations", 200), seed=1)

    assert result["phase_two_start"] == 190
    assert result["evaluations"] == 200 and result["population"]["X"].shape == (10, 2)
    assert ToP(_config(10), kernel=NumPyKernel()).run(_InfeasibleProblem(), ("max_evaluations", 50), seed=1)["phase_two_start"] is None


def test_small_delta_with_every_solution_feasible_starts_phase_two_after_one_generation() -> None:
    result = ToP(_config(10), kernel=NumPyKernel()).run(_LinearFrontProblem(), ("max_evaluations", 200), seed=1)

    assert result["phase_two_start"] == 20
    assert result["population"]["X"].shape == (10, 2)


@pytest.mark.parametrize("extra", [0, 1, 5, 9, 10, 23])
def test_budget_is_exact(extra: int) -> None:
    budget = 10 + extra

    for problem in (_ShiftedFrontProblem(), _LinearFrontProblem()):
        result = ToP(_config(10), kernel=NumPyKernel()).run(problem, ("max_evaluations", budget), seed=3)
        assert result["evaluations"] == budget
        assert result["population"]["X"].shape == (10, problem.n_var)


def test_budget_not_a_multiple_of_the_population_is_exact_with_jmetals_phases() -> None:
    """jMetal's ToPTest runs 100 evaluations for a budget of 95; VAMOS evaluates 95 and decides the same phases."""
    result = ToP(_config(10), kernel=NumPyKernel()).run(_LinearFrontProblem(), ("max_evaluations", 95), seed=1)
    assert result["evaluations"] == 95 and result["phase_two_start"] == 20

    # 195 evaluations: phase 1 at 10 to 170 (170 <= 175.5), phase 2 from 180, the last generation of 5.
    infeasible = ToP(_config(10), kernel=NumPyKernel()).run(_InfeasibleProblem(), ("max_evaluations", 195), seed=1)
    assert infeasible["evaluations"] == 195 and infeasible["phase_two_start"] == 180


def test_partial_phase_one_generation_replaces_only_the_first_positions() -> None:
    problem = _ShiftedFrontProblem()
    algorithm = ToP(_config(10), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 14), seed=4)
    reference = ToP(_config(10), kernel=NumPyKernel())
    reference.initialize(problem, ("max_evaluations", 1000), seed=4)
    state = algorithm.state
    assert state is not None and state.G is not None and algorithm.next_phase() == 1
    X, F, G = state.X.copy(), state.F.copy(), state.G.copy()

    X_off = algorithm.ask()
    np.testing.assert_array_equal(X_off, reference.ask()[:4])
    evaluated = SerialEvalBackend().evaluate(X_off, problem)
    algorithm.tell(evaluated)

    replaces = offspring_replaces_parent(
        constraint_violation(evaluated.G, 4), mean_objectives(evaluated.F), constraint_violation(G[:4], 4), mean_objectives(F[:4])
    )
    expected_X = X.copy()
    expected_X[:4][replaces] = X_off[replaces]
    np.testing.assert_array_equal(state.X, expected_X)
    np.testing.assert_array_equal(state.X[4:], X[4:])
    violation = constraint_violation(state.G, 10)
    assert state.feasible_ratio == feasible_ratio(violation)
    assert state.delta == updated_delta(state.F, violation, 1.0)
    assert state.n_eval == 14 and algorithm.should_terminate()
    with pytest.raises(RuntimeError, match="no remaining evaluation budget"):
        algorithm.ask()


def test_partial_phase_two_generation_selects_from_the_population_and_the_first_offspring() -> None:
    problem = _LinearFrontProblem()
    algorithm = ToP(_config(10), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 27), seed=6)
    algorithm.tell(SerialEvalBackend().evaluate(algorithm.ask(), problem))
    state = algorithm.state
    assert state is not None and algorithm.next_phase() == 2
    X, F = state.X.copy(), state.F.copy()

    X_off = algorithm.ask()
    assert X_off.shape == (7, 2)
    evaluated = SerialEvalBackend().evaluate(X_off, problem)
    algorithm.tell(evaluated)

    selected = environmental_selection(np.vstack([F, evaluated.F]), np.zeros(17), 10)
    np.testing.assert_array_equal(state.X, np.vstack([X, X_off])[selected])
    assert state.n_eval == 27 and state.phase_two_start == 20


def test_budget_must_cover_the_initial_population() -> None:
    problem = _ShiftedFrontProblem()
    with pytest.raises(ValueError, match="max_evaluations >= pop_size"):
        ToP(_config(10), kernel=NumPyKernel()).run(problem, ("max_evaluations", 9), seed=1)
    assert ToP(_config(10), kernel=NumPyKernel()).run(problem, ("max_evaluations", 10), seed=1)["evaluations"] == 10
    with pytest.raises(ValueError, match="pop_size >= 1"):
        ToP({**_config(10), "pop_size": 0}, kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=1)
    # One individual: every DE vector and both tournament candidates are the same solution.
    assert ToP(_config(1), kernel=NumPyKernel()).run(problem, ("max_evaluations", 30), seed=1)["evaluations"] == 30


def test_termination_criteria() -> None:
    problem = _ShiftedFrontProblem()
    assert ToP(_config(10), kernel=NumPyKernel()).run(problem, ("n_gen", 5), seed=1)["evaluations"] == 50

    reachable = {"target_value": 0.0, "reference_point": [11.0, 11.0], "max_evaluations": 500}
    early = ToP(_config(10), kernel=NumPyKernel()).run(problem, ("hv", reachable), seed=1)
    assert early["hv_reached"] is True and early["evaluations"] < 500


def test_run_is_deterministic_and_equals_ask_tell() -> None:
    problem = _ShiftedFrontProblem()
    first = ToP(_config(10), kernel=NumPyKernel()).run(problem, ("max_evaluations", 405), seed=7)
    second = ToP(_config(10), kernel=NumPyKernel()).run(problem, ("max_evaluations", 405), seed=7)
    other = ToP(_config(10), kernel=NumPyKernel()).run(problem, ("max_evaluations", 405), seed=8)

    algorithm = ToP(_config(10), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 405), seed=7)
    while not algorithm.should_terminate():
        X = algorithm.ask()
        algorithm.tell(SerialEvalBackend().evaluate(X, problem))
    stepped = algorithm.result()

    for key in ("X", "F", "G"):
        np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first[key], stepped[key])
    np.testing.assert_array_equal(first["population"]["X"], stepped["population"]["X"])
    assert first["phase_two_start"] == stepped["phase_two_start"] is not None
    assert not np.array_equal(first["population"]["X"], other["population"]["X"])


def test_result_modes_and_constraint_mode_none() -> None:
    problem = _ShiftedFrontProblem()
    population = ToP(_config(10, result_mode="population"), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=2)
    assert population["X"].shape == (10, 4)

    unconstrained = ToP(_config(10, constraint_mode="none"), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=2)
    assert "G" not in unconstrained and "G" not in unconstrained["population"]


def test_the_switch_thresholds_are_configurable() -> None:
    """With a fraction of 0, phase 2 starts at once; with a delta threshold of 0, phase 1 lasts until 90%."""
    problem = _LinearFrontProblem()
    at_once = ToP(_config(10, phase_one_evaluation_fraction=0.0), kernel=NumPyKernel()).run(problem, ("max_evaluations", 200), seed=1)
    assert at_once["phase_two_start"] == 10
    late = ToP(_config(10, delta_threshold=0.0), kernel=NumPyKernel()).run(problem, ("max_evaluations", 200), seed=1)
    assert late["phase_two_start"] == 190
    with pytest.raises(ValueError, match="feasible_ratio_threshold must be in"):
        ToP(_config(10, feasible_ratio_threshold=1.5), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=1)


@pytest.mark.parametrize(("problem", "kwargs"), [("c1dtlz1", {"n_var": 7, "n_obj": 3}), ("mw3", {})])
def test_engines_give_the_same_run(problem: str, kwargs: dict[str, int]) -> None:
    """The engine changes the speed of a run, not its result."""
    pytest.importorskip("numba")
    runs = [
        vamos.optimize(problem, algorithm="top", pop_size=20, max_evaluations=1005, seed=3, engine=engine, **kwargs).data
        for engine in ("numpy", "numba")
    ]

    for array in ("X", "F", "G"):
        np.testing.assert_array_equal(runs[1]["population"][array], runs[0]["population"][array])
    assert runs[1]["phase_two_start"] == runs[0]["phase_two_start"]


def test_variation_choices_and_encodings_are_validated() -> None:
    problem = _ShiftedFrontProblem()
    with pytest.raises(ValueError, match="crossover must be 'de'"):
        ToP({**_config(10), "crossover": ("sbx", {"prob": 1.0, "eta": 20.0})}, kernel=NumPyKernel()).run(
            problem, ("max_evaluations", 100), seed=1
        )
    with pytest.raises(ValueError, match="f must be positive"):
        ToP({**_config(10), "crossover": ("de", {"cr": 1.0, "f": 0.0})}, kernel=NumPyKernel()).run(
            problem, ("max_evaluations", 100), seed=1
        )
    with pytest.raises(ValueError, match="cr must be in"):
        ToP({**_config(10), "crossover": ("de", {"cr": 1.5})}, kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=1)
    for name in ("bin_knapsack", "tsp6", "int_alloc", "mixed_design"):
        discrete = make_problem_selection(name).instantiate()
        with pytest.raises(ValueError, match="needs real variables"):
            ToP(_config(10), kernel=NumPyKernel()).run(discrete, ("max_evaluations", 100), seed=1)
        with pytest.raises((ValueError, ConfigurationError), match="needs real variables"):
            vamos.optimize(discrete, algorithm="top", max_evaluations=300, seed=1, engine="numpy")


def test_other_real_mutations_and_repairs_run() -> None:
    problem = _ShiftedFrontProblem()
    for mutation, repair in (
        (("gaussian", {"prob": 0.25, "sigma": 0.1}), ("reflect", {})),
        (("uniform_reset", {"prob": 0.25}), ("random", {})),
    ):
        result = ToP(_config(10, mutation=mutation, repair=repair), kernel=NumPyKernel()).run(problem, ("max_evaluations", 205), seed=1)
        X = result["population"]["X"]
        assert result["evaluations"] == 205 and np.all((X >= 0.0) & (X <= 1.0))


def test_mw3_regression() -> None:
    """A short run on MW3 ends feasible and near the front.

    With 10,000 evaluations, seeds 1 to 10 give a median IGD of 2.8e-2 against VAMOS's front (seed 2:
    2.0e-2); in three of them the population converges to one end of the front (IGD 0.92), as in 4 of
    the 15 runs of jMetal's ToPIT. The threshold only catches a broken algorithm. The comparison with
    jMetal and the published values is in docs/algorithms/top.md.
    """
    problem = make_problem_selection("mw3").instantiate()
    config = ToPConfig.default(n_var=problem.n_var).to_dict()

    result = ToP(config, kernel=NumPyKernel()).run(problem, ("max_evaluations", 10_000), seed=2)

    front = np.loadtxt(Path(vamos.__file__).parent / "resources" / "reference_fronts" / "mw3.csv", delimiter=",")
    population = result["population"]
    assert population["F"].shape == (100, 2)
    assert np.all(population["G"] <= 0.0)
    distances = np.linalg.norm(front[:, None, :] - population["F"][None, :, :], axis=2)
    assert float(distances.min(axis=1).mean()) < 0.05
    assert result["phase_two_start"] == 2000


# ---------------------------------------------------------------------------
# Configuration, registry and tuning space
# ---------------------------------------------------------------------------


def test_default_configuration_is_platemos() -> None:
    config = ToPConfig.default(n_var=10)

    assert config.pop_size == 100
    assert config.crossover == ("de", {"cr": 1.0, "f": 0.5})
    assert config.mutation == ("pm", {"prob": 0.1, "eta": 20.0})
    assert (config.delta_threshold, config.feasible_ratio_threshold, config.phase_one_evaluation_fraction) == (0.2, 1.0 / 3.0, 0.9)
    assert ToPConfig.from_dict(json.loads(config.to_json())) == config


def test_configuration_validates_the_variation_and_the_thresholds() -> None:
    builder = ToPConfig.builder().pop_size(10).mutation("pm", prob="1/n", eta=20.0)
    with pytest.raises(ConfigurationError, match="crossover must be 'de'"):
        builder.crossover("sbx", prob=1.0, eta=20.0).build()
    with pytest.raises(ConfigurationError, match="parameter f must be positive"):
        builder.crossover("de", cr=1.0, f=-0.5).build()
    with pytest.raises(ConfigurationError, match="cr"):
        builder.crossover("de", cr=2.0, f=0.5).build()
    builder.crossover("de", cr=0.9, f=0.7)
    with pytest.raises(ConfigurationError, match="phase_one_evaluation_fraction must be in"):
        builder.phase_one_evaluation_fraction(1.2).build()
    config = builder.phase_one_evaluation_fraction(0.8).delta_threshold(0.1).feasible_ratio_threshold(0.5).build()
    assert (config.delta_threshold, config.feasible_ratio_threshold, config.phase_one_evaluation_fraction) == (0.1, 0.5, 0.8)


def test_registry_and_optimize() -> None:
    assert "top" in get_builtin_algorithm_names()
    assert callable(resolve_algorithm("top"))
    assert initial_evaluations("top", 100) == 100

    result = vamos.optimize("mw3", algorithm="top", pop_size=20, max_evaluations=105, seed=1, engine="numpy")

    assert result.data["evaluations"] == 105
    assert result.meta["run_artifact_resolved_spec"]["algorithm"]["component_id"] == "vamos.algorithm:top@1"
    with pytest.raises(ConfigurationError, match="max_evaluations must be >= pop_size"):
        vamos.optimize("mw3", algorithm="top", pop_size=20, max_evaluations=19, seed=1)


def test_study_planning_checks_the_budget() -> None:
    from vamos.experiment.study.errors import UnresolvedStudyTaskError

    spec = vamos.StudySpec(problems=["mw3"], algorithms=["top"], seeds=[0], pop_size=100, max_evaluations=99)
    with pytest.raises(UnresolvedStudyTaskError) as excinfo:
        vamos.plan_study(spec)
    assert excinfo.value.reason == "INVALID_EVALUATION_BUDGET"

    assert vamos.plan_study(vamos.StudySpec(problems=["mw3"], algorithms=["top"], seeds=[0], pop_size=100, max_evaluations=150)) is not None


def test_tuning_space_builds_differential_evolution_configurations() -> None:
    space = build_top_config_space()
    names = {param.name for param in space.params}
    assert {
        "pop_size",
        "de_cr",
        "de_f",
        "mutation",
        "delta_threshold",
        "feasible_ratio_threshold",
        "phase_one_evaluation_fraction",
    } <= names
    assert not {"crossover", "crossover_prob"} & names
    rng = np.random.default_rng(0)
    mutations = set()
    for _ in range(40):
        assignment = space.sample(rng)
        config = config_from_assignment("top", assignment)
        assert isinstance(config, ToPConfig)
        assert config.crossover == ("de", {"cr": assignment["de_cr"], "f": assignment["de_f"]})
        assert config.delta_threshold == assignment["delta_threshold"]
        mutations.add(config.mutation[0])
        assert 20 <= config.pop_size <= 200
    assert len(mutations) > 3

    platemo = config_from_assignment("top", {"pop_size": 100, "mutation": "pm"})
    assert platemo.crossover == ("de", {"cr": 1.0, "f": 0.5})
    assert platemo.mutation == ("pm", {"eta": 20.0})
    assert (platemo.delta_threshold, platemo.feasible_ratio_threshold, platemo.phase_one_evaluation_fraction) == (0.2, 1.0 / 3.0, 0.9)
