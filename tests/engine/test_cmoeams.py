"""CMOEA-MS (Tian et al., 2022): parity with jMetal's port, the two stages, the variations, exact budgets and the configuration."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.algorithms import CMOEAMSConfig
from vamos.engine.algorithm.catalog import initial_evaluations
from vamos.engine.algorithm.cmoeams import (
    CMOEAMS,
    cmoeams_tournament,
    cosine_distances,
    environmental_selection,
    exponential_crossover,
    is_stage_b,
    normalized_violation,
    shift_based_density,
    sorted_indices,
    spea2_fitness,
    stage_a_fitness,
    stage_b_fitness,
    tournament_winners,
    truncation,
)
from vamos.engine.algorithm.registry import get_builtin_algorithm_names, resolve_algorithm
from vamos.engine.operators.impl.real.crossover_differential import DEMatingCrossover
from vamos.engine.tuning import build_cmoeams_config_space, config_from_assignment
from vamos.foundation.eval.backends import SerialEvalBackend
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.kernel.operator_primitives import sbx_crossover_pairs
from vamos.foundation.problem.registry import make_problem_selection

FIXTURE = Path(__file__).parent / "data" / "cmoeams_jmetal_reference.json"
_PAYLOAD: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
POPULATION_CASES: list[dict[str, Any]] = _PAYLOAD["populations"]
POINTS_CASES: list[dict[str, Any]] = _PAYLOAD["points"]
STAGE_RULE_CASES: list[dict[str, Any]] = _PAYLOAD["stage_rule"]
DE_CASES: list[dict[str, Any]] = _PAYLOAD["differential_evolution"]
SBX_CASES: list[dict[str, Any]] = _PAYLOAD["sbx"]


def _decoded(values: Any) -> np.ndarray:
    """The fixture's floats, with ``"inf"``, ``"-inf"`` and ``"nan"`` for the non-finite ones."""
    raw = np.asarray(values, dtype=object)
    return np.asarray([float(v) for v in raw.ravel()], dtype=float).reshape(raw.shape)


def _assert_floats(actual: Any, expected: Any) -> None:
    """Floats to 1e-12, with ``NaN`` at the same places."""
    np.testing.assert_allclose(np.asarray(actual, dtype=float), _decoded(expected), rtol=0.0, atol=1e-12, equal_nan=True)


def _selection_kernels() -> list[Any]:
    kernels: list[Any] = [pytest.param(None, id="numpy")]
    try:
        from vamos.foundation.kernel.numba_backend import NumbaKernel
    except ImportError:
        return kernels
    try:
        compiled = NumbaKernel().selection_kernels()
    except (ImportError, RuntimeError):
        return kernels
    return [*kernels, pytest.param(compiled, id="numba")]


class _Replay:
    """A generator that returns fixed arrays in order, for operators that draw their numbers in blocks."""

    def __init__(self, *arrays: Any) -> None:
        self._arrays = [np.asarray(array) for array in arrays]

    def random(self, size: Any = None) -> np.ndarray:
        array = self._arrays.pop(0).astype(float)
        assert array.shape == (size if isinstance(size, tuple) else (size,))
        return array.copy()

    def integers(self, low: int, high: int | None = None, size: Any = None) -> np.ndarray:
        return self._arrays.pop(0).astype(int).copy()


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
    """Two variables, f = (x1, x2), and a constraint violated everywhere."""

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
    """f = (x1, 1 - x1 + x2) without constraints, so every solution is feasible."""

    n_var = 3
    n_obj = 2
    xl = np.zeros(3)
    xu = np.ones(3)
    encoding = "real"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        out["F"] = np.column_stack([X[:, 0], 1.0 - X[:, 0] + X[:, 1] + X[:, 2]])


def _config(pop_size: int = 10, *, de: bool = False, **fields: Any) -> dict[str, Any]:
    builder = CMOEAMSConfig.builder().pop_size(pop_size).mutation("pm", prob="1/n", eta=20.0)
    builder = builder.crossover("de", cr=1.0, f=0.5) if de else builder.crossover("sbx", prob=1.0, eta=20.0)
    config = builder.build().to_dict()
    config.update(fields)
    return config


# ---------------------------------------------------------------------------
# Parity with jMetal's CMOEAMSFitness, CMOEAMSSelection, CMOEAMS.isStageB and operators
# ---------------------------------------------------------------------------


def _population(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The objectives and VAMOS's constraint values (jMetal's negated: VAMOS's positive values are violated)."""
    F = _decoded(case["F"])
    G = -_decoded(case["constraints"]).reshape(F.shape[0], -1)
    return F, G


@pytest.mark.parametrize("case", POPULATION_CASES, ids=[case["label"] for case in POPULATION_CASES])
def test_violation_density_and_fitness_match_jmetal(case: dict[str, Any]) -> None:
    F, G = _population(case)

    violation = normalized_violation(G, F.shape[0])

    _assert_floats(violation, case["violation"])
    _assert_floats(shift_based_density(F), case["density"])
    _assert_floats(stage_a_fitness(F, violation), case["stage_a_fitness"])
    _assert_floats(stage_b_fitness(F, violation), case["stage_b_fitness"])
    if "cosine_distances" in case:
        _assert_floats(cosine_distances(F), case["cosine_distances"])


@pytest.mark.parametrize("kernels", _selection_kernels())
@pytest.mark.parametrize("case", POPULATION_CASES, ids=[case["label"] for case in POPULATION_CASES])
def test_selections_match_jmetal(case: dict[str, Any], kernels: Any) -> None:
    """The environmental selection of both stages, the truncation and the tournaments, in jMetal's order."""
    F, G = _population(case)
    violation = normalized_violation(G, F.shape[0])
    fitness_a, fitness_b = stage_a_fitness(F, violation), stage_b_fitness(F, violation)

    for size, expected_a, expected_b in zip(case["sizes"], case["selected_a"], case["selected_b"], strict=True):
        assert environmental_selection(fitness_a, F, size, kernels).tolist() == expected_a, size
        assert environmental_selection(fitness_b, F, size, kernels).tolist() == expected_b, size
    for removed, expected in zip(case["removed"], case["truncated"], strict=True):
        assert truncation(F, removed, kernels).tolist() == expected, removed
    draws = np.asarray(case["draws"])
    assert tournament_winners(fitness_a, draws).tolist() == case["winners_a"]
    assert tournament_winners(fitness_b, draws).tolist() == case["winners_b"]


@pytest.mark.parametrize("case", POINTS_CASES, ids=[case["label"] for case in POINTS_CASES])
def test_fitness_of_points_matches_jmetal(case: dict[str, Any]) -> None:
    points = _decoded(case["points"])

    _assert_floats(spea2_fitness(points, _decoded(case["violation"])), case["fitness"])
    _assert_floats(cosine_distances(points), case["cosine_distances"])


def test_stage_rule_matches_jmetal() -> None:
    for case in STAGE_RULE_CASES:
        decision = is_stage_b(_decoded(case["violation"]), case["threshold"], case["evaluations"], case["max_evaluations"])
        assert decision == case["stage_b"], case


@pytest.mark.parametrize("case", DE_CASES, ids=[case["label"] for case in DE_CASES])
def test_differential_evolution_matches_jmetal_on_fixed_draws(case: dict[str, Any]) -> None:
    """RAND_1_BIN is VAMOS's de operator on ``[x_i, x_r1, x_r2]``; RAND_1_EXP the exponential crossover; both clamped.

    jMetal draws ``j_rand`` and then, for the exponential crossover, its first variable; a draw
    equal to CR does not cross.
    """
    vectors = np.asarray(case["vectors"], dtype=float)
    first, second, current = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    lower, upper = np.asarray(case["lower"]), np.asarray(case["upper"])
    draws, positions = np.asarray(case["draws"]), np.asarray(case["positions"])
    n_var = lower.size

    if case["variant"] == "rand_1_bin":
        operator = DEMatingCrossover(F=case["f"], CR=case["cr"], lower=lower, upper=upper)
        groups = np.stack([current, first, second], axis=1)
        trials = np.asarray(operator(groups, _Replay(draws, positions[:, 0])))[:, 0, :]  # type: ignore[arg-type]  # a replay of fixed draws
        used = [n_var] * draws.shape[0]
    else:
        mutant = current + case["f"] * (first - second)
        trials = exponential_crossover(current, mutant, positions[:, 1], draws, case["cr"])
        continued = np.cumprod(draws[:, : n_var - 1] < case["cr"], axis=1).sum(axis=1)
        used = np.minimum(continued + 1, n_var).tolist()

    assert used == case["draws_used"]
    _assert_floats(np.clip(trials, lower, upper), case["children"])


@pytest.mark.parametrize("case", SBX_CASES, ids=[case["label"] for case in SBX_CASES])
def test_sbx_with_the_second_parent_first_gives_jmetals_children(case: dict[str, Any]) -> None:
    """VAMOS's SBX on ``(second, first)`` gives jMetal's ``(first child, second child)`` of each crossed pair.

    jMetal gives a variable that it does not cross to the other child and VAMOS leaves it with its
    own parent, so the order of the parents puts each child at jMetal's position. A pair that is not
    crossed (a draw above the probability) is copied in the order of the parents, so VAMOS's
    children are then jMetal's swapped: the two children are the same, at swapped positions.
    """
    parents = np.asarray(case["parents"], dtype=float)
    pair_draws = np.asarray(case["pair_draws"])
    crossed = np.flatnonzero(pair_draws <= case["prob"])
    draws = [np.asarray(case[key])[crossed] for key in ("var_draws", "spread_draws", "swap_draws")]
    lower, upper = np.asarray(case["lower"]), np.asarray(case["upper"])

    children = sbx_crossover_pairs(
        parents[:, ::-1, :].copy(),
        rng=_Replay(pair_draws, *draws),  # type: ignore[arg-type]  # a replay of fixed draws
        lower=lower,
        upper=upper,
        prob_crossover=case["prob"],
        eta=case["eta"],
        kernels=None,
    )

    expected = _decoded(case["children"])
    not_crossed = np.flatnonzero(pair_draws > case["prob"])
    np.testing.assert_allclose(np.clip(children, lower, upper)[crossed], expected[crossed], rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(children[not_crossed][:, ::-1, :], expected[not_crossed])
    assert case["draws_used"] == len(case["sequence"])


def test_fixture_exercises_the_edge_cases() -> None:
    """NaN densities, cosine distances and fitness, ties, an infinite violation, 2 to 5 objectives and truncations."""
    seen = set()
    for case in POPULATION_CASES:
        F, G = _population(case)
        seen.add(f"{F.shape[1]} objectives")
        if np.isnan(_decoded(case["density"])).any():
            seen.add("nan density")
        if np.isnan(cosine_distances(F)).any():
            seen.add("nan cosine distance")
        if np.isnan(_decoded(case["stage_a_fitness"])).any() or np.isnan(_decoded(case["stage_b_fitness"])).any():
            seen.add("nan fitness")
        fitness = _decoded(case["stage_b_fitness"])
        if np.unique(fitness[~np.isnan(fitness)]).size < np.count_nonzero(~np.isnan(fitness)):
            seen.add("tied fitness")
        if np.isinf(G).any():
            seen.add("infinite violation")
        if np.isnan(G).any():
            seen.add("nan constraint")
        if any(np.count_nonzero(_decoded(case["stage_b_fitness"]) < 1.0) > size for size in case["sizes"]):
            seen.add("truncated selection")
    for case in POINTS_CASES:
        if np.isnan(_decoded(case["violation"])).any():
            seen.add("nan violation")
    assert {"2 objectives", "3 objectives", "5 objectives"} <= seen
    assert {
        "nan density",
        "nan cosine distance",
        "nan fitness",
        "tied fitness",
        "infinite violation",
        "nan constraint",
        "truncated selection",
        "nan violation",
    } <= seen
    assert {True, False} == {case["stage_b"] for case in STAGE_RULE_CASES}


# ---------------------------------------------------------------------------
# The cases computed by hand in jMetal's CMOEAMSFitnessTest and CMOEAMSSelectionTest
# ---------------------------------------------------------------------------

_FOUR = np.array([[0.0, 4.0], [2.0, 0.0], [1.0, 2.0], [2.0, 4.0]])


def test_violation_is_normalized_per_constraint_and_averaged() -> None:
    # jMetal's constraints {{-2, -4}, {-1, 1}, {3, -1}}, negated: violations [2, 4], [1, 0] and [0, 1].
    G = np.array([[2.0, 4.0], [1.0, -1.0], [-3.0, 1.0]])
    np.testing.assert_allclose(normalized_violation(G, 3), [1.0, 0.25, 0.125], rtol=0.0, atol=1e-12)
    # A constraint that no row violates contributes zero to the mean of two.
    np.testing.assert_allclose(normalized_violation(np.array([[2.0, 0.0], [1.0, -1.0], [-3.0, -0.5]]), 3), [0.5, 0.25, 0.0])
    np.testing.assert_array_equal(normalized_violation(np.array([[np.nan], [3.0]]), 2), [0.0, 1.0])
    np.testing.assert_array_equal(normalized_violation(None, 3), np.zeros(3))


def test_density_uses_the_shifted_neighbour_after_the_solution_itself() -> None:
    # Normalized, (0, 1), (1, 0), (0.5, 0.5) and (1, 1); floor(sqrt(4)) + 1 = 3.
    np.testing.assert_allclose(shift_based_density(_FOUR), [1.0 / 3.0, 1.0 / 3.0, 1.0 / 2.5, 1.0 / 2.0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(shift_based_density(_FOUR * [10.0, 0.5] + [-3.0, 7.0]), shift_based_density(_FOUR), atol=1e-12)
    assert np.isnan(shift_based_density(np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]))).all()
    with pytest.raises(ValueError, match="at least two"):
        shift_based_density(np.array([[1.0, 2.0]]))


def test_fitness_of_both_stages() -> None:
    violation = np.array([0.0, 0.0, 0.0, 0.5])
    # Stage B: the three feasible solutions dominate the fourth; D uses the second smallest cosine distance.
    first, second = 1.0 / (3.0 - 2.0 / math.sqrt(5.0)), 1.0 / (3.0 - 1.0 / math.sqrt(5.0))
    np.testing.assert_allclose(stage_b_fitness(_FOUR, violation), [first, second, first, 3.0 + first], rtol=0.0, atol=1e-12)
    # Stage A: the points [SDE, CV] are (1/3, 0), (1/3, 0), (0.4, 0) and (0.5, 0.5), so R = [0, 0, 4, 5].
    fourth = 1.0 / (3.0 - math.sqrt(0.5))
    np.testing.assert_allclose(stage_a_fitness(_FOUR, violation), [0.5, 0.5, 4.5, 5.0 + fourth], rtol=0.0, atol=1e-12)
    # A smaller violation dominates whatever the objectives; a lone solution has D = 0.
    np.testing.assert_allclose(spea2_fitness(np.array([[5.0, 5.0], [1.0, 1.0]]), np.array([0.1, 0.2])), [0.5, 1.5])
    np.testing.assert_array_equal(spea2_fitness(np.array([[1.0, 2.0]]), np.zeros(1)), [0.0])


def test_cosine_distances_and_zero_vectors() -> None:
    distances = cosine_distances(np.array([[1.0, 0.0], [0.0, 3.0], [2.0, 2.0], [-1.0, 0.0]]))
    np.testing.assert_allclose(distances[0], [0.0, 1.0, 1.0 - math.sqrt(0.5), 2.0], atol=1e-12)
    zero = cosine_distances(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert np.isnan(zero[0]).all() and np.isnan(zero[1, 0]) and zero[1, 1] == pytest.approx(0.0, abs=1e-12)


_DIRECTIONS = np.array([[1.0, 0.0], [math.cos(math.radians(10.0)), math.sin(math.radians(10.0))], [1.0, 1.0], [0.0, 2.0]])


def test_environmental_selection_and_truncation() -> None:
    assert environmental_selection(np.array([0.4, 2.3, 1.5, 0.3, 3.1]), np.zeros((5, 2)), 3).tolist() == [0, 2, 3]
    # A tie keeps the first solution and NaN goes last.
    assert environmental_selection(np.array([np.nan, 1.5, 1.5, 0.2]), np.zeros((4, 2)), 3).tolist() == [1, 2, 3]
    fitness = np.array([0.4, 0.45, 0.3, 0.2])
    assert environmental_selection(fitness, _DIRECTIONS, 3).tolist() == [0, 2, 3]
    assert environmental_selection(fitness, _DIRECTIONS, 2).tolist() == [0, 3]
    # The dominated solution does not take part in the truncation.
    with_dominated = np.vstack([_DIRECTIONS, [math.cos(math.radians(44.0)), math.sin(math.radians(44.0))]])
    assert environmental_selection(np.append(fitness, 2.5), with_dominated, 3).tolist() == [0, 2, 3]
    assert truncation(_DIRECTIONS, 2).tolist() == [False, True, True, False]
    assert truncation(_DIRECTIONS * np.arange(1.0, 5.0)[:, None], 2).tolist() == [False, True, True, False]
    assert truncation(np.array([[1.0, 0.0], [0.0, 1.0]]), 1).tolist() == [True, False]
    with pytest.raises(ValueError, match="cannot select"):
        environmental_selection(np.array([0.4, 0.5]), np.zeros((2, 2)), 3)


def test_tournament_takes_the_lower_fitness_then_the_first_and_nan_loses() -> None:
    assert tournament_winners(np.array([3.0, 1.0, 2.0]), np.array([[0, 1], [2, 0], [0, 0], [2, 1]])).tolist() == [1, 2, 0, 1]
    assert tournament_winners(np.array([1.0, 1.0]), np.array([[1, 0], [0, 1]])).tolist() == [0, 0]
    assert tournament_winners(np.array([np.nan, 5.0]), np.array([[0, 1]])).tolist() == [1]
    assert sorted_indices(np.array([np.nan, 2.0, 1.0, 2.0, np.inf, 1.0])).tolist() == [2, 5, 1, 3, 4, 0]
    fitness = np.array([0.4, 1.2, 0.2, np.nan, 3.0, 0.2])
    rng = np.random.default_rng(3)
    np.testing.assert_array_equal(
        cmoeams_tournament(fitness, 40, rng), tournament_winners(fitness, np.random.default_rng(3).integers(0, 6, size=(40, 2)))
    )


def test_stage_b_needs_a_ratio_above_lambda_and_ten_percent_of_the_evaluations() -> None:
    three_of_four = np.array([0.0, 0.0, 0.0, 0.5])
    assert is_stage_b(three_of_four, 0.5, 1000, 10000)
    assert not is_stage_b(three_of_four, 0.5, 999, 10000)
    assert not is_stage_b(np.array([0.0, 0.0, 0.1, 0.2]), 0.5, 5000, 10000)
    assert not is_stage_b(three_of_four, 0.8, 5000, 10000)
    assert not is_stage_b(np.array([0.0, 0.0, np.nan, 0.5]), 0.5, 5000, 10000)
    # 10% of the evaluations is a bound that counts: 15 of 150 are enough, 14 are not.
    assert is_stage_b(np.zeros(4), 0.5, 15, 150) and not is_stage_b(np.zeros(4), 0.5, 14, 150)


# ---------------------------------------------------------------------------
# The algorithm: variations, stages, budget and results
# ---------------------------------------------------------------------------


def test_initial_fitness_is_that_of_stage_a() -> None:
    problem = _ShiftedFrontProblem()
    algorithm = CMOEAMS(_config(10), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 100), seed=4)
    state = algorithm.state
    assert state is not None and state.fitness is not None

    np.testing.assert_array_equal(state.fitness, stage_a_fitness(state.F, normalized_violation(state.G, 10)))
    assert state.offspring_size == 10 and not state.stage_b


def test_genetic_ask_pairs_parent_i_with_parent_i_plus_half_and_keeps_both_children() -> None:
    """The draws replayed: ``pop_size`` tournaments, the crossover of ``(second, first)``, then the mutation of the children."""
    problem = make_problem_selection("mw3", n_var=6).instantiate()
    algorithm = CMOEAMS(_config(12), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 1000), seed=5)
    state = algorithm.state
    assert state is not None and state.variation is not None and state.fitness is not None
    rng = copy.deepcopy(state.rng)
    pool = cmoeams_tournament(state.fitness, 12, rng)
    parents = np.stack([state.X[pool[6:]], state.X[pool[:6]]], axis=1).reshape(12, 6)
    children = state.variation.recombine(parents, rng)
    # The first children of the pairs at positions 0 to 5, the second ones at 6 to 11.
    expected = state.variation.mutate(np.concatenate([children[0::2], children[1::2]]), rng)

    np.testing.assert_array_equal(algorithm.ask(), expected)


@pytest.mark.parametrize("variant", ["rand_1_bin", "rand_1_exp"])
def test_differential_ask_creates_the_child_of_each_solution_from_two_tournaments(variant: str) -> None:
    problem = make_problem_selection("mw3", n_var=6).instantiate()
    algorithm = CMOEAMS(_config(10, de=True, crossover=("de", {"cr": 0.7, "f": 0.5}), de_variant=variant), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 1000), seed=6)
    state = algorithm.state
    assert state is not None and state.variation is not None and state.fitness is not None
    rng = copy.deepcopy(state.rng)
    X = state.X
    first, second = cmoeams_tournament(state.fitness, 10, rng), cmoeams_tournament(state.fitness, 10, rng)
    if variant == "rand_1_bin":
        trial = state.variation.recombine(np.stack([X, X[first], X[second]], axis=1).reshape(30, 6), rng)
    else:
        start = rng.integers(0, 6, size=10)
        trial = np.clip(exponential_crossover(X, X + 0.5 * (X[first] - X[second]), start, rng.random((10, 6)), 0.7), problem.xl, problem.xu)
    expected = state.variation.mutate(trial, rng)

    np.testing.assert_array_equal(algorithm.ask(), expected)


def test_tell_computes_the_fitness_of_the_union_in_its_stage_and_selects() -> None:
    problem = _ShiftedFrontProblem()
    algorithm = CMOEAMS(_config(10), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 1000), seed=7)
    state = algorithm.state
    assert state is not None and state.G is not None
    for _ in range(3):
        X, F, G = state.X.copy(), state.F.copy(), state.G.copy()
        X_off = algorithm.ask()
        evaluated = SerialEvalBackend().evaluate(X_off, problem)
        algorithm.tell(evaluated)

        F_union, G_union = np.vstack([F, evaluated.F]), np.vstack([G, evaluated.G])
        violation = normalized_violation(G_union, 20)
        stage_b = is_stage_b(violation, 0.5, state.n_eval, 1000)
        fitness = stage_b_fitness(F_union, violation) if stage_b else stage_a_fitness(F_union, violation)
        selected = environmental_selection(fitness, F_union, 10)
        np.testing.assert_array_equal(state.X, np.vstack([X, X_off])[selected])
        np.testing.assert_array_equal(state.fitness, fitness[selected])
        assert state.stage_b == stage_b


def test_stage_b_starts_at_ten_percent_of_the_evaluations() -> None:
    """Without constraints every solution is feasible: stage B starts once 10% of the evaluations are done."""
    problem = _LinearFrontProblem()
    # Generations end at 20, ..., 1000; 100 is 10% of 1000, so the 8 generations up to 90 are in stage A.
    result = CMOEAMS(_config(10), kernel=NumPyKernel()).run(problem, ("max_evaluations", 1000), seed=1)
    assert result["stage_b_generations"] == 99 - 8 and result["stage"] == "B"
    # DE generations of 5 end at 10, 15, 20, ...: the one that reaches 15, 10% of 150, is already in stage B.
    de = CMOEAMS(_config(5, de=True), kernel=NumPyKernel()).run(problem, ("max_evaluations", 150), seed=1)
    assert de["stage_b_generations"] == 29 - 1
    # A last generation of 5 offspring reaches 15 >= 1.5 evaluations, as jMetal's 20 do.
    partial = CMOEAMS(_config(10), kernel=NumPyKernel()).run(problem, ("max_evaluations", 15), seed=1)
    assert partial["evaluations"] == 15 and partial["stage"] == "B" and partial["stage_b_generations"] == 1
    # Without a feasible solution, or with lambda = 1, the fitness is always stage A's.
    assert (
        CMOEAMS(_config(10), kernel=NumPyKernel()).run(_InfeasibleProblem(), ("max_evaluations", 300), seed=1)["stage_b_generations"] == 0
    )
    never = CMOEAMS(_config(10, feasible_ratio_threshold=1.0), kernel=NumPyKernel()).run(problem, ("max_evaluations", 300), seed=1)
    assert never["stage_b_generations"] == 0 and never["stage"] == "A"


@pytest.mark.parametrize("de", [False, True])
@pytest.mark.parametrize("extra", [0, 1, 5, 9, 10, 23])
def test_budget_is_exact(de: bool, extra: int) -> None:
    budget = 10 + extra
    for problem in (_ShiftedFrontProblem(), _LinearFrontProblem()):
        result = CMOEAMS(_config(10, de=de), kernel=NumPyKernel()).run(problem, ("max_evaluations", budget), seed=3)
        assert result["evaluations"] == budget
        assert result["population"]["X"].shape == (10, problem.n_var)


def test_odd_population_creates_one_child_fewer_per_generation() -> None:
    """jMetal's CMOEAMSTest: with N = 5 the GA variation creates 4 children per generation."""
    algorithm = CMOEAMS(_config(5), kernel=NumPyKernel())
    algorithm.initialize(_ShiftedFrontProblem(), ("max_evaluations", 21), seed=1)
    state = algorithm.state
    assert state is not None and state.offspring_size == 4
    assert algorithm.ask().shape == (4, 4)
    # jMetal runs 21 evaluations for a budget of 20; VAMOS evaluates 20.
    result = CMOEAMS(_config(5), kernel=NumPyKernel()).run(_ShiftedFrontProblem(), ("max_evaluations", 20), seed=1)
    assert result["evaluations"] == 20 and result["generation"] == 4


@pytest.mark.parametrize("de", [False, True])
def test_partial_last_generation_selects_from_the_population_and_the_first_offspring(de: bool) -> None:
    problem = _ShiftedFrontProblem()
    algorithm = CMOEAMS(_config(10, de=de), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 17), seed=4)
    reference = CMOEAMS(_config(10, de=de), kernel=NumPyKernel())
    reference.initialize(problem, ("max_evaluations", 1000), seed=4)
    state = algorithm.state
    assert state is not None and state.G is not None
    X, F, G = state.X.copy(), state.F.copy(), state.G.copy()

    X_off = algorithm.ask()
    np.testing.assert_array_equal(X_off, reference.ask()[:7])
    evaluated = SerialEvalBackend().evaluate(X_off, problem)
    algorithm.tell(evaluated)

    F_union, G_union = np.vstack([F, evaluated.F]), np.vstack([G, evaluated.G])
    violation = normalized_violation(G_union, 17)
    fitness = stage_b_fitness(F_union, violation) if is_stage_b(violation, 0.5, 17, 17) else stage_a_fitness(F_union, violation)
    np.testing.assert_array_equal(state.X, np.vstack([X, X_off])[environmental_selection(fitness, F_union, 10)])
    assert state.n_eval == 17 and algorithm.should_terminate()
    with pytest.raises(RuntimeError, match="no remaining evaluation budget"):
        algorithm.ask()


def test_budget_must_cover_the_initial_population() -> None:
    problem = _ShiftedFrontProblem()
    with pytest.raises(ValueError, match="max_evaluations >= pop_size"):
        CMOEAMS(_config(10), kernel=NumPyKernel()).run(problem, ("max_evaluations", 9), seed=1)
    assert CMOEAMS(_config(10), kernel=NumPyKernel()).run(problem, ("max_evaluations", 10), seed=1)["evaluations"] == 10
    with pytest.raises(ValueError, match="pop_size >= 2"):
        CMOEAMS({**_config(10), "pop_size": 1}, kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=1)
    assert CMOEAMS(_config(2), kernel=NumPyKernel()).run(problem, ("max_evaluations", 30), seed=1)["evaluations"] == 30


def test_termination_criteria() -> None:
    problem = _ShiftedFrontProblem()
    assert CMOEAMS(_config(10), kernel=NumPyKernel()).run(problem, ("n_gen", 5), seed=1)["evaluations"] == 50

    reachable = {"target_value": 0.0, "reference_point": [11.0, 11.0], "max_evaluations": 500}
    early = CMOEAMS(_config(10), kernel=NumPyKernel()).run(problem, ("hv", reachable), seed=1)
    assert early["hv_reached"] is True and early["evaluations"] < 500


@pytest.mark.parametrize("de", [False, True])
def test_run_is_deterministic_and_equals_ask_tell(de: bool) -> None:
    problem = _ShiftedFrontProblem()
    first = CMOEAMS(_config(10, de=de), kernel=NumPyKernel()).run(problem, ("max_evaluations", 405), seed=7)
    second = CMOEAMS(_config(10, de=de), kernel=NumPyKernel()).run(problem, ("max_evaluations", 405), seed=7)
    other = CMOEAMS(_config(10, de=de), kernel=NumPyKernel()).run(problem, ("max_evaluations", 405), seed=8)

    algorithm = CMOEAMS(_config(10, de=de), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 405), seed=7)
    while not algorithm.should_terminate():
        X = algorithm.ask()
        algorithm.tell(SerialEvalBackend().evaluate(X, problem))
    stepped = algorithm.result()

    for key in ("X", "F", "G"):
        np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first[key], stepped[key])
    np.testing.assert_array_equal(first["population"]["X"], stepped["population"]["X"])
    assert first["stage_b_generations"] == stepped["stage_b_generations"]
    assert not np.array_equal(first["population"]["X"], other["population"]["X"])


def test_result_modes_and_constraint_mode_none() -> None:
    problem = _ShiftedFrontProblem()
    population = CMOEAMS(_config(10, result_mode="population"), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=2)
    assert population["X"].shape == (10, 4)

    unconstrained = CMOEAMS(_config(10, constraint_mode="none"), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=2)
    assert "G" not in unconstrained and "G" not in unconstrained["population"]


@pytest.mark.parametrize(("problem", "kwargs", "de"), [("c1dtlz1", {"n_var": 7, "n_obj": 3}, False), ("mw3", {}, False), ("mw3", {}, True)])
def test_engines_give_the_same_run(problem: str, kwargs: dict[str, int], de: bool) -> None:
    """The engine changes the speed of a run, not its result."""
    pytest.importorskip("numba")
    options: dict[str, Any] = {"algorithm_config": CMOEAMSConfig.from_dict(_config(20, de=True))} if de else {"pop_size": 20}
    runs = [
        vamos.optimize(problem, algorithm="cmoeams", max_evaluations=1005, seed=3, engine=engine, **options, **kwargs).data
        for engine in ("numpy", "numba")
    ]

    for array in ("X", "F", "G"):
        np.testing.assert_array_equal(runs[1]["population"][array], runs[0]["population"][array])
    assert runs[1]["stage_b_generations"] == runs[0]["stage_b_generations"]


def test_variation_choices_are_validated() -> None:
    problem = _ShiftedFrontProblem()
    with pytest.raises(ValueError, match="two parents and two children"):
        CMOEAMS(_config(10, crossover=("pcx", {})), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=1)
    with pytest.raises(ValueError, match="cr and f"):
        CMOEAMS(_config(10, crossover=("de", {"cr": 1.0, "scale": 0.5})), kernel=NumPyKernel()).run(
            problem, ("max_evaluations", 100), seed=1
        )
    with pytest.raises(ValueError, match="cr must be in"):
        CMOEAMS(_config(10, crossover=("de", {"cr": 1.5})), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=1)
    with pytest.raises(ValueError, match="f must be positive"):
        CMOEAMS(_config(10, crossover=("de", {"f": 0.0})), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=1)
    with pytest.raises(ValueError, match="de_variant applies to the DE variation"):
        CMOEAMS(_config(10, de_variant="rand_1_exp"), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=1)
    with pytest.raises(ValueError, match="de_variant must be one of"):
        CMOEAMS(_config(10, de=True, de_variant="best_1_bin"), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=1)
    with pytest.raises(ValueError, match="feasible_ratio_threshold must be in"):
        CMOEAMS(_config(10, feasible_ratio_threshold=1.5), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=1)
    binary = make_problem_selection("bin_knapsack").instantiate()
    with pytest.raises(ValueError, match="needs real variables"):
        CMOEAMS(_config(10, crossover=("de", {}), mutation=("bitflip", {"prob": 0.1})), kernel=NumPyKernel()).run(
            binary, ("max_evaluations", 100), seed=1
        )


@pytest.mark.parametrize(
    ("problem_name", "crossover", "mutation"),
    [
        ("bin_knapsack", ("hux", {"prob": 0.9}), ("bitflip", {"prob": 0.1})),
        ("tsp6", ("ox", {}), ("swap", {"prob": 0.1})),
        ("int_alloc", ("uniform", {"prob": 0.9}), ("reset", {"prob": 0.1})),
        ("mixed_design", ("mixed", {"prob": 0.9}), ("mixed", {"prob": 0.1})),
    ],
)
def test_ga_variation_runs_discrete_encodings(problem_name: str, crossover: tuple[str, dict[str, Any]], mutation: Any) -> None:
    problem = make_problem_selection(problem_name).instantiate()

    result = CMOEAMS(_config(10, crossover=crossover, mutation=mutation), kernel=NumPyKernel()).run(
        problem, ("max_evaluations", 63), seed=1
    )

    assert result["evaluations"] == 63
    X = np.asarray(result["population"]["X"], dtype=float)
    assert np.all((X >= problem.xl) & (X <= problem.xu))
    if problem_name == "tsp6":
        np.testing.assert_array_equal(np.sort(X, axis=1), np.tile(np.arange(problem.n_var), (10, 1)))


def test_other_real_variations_mutations_and_repairs_run() -> None:
    problem = _ShiftedFrontProblem()
    for crossover, mutation, repair in (
        (("blx_alpha", {"prob": 0.9, "alpha": 0.5}), ("gaussian", {"prob": 0.25, "sigma": 0.1}), ("reflect", {})),
        (("arithmetic", {"prob": 0.9}), ("uniform_reset", {"prob": 0.25}), ("random", {})),
        (("de", {"cr": 0.3, "f": 0.9}), ("pm", {"prob": 0.25, "eta": 20.0}), ("reflect", {})),
    ):
        config = _config(10, crossover=crossover, mutation=mutation, repair=repair)
        for variant in ("rand_1_bin", "rand_1_exp") if crossover[0] == "de" else ("rand_1_bin",):
            result = CMOEAMS({**config, "de_variant": variant}, kernel=NumPyKernel()).run(problem, ("max_evaluations", 205), seed=1)
            X = result["population"]["X"]
            assert result["evaluations"] == 205 and np.all((X >= 0.0) & (X <= 1.0))


def test_mw3_regression() -> None:
    """A run of jMetal's CMOEAMSIT setting on MW3 ends feasible and near the front.

    With 25,000 evaluations, seeds 1 to 31 give a median IGD of 6.7e-3 against VAMOS's front (seed
    12: 6.5e-3); two of them miss a part of the front (IGD 0.1), as four of jMetal's 31 runs do. The
    threshold only catches a broken algorithm. The comparison with jMetal and the published values is
    in docs/algorithms/cmoeams.md.
    """
    problem = make_problem_selection("mw3").instantiate()
    config = CMOEAMSConfig.default(n_var=problem.n_var).to_dict()

    result = CMOEAMS(config, kernel=NumPyKernel()).run(problem, ("max_evaluations", 25_000), seed=12)

    front = np.loadtxt(Path(vamos.__file__).parent / "resources" / "reference_fronts" / "mw3.csv", delimiter=",")
    population = result["population"]
    assert population["F"].shape == (100, 2)
    assert np.all(population["G"] <= 0.0)
    distances = np.linalg.norm(front[:, None, :] - population["F"][None, :, :], axis=2)
    assert float(distances.min(axis=1).mean()) < 0.02
    assert result["stage"] == "B"


# ---------------------------------------------------------------------------
# Configuration, registry and tuning space
# ---------------------------------------------------------------------------


def test_default_configuration_is_the_papers() -> None:
    config = CMOEAMSConfig.default(n_var=10)

    assert config.pop_size == 100
    assert config.crossover == ("sbx", {"prob": 1.0, "eta": 20.0})
    assert config.mutation == ("pm", {"prob": 0.1, "eta": 20.0})
    assert config.feasible_ratio_threshold == 0.5 and config.de_variant == "rand_1_bin"
    assert CMOEAMSConfig.from_dict(json.loads(config.to_json())) == config


def test_configuration_validates_the_variation_and_lambda() -> None:
    builder = CMOEAMSConfig.builder().pop_size(10).mutation("pm", prob="1/n", eta=20.0).crossover("sbx", prob=1.0, eta=20.0)
    with pytest.raises(ConfigurationError, match="de_variant applies"):
        builder.de_variant("rand_1_exp").build()
    with pytest.raises(ConfigurationError, match="de_variant must be one of"):
        builder.crossover("de", cr=1.0, f=0.5).de_variant("rand_2_bin").build()
    with pytest.raises(ConfigurationError, match="takes the parameters cr and f"):
        builder.crossover("de", cr=1.0, scale=0.5).de_variant("rand_1_bin").build()
    with pytest.raises(ConfigurationError, match="feasible_ratio_threshold"):
        builder.crossover("de", cr=1.0, f=0.5).feasible_ratio_threshold(1.2).build()
    config = builder.feasible_ratio_threshold(0.3).de_variant("RAND_1_EXP").build()
    assert (config.feasible_ratio_threshold, config.de_variant) == (0.3, "rand_1_exp")


def test_registry_and_optimize() -> None:
    assert "cmoeams" in get_builtin_algorithm_names()
    assert callable(resolve_algorithm("cmoeams"))
    assert initial_evaluations("cmoeams", 100) == 100

    result = vamos.optimize("mw3", algorithm="cmoeams", pop_size=20, max_evaluations=105, seed=1, engine="numpy")

    assert result.data["evaluations"] == 105
    assert result.meta["run_artifact_resolved_spec"]["algorithm"]["component_id"] == "vamos.algorithm:cmoeams@1"
    with pytest.raises(ConfigurationError, match="max_evaluations must be >= pop_size"):
        vamos.optimize("mw3", algorithm="cmoeams", pop_size=20, max_evaluations=19, seed=1)


def test_study_planning_checks_the_budget() -> None:
    from vamos.experiment.study.errors import UnresolvedStudyTaskError

    spec = vamos.StudySpec(problems=["mw3"], algorithms=["cmoeams"], seeds=[0], pop_size=100, max_evaluations=99)
    with pytest.raises(UnresolvedStudyTaskError) as excinfo:
        vamos.plan_study(spec)
    assert excinfo.value.reason == "INVALID_EVALUATION_BUDGET"

    assert (
        vamos.plan_study(vamos.StudySpec(problems=["mw3"], algorithms=["cmoeams"], seeds=[0], pop_size=100, max_evaluations=150))
        is not None
    )


def test_tuning_space_builds_ga_and_de_configurations() -> None:
    space = build_cmoeams_config_space()
    names = {param.name for param in space.params}
    assert {"pop_size", "crossover", "crossover_prob", "mutation", "feasible_ratio_threshold"} <= names
    rng = np.random.default_rng(0)
    crossovers, variants = set(), set()
    for _ in range(60):
        assignment = space.sample(rng)
        config = config_from_assignment("cmoeams", assignment)
        assert isinstance(config, CMOEAMSConfig)
        crossovers.add(config.crossover[0])
        assert config.feasible_ratio_threshold == assignment["feasible_ratio_threshold"]
        assert 20 <= config.pop_size <= 200
        if config.crossover[0] == "de":
            assert config.crossover == ("de", {"cr": assignment["de_cr"], "f": assignment["de_f"]})
            assert "crossover_prob" not in assignment
            variants.add(config.de_variant)
        else:
            assert config.de_variant == "rand_1_bin" and "de_variant" not in assignment
    assert crossovers == {"sbx", "blx_alpha", "arithmetic", "de"}
    assert variants == {"rand_1_bin", "rand_1_exp"}

    paper = config_from_assignment("cmoeams", {"pop_size": 100, "crossover": "sbx", "mutation": "pm"})
    assert paper.crossover == ("sbx", {"prob": 1.0, "eta": 20.0}) and paper.feasible_ratio_threshold == 0.5
    platemo_de = config_from_assignment("cmoeams", {"pop_size": 100, "crossover": "de", "mutation": "pm"})
    assert platemo_de.crossover == ("de", {"cr": 1.0, "f": 0.5}) and platemo_de.de_variant == "rand_1_bin"
