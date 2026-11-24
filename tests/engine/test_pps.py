"""PPS-MOEA/D (Fan et al., 2019): the paper's equations and algorithms, the two stages, exact budgets and the configuration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.algorithms import PPSConfig
from vamos.engine.algorithm.catalog import initial_evaluations
from vamos.engine.algorithm.pps import PAPER_POP_SIZE, PAPER_SETTINGS, PPS
from vamos.engine.algorithm.pps.stages import (
    ZERO_WEIGHT,
    change_rate,
    control_generation,
    distinct_rows,
    feasible_ratio,
    feasible_selection,
    first_replacements,
    ideal_and_nadir,
    next_epsilon,
    overall_violation,
    pull_replaces,
    push_replaces,
    relative_change,
    safe_weights,
    tchebycheff_values,
)
from vamos.engine.algorithm.registry import get_builtin_algorithm_names, resolve_algorithm
from vamos.foundation.eval.backends import SerialEvalBackend
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.registry import make_problem_selection


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
    """f = (x1, x2) and a constraint violated everywhere, by 1 + x1."""

    n_var = 2
    n_obj = 2
    n_constraints = 1
    xl = np.zeros(2)
    xu = np.ones(2)
    encoding = "real"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        out["F"] = X.copy()
        out["G"] = (1.0 + X[:, 0])[:, None]


class _FlatProblem:
    """Constant objectives, so the ideal and nadir points never change, and a constraint x1 <= 0.2 that records its largest violation."""

    n_var = 3
    n_obj = 2
    n_constraints = 1
    xl = np.zeros(3)
    xu = np.ones(3)
    encoding = "real"

    def __init__(self) -> None:
        self.largest_violation = -1.0

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        out["F"] = np.tile([1.0, 2.0], (X.shape[0], 1))
        out["G"] = (X[:, 1] - 0.2)[:, None]
        self.largest_violation = max(self.largest_violation, float(np.max(np.maximum(out["G"], 0.0))))


def _config(pop_size: int, **fields: Any) -> dict[str, Any]:
    return {**PPSConfig.default(pop_size=pop_size, n_var=4).to_dict(), **fields}


def _run(problem: Any, budget: int, seed: int, pop_size: int = 10, **fields: Any) -> dict[str, Any]:
    return PPS(_config(pop_size, **fields), kernel=NumPyKernel()).run(problem, ("max_evaluations", budget), seed=seed)


# ---------------------------------------------------------------------------
# The equations and algorithms of section 3
# ---------------------------------------------------------------------------


def test_tchebycheff_divides_by_the_weights_and_counts_a_zero_weight_as_small() -> None:
    """Eq. 7: max_j |f_j - z*_j| / lambda_j."""
    F = np.array([[3.0, 2.0], [1.0, 5.0]])
    weights = safe_weights(np.array([[0.25, 0.75], [0.0, 1.0]]))

    values = tchebycheff_values(F, weights, np.array([1.0, 1.0]))

    assert values[0] == max(2.0 / 0.25, 1.0 / 0.75)
    # A zero weight makes its term dominate, as for the neighbouring weight vectors: 0 / 1e-4 here, then 4 / 1.
    assert weights[1, 0] == ZERO_WEIGHT and values[1] == 4.0
    assert tchebycheff_values(np.array([[2.0, 1.0]]), weights[1:], np.zeros(2))[0] == 2.0 / ZERO_WEIGHT


def test_ideal_and_nadir_points_ignore_nan_as_algorithm_4() -> None:
    F = np.array([[1.0, np.nan], [3.0, 4.0], [-2.0, 6.0]])

    ideal, nadir = ideal_and_nadir(F)

    np.testing.assert_array_equal(ideal, [-2.0, 4.0])
    np.testing.assert_array_equal(nadir, [3.0, 6.0])
    ideal, nadir = ideal_and_nadir(np.array([[np.nan], [np.nan]]))
    assert (ideal[0], nadir[0]) == (1e30, -1e30)


def test_change_rate_is_the_largest_relative_change_of_the_ideal_and_nadir_points() -> None:
    """Eqs. 4-6, with Delta = 1e-6 in the denominators."""
    assert relative_change(np.array([1.0, 2.2]), np.array([1.0, 2.0])) == pytest.approx(0.1)
    # A previous value of zero divides by Delta.
    assert relative_change(np.array([1e-9]), np.array([0.0])) == pytest.approx(1e-3)
    assert change_rate(np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([4.0, 2.0]), np.array([5.0, 2.0])) == pytest.approx(0.2)
    assert change_rate(np.zeros(2), np.zeros(2), np.ones(2), np.ones(2)) == 0.0


def test_epsilon_shrinks_below_alpha_and_follows_the_schedule_otherwise() -> None:
    """Eq. 8: (1 - tau) epsilon(k - 1) if rf_k < alpha, else epsilon(0) (1 - k / Tc) ** cp."""
    settings = {"tao": 0.1, "threshold": 0.95, "cp": 2.0, "tc": 800}

    assert next_epsilon(5.0, 8.0, 100, 0.5, **settings) == pytest.approx(4.5)
    assert next_epsilon(5.0, 8.0, 100, 0.95, **settings) == pytest.approx(8.0 * (1.0 - 100 / 800) ** 2)
    assert next_epsilon(5.0, 8.0, 100, 1.0, **settings) == pytest.approx(6.125)


def test_control_generation_is_a_fraction_of_the_generations() -> None:
    """The paper's Tc = 800 is 0.8 of the 1000 generations of 300,000 evaluations of 300 solutions."""
    assert control_generation(0.8, 300_000, 300) == 800
    assert control_generation(0.8, 10_000, 100) == 80
    assert control_generation(0.001, 100, 100) == 1
    assert control_generation(0.8, 10, 100) == 1


def test_push_replaces_ties_and_better_values() -> None:
    """Algorithm 1: g(y) <= g(x)."""
    np.testing.assert_array_equal(push_replaces(np.array([1.0, 1.0, 1.0]), np.array([2.0, 1.0, 0.5])), [True, True, False])


def test_pull_replacement_follows_algorithm_2() -> None:
    child_values = np.full(6, 1.0)
    values = np.array([2.0, 0.5, 2.0, 0.5, 0.5, 2.0])
    phi = np.array([0.1, 0.1, 0.3, 0.3, 0.6, 0.2])

    replaces = pull_replaces(child_values, 0.3, values, phi, epsilon=0.2)

    # Both within epsilon (child 0.3 is not): no. Equal violations: the values decide. Otherwise the smaller violation wins.
    np.testing.assert_array_equal(replaces, [False, False, True, False, True, False])
    within = pull_replaces(child_values, 0.1, values, phi, epsilon=0.2)
    np.testing.assert_array_equal(within, [True, False, True, True, True, True])
    # With epsilon 0, it is Deb's feasibility rules with the Tchebycheff value among feasible solutions.
    np.testing.assert_array_equal(pull_replaces(np.ones(2), 0.0, np.array([2.0, 0.5]), np.zeros(2), 0.0), [True, False])


def test_replacements_stop_at_the_limit_in_the_candidates_order() -> None:
    np.testing.assert_array_equal(first_replacements(np.array([False, True, True, False, True]), 2), [1, 2])
    assert first_replacements(np.zeros(3, dtype=bool), 2).size == 0


def test_violation_is_infinite_for_nan_constraints() -> None:
    phi = overall_violation(np.array([[0.5, -1.0], [np.nan, 0.0], [-1.0, -2.0]]), 3)
    np.testing.assert_array_equal(phi, [0.5, np.inf, 0.0])
    np.testing.assert_array_equal(overall_violation(None, 2), [0.0, 0.0])
    # rf_k is the ratio of feasible to infeasible solutions: one to two here, infinite without infeasible ones.
    assert feasible_ratio(phi) == 0.5
    assert feasible_ratio(np.zeros(3)) == math.inf and feasible_ratio(np.ones(2)) == 0.0


def test_feasible_selection_keeps_every_distinct_feasible_solution_up_to_the_size() -> None:
    """Algorithm 8, lines 2-5, on the union of line 37 as a set."""
    X = np.array([[0.0], [1.0], [0.0], [2.0], [3.0]])
    F = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0], [0.0, 3.0], [5.0, 5.0]])
    phi = np.array([0.0, 0.0, 0.0, 0.0, 1.0])

    np.testing.assert_array_equal(distinct_rows(X), [0, 1, 3, 4])
    # Dominated feasible solutions stay while they fit; the duplicate and the infeasible solution do not.
    np.testing.assert_array_equal(feasible_selection(X, F, phi, 3, NumPyKernel()), [0, 1, 3])


def test_feasible_selection_selects_as_nsga2_when_there_are_more() -> None:
    """Algorithm 8, lines 7-16: the first fronts, then the largest crowding distances."""
    rng = np.random.default_rng(5)
    F = rng.random((30, 2))
    X = rng.random((30, 3))
    phi = np.where(rng.random(30) < 0.3, 0.5, 0.0)
    feasible = np.flatnonzero(phi == 0.0)

    selected = feasible_selection(X, F, phi, 8, NumPyKernel())

    _, _, expected = NumPyKernel().nsga2_survival(X[feasible], F[feasible], X[:0], F[:0], 8, return_indices=True)
    np.testing.assert_array_equal(selected, feasible[expected])


# ---------------------------------------------------------------------------
# The two stages
# ---------------------------------------------------------------------------


def _step(algorithm: PPS, problem: Any, generations: int) -> None:
    for _ in range(generations * algorithm.state.pop_size if algorithm.state is not None else 0):
        algorithm.tell(SerialEvalBackend().evaluate(algorithm.ask(), problem))


def test_push_stage_ignores_the_constraints() -> None:
    problem = _ShiftedFrontProblem()
    algorithm = PPS(_config(10, delta=1.0), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 1000), seed=1)
    state = algorithm.state
    assert state is not None

    child = algorithm.ask()
    # The best Tchebycheff value of every subproblem, and the most violated solution.
    algorithm.tell({"F": np.array([[-1.0, -1.0]]), "G": np.array([[100.0]])})

    assert state.push_stage and int(np.sum(np.all(state.X == child, axis=1))) == 2
    np.testing.assert_array_equal(state.ideal, [-1.0, -1.0])
    assert state.max_violation == 100.0


def test_switch_happens_when_the_points_stop_changing_with_the_largest_violation_seen() -> None:
    """Lines 5-12: r_k is first computed at k = l + 1; epsilon(0) is the largest violation of every solution evaluated."""
    problem = _FlatProblem()
    algorithm = PPS(_config(10, switch_window=3), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 2000), seed=2)
    state = algorithm.state
    assert state is not None

    _step(algorithm, problem, 3)
    assert state.push_stage and state.rate == 1.0 and state.k == 4
    largest = problem.largest_violation
    phi = state.phi.copy()
    algorithm.ask()

    assert not state.push_stage and state.switch_generation == 4 and state.rate == 0.0
    assert state.initial_epsilon == largest > 0.0
    # Eq. 8 already applies in the generation of the switch.
    assert state.settings is not None
    expected = next_epsilon(largest, largest, 4, feasible_ratio(phi), tao=0.1, threshold=0.95, cp=2.0, tc=state.settings.tc)
    assert state.epsilon == expected


def test_pull_stage_rejects_a_violated_solution_and_accepts_a_feasible_one() -> None:
    """r_k starts at 1.0 (line 2), so a threshold of at least 1 switches in the first generation."""
    problem = _ShiftedFrontProblem()
    algorithm = PPS(_config(10, delta=1.0, switch_threshold=1.0), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 1000), seed=3)
    state = algorithm.state
    assert state is not None

    algorithm.ask()
    assert not state.push_stage and state.switch_generation == 1
    X = state.X.copy()
    algorithm.tell({"F": np.array([[-1.0, -1.0]]), "G": np.array([[1e6]])})
    np.testing.assert_array_equal(state.X, X)

    child = algorithm.ask()
    algorithm.tell({"F": np.array([[-2.0, -2.0]]), "G": np.array([[-1.0]])})
    assert int(np.sum(np.all(state.X == child, axis=1))) == 2


def test_epsilon_is_zero_from_the_control_generation_and_the_push_stage_then_stays() -> None:
    """A search still in the push stage at Tc stays there: the pseudocode switches only while k < Tc."""
    result = _run(_FlatProblem(), 500, seed=1, decay_fraction=0.02)

    assert result["stage"] == "push" and result["switch_generation"] is None and result["epsilon"] == 0.0


def test_result_is_the_feasible_set() -> None:
    problem = _ShiftedFrontProblem()
    result = _run(problem, 3000, seed=4, switch_window=5)

    assert result["switch_generation"] is not None
    ns = result["feasible_set"]
    assert 0 < ns["X"].shape[0] <= 10 and np.all(ns["G"] <= 0.0)
    assert np.all(result["G"] <= 0.0)
    # The result is the non-dominated part of NS.
    ranks, _ = NumPyKernel().nsga2_ranking(ns["F"])
    np.testing.assert_array_equal(np.sort(result["F"], axis=0), np.sort(ns["F"][ranks == 0], axis=0))


def test_without_feasible_solutions_the_result_is_the_least_violated() -> None:
    problem = _InfeasibleProblem()
    result = _run(problem, 300, seed=1)
    population = _run(problem, 300, seed=1, result_mode="population")

    assert result["feasible_set"]["X"].shape == (0, 2)
    least = population["G"][:, 0].min()
    assert np.all(result["G"][:, 0] <= least + 1e-12) and result["X"].shape[0] >= 1
    assert population["X"].shape == (10, 2)


def test_without_feasible_solutions_a_nan_constraint_does_not_empty_the_result() -> None:
    # Through tell, a constraint that could not be evaluated reaches PPS as NaN, an infinite violation. The push
    # stage ignores violations, so the new solution, which dominates every other, enters the population.
    problem = _InfeasibleProblem()
    algorithm = PPS(_config(10), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 100), seed=1)
    child = algorithm.ask()
    algorithm.tell({"X": child, "F": np.full((1, 2), -1.0), "G": np.full((1, 1), np.nan)})
    result = algorithm.result()

    G = result["population"]["G"][:, 0]
    assert np.isnan(G).any() and np.isfinite(G).any()
    # The result is the least violated of the solutions whose violation is known.
    assert result["X"].shape[0] >= 1 and np.all(result["G"][:, 0] <= np.nanmin(G) + 1e-12)


# ---------------------------------------------------------------------------
# Budgets, determinism and engines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [0, 1, 7, 10, 23])
def test_budget_is_exact_and_a_partial_generation_reaches_the_feasible_set(extra: int) -> None:
    problem = _ShiftedFrontProblem()
    algorithm = PPS(_config(10), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 10 + extra), seed=3)
    while not algorithm.should_terminate():
        algorithm.tell(SerialEvalBackend().evaluate(algorithm.ask(), problem))
    state = algorithm.state
    assert state is not None and state.n_eval == 10 + extra and state.cursor == extra % 10

    result = algorithm.result()

    X = np.vstack([state.ns_X, state.X]) if state.cursor else state.ns_X
    F = np.vstack([state.ns_F, state.F]) if state.cursor else state.ns_F
    phi = np.concatenate([state.ns_phi, state.phi]) if state.cursor else state.ns_phi
    np.testing.assert_array_equal(result["feasible_set"]["X"], X[feasible_selection(X, F, phi, 10, NumPyKernel())])
    with pytest.raises(RuntimeError, match="no remaining evaluation budget"):
        algorithm.ask()


def test_budget_must_cover_the_initial_population() -> None:
    with pytest.raises(ValueError, match="max_evaluations >= pop_size"):
        _run(_ShiftedFrontProblem(), 9, seed=1)
    assert _run(_ShiftedFrontProblem(), 10, seed=1)["evaluations"] == 10


def test_run_is_deterministic_and_equals_ask_tell() -> None:
    problem = _ShiftedFrontProblem()
    first = _run(problem, 437, seed=7)
    second = _run(problem, 437, seed=7)
    other = _run(problem, 437, seed=8)

    algorithm = PPS(_config(10), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 437), seed=7)
    while not algorithm.should_terminate():
        algorithm.tell(SerialEvalBackend().evaluate(algorithm.ask(), problem))
    stepped = algorithm.result()

    for key in ("X", "F", "G"):
        np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first[key], stepped[key])
    np.testing.assert_array_equal(first["population"]["X"], stepped["population"]["X"])
    assert not np.array_equal(first["population"]["X"], other["population"]["X"])


def test_termination_criteria() -> None:
    problem = _ShiftedFrontProblem()
    assert PPS(_config(10), kernel=NumPyKernel()).run(problem, ("n_gen", 5), seed=1)["evaluations"] == 50
    reachable = {"target_value": 0.0, "reference_point": [11.0, 11.0], "max_evaluations": 500}
    early = PPS(_config(10), kernel=NumPyKernel()).run(problem, ("hv", reachable), seed=1)
    assert early["hv_reached"] is True and early["evaluations"] < 500 and early["evaluations"] % 10 == 0


@pytest.mark.parametrize(("problem", "pop_size", "kwargs"), [("mw3", 20, {}), ("c1dtlz1", 15, {"n_var": 7, "n_obj": 3})])
def test_engines_give_the_same_run(problem: str, pop_size: int, kwargs: dict[str, int]) -> None:
    """The engine changes the speed of a run, not its result."""
    instance = make_problem_selection(problem, **kwargs).instantiate()
    config = {**PPSConfig.default(pop_size=pop_size, n_var=instance.n_var).to_dict(), "switch_window": 3}
    if instance.n_obj == 3:
        config["weight_vectors"] = {"divisions": 4}
    results = {}
    for engine in ("numpy", "numba", "moocore"):
        pytest.importorskip(engine)
        results[engine] = PPS(config, kernel=resolve_kernel(engine)).run(instance, ("max_evaluations", 2000), seed=5)
    reference = results.pop("numpy")
    for result in results.values():
        for key in ("X", "F"):
            np.testing.assert_array_equal(result[key], reference[key])
        np.testing.assert_array_equal(result["population"]["X"], reference["population"]["X"])


def test_mw3_regression() -> None:
    """A short run on MW3 ends feasible and near the front.

    With 10,000 evaluations of 100 solutions the search does not switch before Tc (80 generations),
    and the feasible set of the push stage covers MW3's front: seeds 1 to 10 give IGDs from 0.012 to
    0.017 against VAMOS's front. The threshold only catches a broken algorithm.
    """
    problem = make_problem_selection("mw3").instantiate()
    config = PPSConfig.default(pop_size=100, n_var=problem.n_var).to_dict()

    result = PPS(config, kernel=NumPyKernel()).run(problem, ("max_evaluations", 10_000), seed=2)

    front = np.loadtxt(Path(vamos.__file__).parent / "resources" / "reference_fronts" / "mw3.csv", delimiter=",")
    assert np.all(result["G"] <= 0.0) and result["F"].shape[0] > 50
    distances = np.linalg.norm(front[:, None, :] - result["F"][None, :, :], axis=2)
    assert float(distances.min(axis=1).mean()) < 0.03


# ---------------------------------------------------------------------------
# Configuration, registry and optimize
# ---------------------------------------------------------------------------


def test_default_configuration_is_the_papers() -> None:
    config = PPSConfig.default(n_var=10)

    assert config.pop_size == PAPER_POP_SIZE == 300
    assert config.crossover == ("de", {"cr": 1.0, "f": 0.5})
    assert config.mutation == ("pm", {"prob": 0.1, "eta": 20.0})
    fields = {name: getattr(config, name) for name in PAPER_SETTINGS}
    assert (
        fields
        == PAPER_SETTINGS
        == {
            "neighbor_size": 30,
            "delta": 0.9,
            "replace_limit": 2,
            "switch_threshold": 1e-3,
            "switch_window": 20,
            "feasibility_ratio_threshold": 0.95,
            "tao": 0.1,
            "cp": 2.0,
            "decay_fraction": 0.8,
        }
    )
    assert PPSConfig.from_dict(json.loads(config.to_json())) == config
    assert PPSConfig.default(pop_size=20).neighbor_size == 20


def test_configuration_validates_its_fields() -> None:
    builder = PPSConfig.builder().pop_size(10).mutation("pm", prob="1/n", eta=20.0)
    with pytest.raises(ConfigurationError, match="crossover must be 'de'"):
        builder.crossover("sbx", prob=1.0, eta=20.0).build()
    builder.crossover("de", cr=1.0, f=0.5)
    with pytest.raises(ConfigurationError, match="neighbor_size must be an integer in \\[2, 10\\]"):
        builder.neighbor_size(11).build()
    builder.neighbor_size(5)
    for name, value in (("tao", 1.0), ("cp", 0.0), ("decay_fraction", 0.0), ("delta", 1.5), ("feasibility_ratio_threshold", -0.1)):
        with pytest.raises(ConfigurationError, match=f"{name} must be in"):
            PPSConfig.from_dict({**builder.build().to_dict(), name: value})
    with pytest.raises(ConfigurationError, match="switch_window must be an integer"):
        builder.switch_window(0).build()
    config = builder.switch_window(5).switch_threshold(0.01).tao(0.2).cp(3).decay_fraction(0.5).replace_limit(1).delta(0.8).build()
    assert (config.switch_window, config.switch_threshold, config.tao, config.cp, config.decay_fraction) == (5, 0.01, 0.2, 3.0, 0.5)
    assert (config.replace_limit, config.delta, config.feasibility_ratio_threshold) == (1, 0.8, 0.95)
    # alpha bounds a ratio of feasible to infeasible solutions, so it may exceed 1: 2 asks for two thirds feasible.
    assert builder.feasibility_ratio_threshold(2.0).build().feasibility_ratio_threshold == 2.0


def test_default_population_has_weight_vectors_for_the_objectives() -> None:
    from vamos.engine.algorithm.components.weight_vectors import largest_weighted_population

    # 300 is the lattice of 23 divisions with three objectives; with four, the largest population up to 300 with weights.
    assert PPSConfig.default(n_obj=2).pop_size == PPSConfig.default(n_obj=3).pop_size == 300
    four = PPSConfig.default(n_obj=4)
    assert four.pop_size == largest_weighted_population(300, 4) < 300
    assert PPSConfig.default(pop_size=100, n_obj=4).pop_size == 100
    # Without pop_size, optimize takes the largest population up to its quick default, 200, with weights: 190 with three objectives.
    result = vamos.optimize("lircmop13", algorithm="pps", max_evaluations=600, seed=1, engine="numpy")
    assert result.data["population"]["F"].shape == (190, 3)
    assert vamos.optimize("lircmop2", algorithm="pps", max_evaluations=400, seed=1, engine="numpy").data["population"]["F"].shape == (
        200,
        2,
    )


def test_tuning_space_builds_the_default_and_adapts_the_population() -> None:
    from vamos.engine.tuning import build_pps_config_space, config_from_assignment
    from vamos.engine.tuning.grammar import builtin_parameter_space
    from vamos.engine.tuning.grammar.default_check import default_assignment

    space = builtin_parameter_space("pps")
    assert (space.algorithm, space.encoding, build_pps_config_space().grammar) == ("pps", "real", space)
    assert {"switch_threshold", "switch_window", "feasibility_ratio_threshold", "tao", "cp", "decay_fraction"} <= set(space.names())

    def build(pop_size: int, n_obj: int) -> PPSConfig:
        config = config_from_assignment("pps", default_assignment(space, pop_size=pop_size, n_obj=n_obj))
        assert isinstance(config, PPSConfig)
        return config

    # The defaults of the space are the paper's settings.
    default = build(300, 2)
    assert {name: getattr(default, name) for name in PAPER_SETTINGS} == PAPER_SETTINGS
    assert (default.crossover, default.mutation) == (("de", {"cr": 1.0, "f": 0.5}), ("polynomial", {"prob": "1.0/n", "eta": 20.0}))
    # A population without weight vectors for the objectives gives way to the largest one that has them, as MOEA/D's.
    assert build(200, 3).pop_size == 190
    assert build(10, 2).neighbor_size == 10


def test_run_validates_the_variation_and_the_encoding() -> None:
    problem = _ShiftedFrontProblem()
    with pytest.raises(ValueError, match="crossover must be 'de'"):
        PPS({**_config(10), "crossover": ("sbx", {})}, kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=1)
    with pytest.raises(ValueError, match="neighbor_size must be an integer"):
        PPS({**_config(10), "neighbor_size": 1}, kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=1)
    binary = make_problem_selection("bin_knapsack").instantiate()
    with pytest.raises(ValueError, match="needs real variables"):
        vamos.optimize(binary, algorithm="pps", pop_size=10, max_evaluations=100, seed=1)


def test_registry_and_optimize() -> None:
    assert "pps" in get_builtin_algorithm_names()
    assert callable(resolve_algorithm("pps"))
    assert initial_evaluations("pps", 100) == 100

    result = vamos.optimize("mw3", algorithm="pps", pop_size=20, max_evaluations=105, seed=1, engine="numpy")

    assert result.data["evaluations"] == 105
    assert result.meta["run_artifact_resolved_spec"]["algorithm"]["component_id"] == "vamos.algorithm:pps@1"
    three = vamos.optimize("c1dtlz1", algorithm="pps", pop_size=91, max_evaluations=200, seed=1, engine="numpy")
    assert three.data["evaluations"] == 200


def test_a_variation_override_sets_cr_and_f_in_either_case() -> None:
    from vamos.engine.algorithm.builders import build_pps_algorithm

    problem = _ShiftedFrontProblem()
    for crossover in (("de", {"CR": 0.7, "F": 0.4}), ("de", {"cr": 0.7, "f": 0.4})):
        _, config = build_pps_algorithm(kernel=NumPyKernel(), problem=problem, pop_size=10, pps_variation={"crossover": crossover})
        assert isinstance(config, PPSConfig) and config.crossover == ("de", {"cr": 0.7, "f": 0.4})
    with pytest.raises(ValueError, match="PPS's crossover takes cr or CR"):
        build_pps_algorithm(kernel=NumPyKernel(), problem=problem, pop_size=10, pps_variation={"crossover": ("de", {"cr": 0.7, "CR": 0.4})})
