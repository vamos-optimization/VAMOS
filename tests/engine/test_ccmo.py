"""CCMO (Tian et al., 2021): parity with jMetal's port, the variations, exact budgets and the configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.algorithms import CCMOConfig
from vamos.engine.algorithm.ccmo import (
    CCMO,
    ccmo_environmental_selection,
    ccmo_fitness,
    ccmo_tournament,
    constraint_first_dominance,
)
from vamos.engine.algorithm.registry import get_builtin_algorithm_names, resolve_algorithm
from vamos.engine.tuning import build_ccmo_config_space, config_from_assignment
from vamos.engine.variation.pipeline import VariationPipeline
from vamos.foundation.eval.backends import SerialEvalBackend
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.registry import make_problem_selection

FIXTURE = Path(__file__).parent / "data" / "ccmo_jmetal_selection.json"
CASES: list[dict[str, Any]] = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


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


def _config(pop_size: int = 10, *, de: bool = False, **fields: Any) -> dict[str, Any]:
    builder = CCMOConfig.builder().pop_size(pop_size).mutation("pm", prob="1/n", eta=20.0)
    builder = builder.crossover("de", cr=1.0, f=0.5) if de else builder.crossover("sbx", prob=1.0, eta=20.0)
    config = builder.build().to_dict()
    config.update(fields)
    return config


# ---------------------------------------------------------------------------
# Parity with jMetal's CCMOFitness and CCMOEnvironmentalSelection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[case["label"] for case in CASES])
def test_fitness_matches_jmetal(case: dict[str, Any]) -> None:
    F = np.array(case["F"])
    violation = np.array(case["violation"])

    fitness, _ = ccmo_fitness(F, violation)
    helper_fitness, _ = ccmo_fitness(F, np.zeros_like(violation))

    np.testing.assert_allclose(fitness, case["fitness"], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(helper_fitness, case["helper_fitness"], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("kernels", _selection_kernels())
@pytest.mark.parametrize("case", CASES, ids=[case["label"] for case in CASES])
def test_environmental_selection_matches_jmetal(case: dict[str, Any], kernels: Any) -> None:
    F = np.array(case["F"])
    violation = np.array(case["violation"])
    for expected in case["selections"]:
        used = violation if expected["with_constraints"] else np.zeros_like(violation)

        selected, fitness = ccmo_environmental_selection(F, used, expected["size"], kernels)

        assert selected.tolist() == expected["selected"], (expected["size"], expected["with_constraints"])
        np.testing.assert_allclose(fitness, expected["fitness"], rtol=0.0, atol=1e-12)


def test_fixture_exercises_every_branch_of_the_selection() -> None:
    """Fewer, as many and more non-dominated rows than the population size, with and without constraints."""
    kinds = set()
    for case in CASES:
        violation = np.array(case["violation"])
        for expected in case["selections"]:
            used = violation if expected["with_constraints"] else np.zeros_like(violation)
            fitness, _ = ccmo_fitness(np.array(case["F"]), used)
            nondominated = int(np.sum(fitness < 1.0))
            kinds.add((int(np.sign(nondominated - expected["size"])), expected["with_constraints"]))
    assert kinds == {(sign, flag) for sign in (-1, 0, 1) for flag in (True, False)}


# ---------------------------------------------------------------------------
# Dominance, fitness and mating selection
# ---------------------------------------------------------------------------


def test_dominance_puts_the_violation_first_and_compares_equal_violations_by_pareto() -> None:
    F = np.array([[1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [0.0, 0.0]])
    violation = np.array([2.0, 2.0, 0.5, 0.5])

    dom = constraint_first_dominance(F, violation)

    # Smaller violation first, whatever the objectives; equal violations by Pareto dominance.
    assert dom[2, 0] and dom[2, 1] and dom[3, 2] and dom[0, 1]
    assert not dom[0, 2] and not dom[1, 0] and not dom.diagonal().any()


def test_fitness_of_the_jmetal_example() -> None:
    """a, b and c are non-dominated, b dominates d and all dominate e; k = 2 (jMetal's CCMOFitnessTest)."""
    F = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
    density = 1.0 / (np.sqrt(5.0) + 2.0)

    fitness, _ = ccmo_fitness(F, np.zeros(5))

    expected = [density, density, density, 2 + density, 5 + 1.0 / (np.sqrt(17.0) + 2.0)]
    np.testing.assert_allclose(fitness, expected, rtol=0.0, atol=1e-12)
    assert ccmo_fitness(np.array([[1.0, 2.0]]), np.zeros(1))[0].tolist() == [0.0]


def test_tournament_prefers_the_lower_fitness_and_breaks_ties_by_position() -> None:
    fitness = np.array([0.3, 0.1, 0.3, 0.1])
    draws = np.random.default_rng(5).integers(0, 4, size=(200, 2))

    winners = ccmo_tournament(fitness, 200, np.random.default_rng(5))

    for (first, second), winner in zip(draws, winners, strict=True):
        if fitness[first] != fitness[second]:
            assert winner == (first if fitness[first] < fitness[second] else second)
        else:
            assert winner == min(first, second)


# ---------------------------------------------------------------------------
# The algorithm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("de", "per_population"), [(False, 5), (True, 10)])
def test_each_population_creates_its_offspring(de: bool, per_population: int) -> None:
    algorithm = CCMO(_config(10, de=de), kernel=NumPyKernel())
    algorithm.initialize(_ShiftedFrontProblem(), ("max_evaluations", 1000), seed=1)

    X = algorithm.ask()

    assert X.shape == (2 * per_population, 4)
    state = algorithm.state
    assert state is not None and state.pending_from_main == per_population
    assert state.n_eval == 20


def test_odd_population_keeps_one_child_per_pair() -> None:
    algorithm = CCMO(_config(7), kernel=NumPyKernel())
    algorithm.initialize(_ShiftedFrontProblem(), ("max_evaluations", 1000), seed=1)

    assert algorithm.ask().shape == (6, 4)


@pytest.mark.parametrize("de", [False, True])
@pytest.mark.parametrize("extra", [0, 1, 4, 9, 10, 23])
def test_budget_is_exact(de: bool, extra: int) -> None:
    budget = 20 + extra

    result = CCMO(_config(10, de=de), kernel=NumPyKernel()).run(_ShiftedFrontProblem(), ("max_evaluations", budget), seed=3)

    assert result["evaluations"] == budget
    assert result["population"]["X"].shape == (10, 4)
    assert result["helper_population"]["X"].shape == (10, 4)


def test_last_generation_cuts_the_offspring_of_population_two_first() -> None:
    """With fewer evaluations left than offspring, ask evaluates the first ones: population 1's first."""
    algorithm = CCMO(_config(10), kernel=NumPyKernel())
    problem = _ShiftedFrontProblem()
    algorithm.initialize(problem, ("max_evaluations", 27), seed=4)
    reference = CCMO(_config(10), kernel=NumPyKernel())
    reference.initialize(problem, ("max_evaluations", 1000), seed=4)

    X = algorithm.ask()
    full = reference.ask()

    assert X.shape == (7, 4)
    state = algorithm.state
    assert state is not None and state.pending_from_main == 5
    np.testing.assert_array_equal(X, full[:7])
    algorithm.tell(SerialEvalBackend().evaluate(X, problem))
    assert algorithm.should_terminate()
    with pytest.raises(RuntimeError, match="no remaining evaluation budget"):
        algorithm.ask()


def test_budget_must_cover_both_initial_populations() -> None:
    with pytest.raises(ValueError, match=r"max_evaluations >= 2 \* pop_size"):
        CCMO(_config(10), kernel=NumPyKernel()).run(_ShiftedFrontProblem(), ("max_evaluations", 19), seed=1)
    with pytest.raises(ValueError, match="pop_size >= 2"):
        CCMO(_config(1), kernel=NumPyKernel()).run(_ShiftedFrontProblem(), ("max_evaluations", 100), seed=1)


def test_termination_criteria() -> None:
    problem = _ShiftedFrontProblem()
    by_generations = CCMO(_config(10), kernel=NumPyKernel()).run(problem, ("n_gen", 5), seed=1)
    assert by_generations["evaluations"] == 50

    reachable = {"target_value": 0.0, "reference_point": [11.0, 11.0], "max_evaluations": 500}
    early = CCMO(_config(10), kernel=NumPyKernel()).run(problem, ("hv", reachable), seed=1)
    assert early["hv_reached"] is True and early["evaluations"] < 500


def test_run_is_deterministic_and_equals_ask_tell() -> None:
    problem = _ShiftedFrontProblem()
    first = CCMO(_config(10), kernel=NumPyKernel()).run(problem, ("max_evaluations", 205), seed=7)
    second = CCMO(_config(10), kernel=NumPyKernel()).run(problem, ("max_evaluations", 205), seed=7)
    other = CCMO(_config(10), kernel=NumPyKernel()).run(problem, ("max_evaluations", 205), seed=8)

    algorithm = CCMO(_config(10), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 205), seed=7)
    while not algorithm.should_terminate():
        X = algorithm.ask()
        algorithm.tell(SerialEvalBackend().evaluate(X, problem))
    stepped = algorithm.result()

    for key in ("X", "F", "G"):
        np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first[key], stepped[key])
    assert not np.array_equal(first["population"]["X"], other["population"]["X"])


def test_population_one_solves_the_problem_and_population_two_ignores_the_constraints() -> None:
    result = CCMO(_config(20), kernel=NumPyKernel()).run(_ShiftedFrontProblem(), ("max_evaluations", 3000), seed=2)

    population, helper = result["population"], result["helper_population"]
    assert np.all(population["G"] <= 0.0)
    assert np.all(result["G"] <= 0.0)
    # The helper population converges to the unconstrained front, x1 = 0, which is infeasible.
    assert np.mean(helper["G"][:, 0] > 0.0) > 0.5


def test_population_is_sorted_by_fitness() -> None:
    algorithm = CCMO(_config(10), kernel=NumPyKernel())
    algorithm.run(_ShiftedFrontProblem(), ("max_evaluations", 100), seed=2)

    state = algorithm.state
    assert state is not None and state.fitness is not None and state.helper_fitness is not None
    assert np.all(np.diff(state.fitness) >= 0.0)
    assert np.all(np.diff(state.helper_fitness) >= 0.0)


def test_result_modes_and_constraint_mode_none() -> None:
    problem = _ShiftedFrontProblem()
    population = CCMO(_config(10, result_mode="population"), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=2)
    assert population["X"].shape == (10, 4)

    unconstrained = CCMO(_config(10, constraint_mode="none"), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=2)
    assert "G" not in unconstrained and "G" not in unconstrained["population"]


def test_variation_choices_are_validated() -> None:
    problem = _ShiftedFrontProblem()
    with pytest.raises(ValueError, match="two by two"):
        CCMO(_config(10, crossover=("pcx", {})), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=1)
    with pytest.raises(ValueError, match="cr and f"):
        CCMO(_config(10, crossover=("de", {"cr": 1.0, "scale": 0.5})), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=1)
    binary = make_problem_selection("bin_knapsack").instantiate()
    with pytest.raises(ValueError, match="needs real variables"):
        CCMO(_config(10, crossover=("de", {}), mutation=("bitflip", {"prob": 0.1})), kernel=NumPyKernel()).run(
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

    result = CCMO(_config(10, crossover=crossover, mutation=mutation), kernel=NumPyKernel()).run(problem, ("max_evaluations", 63), seed=1)

    assert result["evaluations"] == 63
    X = np.asarray(result["population"]["X"], dtype=float)
    assert np.all((X >= problem.xl) & (X <= problem.xu))
    if problem_name == "tsp6":
        np.testing.assert_array_equal(np.sort(X, axis=1), np.tile(np.arange(problem.n_var), (10, 1)))


def test_pipeline_recombine_then_mutate_is_produce_offspring() -> None:
    xl, xu = np.zeros(5), np.ones(5)
    pipeline = VariationPipeline(
        encoding="real",
        cross_method="sbx",
        cross_params={"prob": 1.0, "eta": 20.0},
        mut_method="pm",
        mut_params={"prob": 0.5, "eta": 20.0},
        xl=xl,
        xu=xu,
        workspace=None,
    )
    parents = np.random.default_rng(0).random((8, 5))

    produced = pipeline.produce_offspring(parents.copy(), np.random.default_rng(1))
    rng = np.random.default_rng(1)
    stepped = pipeline.mutate(pipeline.recombine(parents.copy(), rng), rng)

    np.testing.assert_array_equal(produced, stepped)


def test_mw3_regression() -> None:
    """A short run on MW3 ends feasible and near the front.

    With 10,000 evaluations, seeds 1 to 10 reach IGD 7.0e-3 to 1.17e-2 against VAMOS's front (1.04e-2 for
    seed 1); the threshold only catches a broken algorithm. The comparison with the published values,
    at 100,000 evaluations, is in docs/algorithms/ccmo.md.
    """
    problem = make_problem_selection("mw3").instantiate()
    config = CCMOConfig.default(n_var=problem.n_var).to_dict()

    result = CCMO(config, kernel=NumPyKernel()).run(problem, ("max_evaluations", 10_000), seed=1)

    front = np.loadtxt(Path(vamos.__file__).parent / "resources" / "reference_fronts" / "mw3.csv", delimiter=",")
    population = result["population"]
    assert np.all(population["G"] <= 0.0)
    distances = np.linalg.norm(front[:, None, :] - population["F"][None, :, :], axis=2)
    assert float(distances.min(axis=1).mean()) < 0.02


# ---------------------------------------------------------------------------
# Configuration, registry and tuning space
# ---------------------------------------------------------------------------


def test_default_configuration_is_the_papers() -> None:
    config = CCMOConfig.default(n_var=15)

    assert config.pop_size == 100
    assert config.crossover == ("sbx", {"prob": 1.0, "eta": 20.0})
    assert config.mutation == ("pm", {"prob": 1.0 / 15, "eta": 20.0})
    assert CCMOConfig.from_dict(json.loads(config.to_json())) == config
    assert CCMOConfig.default(encoding="binary").crossover[0] != "sbx"


def test_registry_and_optimize() -> None:
    assert "ccmo" in get_builtin_algorithm_names()
    assert callable(resolve_algorithm("ccmo"))

    result = vamos.optimize("mw3", algorithm="ccmo", pop_size=20, max_evaluations=100, seed=1, engine="numpy")

    assert result.data["evaluations"] == 100
    assert result.meta["run_artifact_resolved_spec"]["algorithm"]["component_id"] == "vamos.algorithm:ccmo@1"
    with pytest.raises(ConfigurationError, match="initial populations of ccmo"):
        vamos.optimize("mw3", algorithm="ccmo", pop_size=20, max_evaluations=30, seed=1)


def test_study_planning_rejects_a_budget_below_both_initial_populations(tmp_path: Path) -> None:
    from vamos.experiment.study.errors import UnresolvedStudyTaskError

    spec = vamos.StudySpec(problems=["mw3"], algorithms=["ccmo"], seeds=[0], pop_size=20, max_evaluations=30)
    with pytest.raises(UnresolvedStudyTaskError) as excinfo:
        vamos.plan_study(spec)
    assert excinfo.value.reason == "INVALID_EVALUATION_BUDGET"
    assert "initial populations of ccmo" in str(excinfo.value.expected)

    planned = vamos.plan_study(vamos.StudySpec(problems=["mw3"], algorithms=["ccmo"], seeds=[0], pop_size=20, max_evaluations=40))
    assert planned is not None


def test_tuning_space_builds_ga_and_de_configurations() -> None:
    space = build_ccmo_config_space()
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(60):
        assignment = space.sample(rng)
        config = config_from_assignment("ccmo", assignment)
        assert isinstance(config, CCMOConfig)
        name, params = config.crossover
        seen.add(name)
        if name == "de":
            assert "crossover_prob" not in assignment
            assert params == {"cr": assignment["de_cr"], "f": assignment["de_f"]}
        else:
            assert params["prob"] == assignment["crossover_prob"]
    assert seen == {"sbx", "blx_alpha", "arithmetic", "de"}

    paper = config_from_assignment("ccmo", {"pop_size": 100, "crossover": "sbx", "mutation": "pm"})
    assert paper.crossover == ("sbx", {"prob": 1.0, "eta": 20.0})
    assert paper.mutation == ("pm", {"eta": 20.0})
    de = config_from_assignment("ccmo", {"pop_size": 100, "crossover": "de", "mutation": "pm"})
    assert de.crossover == ("de", {"cr": 1.0, "f": 0.5})
