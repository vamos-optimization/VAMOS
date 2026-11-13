"""C-TAEA (Li et al., 2019): parity with jMetal's port, exact budgets, the archives and the configuration."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.algorithms import CTAEAConfig
from vamos.engine.algorithm.catalog import initial_evaluations
from vamos.engine.algorithm.ctaea import (
    CTAEA,
    associate,
    convergence_archive_update,
    ctaea_tournament,
    ctaea_weight_vectors,
    diversity_archive_update,
    front_ranks,
    mating_proportions,
    tchebycheff_in_own_subregion,
    tournament_winners,
    weight_vector_count,
    worst_of_the_most_crowded_subregion,
)
from vamos.engine.algorithm.registry import get_builtin_algorithm_names, resolve_algorithm
from vamos.engine.tuning import build_ctaea_config_space, config_from_assignment
from vamos.foundation.constraints.utils import compute_violation
from vamos.foundation.eval.backends import SerialEvalBackend
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.registry import make_problem_selection

FIXTURE = Path(__file__).parent / "data" / "ctaea_jmetal_archives.json"
_PAYLOAD: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
WEIGHT_CASES: list[dict[str, Any]] = _PAYLOAD["weight_vectors"]
TOURNAMENT_CASES: list[dict[str, Any]] = _PAYLOAD["tournaments"]
ARCHIVE_CASES: list[dict[str, Any]] = _PAYLOAD["archives"]


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


def _config(pop_size: int = 10, **fields: Any) -> dict[str, Any]:
    config = (
        CTAEAConfig.builder().pop_size(pop_size).crossover("sbx", prob=1.0, eta=20.0).mutation("pm", prob="1/n", eta=20.0).build().to_dict()
    )
    config.update(fields)
    return config


def _three_objectives() -> Any:
    return make_problem_selection("c1dtlz1", n_var=7, n_obj=3).instantiate()


def _block(case: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    n_obj = len(case["weights"][0])
    return np.asarray(case[key]["F"], dtype=float).reshape(-1, n_obj), np.asarray(case[key]["violation"], dtype=float)


# ---------------------------------------------------------------------------
# Parity with jMetal's UniformWeightVectors, Subregions, archive updates, MatingProportions and tournament
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", WEIGHT_CASES, ids=[f"{case['requested']}-{case['n_obj']}" for case in WEIGHT_CASES])
def test_weight_vectors_match_jmetal(case: dict[str, Any]) -> None:
    weights = ctaea_weight_vectors(case["requested"], case["n_obj"])

    expected = np.asarray(case["weights"])
    assert weights.shape == expected.shape
    np.testing.assert_allclose(weights, expected, rtol=0.0, atol=1e-12)
    assert weight_vector_count(case["requested"], case["n_obj"]) == expected.shape[0]


@pytest.mark.parametrize("case", TOURNAMENT_CASES, ids=[case["label"] for case in TOURNAMENT_CASES])
def test_tournament_decisions_match_jmetal_on_fixed_draws(case: dict[str, Any]) -> None:
    F = np.asarray(case["F"], dtype=float)
    feasible = np.asarray(case["violation"]) == 0.0

    winners = tournament_winners(F, feasible, np.asarray(case["draws"]))

    assert winners.tolist() == case["winners"]


@pytest.mark.parametrize("case", ARCHIVE_CASES, ids=[case["label"] for case in ARCHIVE_CASES])
def test_archive_updates_match_jmetal(case: dict[str, Any]) -> None:
    weights = np.asarray(case["weights"], dtype=float)
    convergence_F, convergence_cv = _block(case, "convergence")
    diversity_F, _ = _block(case, "diversity")
    offspring_F, offspring_cv = _block(case, "offspring")
    candidates = np.vstack([convergence_F, offspring_F])
    violation = np.concatenate([convergence_cv, offspring_cv])

    np.testing.assert_array_equal(ctaea_weight_vectors(case["requested"], case["n_obj"]), weights)
    assert associate(candidates, weights).tolist() == case["regions"]
    np.testing.assert_allclose(tchebycheff_in_own_subregion(candidates, weights), case["tchebycheff"], rtol=0.0, atol=1e-12)
    assert worst_of_the_most_crowded_subregion(candidates, weights) == case["worst"]

    convergence = convergence_archive_update(candidates, violation, weights)
    assert convergence.tolist() == case["convergence_update"]
    diversity_candidates = np.vstack([diversity_F, offspring_F])
    diversity = diversity_archive_update(candidates[convergence], diversity_candidates, weights)
    assert diversity.tolist() == case["diversity_update"]

    proportions = mating_proportions(candidates[convergence], diversity_candidates[diversity])
    shares = [proportions.convergence_share, proportions.diversity_share, proportions.non_dominated_share_of_convergence_archive]
    np.testing.assert_allclose(shares, case["proportions"], rtol=0.0, atol=1e-12)


def test_fixture_exercises_every_branch_of_the_convergence_archive() -> None:
    """Exactly N, more (with removals) and fewer (with a trimmed last front) feasible candidates; first updates; 2, 3 and 5 objectives."""
    kinds = set()
    objectives = set()
    for case in ARCHIVE_CASES:
        weights = np.asarray(case["weights"], dtype=float)
        size = weights.shape[0]
        objectives.add(weights.shape[1])
        convergence_F, convergence_cv = _block(case, "convergence")
        offspring_F, offspring_cv = _block(case, "offspring")
        F = np.vstack([convergence_F, offspring_F])
        violation = np.concatenate([convergence_cv, offspring_cv])
        feasible = violation == 0.0
        if convergence_F.shape[0] == 0:
            kinds.add("first update")
        if offspring_F.shape[0] < size:
            kinds.add("partial offspring")
        if feasible.sum() == size:
            kinds.add("exactly N")
        elif feasible.sum() > size:
            fronts = np.cumsum(np.bincount(front_ranks(F[feasible])))
            if fronts[np.searchsorted(fronts, size)] > size:
                kinds.add("truncation")
        else:
            kinds.add("all infeasible" if not feasible.any() else "partly feasible")
            criteria = np.column_stack([violation[~feasible], tchebycheff_in_own_subregion(F[~feasible], weights)])
            fronts = feasible.sum() + np.cumsum(np.bincount(front_ranks(criteria)))
            if fronts[min(np.searchsorted(fronts, size), fronts.size - 1)] > size:
                kinds.add("trimmed last front")
    assert kinds == {
        "first update",
        "partial offspring",
        "exactly N",
        "truncation",
        "all infeasible",
        "partly feasible",
        "trimmed last front",
    }
    assert {2, 3, 5} <= objectives


# ---------------------------------------------------------------------------
# Weight vectors, subregions, archives and mating selection
# ---------------------------------------------------------------------------


def test_weight_vectors_follow_the_requested_size() -> None:
    assert weight_vector_count(100, 2) == 100
    assert weight_vector_count(100, 3) == 91
    # One division per objective already gives M vectors, more than requested.
    assert weight_vector_count(2, 3) == 3
    # H1 = 4 < M = 5 adds an inner layer with H2 = 3, shrunk towards the centre.
    weights = ctaea_weight_vectors(105, 5)
    assert weights.shape == (105, 5)
    np.testing.assert_allclose(weights[70], [0.1, 0.1, 0.1, 0.1, 0.6], rtol=0.0, atol=1e-12)
    assert np.all(weights >= 1e-6)
    with pytest.raises(ValueError, match="at least two objectives"):
        ctaea_weight_vectors(10, 1)
    with pytest.raises(ValueError, match="at least one requested"):
        ctaea_weight_vectors(0, 2)


def test_association_takes_the_largest_cosine_and_the_first_in_a_tie() -> None:
    weights = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    points = np.array([[2.0, 0.1], [1.0, 1.0], [0.0, 3.0], [0.0, 0.0], [1.0, 3.0]])

    assert associate(points, weights).tolist() == [0, 1, 2, 0, 2]
    # Two weight vectors at the same angle: the first one.
    assert associate(np.array([[1.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]])).tolist() == [0]


def test_tournament_rule_on_fixed_draws() -> None:
    """a dominates b; c is non-dominated with a; d and e are infeasible."""
    F = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 3.0], [0.0, 0.0], [0.1, 0.1]])
    feasible = np.array([True, True, True, False, False])
    draws = np.array([[1, 0, 4], [0, 1, 4], [0, 2, 4], [2, 0, 4], [3, 1, 0], [1, 3, 0], [3, 4, 2], [4, 3, 1]])

    winners = tournament_winners(F, feasible, draws)

    # Second dominates first; first kept; non-dominated pairs keep the first; one feasible wins; both infeasible: the third.
    assert winners.tolist() == [0, 0, 0, 2, 1, 1, 2, 1]


def test_tournament_draws_three_rows_with_replacement() -> None:
    F = np.random.default_rng(0).random((7, 2))
    feasible = np.arange(7) % 2 == 0
    draws = np.random.default_rng(3).integers(0, 7, size=(50, 3))

    winners = ctaea_tournament(F, feasible, 50, np.random.default_rng(3))

    np.testing.assert_array_equal(winners, tournament_winners(F, feasible, draws))


def test_mating_proportions_split_the_union_in_halves() -> None:
    convergence = np.array([[1.0, 4.0], [2.0, 2.0], [3.0, 3.0]])
    diversity = np.array([[0.5, 5.0], [4.0, 1.0], [5.0, 5.0]])

    proportions = mating_proportions(convergence, diversity)

    # Non-dominated in the union: (1, 4), (2, 2), (0.5, 5) and (4, 1); in the CA alone: (1, 4) and (2, 2).
    assert (proportions.convergence_share, proportions.diversity_share) == (2 / 6, 2 / 6)
    assert proportions.non_dominated_share_of_convergence_archive == 2 / 3
    assert not proportions.first_parent_from_convergence_archive


def test_convergence_archive_keeps_the_feasible_and_completes_with_the_least_violating() -> None:
    weights = ctaea_weight_vectors(4, 2)
    F = np.array([[0.1, 0.9], [0.9, 0.1], [0.5, 0.5], [0.2, 0.2], [0.3, 0.3], [0.6, 0.6]])
    violation = np.array([0.0, 0.0, 2.0, 1.0, 3.0, 1.0])

    selected = convergence_archive_update(F, violation, weights)

    # Four weight vectors. On (violation, Tchebycheff with the ideal point (0.2, 0.2)), the infeasible rows
    # are (2, 0.9), (1, 0), (3, 0.3) and (1, 1.2): row 3 is the first front, rows 2, 4 and 5 the second.
    # Both feasible rows, row 3 and the second front make six; the two largest violations of the
    # second front, rows 4 and 2, go.
    assert weights.shape[0] == 4
    assert selected.tolist() == [0, 1, 3, 5]


# ---------------------------------------------------------------------------
# The algorithm
# ---------------------------------------------------------------------------


def test_crossover_receives_the_second_parent_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mating selection replayed: the i-th group is (second parent, first parent), and one child per group is kept.

    VAMOS's SBX leaves a variable it does not cross with its own parent, while jMetal's gives it to the
    other child: with this order, the kept child inherits those variables from the second parent, as
    jMetal's first child does.
    """
    algorithm = CTAEA(_config(100), kernel=NumPyKernel())
    algorithm.initialize(_three_objectives(), ("max_evaluations", 1000), seed=5)
    state = algorithm.state
    assert state is not None and state.variation is not None and state.diversity_F is not None and state.diversity_X is not None
    rng = copy.deepcopy(state.rng)
    n = state.offspring_size
    proportions = mating_proportions(state.F, state.diversity_F)
    feasible = compute_violation(state.G, n=n) == 0.0
    diversity_feasible = compute_violation(state.diversity_G, n=n) == 0.0
    if proportions.first_parent_from_convergence_archive:
        first = state.X[ctaea_tournament(state.F, feasible, n, rng)]
    else:
        first = state.diversity_X[ctaea_tournament(state.diversity_F, diversity_feasible, n, rng)]
    from_convergence = rng.random(n) < proportions.non_dominated_share_of_convergence_archive
    in_convergence = state.X[ctaea_tournament(state.F, feasible, n, rng)]
    in_diversity = state.diversity_X[ctaea_tournament(state.diversity_F, diversity_feasible, n, rng)]
    second = np.where(from_convergence[:, None], in_convergence, in_diversity)
    seen: dict[str, np.ndarray] = {}
    recombine = state.variation.recombine

    def spy(parents: np.ndarray, generator: np.random.Generator) -> np.ndarray:
        seen["parents"] = parents.copy()
        children: np.ndarray = recombine(parents, generator)
        seen["children"] = children.copy()
        return children

    monkeypatch.setattr(state.variation, "recombine", spy)

    X = algorithm.ask()

    np.testing.assert_array_equal(seen["parents"][0::2], second)
    np.testing.assert_array_equal(seen["parents"][1::2], first)
    assert X.shape == (n, 7) and seen["children"].shape == (2 * n, 7)


def test_one_offspring_per_weight_vector() -> None:
    algorithm = CTAEA(_config(100), kernel=NumPyKernel())
    algorithm.initialize(_three_objectives(), ("max_evaluations", 1000), seed=1)

    X = algorithm.ask()

    state = algorithm.state
    assert state is not None and state.weights is not None
    assert state.weights.shape == (91, 3)
    assert X.shape == (91, 7)
    assert state.n_eval == 91 and state.X.shape == (91, 7)
    assert state.diversity_X is not None and state.diversity_X.shape == (91, 7)


@pytest.mark.parametrize("extra", [0, 1, 4, 9, 10, 23])
def test_budget_is_exact(extra: int) -> None:
    budget = 10 + extra

    result = CTAEA(_config(10), kernel=NumPyKernel()).run(_ShiftedFrontProblem(), ("max_evaluations", budget), seed=3)

    assert result["evaluations"] == budget
    assert result["population"]["X"].shape == (10, 4)
    assert result["diversity_archive"]["X"].shape == (10, 4)
    assert result["weights"].shape == (10, 2)


def test_last_generation_evaluates_the_first_offspring_and_updates_both_archives() -> None:
    """With fewer evaluations left than weight vectors, ask evaluates the first offspring of a full generation."""
    problem = _three_objectives()
    algorithm = CTAEA(_config(100), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 91 + 7), seed=4)
    reference = CTAEA(_config(100), kernel=NumPyKernel())
    reference.initialize(problem, ("max_evaluations", 1000), seed=4)

    X = algorithm.ask()
    full = reference.ask()

    assert X.shape == (7, 7)
    np.testing.assert_array_equal(X, full[:7])
    state = algorithm.state
    assert state is not None and state.weights is not None and state.G is not None and state.diversity_F is not None
    F, G, diversity_F = state.F.copy(), state.G.copy(), state.diversity_F.copy()
    evaluated = SerialEvalBackend().evaluate(X, problem)
    algorithm.tell(evaluated)

    union = np.vstack([F, evaluated.F])
    kept = convergence_archive_update(union, compute_violation(np.vstack([G, evaluated.G])), state.weights)
    np.testing.assert_array_equal(state.F, union[kept])
    diversity_union = np.vstack([diversity_F, evaluated.F])
    np.testing.assert_array_equal(state.diversity_F, diversity_union[diversity_archive_update(state.F, diversity_union, state.weights)])
    assert state.n_eval == 98 and algorithm.should_terminate()
    with pytest.raises(RuntimeError, match="no remaining evaluation budget"):
        algorithm.ask()


def test_budget_must_cover_the_initial_population() -> None:
    problem = _three_objectives()
    with pytest.raises(ValueError, match=r"max_evaluations >= 91"):
        CTAEA(_config(100), kernel=NumPyKernel()).run(problem, ("max_evaluations", 90), seed=1)
    # 91 weight vectors from 100 requested: 95 evaluations leave one generation of four offspring.
    assert CTAEA(_config(100), kernel=NumPyKernel()).run(problem, ("max_evaluations", 95), seed=1)["evaluations"] == 95
    with pytest.raises(ValueError, match="pop_size >= 1"):
        CTAEA({**_config(10), "pop_size": 0}, kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=1)


def test_termination_criteria() -> None:
    problem = _ShiftedFrontProblem()
    by_generations = CTAEA(_config(10), kernel=NumPyKernel()).run(problem, ("n_gen", 5), seed=1)
    assert by_generations["evaluations"] == 50
    # n_gen counts generations of N, the number of weight vectors.
    assert CTAEA(_config(100), kernel=NumPyKernel()).run(_three_objectives(), ("n_gen", 2), seed=1)["evaluations"] == 182

    reachable = {"target_value": 0.0, "reference_point": [11.0, 11.0], "max_evaluations": 500}
    early = CTAEA(_config(10), kernel=NumPyKernel()).run(problem, ("hv", reachable), seed=1)
    assert early["hv_reached"] is True and early["evaluations"] < 500


def test_run_is_deterministic_and_equals_ask_tell() -> None:
    problem = _ShiftedFrontProblem()
    first = CTAEA(_config(10), kernel=NumPyKernel()).run(problem, ("max_evaluations", 205), seed=7)
    second = CTAEA(_config(10), kernel=NumPyKernel()).run(problem, ("max_evaluations", 205), seed=7)
    other = CTAEA(_config(10), kernel=NumPyKernel()).run(problem, ("max_evaluations", 205), seed=8)

    algorithm = CTAEA(_config(10), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 205), seed=7)
    while not algorithm.should_terminate():
        X = algorithm.ask()
        algorithm.tell(SerialEvalBackend().evaluate(X, problem))
    stepped = algorithm.result()

    for key in ("X", "F", "G"):
        np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first[key], stepped[key])
    np.testing.assert_array_equal(first["diversity_archive"]["X"], stepped["diversity_archive"]["X"])
    assert not np.array_equal(first["population"]["X"], other["population"]["X"])


def test_the_convergence_archive_is_feasible_and_the_diversity_archive_ignores_the_constraints() -> None:
    result = CTAEA(_config(20), kernel=NumPyKernel()).run(_ShiftedFrontProblem(), ("max_evaluations", 3000), seed=2)

    population, diversity = result["population"], result["diversity_archive"]
    assert population["X"].shape == (20, 4) and diversity["X"].shape == (20, 4)
    assert np.all(population["G"] <= 0.0)
    assert np.all(result["G"] <= 0.0)
    # The DA keeps solutions near the unconstrained front, x1 = 0, which is infeasible.
    assert np.any(diversity["G"][:, 0] > 0.0)


def test_result_modes_and_constraint_mode_none() -> None:
    problem = _ShiftedFrontProblem()
    population = CTAEA(_config(10, result_mode="population"), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=2)
    assert population["X"].shape == (10, 4)

    unconstrained = CTAEA(_config(10, constraint_mode="none"), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=2)
    assert "G" not in unconstrained and "G" not in unconstrained["population"]
    assert "G" not in unconstrained["diversity_archive"]


class _SingleObjective:
    n_var = 2
    n_obj = 1
    xl = np.zeros(2)
    xu = np.ones(2)
    encoding = "real"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        out["F"] = X.sum(axis=1, keepdims=True)


@pytest.mark.parametrize(("problem", "kwargs"), [("c1dtlz1", {"n_var": 7, "n_obj": 3}), ("mw3", {})])
def test_engines_give_the_same_run(problem: str, kwargs: dict[str, int]) -> None:
    """The engine changes the speed of a run, not its result: both archives are the same."""
    pytest.importorskip("numba")
    runs = [
        vamos.optimize(problem, algorithm="ctaea", pop_size=20, max_evaluations=605, seed=3, engine=engine, **kwargs).data
        for engine in ("numpy", "numba")
    ]

    for key in ("population", "diversity_archive"):
        for array in ("X", "F", "G"):
            np.testing.assert_array_equal(runs[1][key][array], runs[0][key][array])
    np.testing.assert_array_equal(runs[1]["F"], runs[0]["F"])


def test_variation_choices_and_objectives_are_validated() -> None:
    problem = _ShiftedFrontProblem()
    for crossover in (("pcx", {}), ("de", {})):
        with pytest.raises(ValueError, match="two by two"):
            CTAEA(_config(10, crossover=crossover), kernel=NumPyKernel()).run(problem, ("max_evaluations", 100), seed=1)
    with pytest.raises(ValueError, match="at least two objectives"):
        CTAEA(_config(10), kernel=NumPyKernel()).run(_SingleObjective(), ("max_evaluations", 100), seed=1)


@pytest.mark.parametrize(
    ("problem_name", "crossover", "mutation"),
    [
        ("bin_knapsack", ("hux", {"prob": 0.9}), ("bitflip", {"prob": 0.1})),
        ("tsp6", ("ox", {}), ("swap", {"prob": 0.1})),
        ("int_alloc", ("uniform", {"prob": 0.9}), ("reset", {"prob": 0.1})),
        ("mixed_design", ("mixed", {"prob": 0.9}), ("mixed", {"prob": 0.1})),
    ],
)
def test_variation_runs_discrete_encodings(problem_name: str, crossover: tuple[str, dict[str, Any]], mutation: Any) -> None:
    problem = make_problem_selection(problem_name).instantiate()

    result = CTAEA(_config(10, crossover=crossover, mutation=mutation), kernel=NumPyKernel()).run(problem, ("max_evaluations", 63), seed=1)

    assert result["evaluations"] == 63
    X = np.asarray(result["population"]["X"], dtype=float)
    assert np.all((X >= problem.xl) & (X <= problem.xu))
    if problem_name == "tsp6":
        np.testing.assert_array_equal(np.sort(X, axis=1), np.tile(np.arange(problem.n_var), (X.shape[0], 1)))


def test_mw3_regression() -> None:
    """A short run on MW3 ends feasible and near the front.

    With 10,000 evaluations, seeds 1 to 10 reach IGD 9.6e-3 to 1.5e-2 against VAMOS's front (1.03e-2 for
    seed 1); the threshold only catches a broken algorithm. The comparison with the published values
    is in docs/algorithms/ctaea.md.
    """
    problem = make_problem_selection("mw3").instantiate()
    config = CTAEAConfig.default(n_var=problem.n_var).to_dict()

    result = CTAEA(config, kernel=NumPyKernel()).run(problem, ("max_evaluations", 10_000), seed=1)

    front = np.loadtxt(Path(vamos.__file__).parent / "resources" / "reference_fronts" / "mw3.csv", delimiter=",")
    population = result["population"]
    assert population["F"].shape == (100, 2)
    assert np.all(population["G"] <= 0.0)
    distances = np.linalg.norm(front[:, None, :] - population["F"][None, :, :], axis=2)
    assert float(distances.min(axis=1).mean()) < 0.03


# ---------------------------------------------------------------------------
# Configuration, registry and tuning space
# ---------------------------------------------------------------------------


def test_default_configuration_is_the_papers() -> None:
    config = CTAEAConfig.default(n_var=12)

    assert config.pop_size == 100
    assert config.crossover == ("sbx", {"prob": 1.0, "eta": 20.0})
    assert config.mutation == ("pm", {"prob": 1.0 / 12, "eta": 20.0})
    assert CTAEAConfig.from_dict(json.loads(config.to_json())) == config
    assert CTAEAConfig.default(encoding="binary").crossover[0] != "sbx"


def test_registry_and_optimize() -> None:
    assert "ctaea" in get_builtin_algorithm_names()
    assert callable(resolve_algorithm("ctaea"))

    result = vamos.optimize("mw3", algorithm="ctaea", pop_size=20, max_evaluations=100, seed=1, engine="numpy")

    assert result.data["evaluations"] == 100
    assert result.meta["run_artifact_resolved_spec"]["algorithm"]["component_id"] == "vamos.algorithm:ctaea@1"
    with pytest.raises(ConfigurationError, match="max_evaluations must be >= pop_size"):
        vamos.optimize("mw3", algorithm="ctaea", pop_size=20, max_evaluations=19, seed=1)


def test_optimize_sizes_the_budget_by_the_weight_vectors() -> None:
    """100 requested vectors give 91 with three objectives: a budget of 95 is enough, 90 is not."""
    assert initial_evaluations("ctaea", 100, n_obj=3) == 91
    assert initial_evaluations("ctaea", 2, n_obj=3) == 3
    assert initial_evaluations("ctaea", 100) == 100

    problem = _three_objectives()
    result = vamos.optimize(problem, algorithm="ctaea", pop_size=100, max_evaluations=95, seed=1, engine="numpy")
    assert result.data["evaluations"] == 95
    with pytest.raises(ConfigurationError, match=r"max_evaluations must be >= 91 .* one individual per weight vector"):
        vamos.optimize(problem, algorithm="ctaea", pop_size=100, max_evaluations=90, seed=1)
    with pytest.raises(ConfigurationError, match=r"max_evaluations must be >= 3 .* one individual per weight vector"):
        vamos.optimize(problem, algorithm="ctaea", pop_size=2, max_evaluations=2, seed=1)


def test_study_planning_sizes_the_budget_by_the_weight_vectors() -> None:
    from vamos.experiment.study.errors import UnresolvedStudyTaskError

    spec = vamos.StudySpec(problems=["c1dtlz1"], algorithms=["ctaea"], seeds=[0], pop_size=100, max_evaluations=90)
    with pytest.raises(UnresolvedStudyTaskError) as excinfo:
        vamos.plan_study(spec)
    assert excinfo.value.reason == "INVALID_EVALUATION_BUDGET"
    assert "91 evaluations of the initial population of ctaea" in str(excinfo.value.expected)

    planned = vamos.plan_study(vamos.StudySpec(problems=["c1dtlz1"], algorithms=["ctaea"], seeds=[0], pop_size=100, max_evaluations=95))
    assert planned is not None


def test_tuning_space_builds_two_parent_configurations() -> None:
    space = build_ctaea_config_space()
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(40):
        assignment = space.sample(rng)
        config = config_from_assignment("ctaea", assignment)
        assert isinstance(config, CTAEAConfig)
        name, params = config.crossover
        seen.add(name)
        assert params["prob"] == assignment["crossover_prob"]
        assert 20 <= config.pop_size <= 200
    assert seen == {"sbx", "blx_alpha", "arithmetic"}

    paper = config_from_assignment("ctaea", {"pop_size": 100, "crossover": "sbx", "mutation": "pm"})
    assert paper.crossover == ("sbx", {"prob": 1.0, "eta": 20.0})
    assert paper.mutation == ("pm", {"eta": 20.0})
