from __future__ import annotations

import numpy as np
import pytest

from vamos.engine.algorithm.agemoea import AGEMOEA
from vamos.engine.algorithm.agemoea.geometry import _survival_score, age_survival
from vamos.engine.algorithm.agemoea.survival import constraint_aware_age_survival
from vamos.engine.algorithm.config import AGEMOEAConfig, RVEAConfig
from vamos.engine.algorithm.rvea import RVEA
from vamos.engine.algorithm.rvea.rvea import (
    _calc_gamma,
    _calc_V,
    _constraint_aware_apd_survival,
    _generate_reference_vectors,
)
from vamos.foundation.kernel.numpy_backend import NumPyKernel


class _ConstrainedIdentityProblem:
    n_var = 2
    n_obj = 2
    n_constraints = 1
    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])
    encoding = "real"

    def evaluate(self, X, out):
        out["F"] = np.asarray(X[:, :2], dtype=float).copy()
        out["G"] = (0.5 - X[:, 0])[:, None]


class _UnconstrainedIdentityProblem:
    n_var = 2
    n_obj = 2
    n_constraints = 0
    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])
    encoding = "real"

    def evaluate(self, X, out):
        out["F"] = np.asarray(X[:, :2], dtype=float).copy()


class _MissingConstraintBackend:
    def evaluate(self, X, problem):
        del problem
        return {"F": np.asarray(X[:, :2], dtype=float).copy()}


def _agemoea_config(*, constraint_mode: str = "feasibility") -> dict:
    return (
        AGEMOEAConfig.builder()
        .pop_size(6)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("polynomial", prob=0.2, eta=20.0)
        .constraint_mode(constraint_mode)
        .result_mode("population")
        .build()
        .to_dict()
    )


def _rvea_config(*, constraint_mode: str = "feasibility") -> dict:
    return (
        RVEAConfig.builder()
        .pop_size(6)
        .n_partitions(5)
        .alpha(2.0)
        .adapt_freq(0.1)
        .crossover("sbx", prob=1.0, eta=30.0)
        .mutation("polynomial", prob=0.2, eta=20.0)
        .constraint_mode(constraint_mode)
        .result_mode("population")
        .build()
        .to_dict()
    )


def test_agemoea_feasibility_survival_keeps_feasible_then_least_violating() -> None:
    F = np.array(
        [
            [5.0, 5.0],
            [0.0, 0.0],
            [4.0, 4.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ]
    )
    G = np.array([[-0.1], [0.4], [-0.2], [0.1], [0.3], [0.2]])

    survivors, _ = constraint_aware_age_survival(F, G, 4, NumPyKernel(), "feasibility")

    assert set(survivors.tolist()) == {0, 2, 3, 5}


@pytest.mark.parametrize("scale, shift", [((1.0, 1.0), (0.0, 0.0)), ((2.0, 3.0), (10.0, 20.0))])
@pytest.mark.parametrize("reverse_rows", [False, True])
def test_age_survival_scores_the_truncated_front(reverse_rows: bool, scale: tuple, shift: tuple) -> None:
    # Front 0 holds 3 points and N = 5, so two of the six front-1 points survive.
    # The author's code (PlatEMO EnvironmentalSelection.m, ``for i = 2:MaxFNo``)
    # scores every later front, the truncated one included, by proximity to the
    # ideal point; pymoo 0.6.2 skips the truncated front and cuts it by index.
    # Shifting and scaling the objectives must not change which rows survive.
    front0 = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
    front1 = [(0.05, 3.0), (0.3, 2.0), (0.55, 0.6), (1.2, 0.3), (3.0, 0.05), (4.0, 0.04)]
    if reverse_rows:
        front1 = front1[::-1]
    base = np.array(front0 + front1)
    F = base * np.array(scale) + np.array(shift)

    survivors, _ = age_survival(F, 5, NumPyKernel())

    kept = {tuple(row) for row in base[survivors].tolist()}
    assert kept == {*front0, (0.55, 0.6), (1.2, 0.3)}


@pytest.mark.parametrize("shift", [(0.0, 0.0), (100.0, 0.0), (0.0, -50.0), (0.0, 0.1234567890123)])
def test_age_survival_with_fewer_first_front_points_than_objectives_ignores_the_origin(shift: tuple) -> None:
    # With fewer first-front points than objectives there is no hyperplane, and
    # the later fronts are normalized by the first front's spread above its
    # ideal point. Translating the objectives must not change the survivors.
    base = np.array([[0.0, 0.0], [1.0, 3.0], [3.0, 2.0]])

    survivors, _ = age_survival(base + np.array(shift), 2, NumPyKernel())

    assert {tuple(row) for row in base[survivors].tolist()} == {(0.0, 0.0), (1.0, 3.0)}


def _random_population(rng: np.random.Generator, trial: int, n_obj: int) -> tuple[np.ndarray, int]:
    n = int(rng.integers(6, 40))
    W = rng.random((n, n_obj))
    if trial % 4 == 0:
        F = W / W.sum(axis=1, keepdims=True) + rng.random((n, 1)) * 0.5  # several fronts
    elif trial % 4 == 1:
        F = W
    elif trial % 4 == 2:
        F = W**3
    else:
        F = W.copy()
        F[0] = W.min(axis=0) * rng.random(n_obj)  # a single point dominates the rest
    return F, int(rng.integers(2, n))


@pytest.mark.parametrize("n_obj", [2, 3])
def test_age_survival_does_not_depend_on_the_objectives_origin(n_obj: int) -> None:
    # Every score is measured from the first front's ideal point, so a
    # translation of the objectives, round or not, keeps the same survivors.
    # Per-objective scaling is not covered: the extreme-point search runs on
    # unnormalized objectives with axes offset by 1e-6, as in PlatEMO, pymoo
    # and jMetal, so it can pick other corners when the units change.
    kernel = NumPyKernel()
    rng = np.random.default_rng(40 + n_obj)
    for trial in range(60):
        F, keep = _random_population(rng, trial, n_obj)
        expected, _ = age_survival(F, keep, kernel)
        shifted, _ = age_survival(F + rng.uniform(-100.0, 100.0, n_obj), keep, kernel)
        np.testing.assert_array_equal(shifted, expected, err_msg=f"trial {trial}")


def _age_survival_scores(F: np.ndarray) -> np.ndarray:
    """Score every row as AGE-MOEA's survival does (PlatEMO's EnvironmentalSelection, ideal-centred)."""
    kernel = NumPyKernel()
    ranks, _ = kernel.nsga2_ranking(F)
    first = ranks == 0
    ideal = F[first].min(axis=0)
    scores = np.zeros(F.shape[0])
    scores[first], p, normalization = _survival_score(F[first], ideal, kernel.selection_kernels())
    distance = np.sum(np.abs((F[~first] - ideal) / normalization) ** p, axis=1) ** (1.0 / p)
    scores[~first] = 1.0 / np.maximum(distance, 1e-12)
    return scores


def test_agemoea_tournament_uses_the_survival_score() -> None:
    # PlatEMO (AGEMOEA.m, ``TournamentSelection(2, N, FrontNo, -CrowdDis)``) and
    # pymoo break rank ties with the survival score, not NSGA-II crowding.
    algorithm = AGEMOEA(_agemoea_config(constraint_mode="none"), NumPyKernel())
    algorithm.initialize(_UnconstrainedIdentityProblem(), termination=("max_evaluations", 18), seed=5)
    state = algorithm.state
    assert state is not None
    np.testing.assert_allclose(state.selection_crowding, _age_survival_scores(state.F))

    parents_F = state.F.copy()
    offspring_F = algorithm.ask()[:, :2].copy()
    algorithm.tell(offspring_F)

    union = np.vstack([parents_F, offspring_F])
    rows = [int(np.flatnonzero(np.all(union == row, axis=1))[0]) for row in state.F]
    np.testing.assert_allclose(state.selection_crowding, _age_survival_scores(union)[rows])


def test_rvea_feasibility_survival_keeps_feasible_then_least_violating() -> None:
    F = np.array(
        [
            [5.0, 5.0],
            [0.0, 0.0],
            [4.0, 4.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ]
    )
    G = np.array([[-0.1], [0.4], [-0.2], [0.1], [0.3], [0.2]])
    V = _calc_V(_generate_reference_vectors(2, 5))
    gamma = _calc_gamma(V)

    survivors, ideal, nadir = _constraint_aware_apd_survival(
        F,
        G,
        V,
        gamma,
        np.full(2, np.inf),
        4,
        1,
        4,
        2.0,
        "feasibility",
    )

    assert set(survivors.tolist()) == {0, 2, 3, 5}
    np.testing.assert_array_equal(ideal, np.array([4.0, 4.0]))
    np.testing.assert_array_equal(nadir, np.array([5.0, 5.0]))


def test_rvea_all_infeasible_objectives_do_not_seed_geometry() -> None:
    V = _calc_V(_generate_reference_vectors(2, 5))
    gamma = _calc_gamma(V)
    initial_ideal = np.full(2, np.inf)
    infeasible_F = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ]
    )
    infeasible_G = np.ones((6, 1))

    _, provisional_ideal, provisional_nadir = _constraint_aware_apd_survival(
        infeasible_F,
        infeasible_G,
        V,
        gamma,
        initial_ideal,
        4,
        1,
        4,
        2.0,
        "feasibility",
    )

    assert np.isinf(provisional_ideal).all()
    assert provisional_nadir is None

    mixed_F = np.array(
        [
            [4.0, 4.0],
            [5.0, 5.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ]
    )
    mixed_G = np.array([[-0.2], [-0.1], [0.1], [0.2], [0.3], [0.4]])

    _, feasible_ideal, feasible_nadir = _constraint_aware_apd_survival(
        mixed_F,
        mixed_G,
        V,
        gamma,
        provisional_ideal,
        4,
        2,
        4,
        2.0,
        "feasibility",
    )

    np.testing.assert_array_equal(feasible_ideal, np.array([4.0, 4.0]))
    np.testing.assert_array_equal(feasible_nadir, np.array([5.0, 5.0]))


@pytest.mark.parametrize(
    ("algorithm_cls", "config"),
    [
        (AGEMOEA, _agemoea_config()),
        (RVEA, _rvea_config()),
    ],
)
def test_active_constraint_mode_exposes_population_aligned_g(algorithm_cls, config) -> None:
    result = algorithm_cls(config, NumPyKernel()).run(
        _ConstrainedIdentityProblem(),
        termination=("max_evaluations", 12),
        seed=7,
    )

    assert "G" in result
    assert "G" in result["population"]
    # RVEA keeps one survivor per non-empty reference vector, so its population can be smaller.
    n_rows = result["population"]["X"].shape[0]
    assert result["population"]["G"].shape == (n_rows, 1)
    assert n_rows == 6 if algorithm_cls is AGEMOEA else 1 <= n_rows <= 6
    np.testing.assert_array_equal(result["G"], result["population"]["G"])
    expected_g = (0.5 - result["population"]["X"][:, 0])[:, None]
    np.testing.assert_allclose(result["population"]["G"], expected_g)


@pytest.mark.parametrize(
    ("algorithm_cls", "config"),
    [
        (AGEMOEA, _agemoea_config()),
        (RVEA, _rvea_config()),
    ],
)
def test_active_constraint_mode_rejects_missing_initial_g(algorithm_cls, config) -> None:
    algorithm = algorithm_cls(config, NumPyKernel())

    with pytest.raises(ValueError, match="requires constraint values G"):
        algorithm.initialize(
            _ConstrainedIdentityProblem(),
            termination=("max_evaluations", 12),
            seed=5,
            eval_strategy=_MissingConstraintBackend(),
        )


@pytest.mark.parametrize(
    ("algorithm_cls", "config_factory"),
    [
        (AGEMOEA, _agemoea_config),
        (RVEA, _rvea_config),
    ],
)
def test_constraint_mode_none_preserves_unconstrained_behavior(algorithm_cls, config_factory) -> None:
    config = config_factory(constraint_mode="none")
    constrained_result = algorithm_cls(config, NumPyKernel()).run(
        _ConstrainedIdentityProblem(),
        termination=("max_evaluations", 18),
        seed=11,
    )
    unconstrained_result = algorithm_cls(config, NumPyKernel()).run(
        _UnconstrainedIdentityProblem(),
        termination=("max_evaluations", 18),
        seed=11,
    )

    assert "G" not in constrained_result
    assert "G" not in constrained_result["population"]
    np.testing.assert_array_equal(constrained_result["X"], unconstrained_result["X"])
    np.testing.assert_array_equal(constrained_result["F"], unconstrained_result["F"])


@pytest.mark.parametrize(
    ("algorithm_cls", "config"),
    [
        (AGEMOEA, _agemoea_config()),
        (RVEA, _rvea_config()),
    ],
)
def test_constrained_ask_tell_rejects_missing_offspring_g(algorithm_cls, config) -> None:
    problem = _ConstrainedIdentityProblem()
    algorithm = algorithm_cls(config, NumPyKernel())
    algorithm.initialize(problem, termination=("max_evaluations", 12), seed=3)
    X = algorithm.ask()
    F = X[:, :2].copy()

    with pytest.raises(ValueError, match="requires constraint values G"):
        algorithm.tell(F)


def test_apd_survival_keeps_one_survivor_per_vector() -> None:
    from vamos.engine.algorithm.rvea.survival import apd_survival

    V = _calc_V(np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]))
    gamma = _calc_gamma(V)
    F = np.array([[0.01, 5.0], [0.02, 9.0], [1.0, 1.0], [1.1, 1.1]])

    survivors, _, _ = apd_survival(F, V, gamma, np.full(2, np.inf), 3, 1, 10, 2.0)

    assert sorted(survivors.tolist()) == [0, 2]


def test_rvea_parents_are_a_permutation() -> None:
    from vamos.foundation.problem.zdt1 import ZDT1Problem

    cfg = RVEAConfig.default(pop_size=20, n_var=6, n_obj=2)
    algorithm = RVEA(cfg.to_dict(), kernel=NumPyKernel())
    algorithm.initialize(ZDT1Problem(n_var=6), termination=("max_evaluations", 100), seed=4)
    st = algorithm._st
    assert st is not None
    seen: list[np.ndarray] = []
    produce = st.variation.produce_offspring

    def spy(parents, rng):  # type: ignore[no-untyped-def]
        seen.append(np.array(parents))
        return produce(parents, rng)

    st.variation.produce_offspring = spy
    population = st.X.copy()
    algorithm.ask()

    rows = [int(np.flatnonzero(np.all(population == parent, axis=1))[0]) for parent in seen[0]]
    assert sorted(rows) == list(range(population.shape[0]))
