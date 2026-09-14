from __future__ import annotations

import numpy as np
import pytest

from vamos.engine.algorithm.agemoea import AGEMOEA
from vamos.engine.algorithm.agemoea.agemoea import _constraint_aware_age_survival
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

    survivors = _constraint_aware_age_survival(F, G, 4, NumPyKernel(), "feasibility")

    assert set(survivors.tolist()) == {0, 2, 3, 5}


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
    assert result["population"]["G"].shape == (6, 1)
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