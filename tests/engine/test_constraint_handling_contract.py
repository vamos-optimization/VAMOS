"""Constraint handling of the built-in algorithms: feasible solutions come before infeasible ones."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.algorithms import (
    AGEMOEAConfig,
    IBEAConfig,
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    RVEAConfig,
    SMPSOConfig,
    SMSEMOAConfig,
    SPEA2Config,
)
from vamos.engine.algorithm.ibea.helpers import environmental_selection
from vamos.engine.algorithm.nsgaiii.helpers import nsgaiii_survival
from vamos.foundation.kernel.registry import resolve_kernel

CONFIG_CLASSES: dict[str, Any] = {
    "agemoea": AGEMOEAConfig,
    "ibea": IBEAConfig,
    "moead": MOEADConfig,
    "nsgaii": NSGAIIConfig,
    "nsgaiii": NSGAIIIConfig,
    "rvea": RVEAConfig,
    "smpso": SMPSOConfig,
    "smsemoa": SMSEMOAConfig,
    "spea2": SPEA2Config,
}
REFERENCE_DIRECTION_ALGORITHMS = {"moead", "nsgaiii", "rvea"}


def _selection_kernels(engine: str) -> Any:
    if engine == "numba":
        pytest.importorskip("numba")
    return resolve_kernel(engine).selection_kernels()


# ---- IBEA -------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_ibea_selection_removes_infeasible_solutions_first(engine: str) -> None:
    # The two infeasible solutions dominate the feasible ones, so only the constraint penalty can keep the latter.
    X = np.arange(4, dtype=float)[:, None]
    F = np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 0.9]])
    G = np.array([[1.0], [2.0], [-1.0], [-0.5]])

    survivors, _, kept_G, _ = environmental_selection(X, F, G, 2, "epsilon", 0.05, _selection_kernels(engine))

    assert sorted(survivors[:, 0].tolist()) == [2.0, 3.0]
    assert np.all(kept_G <= 0.0)


@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_ibea_selection_keeps_the_least_violation_when_nothing_is_feasible(engine: str) -> None:
    X = np.arange(4, dtype=float)[:, None]
    F = np.array([[0.0, 1.0], [0.3, 0.7], [0.6, 0.4], [1.0, 0.0]])
    G = np.array([[3.0], [0.5], [2.0], [0.1]])

    survivors, _, _, _ = environmental_selection(X, F, G, 2, "epsilon", 0.05, _selection_kernels(engine))

    assert sorted(survivors[:, 0].tolist()) == [1.0, 3.0]


# ---- NSGA-III ---------------------------------------------------------------


def _nsgaiii_survive(
    F_par: np.ndarray, G_par: np.ndarray, F_off: np.ndarray, G_off: np.ndarray, pop_size: int
) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    ref_norm = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    X_par = np.zeros((F_par.shape[0], 2))
    X_off = np.ones((F_off.shape[0], 2))
    _, F_new, G_new, indices, *_ = nsgaiii_survival(
        X_par,
        F_par,
        G_par,
        X_off,
        F_off,
        G_off,
        pop_size,
        ref_norm,
        np.random.default_rng(0),
        np.full(2, np.inf),
        None,
        np.full(2, -np.inf),
    )
    assert G_new is not None
    return np.asarray(indices), G_new


def test_nsgaiii_survival_keeps_feasible_solutions_before_dominating_infeasible_ones() -> None:
    F_par = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    F_off = F_par + 1.0  # Every offspring is dominated by a parent.

    indices, G_new = _nsgaiii_survive(F_par, np.ones((3, 1)), F_off, -np.ones((3, 1)), 3)

    assert sorted(indices.tolist()) == [3, 4, 5]
    assert np.all(G_new <= 0.0)


def test_nsgaiii_survival_fills_with_the_least_violation() -> None:
    F_par = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    G_par = np.array([[0.4], [0.1], [0.9]])
    F_off = F_par + 1.0
    G_off = np.array([[-1.0], [0.2], [0.3]])

    indices, _ = _nsgaiii_survive(F_par, G_par, F_off, G_off, 3)

    assert sorted(indices.tolist()) == [1, 3, 4]  # The feasible offspring, then violations 0.1 and 0.2.


def test_nsgaiii_ask_tell_requires_offspring_constraints_on_a_constrained_problem() -> None:
    from vamos.engine.algorithm.nsgaiii.nsgaiii import NSGAIII
    from vamos.foundation.kernel.numpy_backend import NumPyKernel

    config = NSGAIIIConfig.default(n_var=3, n_obj=2)
    algorithm = NSGAIII(config.to_dict(), kernel=NumPyKernel())
    algorithm.initialize(_problem(), ("max_evaluations", 10 * int(config.to_dict()["pop_size"])), seed=0)
    X = algorithm.ask()

    with pytest.raises(ValueError, match="requires constraint values G"):
        algorithm.tell({"F": _objectives(X)})
    with pytest.raises(ValueError, match="align"):
        algorithm.tell({"F": _objectives(X), "G": _constraints(X)[:-1]})
    algorithm.tell({"F": _objectives(X), "G": _constraints(X)})  # The offspring is still pending after a rejection.

    assert algorithm.state is not None and algorithm.state.G is not None


def test_nsgaiii_moves_the_population_into_the_feasible_region() -> None:
    def feasible_share(mode: str) -> float:
        config = dataclasses.replace(NSGAIIIConfig.default(n_var=3, n_obj=2), constraint_mode=mode)
        pop_size = int(config.to_dict()["pop_size"])
        result = vamos.optimize(
            _problem(), algorithm="nsgaiii", algorithm_config=config, max_evaluations=6 * pop_size, seed=3, engine="numpy"
        )
        return float(np.mean(_constraints(result.data["population"]["X"])[:, 0] <= 0.0))

    assert feasible_share("feasibility") == 1.0
    assert feasible_share("none") < 0.5


# ---- constraint_mode --------------------------------------------------------


@pytest.mark.parametrize("value", ["penalty", "epsilon", "feasible", "nonsense", ""])
@pytest.mark.parametrize("name", sorted(CONFIG_CLASSES))
def test_constraint_mode_rejects_values_that_no_algorithm_implements(name: str, value: str) -> None:
    with pytest.raises(ValueError, match="constraint_mode"):
        CONFIG_CLASSES[name].builder().constraint_mode(value)


@pytest.mark.parametrize("value", ["penalty", "epsilon"])
def test_configuration_objects_reject_unimplemented_constraint_modes(value: str) -> None:
    config = NSGAIIConfig.default(pop_size=8, n_var=3)

    with pytest.raises(ValueError, match="constraint_mode"):
        dataclasses.replace(config, constraint_mode=value)
    with pytest.raises(ValueError, match="constraint_mode"):
        NSGAIIConfig.from_dict({**config.to_dict(), "constraint_mode": value})


@pytest.mark.parametrize("value, expected", [("none", "none"), ("feasibility", "feasibility"), (" Feasibility ", "feasibility")])
def test_supported_constraint_modes_are_accepted(value: str, expected: str) -> None:
    config = NSGAIIConfig.builder().pop_size(8).crossover("sbx").mutation("pm").constraint_mode(value).build()

    assert config.constraint_mode == expected


@pytest.mark.parametrize("name", sorted(CONFIG_CLASSES))
def test_constraint_handling_can_be_disabled_on_a_constrained_problem(name: str) -> None:
    config = dataclasses.replace(_config(name), constraint_mode="none")
    pop_size = int(config.to_dict()["pop_size"])

    result = vamos.optimize(_problem(), algorithm=name, algorithm_config=config, max_evaluations=2 * pop_size, seed=3, engine="numpy")

    assert np.asarray(result.F).shape[1] == 2


# ---- final results ----------------------------------------------------------


def _objectives(X: np.ndarray) -> np.ndarray:
    # Objectives ignore x0, so infeasible solutions (x0 > 0.2) are as good as feasible ones and often dominate them.
    return np.column_stack([X[:, 1], 1.0 - X[:, 1] + X[:, 2]])


def _constraints(X: np.ndarray) -> np.ndarray:
    return (X[:, 0] - 0.2)[:, None]


def _config(name: str) -> Any:
    cls = CONFIG_CLASSES[name]
    if name in REFERENCE_DIRECTION_ALGORITHMS:
        config = cls.default(n_var=3, n_obj=2)
    else:
        config = cls.default(pop_size=20, n_var=3)
    return dataclasses.replace(config, result_mode="non_dominated")


def _problem() -> Any:
    return vamos.make_problem(
        _objectives,
        n_var=3,
        n_obj=2,
        bounds=[(0.0, 1.0)] * 3,
        encoding="real",
        constraints=_constraints,
        n_constraints=1,
        vectorized=True,
    )


@pytest.mark.parametrize("name", sorted(CONFIG_CLASSES))
def test_non_dominated_results_contain_only_feasible_solutions_when_any_exist(name: str) -> None:
    config = _config(name)
    pop_size = int(config.to_dict()["pop_size"])

    # A budget of one population returns the random initial population, which always mixes both kinds.
    result = vamos.optimize(_problem(), algorithm=name, algorithm_config=config, max_evaluations=pop_size, seed=3, engine="numpy")

    population = result.data["population"]["X"]
    population_feasible = _constraints(population)[:, 0] <= 0.0
    assert population_feasible.any() and not population_feasible.all(), "the final population must mix both kinds"
    assert np.all(_constraints(np.asarray(result.X))[:, 0] <= 0.0)


def _dominated_rows(F: np.ndarray) -> int:
    less_equal = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    strictly_less = np.any(F[:, None, :] < F[None, :, :], axis=2)
    return int(np.any(less_equal & strictly_less, axis=0).sum())


@pytest.mark.parametrize("problem_name, n_var, n_obj", [("dtlz1", 7, 3), ("mw1", 15, 2)])
@pytest.mark.parametrize("name", sorted(CONFIG_CLASSES))
def test_non_dominated_result_mode_has_no_dominated_rows(name: str, problem_name: str, n_var: int, n_obj: int) -> None:
    cls = CONFIG_CLASSES[name]
    if name in REFERENCE_DIRECTION_ALGORITHMS:
        config = cls.default(n_var=n_var, n_obj=n_obj)
    else:
        config = cls.default(pop_size=100, n_var=n_var)
    config = dataclasses.replace(config, result_mode="non_dominated")
    problem = vamos.make_problem_selection(problem_name, n_var=n_var, n_obj=n_obj).instantiate()

    result = vamos.optimize(problem, algorithm=name, algorithm_config=config, max_evaluations=1000, seed=1, engine="numpy")

    F = np.asarray(result.F)
    G = result.data.get("G")
    if G is not None and np.all(np.asarray(G) <= 0.0, axis=1).any():
        assert np.all(np.asarray(G) <= 0.0)
    assert _dominated_rows(F) == 0, f"{_dominated_rows(F)} of {F.shape[0]} result rows are dominated"
