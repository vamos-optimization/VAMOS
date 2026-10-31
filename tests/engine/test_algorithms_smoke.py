"""Lightweight smoke tests for the algorithm cores."""

import warnings

import numpy as np
import pytest

from vamos.engine.algorithm.components.termination import HVTracker
from vamos.engine.algorithm.config import (
    IBEAConfig,
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    SMPSOConfig,
    SMSEMOAConfig,
    SPEA2Config,
)
from vamos.engine.algorithm.ibea import IBEA
from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.engine.algorithm.nsgaiii import NSGAIII
from vamos.engine.algorithm.smpso import SMPSO
from vamos.engine.algorithm.smsemoa import SMSEMOA
from vamos.engine.algorithm.spea2 import SPEA2
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.dtlz import DTLZ2Problem
from vamos.foundation.problem.tsp import TSPProblem
from vamos.foundation.problem.zdt1 import ZDT1Problem
from vamos.foundation.problem.zdt2 import ZDT2Problem
from vamos.foundation.quality_indicators.hypervolume import hypervolume

HV_STOP_ALGORITHMS = ("nsgaii", "nsgaiii", "moead", "smsemoa", "spea2", "ibea", "smpso")


def _hv_stop_algorithm(name: str, pop_size: int, n_var: int):  # noqa: ANN202 - algorithm classes share no base type
    if name == "nsgaii":
        return NSGAII(NSGAIIConfig.default(pop_size=pop_size, n_var=n_var).to_dict(), kernel=NumPyKernel())
    if name == "nsgaiii":
        cfg = (
            NSGAIIIConfig.builder()
            .pop_size(pop_size)
            .crossover("sbx", prob=1.0, eta=30.0)
            .mutation("polynomial", prob="1/n", eta=20.0)
            .selection("tournament", size=2)
            .reference_directions(divisions=pop_size - 1)
            .build()
        )
        return NSGAIII(cfg.to_dict(), kernel=NumPyKernel())
    if name == "moead":
        return MOEAD(MOEADConfig.default(pop_size=pop_size, n_var=n_var, n_obj=2).to_dict(), kernel=NumPyKernel())
    if name == "smsemoa":
        return SMSEMOA(SMSEMOAConfig.default(pop_size=pop_size, n_var=n_var).to_dict(), kernel=NumPyKernel())
    if name == "spea2":
        return SPEA2(SPEA2Config.default(pop_size=pop_size, n_var=n_var).to_dict(), kernel=NumPyKernel())
    if name == "ibea":
        return IBEA(IBEAConfig.default(pop_size=pop_size, n_var=n_var).to_dict(), kernel=NumPyKernel())
    if name == "smpso":
        return SMPSO(SMPSOConfig.default(pop_size=pop_size, n_var=n_var).to_dict(), kernel=NumPyKernel())
    raise AssertionError(name)


def _reported_hv_target(result: dict) -> bool:
    # NSGA-III, SMS-EMOA and IBEA still report the flag as ``hv_converged``.
    return bool(result.get("hv_reached", result.get("hv_converged", False)))


def _hv_inside(F: np.ndarray, ref: np.ndarray) -> float:
    inside = F[np.all(F <= ref, axis=1)]
    return hypervolume(inside, ref) if inside.shape[0] else 0.0


@pytest.mark.parametrize("name", HV_STOP_ALGORITHMS)
def test_hv_termination_does_not_stop_before_the_target(name: str) -> None:
    pop_size, n_var, target = 100, 30, 0.6
    ref = np.array([1.1, 1.1])
    algorithm = _hv_stop_algorithm(name, pop_size, n_var)
    termination = (
        "hv",
        {"target_value": target, "reference_point": ref.tolist(), "max_evaluations": 30_000, "allow_ref_expand": True},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = algorithm.run(ZDT1Problem(n_var=n_var), termination=termination, seed=11)

    hv = _hv_inside(np.asarray(result["F"], dtype=float), ref)
    assert result["evaluations"] > pop_size, f"{name} stopped after the initial population with HV {hv:.4f}"
    assert _reported_hv_target(result) is True
    assert hv >= target
    assert any(issubclass(w.category, FutureWarning) and "allow_ref_expand" in str(w.message) for w in caught)


@pytest.mark.parametrize("name", HV_STOP_ALGORITHMS)
def test_hv_termination_ignores_points_outside_the_reference(name: str) -> None:
    pop_size, n_var, target, budget = 20, 30, 0.6, 400
    ref = np.array([1.1, 1.1])
    algorithm = _hv_stop_algorithm(name, pop_size, n_var)
    termination = ("hv", {"target_value": target, "reference_point": ref.tolist(), "max_evaluations": budget})

    # The random initial ZDT1 population lies far outside [1.1, 1.1].
    result = algorithm.run(ZDT1Problem(n_var=n_var), termination=termination, seed=11)

    if _reported_hv_target(result):
        assert _hv_inside(np.asarray(result["F"], dtype=float), ref) >= target
    else:
        assert result["evaluations"] == budget


def test_hv_tracker_scores_only_points_inside_the_reference() -> None:
    tracker = HVTracker({"target_value": 0.25, "reference_point": [1.0, 1.0]}, kernel=NumPyKernel())

    assert tracker.reached(np.array([[5.0, 5.0], [2.0, 0.0]])) is False
    assert tracker.last_value == 0.0
    assert tracker.reached(np.array([[0.5, 0.5], [2.0, 0.0], [0.0, 3.0]])) is True
    assert tracker.last_value == pytest.approx(0.25)


def test_nsgaii_hv_termination_hits_target():
    pop_size = 10
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    )
    algorithm = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=8)
    hv_term = (
        "hv",
        {
            "target_value": 1e-4,
            "reference_point": [5.0, 5.0],
            "max_evaluations": 40,
        },
    )

    result = algorithm.run(problem, termination=hv_term, seed=3)

    assert result["hv_reached"] is True
    # Result contains only non-dominated solutions (may be <= pop_size)
    assert result["F"].shape[0] <= pop_size
    assert result["F"].shape[1] == problem.n_obj
    # Full population still available
    assert result["population"]["F"].shape == (pop_size, problem.n_obj)


def test_nsgaii_steady_state_runs():
    pop_size = 12
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(1)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    )
    algorithm = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=6)
    budget = pop_size + 5

    result = algorithm.run(problem, termination=("max_evaluations", budget), seed=11)

    assert result["evaluations"] == budget
    assert result["F"].shape[0] <= pop_size
    assert result["population"]["F"].shape == (pop_size, problem.n_obj)


def test_smsemoa_smoke_runs_with_small_population():
    pop_size = 8
    cfg = (
        SMSEMOAConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("random")
        .reference_point(offset=1.0, adaptive=True)
        .build()
    )
    algorithm = SMSEMOA(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT2Problem(n_var=6)

    result = algorithm.run(problem, termination=("max_evaluations", pop_size + 6), seed=4)

    # Result contains only non-dominated solutions (may be <= pop_size)
    assert result["F"].shape[0] <= pop_size
    assert result["F"].shape[1] == problem.n_obj
    assert np.isfinite(result["F"]).all()
    # Full population still available
    assert result["population"]["F"].shape == (pop_size, problem.n_obj)


def test_smsemoa_uses_eval_strategy():
    pop_size = 8
    cfg = (
        SMSEMOAConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("random")
        .reference_point(offset=1.0, adaptive=True)
        .build()
    )
    algorithm = SMSEMOA(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT2Problem(n_var=6)

    class CountingBackend:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def evaluate(self, X, problem):  # noqa: ANN001 - protocol-style test double
            from vamos.foundation.eval import EvaluationResult

            self.calls.append(int(X.shape[0]))
            F = np.full((X.shape[0], problem.n_obj), 0.1, dtype=float)
            return EvaluationResult(F=F, G=None)

    backend = CountingBackend()
    algorithm.run(problem, termination=("max_evaluations", pop_size + 3), seed=4, eval_strategy=backend)

    assert backend.calls[0] == pop_size  # initial population
    assert backend.calls[1:] == [1, 1, 1]  # steady-state offspring evaluations


def test_moead_smoke_runs_without_weight_files():
    pop_size = 6
    cfg = (
        MOEADConfig.builder()
        .pop_size(pop_size)
        .neighbor_size(3)
        .delta(0.9)
        .replace_limit(2)
        .crossover("de", cr=1.0, f=0.5)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .weight_vectors(divisions=pop_size - 1)
        .build()
    )
    algorithm = MOEAD(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=8)

    result = algorithm.run(problem, termination=("max_evaluations", pop_size + 4), seed=5)

    # Result contains only non-dominated solutions (may be <= pop_size)
    assert result["F"].shape[0] <= pop_size
    assert result["F"].shape[1] == problem.n_obj
    assert np.isfinite(result["F"]).all()
    # Full population still available
    assert result["population"]["F"].shape == (pop_size, problem.n_obj)


def test_spea2_smoke_runs_with_archive():
    pop_size = 12
    cfg = (
        SPEA2Config.builder()
        .pop_size(pop_size)
        .archive_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    )
    algorithm = SPEA2(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=6)

    result = algorithm.run(problem, termination=("max_evaluations", pop_size + 12), seed=7)

    # The environmental archive is full; the non_dominated result keeps its first front.
    assert result["archive"]["F"].shape[0] == cfg.archive_size
    assert 0 < result["F"].shape[0] <= cfg.archive_size
    assert result["F"].shape[1] == problem.n_obj
    assert np.isfinite(result["F"]).all()


def test_ibea_smoke_indicator_eps():
    pop_size = 10
    cfg = (
        IBEAConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .indicator("eps")
        .kappa(0.05)
        .build()
    )
    algorithm = IBEA(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT2Problem(n_var=6)

    result = algorithm.run(problem, termination=("max_evaluations", pop_size + 10), seed=8)

    # Result contains only non-dominated solutions (may be <= pop_size)
    assert result["F"].shape[0] <= pop_size
    assert result["F"].shape[1] == problem.n_obj
    assert np.isfinite(result["F"]).all()
    # Full population still available
    assert result["population"]["F"].shape == (pop_size, problem.n_obj)


def test_ibea_uses_eval_strategy():
    pop_size = 10
    cfg = (
        IBEAConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .indicator("eps")
        .kappa(0.05)
        .build()
    )
    algorithm = IBEA(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT2Problem(n_var=6)

    class CountingBackend:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def evaluate(self, X, problem):  # noqa: ANN001 - protocol-style test double
            from vamos.foundation.eval import EvaluationResult

            self.calls.append(int(X.shape[0]))
            F = np.full((X.shape[0], problem.n_obj), 0.2, dtype=float)
            return EvaluationResult(F=F, G=None)

    backend = CountingBackend()
    algorithm.run(problem, termination=("max_evaluations", pop_size + 4), seed=8, eval_strategy=backend)

    assert backend.calls[0] == pop_size
    assert backend.calls[1:] == [4]


def test_nsgaiii_uses_eval_strategy():
    pop_size = 6
    cfg = (
        NSGAIIIConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .reference_directions(divisions=2)
        .build()
    )
    algorithm = NSGAIII(cfg.to_dict(), kernel=NumPyKernel())
    problem = DTLZ2Problem(n_var=6, n_obj=3)

    class CountingBackend:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def evaluate(self, X, problem):  # noqa: ANN001 - protocol-style test double
            from vamos.foundation.eval import EvaluationResult

            self.calls.append(int(X.shape[0]))
            F = np.full((X.shape[0], problem.n_obj), 0.3, dtype=float)
            return EvaluationResult(F=F, G=None)

    backend = CountingBackend()
    algorithm.run(problem, termination=("max_evaluations", pop_size * 2), seed=12, eval_strategy=backend)

    assert backend.calls[0] == pop_size
    assert backend.calls[1:] == [pop_size]


def test_smpso_smoke_runs():
    pop_size = 14
    cfg = SMPSOConfig.builder().pop_size(pop_size).archive_size(pop_size).mutation("polynomial", prob="1/n", eta=20.0).build()
    algorithm = SMPSO(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=5)

    result = algorithm.run(problem, termination=("max_evaluations", pop_size * 2), seed=9)

    assert result["F"].shape[0] > 0
    assert result["F"].shape[1] == problem.n_obj
    assert np.isfinite(result["F"]).all()


def test_nsgaii_with_multiprocessing_eval_strategy():
    pop_size = 10
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    )
    algorithm = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=6)
    from vamos.foundation.eval.backends import MultiprocessingEvalBackend

    eval_strategy = MultiprocessingEvalBackend(n_workers=2)
    result = algorithm.run(problem, termination=("max_evaluations", pop_size + 6), seed=10, eval_strategy=eval_strategy)

    # Result contains only non-dominated solutions (may be <= pop_size)
    assert result["F"].shape[0] <= pop_size
    assert result["F"].shape[1] == problem.n_obj
    assert np.isfinite(result["F"]).all()
    # Full population still available
    assert result["population"]["F"].shape == (pop_size, problem.n_obj)


def test_nsgaii_permutation_smoke():
    pop_size = 8
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("ox", prob=0.9)
        .mutation("swap", prob="2/n")
        .selection("tournament", size=2)
        .build()
    )
    algorithm = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    problem = TSPProblem(n_cities=7)

    result = algorithm.run(problem, termination=("max_evaluations", pop_size + 6), seed=6)

    X = result["X"]
    # Result contains only non-dominated solutions (may be <= pop_size)
    assert X.shape[0] <= pop_size
    assert X.shape[1] == problem.n_var
    # Every individual should remain a valid permutation.
    expected = np.arange(problem.n_var)
    assert all(np.array_equal(np.sort(row), expected) for row in X)
    assert result["F"].shape[0] <= pop_size
    assert result["F"].shape[1] == problem.n_obj
    # Full population still available
    assert result["population"]["F"].shape == (pop_size, problem.n_obj)
