import numpy as np
import pytest

from vamos.engine.algorithm.config import MOEADConfig, NSGAIIIConfig, SMPSOConfig, SMSEMOAConfig, SPEA2Config
from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.algorithm.nsgaiii import NSGAIII
from vamos.engine.algorithm.smpso import SMPSO
from vamos.engine.algorithm.smsemoa import SMSEMOA
from vamos.engine.algorithm.spea2 import SPEA2
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.binary import BinaryKnapsackProblem
from vamos.foundation.problem.integer import IntegerResourceAllocationProblem
from vamos.foundation.problem.mixed import MixedDesignProblem
from vamos.foundation.problem.tsp import TSPProblem


def _run_moead(problem, cross, mut, pop_size=12, n_eval=60):
    cfg = (
        MOEADConfig.builder()
        .pop_size(pop_size)
        .neighbor_size(min(5, pop_size))
        .delta(0.9)
        .replace_limit(2)
        .crossover(cross[0], **cross[1])
        .mutation(mut[0], **mut[1])
        .aggregation("tchebycheff")
        .result_mode("population")
    ).build()
    algo = MOEAD(cfg.to_dict(), kernel=NumPyKernel())
    res = algo.run(problem, termination=("max_evaluations", n_eval), seed=0)
    assert "F" in res and res["F"].shape[0] == pop_size
    assert np.isfinite(res["F"]).all()


def _run_smsemoa(problem, cross, mut, pop_size=10, n_eval=40):
    cfg = (
        SMSEMOAConfig.builder()
        .pop_size(pop_size)
        .crossover(cross[0], **cross[1])
        .mutation(mut[0], **mut[1])
        .selection("tournament", size=2)
        .reference_point(offset=0.1, adaptive=True)
        .result_mode("population")
    ).build()
    algo = SMSEMOA(cfg.to_dict(), kernel=NumPyKernel())
    res = algo.run(problem, termination=("max_evaluations", n_eval), seed=0)
    assert "F" in res and res["F"].shape[0] == pop_size
    assert np.isfinite(res["F"]).all()


def _run_nsgaiii(problem, cross, mut, pop_size=12, n_eval=60):
    cfg = (
        NSGAIIIConfig.builder()
        .pop_size(pop_size)
        .crossover(cross[0], **cross[1])
        .mutation(mut[0], **mut[1])
        .selection("tournament", size=2)
        .reference_directions(path=None)
    ).build()
    algo = NSGAIII(cfg.to_dict(), kernel=NumPyKernel())
    res = algo.run(problem, termination=("max_evaluations", n_eval), seed=0)
    assert "F" in res and res["F"].shape[0] >= pop_size
    assert np.isfinite(res["F"]).all()


def _run_spea2(problem, cross, mut, pop_size=12, n_eval=60):
    cfg = (
        SPEA2Config.builder()
        .pop_size(pop_size)
        .archive_size(pop_size)
        .crossover(cross[0], **cross[1])
        .mutation(mut[0], **mut[1])
        .selection("tournament", size=2)
    ).build()
    algo = SPEA2(cfg.to_dict(), kernel=NumPyKernel())
    res = algo.run(problem, termination=("max_evaluations", n_eval), seed=0)
    assert "F" in res and res["F"].shape[0] > 0
    assert np.isfinite(res["F"]).all()


def test_moead_binary_and_integer():
    _run_moead(BinaryKnapsackProblem(n_var=8), ("uniform", {"prob": 0.9}), ("bitflip", {"prob": "1/n"}))
    _run_moead(
        IntegerResourceAllocationProblem(n_var=6, max_per_task=4),
        ("uniform", {"prob": 0.9}),
        ("reset", {"prob": "1/n", "step": 1}),
    )


def test_moead_permutation():
    _run_moead(TSPProblem(n_cities=7), ("ox", {"prob": 0.9}), ("swap", {"prob": "2/n"}), pop_size=10, n_eval=40)


def test_smsemoa_binary_and_integer():
    _run_smsemoa(BinaryKnapsackProblem(n_var=8), ("uniform", {"prob": 0.9}), ("bitflip", {"prob": "1/n"}))
    _run_smsemoa(
        IntegerResourceAllocationProblem(n_var=6, max_per_task=4),
        ("uniform", {"prob": 0.9}),
        ("reset", {"prob": "1/n", "step": 1}),
    )


def test_smsemoa_permutation():
    _run_smsemoa(TSPProblem(n_cities=7), ("aex", {"prob": 0.9}), ("two_opt", {"prob": "2/n"}), pop_size=10, n_eval=40)


def test_nsgaiii_binary_and_integer():
    _run_nsgaiii(BinaryKnapsackProblem(n_var=8), ("uniform", {"prob": 0.9}), ("bitflip", {"prob": "1/n"}))
    _run_nsgaiii(
        IntegerResourceAllocationProblem(n_var=6, max_per_task=4),
        ("uniform", {"prob": 0.9}),
        ("reset", {"prob": "1/n", "step": 1}),
    )


def test_nsgaiii_permutation():
    _run_nsgaiii(TSPProblem(n_cities=7), ("aex", {"prob": 0.9}), ("two_opt", {"prob": "2/n"}), pop_size=10, n_eval=40)


def test_spea2_binary_integer_and_permutation():
    _run_spea2(BinaryKnapsackProblem(n_var=8), ("hux", {"prob": 0.9}), ("segment_inversion", {"prob": "1/n"}))
    _run_spea2(
        IntegerResourceAllocationProblem(n_var=6, max_per_task=4),
        ("sbx", {"prob": 0.9, "eta": 20.0}),
        ("gaussian", {"prob": "1/n", "sigma": 1.0}),
    )
    _run_spea2(TSPProblem(n_cities=7), ("aex", {"prob": 0.9}), ("two_opt", {"prob": "2/n"}), pop_size=10, n_eval=40)


def test_smpso_mixed_mutation_path():
    pop_size = 14
    cfg = (SMPSOConfig.builder().pop_size(pop_size).archive_size(pop_size).mutation("mixed", prob="1/n")).build()
    algo = SMPSO(cfg.to_dict(), kernel=NumPyKernel())
    problem = MixedDesignProblem(n_var=9)
    res = algo.run(problem, termination=("max_evaluations", pop_size * 2), seed=3)
    assert "F" in res and res["F"].shape[0] > 0
    assert np.isfinite(res["F"]).all()


_ADVERTISED_PROBLEMS = {"binary": "zdt5", "integer": "int_alloc", "permutation": "tsp6", "mixed": "mixed_design"}
_DISCRETE_ALGORITHMS = ("agemoea", "ibea", "moead", "nsgaii", "nsgaiii", "rvea", "smsemoa", "spea2")


@pytest.mark.parametrize("encoding", sorted(_ADVERTISED_PROBLEMS))
@pytest.mark.parametrize("algorithm", _DISCRETE_ALGORITHMS)
def test_advertised_operator_methods_work_for_every_algorithm(algorithm: str, encoding: str) -> None:
    """Every name that available_*_methods advertises runs with every algorithm of the encoding (VR-109)."""
    import dataclasses

    import vamos
    from vamos.algorithms import available_crossover_methods, available_mutation_methods
    from vamos.engine.algorithm.config.defaults import build_default_algorithm_config
    from vamos.foundation.problem.registry import make_problem_selection

    problem = make_problem_selection(_ADVERTISED_PROBLEMS[encoding]).instantiate()
    pop_size = None if algorithm in ("moead", "nsgaiii", "rvea") else 12
    failures = []
    for field, methods in (("crossover", available_crossover_methods(encoding)), ("mutation", available_mutation_methods(encoding))):
        for method in methods:
            config = build_default_algorithm_config(
                algorithm, pop_size=pop_size, n_var=problem.n_var, n_obj=problem.n_obj, encoding=encoding
            )
            _, params = getattr(config, field)
            try:
                vamos.optimize(
                    problem,
                    algorithm=algorithm,
                    algorithm_config=dataclasses.replace(config, **{field: (method, dict(params))}),
                    max_evaluations=150,
                    seed=1,
                )
            except ValueError as exc:
                failures.append(f"{field}={method}: {exc}")
    assert failures == []
