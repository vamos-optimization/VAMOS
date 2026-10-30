from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from vamos import StudyResult, optimize
from vamos.engine.algorithm.agemoea import AGEMOEA
from vamos.engine.algorithm.config import (
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
from vamos.engine.algorithm.ibea import IBEA
from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.engine.algorithm.nsgaiii import NSGAIII
from vamos.engine.algorithm.rvea import RVEA
from vamos.engine.algorithm.smpso import SMPSO
from vamos.engine.algorithm.smsemoa import SMSEMOA
from vamos.engine.algorithm.spea2 import SPEA2
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.dtlz import DTLZ2Problem
from vamos.foundation.problem.zdt1 import ZDT1Problem

POP_SIZE = 6


def _nsgaii() -> tuple[Any, int]:
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(POP_SIZE)
        .offspring_size(POP_SIZE)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament")
        .build()
    )
    return NSGAII(cfg.to_dict(), kernel=NumPyKernel()), POP_SIZE


def _moead() -> tuple[Any, int]:
    cfg = (
        MOEADConfig.builder()
        .pop_size(POP_SIZE)
        .neighbor_size(3)
        .delta(0.9)
        .replace_limit(2)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .weight_vectors(divisions=5)
        .build()
    )
    return MOEAD(cfg.to_dict(), kernel=NumPyKernel()), POP_SIZE


def _smsemoa() -> tuple[Any, int]:
    cfg = (
        SMSEMOAConfig.builder()
        .pop_size(POP_SIZE)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("random")
        .reference_point(offset=1.0, adaptive=True)
        .build()
    )
    return SMSEMOA(cfg.to_dict(), kernel=NumPyKernel()), POP_SIZE


def _spea2() -> tuple[Any, int]:
    cfg = (
        SPEA2Config.builder()
        .pop_size(POP_SIZE)
        .archive_size(POP_SIZE)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament")
        .build()
    )
    return SPEA2(cfg.to_dict(), kernel=NumPyKernel()), POP_SIZE


def _ibea() -> tuple[Any, int]:
    cfg = (
        IBEAConfig.builder()
        .pop_size(POP_SIZE)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament")
        .indicator("eps")
        .kappa(0.05)
        .build()
    )
    return IBEA(cfg.to_dict(), kernel=NumPyKernel()), POP_SIZE


def _smpso() -> tuple[Any, int]:
    cfg = SMPSOConfig.default(pop_size=POP_SIZE, n_var=6)
    return SMPSO(cfg.to_dict(), kernel=NumPyKernel()), POP_SIZE


def _nsgaiii() -> tuple[Any, int]:
    cfg = (
        NSGAIIIConfig.builder()
        .pop_size(POP_SIZE)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament")
        .reference_directions(divisions=5)
        .build()
    )
    return NSGAIII(cfg.to_dict(), kernel=NumPyKernel()), POP_SIZE


def _agemoea() -> tuple[Any, int]:
    cfg = AGEMOEAConfig.builder().pop_size(POP_SIZE).crossover("sbx", prob=0.9, eta=20.0).mutation("polynomial", prob=0.1, eta=20.0).build()
    return AGEMOEA(cfg.to_dict(), kernel=NumPyKernel()), POP_SIZE


def _rvea() -> tuple[Any, int]:
    cfg = (
        RVEAConfig.builder()
        .pop_size(POP_SIZE)
        .n_partitions(5)
        .alpha(2.0)
        .adapt_freq(0.1)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("polynomial", prob=0.1, eta=20.0)
        .build()
    )
    return RVEA(cfg.to_dict(), kernel=NumPyKernel()), POP_SIZE


ALGORITHM_FACTORIES: list[tuple[str, Callable[[], tuple[Any, int]]]] = [
    ("nsgaii", _nsgaii),
    ("moead", _moead),
    ("smsemoa", _smsemoa),
    ("spea2", _spea2),
    ("ibea", _ibea),
    ("smpso", _smpso),
    ("nsgaiii", _nsgaiii),
    ("agemoea", _agemoea),
    ("rvea", _rvea),
]


@pytest.mark.parametrize("budget_offset", [0, 1, POP_SIZE - 1])
@pytest.mark.parametrize(
    ("algorithm_name", "factory"),
    ALGORITHM_FACTORIES,
    ids=[name for name, _ in ALGORITHM_FACTORIES],
)
def test_algorithms_enforce_exact_max_evaluations(
    algorithm_name: str,
    factory: Callable[[], tuple[Any, int]],
    budget_offset: int,
) -> None:
    algorithm, pop_size = factory()
    problem = ZDT1Problem(n_var=6)
    budget = pop_size + budget_offset

    result = algorithm.run(problem, termination=("max_evaluations", budget), seed=11)

    assert result["evaluations"] == budget, algorithm_name
    assert "n_eval" not in result


@pytest.mark.parametrize(
    ("algorithm_name", "factory"),
    ALGORITHM_FACTORIES,
    ids=[name for name, _ in ALGORITHM_FACTORIES],
)
def test_algorithms_reject_budget_smaller_than_population(
    algorithm_name: str,
    factory: Callable[[], tuple[Any, int]],
) -> None:
    algorithm, pop_size = factory()
    problem = ZDT1Problem(n_var=6)

    with pytest.raises(ValueError, match="max_evaluations.*pop_size"):
        algorithm.run(problem, termination=("max_evaluations", pop_size - 1), seed=11)


def test_ask_caps_to_remaining_budget_and_rejects_exhausted_budget() -> None:
    algorithm, pop_size = _agemoea()
    problem = ZDT1Problem(n_var=6)
    algorithm.initialize(problem, ("max_evaluations", pop_size + 1), seed=0)

    X = algorithm.ask()
    assert X.shape[0] == 1
    out: dict[str, np.ndarray] = {"F": np.empty((X.shape[0], problem.n_obj), dtype=float)}
    problem.evaluate(X, out)
    algorithm.tell(out["F"])

    with pytest.raises(RuntimeError, match="no remaining evaluation budget"):
        algorithm.ask()


def test_optimize_rejects_budget_smaller_than_population() -> None:
    with pytest.raises(ConfigurationError, match="max_evaluations must be >= pop_size"):
        optimize(ZDT1Problem(n_var=4), algorithm="nsgaii", max_evaluations=3, pop_size=4)


def test_optimize_smpso_default_configuration_runs() -> None:
    result = optimize(
        ZDT1Problem(n_var=6),
        algorithm="smpso",
        max_evaluations=12,
        pop_size=6,
        seed=1,
        engine="numpy",
    )

    assert result.data["evaluations"] == 12
    assert "n_eval" not in result.data


def test_nsgaiii_and_rvea_defaults_choose_compatible_reference_counts() -> None:
    nsgaiii = optimize(
        DTLZ2Problem(n_var=7, n_obj=3),
        algorithm="nsgaiii",
        max_evaluations=92,
        seed=1,
        engine="numpy",
    )
    rvea = optimize(
        DTLZ2Problem(n_var=7, n_obj=3),
        algorithm="rvea",
        max_evaluations=92,
        seed=1,
        engine="numpy",
    )

    assert nsgaiii.data["evaluations"] == 92
    assert nsgaiii.meta["run_artifact_resolved_spec"]["population"]["initial_size"] == 91
    assert rvea.data["evaluations"] == 92
    assert rvea.meta["run_artifact_resolved_spec"]["population"]["initial_size"] == 91


def test_nsgaiii_accepts_n_at_least_h_and_rvea_rejects_incompatible_pop_size() -> None:
    problem = DTLZ2Problem(n_var=7, n_obj=3)

    # NSGA-III keeps its 91 directions (12 divisions) and admits N >= H, as Deb and Jain, jMetal and pymoo do.
    larger = optimize(problem, algorithm="nsgaiii", max_evaluations=100, pop_size=100, seed=1, engine="numpy")
    assert larger.data["evaluations"] == 100
    assert larger.data["reference_directions"].shape == (91, 3)
    assert larger.meta["run_artifact_resolved_spec"]["population"]["initial_size"] == 100

    with pytest.raises(ValueError, match="pop_size"):
        optimize(
            problem,
            algorithm="nsgaiii",
            max_evaluations=100,
            pop_size=90,
            seed=1,
            engine="numpy",
        )

    with pytest.raises(ValueError, match="RVEA requires pop_size"):
        optimize(
            problem,
            algorithm="rvea",
            max_evaluations=100,
            pop_size=100,
            seed=1,
            engine="numpy",
        )


def test_study_result_evaluations_metric_for_ref_direction_algorithms() -> None:
    result = optimize(
        DTLZ2Problem(n_var=7, n_obj=3),
        algorithm="nsgaiii",
        max_evaluations=92,
        seed=[1, 2],
        engine="numpy",
    )

    assert isinstance(result, StudyResult)
    assert np.all(result.metric_values("evaluations") == 92)
    assert result.mean("evaluations") == pytest.approx(92.0)
    assert result.std("evaluations") == pytest.approx(0.0)
    assert result.best_run("evaluations").meta["seed"] == 1
