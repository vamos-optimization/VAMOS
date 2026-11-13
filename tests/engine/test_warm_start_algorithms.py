from typing import Any

import numpy as np
import pytest

import vamos
from vamos.engine.algorithm.config import MOEADConfig, NSGAIIConfig
from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _nsgaii_cfg(pop_size: int) -> NSGAIIConfig:
    return (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    )


def _moead_cfg(pop_size: int) -> MOEADConfig:
    return (
        MOEADConfig.builder()
        .pop_size(pop_size)
        .batch_size(1)
        .neighbor_size(max(2, min(4, pop_size)))
        .delta(0.9)
        .replace_limit(2)
        .crossover("de", cr=1.0, f=0.5)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .weight_vectors(divisions=pop_size - 1)
        .build()
    )


def test_nsgaii_warm_start_matches_full_run() -> None:
    pop_size = 6
    budget_stage1 = pop_size * 2
    budget_full = pop_size * 3

    cfg = _nsgaii_cfg(pop_size)
    problem = ZDT1Problem(n_var=4)

    full = NSGAII(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("max_evaluations", budget_full),
        seed=42,
    )

    stage1 = NSGAII(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("max_evaluations", budget_stage1),
        seed=42,
    )
    checkpoint = stage1["checkpoint"]

    resumed = NSGAII(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("max_evaluations", budget_full),
        seed=123,
        checkpoint=checkpoint,
    )

    assert resumed["evaluations"] == budget_full
    assert np.allclose(full["population"]["F"], resumed["population"]["F"])
    assert np.allclose(full["population"]["X"], resumed["population"]["X"])


def test_moead_warm_start_matches_full_run() -> None:
    pop_size = 6
    budget_stage1 = pop_size + 2
    budget_full = pop_size + 4

    cfg = _moead_cfg(pop_size)
    problem = ZDT1Problem(n_var=4)

    full = MOEAD(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("max_evaluations", budget_full),
        seed=7,
    )

    stage1 = MOEAD(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("max_evaluations", budget_stage1),
        seed=7,
    )
    checkpoint = stage1["checkpoint"]

    resumed = MOEAD(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("max_evaluations", budget_full),
        seed=99,
        checkpoint=checkpoint,
    )

    assert resumed["evaluations"] == budget_full
    assert np.allclose(full["population"]["F"], resumed["population"]["F"])
    assert np.allclose(full["population"]["X"], resumed["population"]["X"])


def _damaged(checkpoint: dict[str, Any], damage: str) -> dict[str, Any]:
    damaged = dict(checkpoint)
    if damage == "rows":
        damaged["X"] = np.asarray(checkpoint["X"])[:3]
        damaged["F"] = np.asarray(checkpoint["F"])[:3]
    elif damage == "missing_F":
        del damaged["F"]
    else:
        damaged["rng_state"] = {"bit_generator": "NotAGenerator", "state": {}}
    return damaged


@pytest.mark.parametrize("algorithm", ["nsgaii", "moead"])
@pytest.mark.parametrize("damage", ["rows", "missing_F", "rng"])
def test_invalid_checkpoint_raises_configuration_error(algorithm: str, damage: str) -> None:
    stage = vamos.optimize("zdt1", algorithm=algorithm, engine="numpy", seed=3, max_evaluations=300)
    checkpoint = _damaged(stage.data["checkpoint"], damage)

    with pytest.raises(ConfigurationError, match="checkpoint"):
        vamos.optimize("zdt1", algorithm=algorithm, engine="numpy", seed=3, max_evaluations=600, checkpoint=checkpoint)


@pytest.mark.parametrize("algorithm", ["nsgaii", "moead"])
def test_constrained_checkpoint_without_constraint_values_raises(algorithm: str) -> None:
    stage = vamos.optimize("mw1", algorithm=algorithm, engine="numpy", seed=3, max_evaluations=300)
    checkpoint = dict(stage.data["checkpoint"])
    assert checkpoint["G"] is not None
    checkpoint["G"] = None

    with pytest.raises(ConfigurationError, match="constraint values G"):
        vamos.optimize("mw1", algorithm=algorithm, engine="numpy", seed=3, max_evaluations=600, checkpoint=checkpoint)


@pytest.mark.parametrize(("key", "value"), [("subproblem_cursor", 10**6), ("subproblem_order", [0, 0, 1])])
def test_moead_checkpoint_with_malformed_subproblem_state_raises(key: str, value: Any) -> None:
    stage = vamos.optimize("zdt1", algorithm="moead", engine="numpy", seed=3, max_evaluations=300)
    checkpoint = dict(stage.data["checkpoint"])
    checkpoint["extra"] = {**checkpoint["extra"], key: value}

    with pytest.raises(ConfigurationError, match=key):
        vamos.optimize("zdt1", algorithm="moead", engine="numpy", seed=3, max_evaluations=600, checkpoint=checkpoint)


def test_moead_default_continuation_matches_uninterrupted() -> None:
    full = vamos.optimize("zdt1", algorithm="moead", engine="numpy", seed=5, max_evaluations=2000)
    stage = vamos.optimize("zdt1", algorithm="moead", engine="numpy", seed=5, max_evaluations=1000)
    resumed = vamos.optimize("zdt1", algorithm="moead", engine="numpy", seed=5, max_evaluations=2000, checkpoint=stage.data["checkpoint"])

    assert resumed.data["evaluations"] == 2000
    assert resumed.data["population"]["F"].tobytes() == full.data["population"]["F"].tobytes()
    assert resumed.data["population"]["X"].tobytes() == full.data["population"]["X"].tobytes()


@pytest.mark.parametrize("algorithm", ["nsgaii", "moead"])
def test_resume_with_budget_below_checkpoint_fails(algorithm: str) -> None:
    stage = vamos.optimize("zdt1", algorithm=algorithm, engine="numpy", seed=3, max_evaluations=300)
    checkpoint = stage.data["checkpoint"]

    # max_evaluations is the total of the resumed run; VAMOS 1.1.0 returned the checkpoint's 300 evaluations silently.
    with pytest.raises(ConfigurationError, match="300"):
        vamos.optimize("zdt1", algorithm=algorithm, engine="numpy", seed=3, max_evaluations=200, checkpoint=checkpoint)

    resumed = vamos.optimize("zdt1", algorithm=algorithm, engine="numpy", seed=3, max_evaluations=300, checkpoint=checkpoint)
    assert resumed.data["evaluations"] == 300
