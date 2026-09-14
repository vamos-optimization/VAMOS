from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.foundation.kernel.numpy_backend import NumPyKernel


class Identity2DProblem:
    def __init__(self, n_var: int = 3) -> None:
        self.n_var = n_var
        self.n_obj = 2
        self.xl = np.zeros(n_var, dtype=float)
        self.xu = np.ones(n_var, dtype=float)
        self.encoding = "real"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        out["F"] = np.asarray(X[:, :2], dtype=float).copy()


def _builder(pop_size: int = 12):
    return (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
    )


def test_steady_state_builder_resolves_one_offspring_and_replacement() -> None:
    cfg = _builder().steady_state().build()

    assert cfg.steady_state is True
    assert cfg.offspring_size == 1
    assert cfg.replacement_size == 1
    assert cfg.to_dict()["offspring_size"] == 1
    assert cfg.to_dict()["replacement_size"] == 1


def test_steady_state_from_dict_resolves_legacy_missing_sizes() -> None:
    payload = NSGAIIConfig.default(pop_size=12, n_var=3).to_dict()
    payload["steady_state"] = True
    payload["offspring_size"] = None
    payload["replacement_size"] = None

    cfg = NSGAIIConfig.from_dict(payload)

    assert cfg.steady_state is True
    assert cfg.offspring_size == 1
    assert cfg.replacement_size == 1


@pytest.mark.parametrize(
    ("method", "message"),
    [
        (lambda builder: builder.offspring_size(2), "offspring_size=1"),
        (lambda builder: builder.replacement_size(2), "replacement_size=1"),
    ],
)
def test_steady_state_rejects_conflicting_explicit_sizes(method, message: str) -> None:
    builder = method(_builder().steady_state())
    with pytest.raises(ValueError, match=message):
        builder.build()


def test_steady_state_flag_drives_one_offspring_runtime_from_raw_mapping() -> None:
    raw = NSGAIIConfig.default(pop_size=12, n_var=3).to_dict()
    raw["steady_state"] = True
    raw["offspring_size"] = None
    raw["replacement_size"] = None

    algo = NSGAII(raw, kernel=NumPyKernel())
    problem = Identity2DProblem()
    algo._initialize_run(problem, termination=("max_evaluations", 14), seed=3, eval_strategy=None, live_viz=None)

    state = algo._st
    assert state is not None
    assert state.offspring_size == 1
    assert state.replacement_size == 1
    assert state.incremental_mode is True
    assert state.incremental_enabled is True

    before = state.n_eval
    offspring = algo.ask()
    assert offspring.shape == (1, problem.n_var)
    algo.tell(SimpleNamespace(F=np.asarray(offspring[:, :2], dtype=float), G=None))
    assert state.n_eval == before + 1
    assert state.X.shape[0] == 12
    assert state.F.shape[0] == 12


def test_steady_state_run_honors_exact_budget() -> None:
    cfg = _builder().steady_state().build()
    result = NSGAII(cfg.to_dict(), kernel=NumPyKernel()).run(
        Identity2DProblem(),
        termination=("max_evaluations", 17),
        seed=11,
    )

    assert result["evaluations"] == 17
    assert result["population"]["X"].shape == (12, 3)
    assert result["population"]["F"].shape == (12, 2)


def test_offspring_size_one_remains_backward_compatible_without_flag() -> None:
    cfg = _builder().offspring_size(1).build()
    assert cfg.steady_state is False
    assert cfg.offspring_size == 1

    algo = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    problem = Identity2DProblem()
    algo._initialize_run(problem, termination=("max_evaluations", 13), seed=5, eval_strategy=None, live_viz=None)

    offspring = algo.ask()
    assert offspring.shape == (1, problem.n_var)
