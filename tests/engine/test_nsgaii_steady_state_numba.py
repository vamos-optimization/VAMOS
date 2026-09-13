from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vamos.engine.algorithm.nsgaii.ask_tell as ask_tell_module
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.engine.algorithm.nsgaii.helpers import fronts_from_ranks
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


class Identity2DProblem:
    def __init__(self, n_var: int = 3) -> None:
        self.n_var = n_var
        self.n_obj = 2
        self.xl = np.zeros(n_var, dtype=float)
        self.xu = np.ones(n_var, dtype=float)
        self.encoding = "real"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        out["F"] = np.asarray(X[:, :2], dtype=float).copy()


class Identity3DProblem:
    def __init__(self, n_var: int = 4) -> None:
        self.n_var = n_var
        self.n_obj = 3
        self.xl = np.zeros(n_var, dtype=float)
        self.xu = np.ones(n_var, dtype=float)
        self.encoding = "real"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        out["F"] = np.asarray(X[:, :3], dtype=float).copy()


def _steady_state_cfg(pop_size: int) -> dict[str, object]:
    return (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .steady_state()
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
        .to_dict()
    )


@pytest.mark.numba
def test_nsgaii_steady_state_numba_uses_full_survival_for_biobjective(monkeypatch: pytest.MonkeyPatch) -> None:
    algo = NSGAII(_steady_state_cfg(10), kernel=NumbaKernel())
    problem = Identity2DProblem()
    algo._initialize_run(problem, termination=("max_evaluations", 20), seed=0, eval_strategy=None, live_viz=None)

    incremental_calls = 0
    original_incremental = ask_tell_module.incremental_insert_fronts

    def _spy_incremental(*args: object, **kwargs: object) -> object:
        nonlocal incremental_calls
        incremental_calls += 1
        return original_incremental(*args, **kwargs)

    survival_calls = 0
    original_survival = algo.kernel.nsga2_survival

    def _spy_survival(*args: object, **kwargs: object) -> object:
        nonlocal survival_calls
        survival_calls += 1
        return original_survival(*args, **kwargs)

    monkeypatch.setattr(ask_tell_module, "incremental_insert_fronts", _spy_incremental)
    monkeypatch.setattr(algo.kernel, "nsga2_survival", _spy_survival)

    _ = algo.ask()
    algo.tell(SimpleNamespace(F=np.array([[-1.0, -1.0]], dtype=float), G=None))

    assert survival_calls == 1
    assert incremental_calls == 0


@pytest.mark.numba
def test_nsgaii_steady_state_numba_incremental_matches_full_survival_for_three_objectives() -> None:
    cfg = _steady_state_cfg(12)
    problem = Identity3DProblem()

    algo_inc = NSGAII(dict(cfg), kernel=NumbaKernel())
    algo_full = NSGAII(dict(cfg), kernel=NumbaKernel())
    algo_inc._initialize_run(problem, termination=("max_evaluations", 24), seed=7, eval_strategy=None, live_viz=None)
    algo_full._initialize_run(problem, termination=("max_evaluations", 24), seed=7, eval_strategy=None, live_viz=None)

    X_off = algo_inc.ask()
    eval_result = SimpleNamespace(F=np.array([[-1.0, -1.0, -1.0]], dtype=float), G=None)

    st_full = algo_full._st
    assert st_full is not None
    st_full.pending_offspring = X_off.copy()
    st_full.incremental_enabled = False

    algo_inc.tell(eval_result)
    algo_full.tell(eval_result)

    st_inc = algo_inc._st
    assert st_inc is not None
    assert st_full is not None

    np.testing.assert_allclose(st_inc.X, st_full.X)
    np.testing.assert_allclose(st_inc.F, st_full.F)

    ranks_full, crowding_full = algo_full.kernel.nsga2_ranking(st_full.F)
    np.testing.assert_array_equal(st_inc.ranks, ranks_full)
    np.testing.assert_allclose(st_inc.crowding, crowding_full, equal_nan=True)
    assert st_inc.fronts == fronts_from_ranks(ranks_full)


@pytest.mark.numba
def test_nsgaii_steady_state_numba_reproducible_with_seed() -> None:
    cfg = _steady_state_cfg(12)
    problem = ZDT1Problem(n_var=8)

    res1 = NSGAII(dict(cfg), kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 24), seed=42)
    res2 = NSGAII(dict(cfg), kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 24), seed=42)

    np.testing.assert_array_equal(res1["population"]["F"], res2["population"]["F"])
    np.testing.assert_array_equal(res1["population"]["X"], res2["population"]["X"])
