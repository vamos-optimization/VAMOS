from __future__ import annotations

import numpy as np

from vamos.engine.algorithm.config import SMSEMOAConfig
from vamos.engine.algorithm.smsemoa import SMSEMOA
from vamos.engine.algorithm.smsemoa.helpers import refresh_selection_cache, survival_selection
from vamos.engine.algorithm.smsemoa.state import SMSEMOAState
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


class SimpleConstrainedProblem:
    def __init__(self) -> None:
        self.n_var = 2
        self.n_obj = 2
        self.n_constraints = 1
        self.xl = np.array([0.0, 0.0])
        self.xu = np.array([1.0, 1.0])

    def evaluate(self, X, out) -> None:  # noqa: ANN001 - protocol-style test double
        out["F"] = np.column_stack((X[:, 0], 1.0 - X[:, 0] + X[:, 1]))
        out["G"] = np.column_stack((0.5 - X[:, 0],))


class CountingKernel(NumPyKernel):
    def __init__(self) -> None:
        super().__init__()
        self.rank_calls = 0
        self.hv_contrib_calls = 0

    def nsga2_ranking(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.rank_calls += 1
        return super().nsga2_ranking(F)

    def hypervolume_contributions(self, points: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
        self.hv_contrib_calls += 1
        return super().hypervolume_contributions(points, reference_point)


class HookKernel(CountingKernel):
    def __init__(self, contributions: np.ndarray) -> None:
        super().__init__()
        self._contributions = np.asarray(contributions, dtype=float)
        self.hook_used = False

    def hypervolume_contributions(self, points: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
        del reference_point
        self.hook_used = True
        return self._contributions[: points.shape[0]].copy()


def _build_state(
    F: np.ndarray,
    *,
    G: np.ndarray | None = None,
    selection_method: str = "tournament",
) -> SMSEMOAState:
    F_arr = np.asarray(F, dtype=float)
    G_arr = None if G is None else np.asarray(G, dtype=float)
    return SMSEMOAState(
        X=np.arange(F_arr.shape[0], dtype=float).reshape(-1, 1),
        F=F_arr.copy(),
        G=None if G_arr is None else G_arr.copy(),
        rng=np.random.default_rng(0),
        pop_size=F_arr.shape[0],
        offspring_size=1,
        constraint_mode="none" if G_arr is None else "feasibility",
        ref_point=F_arr.max(axis=0) + 1.0,
        ref_offset=1.0,
        ref_adaptive=False,
        pressure=2,
        selection_method=selection_method,
    )


def test_smsemoa_initialization_defaults_to_feasibility():
    problem = SimpleConstrainedProblem()
    cfg = (
        SMSEMOAConfig.builder()
        .pop_size(8)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("random")
        .build()
    )

    algo = SMSEMOA(cfg.to_dict(), kernel=NumPyKernel())
    algo.initialize(problem, termination=("max_evaluations", 10), seed=0)

    assert algo.state is not None
    assert algo.state.constraint_mode == "feasibility"
    assert algo.state.G is not None


def test_smsemoa_survival_prefers_lower_constraint_violation():
    kernel = NumPyKernel()
    state = _build_state(
        np.array(
            [
                [5.0, 5.0],
                [6.0, 6.0],
                [0.0, 0.0],
            ]
        ),
        G=np.array([[-0.1], [-0.2], [0.5]]),
    )
    refresh_selection_cache(state, kernel)

    survival_selection(
        state,
        X_child=np.array([[9.0]]),
        F_child=np.array([[10.0, 10.0]]),
        G_child=np.array([[0.2]]),
        kernel=kernel,
    )

    assert 2.0 not in state.X[:, 0]
    assert 9.0 in state.X[:, 0]


def test_smsemoa_tournament_ask_and_tell_reuse_cached_ranking():
    problem = ZDT1Problem(n_var=6)
    kernel = CountingKernel()
    cfg = (
        SMSEMOAConfig.builder()
        .pop_size(10)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    )
    algo = SMSEMOA(cfg.to_dict(), kernel=kernel)
    algo.initialize(problem, termination=("max_evaluations", 14), seed=2)
    rank_calls_after_init = kernel.rank_calls

    X_child = algo.ask()
    assert kernel.rank_calls == rank_calls_after_init

    out: dict[str, np.ndarray] = {"F": np.empty((X_child.shape[0], problem.n_obj), dtype=float)}
    problem.evaluate(X_child, out)
    algo.tell({"F": out["F"], "G": out.get("G")})

    assert kernel.rank_calls == rank_calls_after_init


def test_smsemoa_survival_uses_kernel_hv_contribution_hook():
    kernel = HookKernel(np.array([0.0, 1.0, 1.0, 1.0]))
    state = _build_state(
        np.array(
            [
                [0.0, 4.0],
                [1.0, 3.0],
                [3.0, 1.0],
            ]
        )
    )
    refresh_selection_cache(state, kernel)

    survival_selection(
        state,
        X_child=np.array([[4.0]]),
        F_child=np.array([[4.0, 0.0]]),
        G_child=None,
        kernel=kernel,
    )

    assert kernel.hook_used is True
    assert 0.0 not in state.X[:, 0]
    assert 4.0 in state.X[:, 0]


def test_smsemoa_numba_seeded_run_is_repeatable():
    problem = ZDT1Problem(n_var=6)
    cfg = (
        SMSEMOAConfig.builder()
        .pop_size(8)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    )

    result_a = SMSEMOA(cfg.to_dict(), kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 16), seed=0)
    result_b = SMSEMOA(cfg.to_dict(), kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 16), seed=0)

    np.testing.assert_allclose(result_a["population"]["F"], result_b["population"]["F"])
    np.testing.assert_allclose(result_a["F"], result_b["F"])


class _ScaledZDT1:
    """ZDT1 with the second objective multiplied by ``factor``."""

    def __init__(self, factor: float) -> None:
        self._base = ZDT1Problem(n_var=6)
        self.n_var = self._base.n_var
        self.n_obj = self._base.n_obj
        self.xl = self._base.xl
        self.xu = self._base.xu
        self.encoding = "real"
        self._factor = factor

    def evaluate(self, X, out) -> None:  # noqa: ANN001 - protocol-style test double
        self._base.evaluate(X, out)
        out["F"][:, 1] *= self._factor


def _smsemoa_config(vector: list[float] | None = None) -> SMSEMOAConfig:
    builder = (
        SMSEMOAConfig.builder()
        .pop_size(12)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
    )
    if vector is not None:
        builder = builder.reference_point(vector=vector)
    return builder.build()


def test_smsemoa_selection_is_invariant_to_objective_units() -> None:
    cfg = _smsemoa_config()
    runs = {
        factor: SMSEMOA(cfg.to_dict(), kernel=NumPyKernel()).run(_ScaledZDT1(factor), termination=("max_evaluations", 400), seed=3)
        for factor in (1.0, 2.0**10, 2.0**-6)
    }
    base = runs[1.0]["population"]["X"]
    for factor, result in runs.items():
        assert result["population"]["X"].tobytes() == base.tobytes(), factor


def test_user_reference_vector_is_kept() -> None:
    cfg = _smsemoa_config(vector=[1.1, 1.1])
    result = SMSEMOA(cfg.to_dict(), kernel=NumPyKernel()).run(ZDT1Problem(n_var=6), termination=("max_evaluations", 400), seed=3)

    np.testing.assert_array_equal(result["reference_point"], [1.1, 1.1])
    assert np.all(result["population"]["F"] <= 1.1, axis=1).any()
