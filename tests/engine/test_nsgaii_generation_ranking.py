"""NSGA-II reuses the generation-end ranking for the next selection without changing results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import vamos
from vamos.algorithms import NSGAIIConfig
from vamos.engine.algorithm.nsgaii.state import NSGAIIState, compute_selection_metrics
from vamos.foundation.kernel.numpy_backend import NumPyKernel


class _CountingKernel(NumPyKernel):
    def __init__(self) -> None:
        super().__init__()
        self.ranked_shapes: list[tuple[int, ...]] = []

    def nsga2_ranking(self, F: Any) -> Any:
        self.ranked_shapes.append(tuple(np.shape(F)))
        return super().nsga2_ranking(F)


def _state_with(F: np.ndarray) -> NSGAIIState:
    return NSGAIIState(
        X=np.zeros((F.shape[0], 2)),
        F=F,
        G=None,
        rng=np.random.default_rng(0),
        variation=None,  # type: ignore[arg-type]
        operator_pool=[],
        variation_workspace=None,  # type: ignore[arg-type]
    )


def test_published_ranking_is_reused_once_for_the_same_population() -> None:
    F = np.random.default_rng(1).random((12, 3))
    st = _state_with(F)
    ranks, crowding = NumPyKernel().nsga2_ranking(F)
    st.publish_generation_ranking(F, ranks, crowding)

    reused = st.take_generation_ranking()

    assert reused is not None
    assert reused[0] is ranks and reused[1] is crowding
    assert st.take_generation_ranking() is None


def test_published_ranking_is_rejected_after_in_place_change_or_replacement() -> None:
    F = np.random.default_rng(2).random((10, 2))
    st = _state_with(F)
    ranks, crowding = NumPyKernel().nsga2_ranking(F)

    st.publish_generation_ranking(F, ranks, crowding)
    st.F[3, 0] = -1.0
    assert st.take_generation_ranking() is None

    st.publish_generation_ranking(st.F, ranks, crowding)
    replacement = st.F.copy()
    replacement[0, 1] += 1.0
    st.F = replacement
    assert st.take_generation_ranking() is None

    st.publish_generation_ranking(F.copy(), ranks, crowding)  # not the state's population
    assert st.take_generation_ranking() is None


def test_constrained_selection_ranks_only_the_feasible_rows() -> None:
    rng = np.random.default_rng(3)
    F = rng.random((16, 2))
    G = rng.normal(size=(16, 1))
    kernel = _CountingKernel()

    compute_selection_metrics(kernel, F, G, "feasibility")

    n_feasible = int(np.count_nonzero(G[:, 0] <= 0.0))
    assert kernel.ranked_shapes == [(n_feasible, 2)]


@pytest.mark.parametrize("engine", ["numpy", "numba"])
@pytest.mark.parametrize("problem,kwargs", [("zdt1", {}), ("dtlz2", {"n_obj": 3}), ("c1dtlz1", {"n_obj": 3})])
def test_ranking_reuse_does_not_change_the_trajectory(
    monkeypatch: pytest.MonkeyPatch, engine: str, problem: str, kwargs: dict[str, int]
) -> None:
    if engine == "numba":
        pytest.importorskip("numba")

    def run() -> tuple[np.ndarray, np.ndarray]:
        result = vamos.optimize(problem, algorithm="nsgaii", engine=engine, seed=4, pop_size=24, max_evaluations=480, **kwargs)
        return np.asarray(result.X), np.asarray(result.F)

    X_reused, F_reused = run()
    monkeypatch.setattr(NSGAIIState, "take_generation_ranking", lambda self: None)
    X_fresh, F_fresh = run()

    np.testing.assert_array_equal(X_reused, X_fresh)
    np.testing.assert_array_equal(F_reused, F_fresh)


def test_generational_nsgaii_ranks_the_population_once_less_per_generation() -> None:
    from vamos.engine.algorithm.nsgaii import NSGAII
    from vamos.foundation.problem.registry import make_problem_selection

    problem = make_problem_selection("zdt1", n_var=6).instantiate()
    config = NSGAIIConfig.default(pop_size=10, n_var=6).to_dict()
    kernel = _CountingKernel()
    NSGAII(config, kernel=kernel).run(problem, termination=("max_evaluations", 60), seed=0)

    # 5 generations: the initial and each end-of-generation ranking feed the next
    # selection directly, plus one ranking for the final non-dominated result.
    # Without reuse every ask() ranked the population again (12 calls).
    assert kernel.ranked_shapes == [(10, 2)] * 7
