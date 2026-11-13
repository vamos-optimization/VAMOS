from __future__ import annotations

import numpy as np
import pytest

from vamos.engine.algorithm.config import MOEADConfig
from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.algorithm.moead.helpers import (
    ZERO_WEIGHT_EPS,
    resolve_aggregation_spec,
    tchebycheff,
    update_neighborhood,
    update_neighborhood_batch,
)
from vamos.engine.algorithm.moead.state import MOEADState
from vamos.foundation.constraints.utils import compute_violation
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _weights() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.8, 0.2],
        ],
        dtype=float,
    )


def _build_state(*, constrained: bool) -> MOEADState:
    weights = _weights()
    weights_safe = np.where(weights == 0.0, ZERO_WEIGHT_EPS, weights)
    norms = np.linalg.norm(weights, axis=1)
    weights_unit = weights / norms[:, None]
    agg_id, theta, rho = resolve_aggregation_spec("tchebycheff", {})
    G = None
    cv = None
    if constrained:
        G = np.array([[0.4], [0.1], [-0.2], [0.3]], dtype=float)
        cv = compute_violation(G)

    return MOEADState(
        X=np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float),
        F=np.array(
            [
                [4.0, 1.0],
                [1.0, 4.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ],
            dtype=float,
        ),
        G=G,
        rng=np.random.default_rng(0),
        pop_size=4,
        offspring_size=2,
        constraint_mode="feasibility" if constrained else "none",
        weights=weights,
        weights_safe=weights_safe,
        weights_unit=weights_unit,
        neighbors=np.array(
            [
                [0, 2],
                [1, 2],
                [2, 0],
                [3, 2],
            ],
            dtype=int,
        ),
        ideal=np.array([1.0, 1.0], dtype=float),
        aggregator=tchebycheff,
        aggregation_id=agg_id,
        aggregation_theta=theta,
        aggregation_rho=rho,
        replace_limit=2,
        batch_size=2,
        cv=cv,
    )


def _run_sequential(
    state: MOEADState,
    children: np.ndarray,
    children_f: np.ndarray,
    children_g: np.ndarray | None,
    children_cv: np.ndarray | None,
    candidate_orders: np.ndarray,
    candidate_lengths: np.ndarray,
) -> MOEADState:
    for pos in range(children.shape[0]):
        update_neighborhood(
            st=state,
            idx=pos,
            child=children[pos],
            child_f=children_f[pos],
            child_g=None if children_g is None else children_g[pos],
            cv_penalty=0.0 if children_cv is None else float(children_cv[pos]),
            candidate_order=candidate_orders[pos, : int(candidate_lengths[pos])],
        )
    return state


def test_moead_batch_update_matches_single_updates_unconstrained():
    children = np.array([[9.0], [10.0]], dtype=float)
    children_f = np.array([[1.5, 1.5], [0.5, 5.0]], dtype=float)
    candidate_orders = np.array(
        [
            [0, 2, 3, 1],
            [1, 3, 2, 0],
        ],
        dtype=int,
    )
    candidate_lengths = np.array([4, 4], dtype=np.int64)

    sequential = _run_sequential(
        _build_state(constrained=False),
        children,
        children_f,
        None,
        None,
        candidate_orders,
        candidate_lengths,
    )
    batched = _build_state(constrained=False)
    update_neighborhood_batch(
        st=batched,
        children=children,
        children_f=children_f,
        children_g=None,
        children_cv=None,
        candidate_orders=candidate_orders,
        candidate_lengths=candidate_lengths,
    )

    np.testing.assert_allclose(batched.X, sequential.X)
    np.testing.assert_allclose(batched.F, sequential.F)


def test_moead_batch_update_matches_single_updates_with_constraints():
    children = np.array([[9.0], [10.0]], dtype=float)
    children_f = np.array([[1.5, 1.5], [0.5, 5.0]], dtype=float)
    children_g = np.array([[-0.3], [0.05]], dtype=float)
    children_cv = compute_violation(children_g)
    candidate_orders = np.array(
        [
            [0, 2, 3, 1],
            [1, 3, 2, 0],
        ],
        dtype=int,
    )
    candidate_lengths = np.array([4, 4], dtype=np.int64)

    sequential = _run_sequential(
        _build_state(constrained=True),
        children,
        children_f,
        children_g,
        children_cv,
        candidate_orders,
        candidate_lengths,
    )
    batched = _build_state(constrained=True)
    update_neighborhood_batch(
        st=batched,
        children=children,
        children_f=children_f,
        children_g=children_g,
        children_cv=children_cv,
        candidate_orders=candidate_orders,
        candidate_lengths=candidate_lengths,
    )

    np.testing.assert_allclose(batched.X, sequential.X)
    np.testing.assert_allclose(batched.F, sequential.F)
    assert batched.G is not None
    assert sequential.G is not None
    assert batched.cv is not None
    assert sequential.cv is not None
    np.testing.assert_allclose(batched.G, sequential.G)
    np.testing.assert_allclose(batched.cv, sequential.cv)


@pytest.mark.parametrize(("delta", "use_neighbors"), [(1.0, True), (0.0, False)])
def test_moead_ask_parent_pairs_are_distinct_and_match_sampling_pool(delta: float, use_neighbors: bool):
    problem = ZDT1Problem(n_var=8)
    cfg = (
        MOEADConfig.builder()
        .pop_size(20)
        .batch_size(8)
        .neighbor_size(5)
        .delta(delta)
        .replace_limit(2)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .weight_vectors(divisions=19)
        .build()
    )
    algo = MOEAD(cfg.to_dict(), kernel=NumPyKernel())
    algo._initialize_run(problem, ("max_evaluations", 40), seed=3)
    algo.ask()

    state = algo._st
    assert state is not None
    assert state.pending_parent_pairs is not None
    assert state.pending_active_indices is not None
    assert state.pending_use_neighbors is not None

    parent_pairs = state.pending_parent_pairs
    active = state.pending_active_indices
    np.testing.assert_array_equal(state.pending_use_neighbors, np.full(parent_pairs.shape[0], use_neighbors, dtype=bool))
    assert np.all(parent_pairs[:, 0] != parent_pairs[:, 1])

    if use_neighbors:
        for pos, idx in enumerate(active):
            neighborhood = state.neighbors[idx]
            assert int(parent_pairs[pos, 0]) in neighborhood
            assert int(parent_pairs[pos, 1]) in neighborhood
    else:
        assert np.all(parent_pairs >= 0)
        assert np.all(parent_pairs < state.pop_size)


@pytest.mark.numba
def test_moead_numba_reproducible_with_seed_on_batch_path():
    problem = ZDT1Problem(n_var=8)
    cfg = (
        MOEADConfig.builder()
        .pop_size(20)
        .batch_size(4)
        .neighbor_size(6)
        .delta(0.9)
        .replace_limit(3)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .aggregation("pbi", theta=5.0)
        .weight_vectors(divisions=19)
        .build()
    )

    result_a = MOEAD(cfg.to_dict(), kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 52), seed=0)
    result_b = MOEAD(cfg.to_dict(), kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 52), seed=0)

    np.testing.assert_allclose(result_a["population"]["F"], result_b["population"]["F"])
    np.testing.assert_allclose(result_a["F"], result_b["F"])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "use_numba_variation sets a process-wide override in vamos.engine.operators.impl.flags that stays set "
        "after the run and selects the Numba binary, integer and permutation operators of every later run; "
        "passing the switch through the operator configuration removes it"
    ),
)
def test_moead_numba_flag_does_not_outlive_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from vamos.engine.operators.impl import flags

    monkeypatch.delenv("VAMOS_USE_NUMBA_VARIATION", raising=False)
    cfg = (
        MOEADConfig.builder()
        .pop_size(20)
        .batch_size(4)
        .neighbor_size(6)
        .delta(0.9)
        .replace_limit(3)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .aggregation("pbi", theta=5.0)
        .weight_vectors(divisions=19)
        .use_numba_variation(True)
        .build()
    )
    before = flags.should_use_numba_variation()
    try:
        MOEAD(cfg.to_dict(), kernel=NumPyKernel()).run(ZDT1Problem(n_var=8), termination=("max_evaluations", 40), seed=0)
        # The run's configuration must not choose the operators of the runs that follow it.
        assert flags.should_use_numba_variation() == before, "use_numba_variation outlived the run"
    finally:
        flags.set_numba_variation(None)
