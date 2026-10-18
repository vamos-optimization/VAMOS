import numpy as np
import pytest

pytest.importorskip("numba")

from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.kernel.numba_ops import tournament_winners_numba


@pytest.mark.numba
def test_tournament_winners_numba_orders_by_rank_crowding_then_key():
    ranks = np.array([1, 0, 0, 0], dtype=np.int64)
    crowding = np.array([9.0, 1.0, np.nan, 1.0])
    candidates = np.array([[0, 1, 2], [2, 3, 1]], dtype=np.int64)
    tie_break = np.array([[0.1, 0.2, 0.3], [0.5, 0.9, 0.4]])

    winners = tournament_winners_numba(ranks, crowding, candidates, tie_break)

    # Row 0: rank 0 beats rank 1; NaN crowding counts as -inf.
    # Row 1: candidates 3 and 1 tie on rank and crowding; the smaller key wins.
    np.testing.assert_array_equal(winners, [1, 1])


@pytest.mark.numba
def test_numba_backend_integration_with_explicit_generator():
    kernel = NumbaKernel()
    params = {"prob": 0.5, "eta": 20.0}

    X = np.random.default_rng(4).random((10, 2))
    kernel.polynomial_mutation(X, params, np.random.default_rng(0), 0.0, 1.0)
    assert np.all(np.isfinite(X))
    assert np.all((X >= 0.0) & (X <= 1.0))

    parents = np.random.default_rng(5).random((20, 2))
    offspring = kernel.sbx_crossover(parents, params, np.random.default_rng(1), 0.0, 1.0)
    assert offspring.shape == parents.shape
    assert np.all(np.isfinite(offspring))
    assert np.all((offspring >= 0.0) & (offspring <= 1.0))


@pytest.mark.numba
def test_numba_backend_sbx_keeps_odd_parent_padding_contract():
    kernel = NumbaKernel()
    parents = np.random.default_rng(0).random((5, 3))

    offspring = kernel.sbx_crossover(parents, {"prob": 1.0, "eta": 20.0}, np.random.default_rng(2), 0.0, 1.0)

    # Odd parent counts duplicate the last parent, exactly like NumPyKernel.
    assert offspring.shape == (6, 3)
