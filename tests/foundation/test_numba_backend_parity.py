from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from vamos.foundation.kernel._numba_selection import binary_tournament_winners_numba
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.kernel.numpy_backend import NumPyKernel


@pytest.mark.numba
def test_numba_backend_matches_numpy_on_degenerate_front_crowding():
    F = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.54737175, 0.70416167, 1.52989318, -0.88847629, 2.07505589],
            [1.01036168, -1.57342212, -1.89154442, 1.58056301, 1.29870262],
            [0.78833977, -0.67652331, -0.65683177, 1.46598876, -0.32424641],
            [0.33844874, 0.36929196, 1.41948101, -0.30174454, 0.80555704],
            [1.42785805, -1.29988395, -0.03991281, -0.92532477, -0.82596787],
            [1.74282178, -1.42917564, -0.23688857, -0.58204089, 1.55768685],
            [-1.64796965, 0.05891919, 0.62568412, -0.52170192, 0.38113437],
            [-0.35338667, 0.62066769, -0.30376236, 1.17848572, 1.59986942],
            [-1.7155635, 0.00552614, 0.7059121, 1.66917642, 0.66561383],
            [-0.72078765, -0.36228572, -1.17058539, 1.79848755, -0.44581703],
            [0.60799129, -0.0444921, 0.39808814, -0.78944566, 0.37821116],
            [0.47108403, 0.36465244, 1.25590835, 1.58497449, 1.20924889],
            [-0.45661502, 0.40353047, -0.84477446, 0.79803708, -0.06214161],
            [0.93440549, 1.02249964, -0.7655916, 0.88829499, -0.80212363],
            [0.45900282, 1.13774903, 1.32234728, 1.98797023, 0.40353076],
            [-0.17026033, 0.59637191, -0.72121649, -0.319329, 1.87837721],
            [-0.72807054, 0.32339287, -0.16975732, -0.56571433, 0.31678848],
            [-0.0892354, -1.62635598, 0.6326408, -0.23282097, 1.78592738],
        ],
        dtype=float,
    )

    numpy_kernel = NumPyKernel()
    numba_kernel = NumbaKernel()

    ranks_np, crowding_np = numpy_kernel.nsga2_ranking(F)
    ranks_nb, crowding_nb = numba_kernel.nsga2_ranking(F)

    np.testing.assert_array_equal(ranks_nb, ranks_np)
    np.testing.assert_allclose(crowding_nb, crowding_np, equal_nan=True)


@pytest.mark.numba
def test_numba_backend_survival_matches_numpy_on_partial_front_ties():
    X = np.arange(6, dtype=float).reshape(3, 2)
    X_off = X + 10.0
    F = np.array(
        [
            [-0.09728388, -0.39201547],
            [1.46001355, -0.07135625],
            [0.060988, 1.30536592],
        ],
        dtype=float,
    )
    F_off = np.array(
        [
            [0.20315227, 0.9794765],
            [-0.64510862, 0.31560114],
            [0.57048443, 0.1320587],
        ],
        dtype=float,
    )

    numpy_kernel = NumPyKernel()
    numba_kernel = NumbaKernel()

    X_np, F_np, sel_np = numpy_kernel.nsga2_survival(X, F, X_off, F_off, 3, return_indices=True)
    X_nb, F_nb, sel_nb = numba_kernel.nsga2_survival(X, F, X_off, F_off, 3, return_indices=True)

    np.testing.assert_array_equal(sel_nb, sel_np)
    np.testing.assert_allclose(X_nb, X_np)
    np.testing.assert_allclose(F_nb, F_np)


@pytest.mark.numba
def test_numba_backend_survival_matches_numpy_on_random_biobjective_ties():
    rng = np.random.default_rng(19)
    numpy_kernel = NumPyKernel()
    numba_kernel = NumbaKernel()

    for _ in range(25):
        X = rng.random((12, 4))
        X_off = rng.random((12, 4))
        F = rng.integers(0, 7, size=(12, 2)).astype(float)
        F_off = rng.integers(0, 7, size=(12, 2)).astype(float)

        X_np, F_np, sel_np = numpy_kernel.nsga2_survival(X, F, X_off, F_off, 12, return_indices=True)
        X_nb, F_nb, sel_nb = numba_kernel.nsga2_survival(X, F, X_off, F_off, 12, return_indices=True)

        np.testing.assert_array_equal(sel_nb, sel_np)
        np.testing.assert_allclose(X_nb, X_np)
        np.testing.assert_allclose(F_nb, F_np)


@pytest.mark.numba
def test_numba_backend_matches_numpy_on_non_finite_fronts():
    F = np.array(
        [
            [np.nan, 1.0],
            [0.0, 2.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )

    numpy_kernel = NumPyKernel()
    numba_kernel = NumbaKernel()

    ranks_np, crowding_np = numpy_kernel.nsga2_ranking(F)
    ranks_nb, crowding_nb = numba_kernel.nsga2_ranking(F)

    np.testing.assert_array_equal(ranks_nb, ranks_np)
    np.testing.assert_allclose(crowding_nb, crowding_np, equal_nan=True)


@pytest.mark.numba
def test_numba_backend_matches_numpy_on_random_biobjective_fronts():
    rng = np.random.default_rng(7)
    numpy_kernel = NumPyKernel()
    numba_kernel = NumbaKernel()

    for n_points in (8, 16, 24, 32):
        for _ in range(25):
            F = rng.integers(0, 9, size=(n_points, 2)).astype(float)
            ranks_np, crowding_np = numpy_kernel.nsga2_ranking(F)
            ranks_nb, crowding_nb = numba_kernel.nsga2_ranking(F)
            np.testing.assert_array_equal(ranks_nb, ranks_np)
            np.testing.assert_allclose(crowding_nb, crowding_np, equal_nan=True)


@pytest.mark.numba
def test_numba_backend_matches_numpy_on_random_multiobjective_fronts():
    rng = np.random.default_rng(11)
    numpy_kernel = NumPyKernel()
    numba_kernel = NumbaKernel()

    for shape in ((10, 3), (14, 4), (18, 5)):
        for _ in range(20):
            F = rng.integers(0, 11, size=shape).astype(float)
            ranks_np, crowding_np = numpy_kernel.nsga2_ranking(F)
            ranks_nb, crowding_nb = numba_kernel.nsga2_ranking(F)
            np.testing.assert_array_equal(ranks_nb, ranks_np)
            np.testing.assert_allclose(crowding_nb, crowding_np, equal_nan=True)


@pytest.mark.numba
def test_numba_backend_mutation_matches_numpy_for_same_seed():
    X = np.random.default_rng(123).random((32, 7))
    params = {"prob": 0.35, "eta": 15.0}

    numpy_kernel = NumPyKernel()
    numba_kernel = NumbaKernel()

    X_np = X.copy()
    X_nb = X.copy()
    numpy_kernel.polynomial_mutation(X_np, params, np.random.default_rng(0), 0.0, 1.0)
    numba_kernel.polynomial_mutation(X_nb, params, np.random.default_rng(0), 0.0, 1.0)

    np.testing.assert_array_equal(X_nb, X_np)


@pytest.mark.numba
def test_numba_backend_mutation_repeats_for_same_seed():
    X = np.random.default_rng(321).random((24, 5))
    params = {"prob": 0.5, "eta": 20.0}

    kernel = NumbaKernel()
    X_first = X.copy()
    X_second = X.copy()

    kernel.polynomial_mutation(X_first, params, np.random.default_rng(0), 0.0, 1.0)
    kernel.polynomial_mutation(X_second, params, np.random.default_rng(0), 0.0, 1.0)

    np.testing.assert_allclose(X_first, X_second, rtol=0.0, atol=0.0)


@pytest.mark.numba
def test_numba_backend_tournament_selection_matches_numpy_for_same_seed():
    rng = np.random.default_rng(5)
    ranks = rng.integers(0, 4, size=48, dtype=int)
    crowding = rng.normal(size=48)
    crowding[::11] = np.nan
    crowding[::7] = np.inf

    rng_np = np.random.default_rng(0)
    rng_nb = np.random.default_rng(0)
    selected_np = NumPyKernel().tournament_selection(ranks, crowding, pressure=2, rng=rng_np, n_parents=64)
    selected_nb = NumbaKernel().tournament_selection(ranks, crowding, pressure=2, rng=rng_nb, n_parents=64)

    np.testing.assert_array_equal(selected_nb, selected_np)
    assert rng_nb.bit_generator.state == rng_np.bit_generator.state


@pytest.mark.numba
def test_numba_backend_tournament_selection_uses_numpy_fallback_for_higher_pressure():
    rng = np.random.default_rng(9)
    ranks = rng.integers(0, 5, size=40, dtype=int)
    crowding = rng.normal(size=40)

    numpy_kernel = NumPyKernel()
    numba_kernel = NumbaKernel()

    selected_np = numpy_kernel.tournament_selection(ranks, crowding, pressure=3, rng=np.random.default_rng(2), n_parents=50)
    selected_nb = numba_kernel.tournament_selection(ranks, crowding, pressure=3, rng=np.random.default_rng(2), n_parents=50)

    np.testing.assert_array_equal(selected_nb, selected_np)


@pytest.mark.numba
def test_binary_tournament_matches_numpy_values_and_draws():
    numpy_kernel, numba_kernel = NumPyKernel(), NumbaKernel()
    for seed in range(400):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 60))
        ranks = rng.integers(0, 3, size=n)
        crowding = rng.choice([0.0, 0.5, 1.0, np.inf, np.nan], size=n)
        rng_np, rng_nb = np.random.default_rng(seed), np.random.default_rng(seed)
        expected = numpy_kernel.tournament_selection(ranks, crowding, 2, rng_np, 2 * n)
        actual = numba_kernel.tournament_selection(ranks, crowding, 2, rng_nb, 2 * n)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
        assert rng_nb.bit_generator.state == rng_np.bit_generator.state


@pytest.mark.numba
def test_binary_tournament_preserves_fractional_rank_keys():
    ranks = np.array([0.25, 0.75, 0.5, 0.25])
    crowding = np.zeros(4)
    for seed in range(50):
        expected = NumPyKernel().tournament_selection(ranks, crowding, 2, np.random.default_rng(seed), 16)
        actual = NumbaKernel().tournament_selection(ranks, crowding, 2, np.random.default_rng(seed), 16)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.numba
def test_binary_tournament_kernel_marks_full_ties():
    winners = binary_tournament_winners_numba(
        np.array([0, 0, 1], dtype=np.int64),
        np.array([1.0, 1.0, np.nan]),
        np.array([0, 2, 1], dtype=np.int64),
        np.array([1, 0, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(winners, [-1, 0, 1])


def _adversarial_objectives(rng: np.random.Generator, trial: int, n_rows: int, n_obj: int) -> np.ndarray:
    kind = trial % 5
    if kind == 0:
        return rng.integers(0, 4, size=(n_rows, n_obj)).astype(float)  # heavy ties
    if kind == 1:
        return rng.random((n_rows, n_obj))
    if kind == 2:
        base = rng.random((max(1, n_rows // 3), n_obj))
        return base[rng.integers(0, base.shape[0], size=n_rows)]  # exact duplicates
    if kind == 3:
        F = rng.random((n_rows, n_obj))
        F[:, 0] = 1.0  # degenerate objective
        return F
    F = rng.random((n_rows, n_obj))
    return F / np.linalg.norm(F, axis=1, keepdims=True)  # mostly one front


@pytest.mark.numba
def test_numba_ranking_matches_numpy_on_adversarial_populations():
    rng = np.random.default_rng(2026)
    numpy_kernel, numba_kernel = NumPyKernel(), NumbaKernel()
    for trial in range(1500):
        F = _adversarial_objectives(rng, trial, int(rng.integers(1, 120)), int(rng.integers(1, 11)))
        ranks_np, crowding_np = numpy_kernel.nsga2_ranking(F)
        ranks_nb, crowding_nb = numba_kernel.nsga2_ranking(F)
        np.testing.assert_array_equal(ranks_nb, ranks_np)
        np.testing.assert_array_equal(crowding_nb, crowding_np)


@pytest.mark.numba
def test_numba_survival_matches_numpy_at_front_boundaries():
    rng = np.random.default_rng(77)
    numpy_kernel, numba_kernel = NumPyKernel(), NumbaKernel()
    for trial in range(1500):
        half = int(rng.integers(1, 60))
        F = _adversarial_objectives(rng, trial, 2 * half, int(rng.integers(1, 9)))
        X = rng.random((2 * half, 3))
        ranks, _ = numpy_kernel.nsga2_ranking(F)
        front_sizes = np.cumsum(np.bincount(ranks))
        candidates = {1, 2 * half, int(rng.integers(1, 2 * half + 1)), int(front_sizes[0])}
        candidates.update(int(size) for size in front_sizes[:3])
        for pop_size in sorted(candidates):
            X_np, F_np, sel_np = numpy_kernel.nsga2_survival(X[:half], F[:half], X[half:], F[half:], pop_size, return_indices=True)
            X_nb, F_nb, sel_nb = numba_kernel.nsga2_survival(X[:half], F[:half], X[half:], F[half:], pop_size, return_indices=True)
            np.testing.assert_array_equal(sel_nb, sel_np)
            assert sel_nb.dtype == sel_np.dtype
            np.testing.assert_array_equal(X_nb, X_np)
            np.testing.assert_array_equal(F_nb, F_np)


@pytest.mark.numba
def test_numba_survival_returns_arrays_owned_by_the_caller():
    rng = np.random.default_rng(5)
    kernel = NumbaKernel()
    X, F = rng.random((40, 3)), rng.random((40, 3))
    X_first, F_first = kernel.nsga2_survival(X[:20], F[:20], X[20:], F[20:], 20)
    expected = (X_first.copy(), F_first.copy())
    for _ in range(5):
        X_next, F_next = rng.random((40, 3)), rng.random((40, 3))
        kernel.nsga2_survival(X_next[:20], F_next[:20], X_next[20:], F_next[20:], 20)
    np.testing.assert_array_equal(X_first, expected[0])
    np.testing.assert_array_equal(F_first, expected[1])
