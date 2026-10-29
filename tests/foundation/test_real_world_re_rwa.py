import itertools

import numpy as np
import pytest

from vamos.foundation.problem.registry import make_problem_selection
from vamos.resources import reference_front_path


def _sample(problem, n: int = 4, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xl = problem.xl
    xu = problem.xu
    if np.isscalar(xl) or np.isscalar(xu):
        return rng.uniform(float(xl), float(xu), size=(n, problem.n_var))
    return rng.uniform(np.asarray(xl, dtype=float), np.asarray(xu, dtype=float), size=(n, problem.n_var))


# Wire diameters RE25 can reach in Tanabe and Ishibuchi's reproblem.py (and jMetal 7.9): x3 is bounded to
# [0.09, 0.5] and rounded to the nearest listed diameter, so the 23 listed diameters below 0.092 are unreachable.
_RE25_REFERENCE_DIAMETERS = (
    0.092,
    0.105,
    0.12,
    0.135,
    0.148,
    0.162,
    0.177,
    0.192,
    0.207,
    0.225,
    0.244,
    0.263,
    0.283,
    0.307,
    0.331,
    0.362,
    0.394,
    0.4375,
    0.5,
)


def test_re24_re25_domains_match_reference():
    re24 = make_problem_selection("re24").instantiate()
    np.testing.assert_array_equal(re24.xl, [0.5, 0.5])
    np.testing.assert_array_equal(re24.xu, [4.0, 50.0])

    re25 = make_problem_selection("re25").instantiate()
    n = len(_RE25_REFERENCE_DIAMETERS)
    assert re25.xu[2] == n - 1
    assert re25.mixed_spec["cat_cardinality"].tolist() == [n]
    # f1 = pi^2 * x2 * x3^2 * (x1 + 2) / 4 recovers the diameter selected by each category index.
    X = np.column_stack([np.full(n, 10.0), np.full(n, 1.5), np.arange(n, dtype=float)])
    out = {"F": np.empty((n, 2))}
    re25.evaluate(X, out)
    diameters = np.asarray(_RE25_REFERENCE_DIAMETERS)
    np.testing.assert_allclose(out["F"][:, 0], np.pi**2 * 1.5 * diameters**2 * 12.0 / 4.0, rtol=1e-15)


def test_re24_reference_front_covers_low_cost_designs():
    from vamos.resources import reference_front_path

    front = np.loadtxt(reference_front_path("re24"), delimiter=",", ndmin=2)
    # Tanabe and Ishibuchi's approximated front runs from f1 = 60.5 (f2 = 44.28) to f1 = 481.6 (feasible).
    assert front.shape[1] == 2
    assert front[:, 0].min() <= 61.0
    assert front[:, 1].min() == 0.0
    assert front[:, 0].max() <= 482.0


@pytest.mark.parametrize(
    "key",
    [
        # Zapotecas-Martínez et al. (2023) RWA suite
        "rwa1",
        "rwa2",
        "rwa3",
        "rwa4",
        "rwa5",
        "rwa6",
        "rwa7",
        "rwa8",
        "rwa9",
        "rwa10",
        # Tanabe & Ishibuchi (2020) RE suite
        "re21",
        "re22",
        "re23",
        "re24",
        "re25",
        "re31",
        "re32",
        "re33",
        "re34",
        "re35",
        "re36",
        "re37",
        "re41",
        "re42",
        "re61",
        "re91",
    ],
)
def test_re_and_rwa_shapes_and_finiteness(key: str):
    selection = make_problem_selection(key)
    problem = selection.instantiate()
    X = _sample(problem, n=5)
    out = {"F": np.empty((X.shape[0], problem.n_obj))}
    problem.evaluate(X, out)
    assert out["F"].shape == (X.shape[0], problem.n_obj)
    assert np.isfinite(out["F"]).all()


@pytest.mark.parametrize("key", ["re22", "re23", "re25", "re35"])
def test_re_mixed_problems_expose_mixed_spec(key: str):
    problem = make_problem_selection(key).instantiate()
    assert getattr(problem, "encoding", None) == "mixed"
    assert hasattr(problem, "mixed_spec")


def _evaluate(key: str, X: np.ndarray) -> np.ndarray:
    problem = make_problem_selection(key).instantiate()
    out = {"F": np.empty((X.shape[0], problem.n_obj))}
    problem.evaluate(X, out)
    return out["F"]


def _unit_box_points() -> np.ndarray:
    corners = np.array(list(itertools.product((0.0, 1.0), repeat=4)))
    return np.vstack([corners, np.random.default_rng(7).random((50, 4))])


def test_rwa7_matches_re37_response_surfaces():
    # RWA7, RWA8 and RE37 evaluate the injector response surfaces of Goel et al. (2007) in variables
    # normalized to [0, 1]^4 (Tanabe & Ishibuchi's reproblem.py RE37; jMetal RWA7/RWA8). RE37 returns
    # (TFmax, Xcc, TTmax), RWA7 (Xcc, TFmax, TTmax) and RWA8 (TFmax, TW4, TTmax, Xcc).
    re37 = make_problem_selection("re37").instantiate()
    for key in ("rwa7", "rwa8"):
        problem = make_problem_selection(key).instantiate()
        np.testing.assert_array_equal(np.broadcast_to(problem.xl, (4,)), np.broadcast_to(re37.xl, (4,)))
        np.testing.assert_array_equal(np.broadcast_to(problem.xu, (4,)), np.broadcast_to(re37.xu, (4,)))

    X = _unit_box_points()
    tfmax_xcc_ttmax = _evaluate("re37", X)
    np.testing.assert_allclose(_evaluate("rwa7", X), tfmax_xcc_ttmax[:, [1, 0, 2]], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(_evaluate("rwa8", X)[:, [0, 2, 3]], tfmax_xcc_ttmax[:, [0, 2, 1]], rtol=0.0, atol=1e-12)


def test_rwa7_rwa8_match_jmetal_values():
    # Expected values from jMetal 7.9 RWA7 and RWA8; Tanabe's RE37 gives the same Xcc, TFmax and TTmax.
    X = np.array([[0.25, 0.5, 0.75, 1.0], [0.875, 0.125, 0.625, 0.375]])
    rwa7 = np.array(
        [
            [0.729075, 0.3860462499999999, 0.9134531249999998],
            [0.23782500000000004, 0.8047059375, 0.46131406249999984],
        ]
    )
    rwa8 = np.array(
        [
            [0.3860462499999999, 0.4977350000000001, 0.9134531249999998, 0.729075],
            [0.8047059375, 0.8209126562500001, 0.46131406249999984, 0.23782500000000004],
        ]
    )
    np.testing.assert_allclose(_evaluate("rwa7", X), rwa7, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(_evaluate("rwa8", X), rwa8, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("key", ["rwa7", "rwa8"])
def test_rwa7_rwa8_reference_fronts_match_the_normalized_formulation(key: str):
    front = np.loadtxt(reference_front_path(key), delimiter=",", ndmin=2)
    grid = np.array(list(itertools.product(np.linspace(0.0, 1.0, 6), repeat=4)))
    F = _evaluate(key, np.vstack([grid, np.random.default_rng(11).random((4_000, 4))]))
    assert front.shape[1] == F.shape[1]

    # Every point is an objective vector of the [0, 1]^4 formulation, up to a margin for unsampled extremes.
    low, high = F.min(axis=0), F.max(axis=0)
    margin = 0.05 * (high - low)
    assert np.all(front >= low - margin) and np.all(front <= high + margin)

    # No sampled design dominates a point of the front.
    for chunk in np.array_split(F, 4):
        dominated = np.all(chunk[:, None, :] <= front[None, :, :], axis=2) & np.any(chunk[:, None, :] < front[None, :, :], axis=2)
        assert not dominated.any()
