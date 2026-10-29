import numpy as np
import pytest

from vamos.foundation.problem.registry import make_problem_selection


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
