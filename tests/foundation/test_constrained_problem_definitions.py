"""Registered constrained problems expose their constraints with the VAMOS convention, ``g <= 0`` is feasible."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.foundation.problem.registry import get_problem_specs, make_problem_selection
from vamos.problems import CEC2009CF1


def _sample(problem: Any, rng: np.random.Generator, n: int = 4) -> np.ndarray:
    encoding = str(getattr(problem, "encoding", "real"))
    if encoding == "permutation":
        return np.array([rng.permutation(problem.n_var) for _ in range(n)], dtype=float)
    xl = np.broadcast_to(np.asarray(getattr(problem, "xl", 0.0), dtype=float), (problem.n_var,))
    xu = np.broadcast_to(np.asarray(getattr(problem, "xu", 1.0), dtype=float), (problem.n_var,))
    X = xl + rng.random((n, problem.n_var)) * (xu - xl)
    if encoding == "mixed":
        spec = getattr(problem, "mixed_spec", {})
        discrete = np.concatenate([np.asarray(spec.get("int_idx", []), dtype=int), np.asarray(spec.get("cat_idx", []), dtype=int)])
        X[:, discrete] = np.rint(X[:, discrete])
        return X
    return np.rint(X) if encoding in {"binary", "integer"} else X


@pytest.mark.parametrize("key", sorted(get_problem_specs()))
def test_registered_problems_declare_the_constraints_they_write(key: str) -> None:
    try:
        problem = make_problem_selection(key).instantiate()
    except ImportError as exc:
        pytest.skip(f"{key} needs an optional dependency: {exc}")
    X = _sample(problem, np.random.default_rng(0))
    out = {"F": np.full((X.shape[0], problem.n_obj), np.nan)}

    problem.evaluate(X, out)

    written = 0 if out.get("G") is None else np.atleast_2d(np.asarray(out["G"])).shape[1]
    assert int(getattr(problem, "n_constraints", 0) or 0) == written


@pytest.mark.parametrize("key, n_constraints", [("cec2009_cf1", 1), ("welded_beam", 4)])
def test_constrained_problems_return_their_constraints_to_algorithms(key: str, n_constraints: int) -> None:
    problem = make_problem_selection(key).instantiate()
    X = _sample(problem, np.random.default_rng(1))

    _, G = evaluate_population_with_constraints(problem, X)

    assert G is not None and G.shape == (X.shape[0], n_constraints)


def _cf1_point(x0: float, n_var: int = 10, offset: float = 0.0) -> np.ndarray:
    """A CF1 decision vector whose remaining variables sit ``offset`` away from the Pareto set."""
    j = np.arange(2, n_var + 1)
    rest = np.power(x0, 0.5 * (1.0 + 3.0 * (j - 2.0) / (n_var - 2.0))) + offset
    return np.concatenate([[x0], rest])[None, :]


def test_cf1_uses_the_cec2009_domain() -> None:
    problem = CEC2009CF1(n_var=10)

    assert np.all(problem.xl == 0.0) and np.all(problem.xu == 1.0)


def test_cf1_constraint_is_feasible_when_nonpositive() -> None:
    # CEC2009 CF1 requires c = f1 + f2 - |sin(10*pi*(f1 - f2 + 1))| - 1 >= 0, so VAMOS must store G = -c.
    problem = CEC2009CF1(n_var=10)

    def constraint(X: np.ndarray) -> float:
        out = {"F": np.empty((1, 2))}
        problem.evaluate(X, out)
        return float(out["G"][0, 0])

    assert constraint(_cf1_point(0.1)) == pytest.approx(0.0, abs=1e-12)  # A Pareto-optimal point, c = 0.
    assert constraint(_cf1_point(0.125)) > 0.0  # On f1 + f2 = 1 between optimal points: c = -1, infeasible.
    assert constraint(_cf1_point(0.125, offset=0.6)) < 0.0  # Far from the Pareto set: f1 + f2 > 2, so c > 0.


@pytest.mark.parametrize(
    ("key", "n_var_min"),
    [("cec2009_cf2", 3), ("cec2009_cf3", 3), ("cec2009_cf4", 2), ("cec2009_cf5", 2), ("cec2009_cf6", 4), ("cec2009_cf7", 4)]
    + [(f"cec2009_cf{k}", 5) for k in (8, 9, 10)],
)
def test_cec2009_cf_problems_need_jmetals_minimum_dimension(key: str, n_var_min: int) -> None:
    assert make_problem_selection(key, n_var=n_var_min).instantiate().n_var == n_var_min
    with pytest.raises(ValueError, match="at least"):
        make_problem_selection(key, n_var=n_var_min - 1).instantiate()


@pytest.mark.parametrize("k", (8, 9, 10))
def test_cec2009_cf8_to_cf10_constraint_holds_at_the_top_of_the_front(k: int) -> None:
    # At (0, 0, 1) the constraint divides by 1 - f3^2 = 0; as in jMetal it is then 0, not NaN.
    problem = make_problem_selection(f"cec2009_cf{k}").instantiate()
    X = np.zeros((1, problem.n_var))
    X[0, 0] = 1.0

    F, G = evaluate_population_with_constraints(problem, X)

    assert F[0, 2] == 1.0 and G is not None
    assert G[0, 0] == 0.0


@pytest.mark.parametrize(("n_obj", "k"), [(2, 1), (3, 2), (5, 2), (8, 4), (9, 3)])
def test_zxh_cf_constraints_follow_the_number_of_objectives(n_obj: int, k: int) -> None:
    # One or two radius constraints, then per constrained angle: none (A), two bounds (B), one excluded
    # interval (C) or two (D).
    expected = {1: 1, 2: 2, 5: 1 + 2 * k, 6: 2 + 2 * k, 9: 1 + k, 10: 2 + k, 13: 1 + 2 * k, 14: 2 + 2 * k}
    for variant, n_constraints in expected.items():
        selection = make_problem_selection(f"zxh_cf{variant}", n_obj=n_obj)
        problem = selection.instantiate()
        assert (problem.n_var, problem.n_obj, problem.n_constraints) == (n_obj + 10, n_obj, n_constraints)


def _zxh_point(theta0: float, theta1: float, n_var: int = 13) -> np.ndarray:
    """A three-objective ZXH-CF design on the Pareto set (S = 1, h = 0) with the given angles."""
    a0, a1 = 0.5 * np.pi * theta0, 0.5 * np.pi * theta1
    x = np.full(n_var, 0.2)
    x[:3] = [np.cos(a0), np.sin(a0) * np.cos(a1), np.sin(a0) * np.sin(a1)]
    return x[None, :]


def test_zxh_cf_type_c_excludes_an_interval_of_each_angle() -> None:
    # ZXH-CF11 (r = 1/2): G = [S + h - 1, r - S - h, then min(theta - 1/4, 3/4 - theta) per angle].
    problem = make_problem_selection("zxh_cf11").instantiate()
    _, G = evaluate_population_with_constraints(problem, np.vstack([_zxh_point(0.4, 0.1), _zxh_point(0.1, 0.9)]))

    assert G is not None and G.shape == (2, 4)
    np.testing.assert_allclose(G[0, 2:], [0.15, -0.15], atol=1e-12)
    np.testing.assert_allclose(G[1, 2:], [-0.15, -0.15], atol=1e-12)
    assert np.max(G[1]) <= 1e-12


def test_zxh_cf_type_d_excludes_two_intervals_of_each_angle() -> None:
    # ZXH-CF16 (r = 3/4): per angle, min(theta - 1/10, 2/5 - theta) and min(theta - 7/10, 4/5 - theta).
    problem = make_problem_selection("zxh_cf16").instantiate()
    designs = np.vstack([_zxh_point(0.25, 0.05), _zxh_point(0.75, 0.55), _zxh_point(0.55, 0.95)])
    _, G = evaluate_population_with_constraints(problem, designs)

    assert G is not None and G.shape == (3, 6)
    np.testing.assert_allclose(G[0, 2:], [0.15, -0.45, -0.05, -0.65], atol=1e-12)
    np.testing.assert_allclose(G[1, 2:], [-0.35, 0.05, -0.15, -0.15], atol=1e-12)
    assert np.max(G[2]) <= 1e-12


def test_zxh_cf_ackley_problems_need_a_distance_variable() -> None:
    # The Ackley function averages over the variables after the first M, as jMetal checks.
    assert make_problem_selection("zxh_cf1", n_var=3).instantiate().n_var == 3
    for key in ("zxh_cf3", "zxh_cf8", "zxh_cf9", "zxh_cf14"):
        with pytest.raises(ValueError, match="at least 4 variables"):
            make_problem_selection(key, n_var=3).instantiate()
