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
