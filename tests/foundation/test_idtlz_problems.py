"""IDTLZ1 and IDTLZ2, the inverted DTLZ1 and DTLZ2 of Jain and Deb (2014), as PlatEMO defines them.

VAMOS 1.5.0: PlatEMO's ``IDTLZ1.m`` and ``IDTLZ2.m`` (BIMK/PlatEMO, ``Problems/Multi-objective
optimization/DTLZ``) compute ``f = (1 + g) / 2 - (1 + g) / 2 * s`` and ``f = (1 + g) - (1 + g) * s``, where ``s`` is
DTLZ1's or DTLZ2's position shape and ``g`` its distance function. The expected values below are written out from
those formulas, one objective at a time, with :mod:`math`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from vamos import optimize
from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.foundation.problem.dtlz import DTLZ1Problem, DTLZ2Problem, IDTLZ1Problem, IDTLZ2Problem
from vamos.foundation.problem.registry import available_problem_names, make_problem_selection
from vamos.foundation.problem.resolver import resolve_reference_front_path


def _platemo_objectives(k: int, x: list[float], m: int) -> list[float]:
    """IDTLZ``k`` at ``x`` with ``m`` objectives: f_1 uses x_1 ... x_{M-1}, f_i (1 < i < M) uses x_1 ... x_{M-i} and
    the complement of x_{M-i+1}, and f_M the complement of x_1."""
    position, distance = x[: m - 1], x[m - 1 :]
    if k == 1:
        g = 100.0 * (len(distance) + sum((xi - 0.5) ** 2 - math.cos(20.0 * math.pi * (xi - 0.5)) for xi in distance))
        kept, complement, scale = position, [1.0 - xi for xi in position], (1.0 + g) / 2.0
    else:
        g = sum((xi - 0.5) ** 2 for xi in distance)
        angle = [xi * math.pi / 2.0 for xi in position]
        kept, complement, scale = [math.cos(a) for a in angle], [math.sin(a) for a in angle], 1.0 + g
    objectives = []
    for i in range(1, m + 1):
        shape = math.prod(kept[: m - i])
        if i > 1:
            shape *= complement[m - i]
        objectives.append(scale - scale * shape)
    return objectives


def _points(m: int, n_var: int) -> np.ndarray:
    """Fixed decision vectors: the bounds, the centre, and points that move every variable away from 0.5."""
    rng = np.random.default_rng(1000 + 10 * m + n_var)
    corners = np.array([np.zeros(n_var), np.ones(n_var), np.full(n_var, 0.5)])
    return np.vstack([corners, rng.random((5, n_var))])


def _evaluate(problem: object, X: np.ndarray) -> np.ndarray:
    F, G = evaluate_population_with_constraints(problem, X)
    assert G is None
    return np.asarray(F)


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("m", [2, 3, 5])
def test_objectives_follow_platemos_formulas(k: int, m: int) -> None:
    problem = make_problem_selection(f"idtlz{k}", n_obj=m).instantiate()
    X = _points(m, problem.n_var)

    F = _evaluate(problem, X)

    expected = np.array([_platemo_objectives(k, row.tolist(), m) for row in X])
    np.testing.assert_allclose(F, expected, rtol=1e-12, atol=1e-12)


def test_objectives_at_the_centre_of_the_box() -> None:
    # With every variable at 0.5, g = 0: DTLZ1's shape is (1/4, 1/4, 1/2) and DTLZ2's (1/2, 1/2, sqrt(2)/2).
    centre_1 = np.full((1, 7), 0.5)
    centre_2 = np.full((1, 12), 0.5)

    np.testing.assert_allclose(_evaluate(IDTLZ1Problem(), centre_1), [[0.375, 0.375, 0.25]], rtol=1e-15)
    np.testing.assert_allclose(_evaluate(IDTLZ2Problem(), centre_2), [[0.5, 0.5, 1.0 - math.sqrt(0.5)]], rtol=1e-15)


@pytest.mark.parametrize("m", [2, 3, 6])
def test_idtlz_turns_dtlz_over(m: int) -> None:
    # IDTLZ1 = (1 + g) / 2 - DTLZ1 and IDTLZ2 = (1 + g) - DTLZ2, with the same g (pymoo writes its InvertedDTLZ1 so).
    rng = np.random.default_rng(m)
    X1, X2 = rng.random((20, m + 4)), rng.random((20, m + 9))
    distance = X1[:, m - 1 :] - 0.5
    g1 = 100.0 * (distance.shape[1] + np.sum(distance**2 - np.cos(20.0 * np.pi * distance), axis=1))
    g2 = np.sum((X2[:, m - 1 :] - 0.5) ** 2, axis=1)

    inverted_1 = _evaluate(IDTLZ1Problem(n_obj=m), X1)
    inverted_2 = _evaluate(IDTLZ2Problem(n_obj=m), X2)

    np.testing.assert_allclose(inverted_1, (1.0 + g1)[:, None] / 2.0 - _evaluate(DTLZ1Problem(n_obj=m), X1), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(inverted_2, (1.0 + g2)[:, None] - _evaluate(DTLZ2Problem(n_obj=m), X2), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("key", "m", "n_var"), [("idtlz1", 3, 7), ("idtlz1", 5, 9), ("idtlz2", 3, 12), ("idtlz2", 8, 17)])
def test_registry_resolves_platemos_dimensions(key: str, m: int, n_var: int) -> None:
    # PlatEMO's D = M + 4 for IDTLZ1 and M + 9 for IDTLZ2; the variables lie in [0, 1].
    assert key in available_problem_names()
    selection = make_problem_selection(key, n_obj=m) if m != 3 else make_problem_selection(key)
    problem = selection.instantiate()

    assert (problem.n_var, problem.n_obj, problem.encoding) == (n_var, m, "real")
    assert (problem.xl, problem.xu) == (0.0, 1.0)
    assert selection.spec.allow_n_obj_override


@pytest.mark.parametrize(("cls", "n_var"), [(IDTLZ1Problem, 10), (IDTLZ2Problem, 8)])
def test_a_non_standard_number_of_variables_warns(cls: type, n_var: int) -> None:
    with pytest.warns(UserWarning, match=f"Non-standard {cls.__name__.removesuffix('Problem')} configuration"):
        cls(n_var=n_var, n_obj=3)
    with pytest.raises(ValueError, match="n_var >= n_obj"):
        cls(n_var=2, n_obj=3)


def _pareto_designs(k: int, front: np.ndarray, n_var: int) -> np.ndarray:
    """Designs with g = 0 (every distance variable at 0.5) whose objectives are ``front``: the position variables
    invert DTLZ1's (``s = 1 - 2 f``) or DTLZ2's (``s = 1 - f``) shape."""
    n, m = front.shape
    s = 1.0 - 2.0 * front if k == 1 else 1.0 - front
    X = np.full((n, n_var), 0.5)
    rest = np.ones(n)
    for j in range(m - 1):
        i = m - 1 - j
        if k == 1:
            # s_M = 1 - x_1, then s_{M-1} = x_1 (1 - x_2), and so on.
            X[:, j] = 1.0 - np.divide(s[:, i], rest, out=np.zeros(n), where=rest > 1e-15)
            rest = rest * X[:, j]
        else:
            X[:, j] = 2.0 / np.pi * np.arctan2(s[:, i], np.linalg.norm(s[:, :i], axis=1))
    return np.clip(X, 0.0, 1.0)


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("m", [2, 3])
def test_reference_front_points_are_reached_with_g_zero(k: int, m: int) -> None:
    path = resolve_reference_front_path(f"idtlz{k}", None, n_obj=m)
    assert path is not None and Path(path).name == f"idtlz{k}.{m}d.csv"
    front = np.loadtxt(path, delimiter=",", ndmin=2)
    problem = make_problem_selection(f"idtlz{k}", n_obj=m).instantiate()

    F = _evaluate(problem, _pareto_designs(k, front, problem.n_var))

    np.testing.assert_allclose(F, front, rtol=0.0, atol=1e-9)
    # On the front, IDTLZ1's objectives sum to (M - 1) / 2 and IDTLZ2's complements lie on the unit sphere.
    residual = front.sum(axis=1) - (m - 1) / 2.0 if k == 1 else np.sum((1.0 - front) ** 2, axis=1) - 1.0
    assert np.max(np.abs(residual)) <= 1e-12
    assert front.min() >= 0.0 and front.max() <= (0.5 if k == 1 else 1.0)


@pytest.mark.parametrize("key", ["idtlz1", "idtlz2"])
def test_only_two_and_three_objectives_have_a_packaged_front(key: str) -> None:
    assert Path(str(resolve_reference_front_path(key, None))).name == f"{key}.3d.csv"
    assert resolve_reference_front_path(key, None, n_obj=4) is None


@pytest.mark.parametrize("key", ["idtlz1", "idtlz2"])
def test_nsgaii_runs_on_idtlz(key: str) -> None:
    result = optimize(key, algorithm="nsgaii", max_evaluations=60, pop_size=12, seed=3, engine="numpy")

    assert result.F is not None and result.F.shape[1] == 3
    assert np.isfinite(result.F).all()
