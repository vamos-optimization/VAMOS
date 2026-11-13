"""CRE21-CRE51 (Tanabe and Ishibuchi) and RCM01-RCM50 (CEC 2021): dimensions, rounding, equalities and overlaps.

The values themselves are checked against jMetal in ``test_problem_reference_values.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.foundation.problem.cre import CREProblem
from vamos.foundation.problem.rcm import EQUALITY_TOLERANCE, RCMProblem
from vamos.foundation.problem.rcm._common import java_round
from vamos.foundation.problem.registry import get_problem_specs, make_problem_selection

# (objectives, variables, constraints) of jMetal's classes.
CRE_DIMENSIONS = {
    "cre21": (2, 3, 3),
    "cre22": (2, 4, 4),
    "cre23": (2, 4, 4),
    "cre24": (2, 7, 11),
    "cre25": (2, 4, 1),
    "cre31": (3, 7, 10),
    "cre32": (3, 6, 9),
    "cre51": (5, 3, 7),
}
RCM_DIMENSIONS = {
    1: (2, 4, 2),
    2: (2, 5, 5),
    3: (2, 3, 3),
    4: (2, 4, 4),
    5: (2, 4, 4),
    6: (2, 7, 11),
    7: (2, 4, 1),
    8: (3, 7, 9),
    9: (2, 4, 0),
    10: (2, 2, 2),
    11: (5, 3, 7),
    12: (2, 4, 1),
    13: (3, 7, 11),
    14: (2, 5, 8),
    15: (2, 3, 8),
    16: (2, 2, 2),
    17: (3, 6, 9),
    18: (2, 3, 3),
    19: (3, 10, 10),
    20: (2, 4, 7),
    21: (2, 6, 4),
    22: (2, 9, 6),
    23: (2, 6, 5),
    24: (3, 9, 6),
    25: (2, 2, 2),
    26: (2, 3, 2),
    27: (2, 3, 3),
    28: (2, 7, 8),
    29: (2, 7, 9),
    **dict.fromkeys((30, 31, 32), (2, 25, 24)),
    **dict.fromkeys((33, 34, 35), (2, 30, 29)),
    **dict.fromkeys((36, 37, 38), (2, 28, 24)),
    39: (3, 28, 24),
    **dict.fromkeys((40, 42, 43), (2, 34, 26)),
    **dict.fromkeys((41, 44, 45), (3, 34, 26)),
    46: (4, 34, 26),
    **dict.fromkeys((47, 48), (2, 18, 12)),
    49: (3, 18, 12),
    50: (2, 6, 1),
}
DIMENSIONS = {**CRE_DIMENSIONS, **{f"rcm{k:02d}": dims for k, dims in RCM_DIMENSIONS.items()}}
Array = np.ndarray[Any, Any]


def _evaluate(key: str, X: Array) -> tuple[Array, Array | None]:
    return evaluate_population_with_constraints(make_problem_selection(key).instantiate(), np.atleast_2d(X))


def _uniform(key: str, n: int, seed: int = 0) -> Array:
    problem = make_problem_selection(key).instantiate()
    return np.asarray(problem.xl, dtype=float) + np.random.default_rng(seed).random((n, problem.n_var)) * (problem.xu - problem.xl)


def test_every_cre_and_rcm_problem_is_registered() -> None:
    registered = {key for key in get_problem_specs() if key.startswith(("cre", "rcm"))}
    assert registered == set(DIMENSIONS)


@pytest.mark.parametrize("key", sorted(DIMENSIONS))
def test_dimensions_are_jmetals_and_fixed(key: str) -> None:
    n_obj, n_var, n_constraints = DIMENSIONS[key]
    spec = get_problem_specs()[key]
    problem = CREProblem(key) if key.startswith("cre") else RCMProblem(int(key[3:]))
    assert (problem.n_obj, problem.n_var, problem.n_constraints) == (n_obj, n_var, n_constraints)
    assert (spec.default_n_obj, spec.default_n_var, spec.encoding) == (n_obj, n_var, "real")
    xl, xu = np.asarray(problem.xl), np.asarray(problem.xu)
    assert xl.shape == xu.shape == (n_var,) and np.all(xl < xu)
    with pytest.raises(ValueError, match="fixed number of variables"):
        make_problem_selection(key, n_var=n_var + 1)
    with pytest.raises(ValueError, match="fixed number of objectives"):
        make_problem_selection(key, n_obj=n_obj + 1)


def test_rcm09_has_no_constraints() -> None:
    # The suite's metadata lists one constraint, but its evaluator never sets it; jMetal follows the evaluator.
    problem = RCMProblem(9)
    out: dict[str, Array] = {}
    problem.evaluate(_uniform("rcm09", 5), out)
    assert problem.n_constraints == 0 and "G" not in out
    assert _evaluate("rcm09", _uniform("rcm09", 5))[1] is None


def test_java_round_rounds_halves_up_without_the_error_of_floor_x_plus_half() -> None:
    x = np.array([0.49999999999999994, 0.5, 1.5, 2.5, -0.5, -1.5, -0.49, 16.51, 28.49])
    assert java_round(x).tolist() == [0.0, 1.0, 2.0, 3.0, 0.0, -1.0, 0.0, 17.0, 28.0]
    # np.floor(x + 0.5) rounds the largest double below 0.5 up, Java's Math.round does not.
    assert np.floor(x[0] + 0.5) == 1.0


def test_cre_rounds_half_to_even_and_rcm_rounds_half_up() -> None:
    # CRE24's number of teeth uses Math.rint (18.5 -> 18), RCM06's Math.round (18.5 -> 19).
    cre24 = np.array([3.0, 0.75, 18.0, 8.0, 8.0, 3.5, 5.2])
    rcm06 = cre24.copy()
    for key, x, rounded in (("cre24", cre24, 18.0), ("rcm06", rcm06, 19.0)):
        half, exact = x.copy(), x.copy()
        half[2], exact[2] = 18.5, rounded
        F_half, G_half = _evaluate(key, half)
        F_exact, G_exact = _evaluate(key, exact)
        np.testing.assert_array_equal(F_half, F_exact)
        np.testing.assert_array_equal(G_half, G_exact)


@pytest.mark.parametrize(
    ("key", "columns"),
    [
        ("cre24", [2]),
        ("cre25", [0, 1, 2, 3]),
        ("rcm01", [0, 1]),
        ("rcm06", [2]),
        ("rcm13", [2]),
        ("rcm15", [0, 2]),
        ("rcm19", [0, 1, 2]),
        ("rcm25", [1]),
        ("rcm26", [2]),
        ("rcm28", [4, 5]),
        ("rcm29", [3, 4, 5, 6]),
    ],
)
def test_integer_variables_are_rounded_inside_the_evaluation(key: str, columns: list[int]) -> None:
    problem = make_problem_selection(key).instantiate()
    X = _uniform(key, 20, seed=1)
    X[:, columns] = np.floor(X[:, columns] + 0.5)
    # Within 0.45 of an integer (and inside the bounds), every integer variable rounds to that integer.
    moved = X.copy()
    moved[:, columns] += np.random.default_rng(2).uniform(-0.45, 0.45, (X.shape[0], len(columns)))
    moved = np.clip(moved, problem.xl, problem.xu)
    F, G = _evaluate(key, X)
    F_moved, G_moved = _evaluate(key, moved)
    np.testing.assert_array_equal(F_moved, F)
    assert G is not None and G_moved is not None
    np.testing.assert_array_equal(G_moved, G)


def test_equality_constraints_are_relaxed_to_1e_4() -> None:
    # RCM26: h1 = x1 - 2 exp(-x2) = 0, stored after the inequality as |h1| - 1e-4.
    x2 = 0.7
    for offset, feasible in ((0.0, True), (0.5e-4, True), (-0.9e-4, True), (2e-4, False), (-2e-4, False)):
        _, G = _evaluate("rcm26", np.array([2.0 * np.exp(-x2) + offset, x2, 0.0]))
        assert G is not None
        assert G[0, 1] == pytest.approx(abs(offset) - EQUALITY_TOLERANCE, abs=1e-15)
        assert bool(np.all(G <= 0.0)) is feasible


def test_rcm22_is_feasible_at_the_origin() -> None:
    # Haverly's pooling: no flow meets both inequalities (0 <= 0) and every equality (|h| = 0 <= 1e-4).
    F, G = _evaluate("rcm22", np.zeros(9))
    assert G is not None
    np.testing.assert_array_equal(F, [[0.0, 0.0]])
    np.testing.assert_allclose(G, [[0.0, 0.0, *([-EQUALITY_TOLERANCE] * 4)]], rtol=0.0, atol=1e-18)


def test_sopwm_constraints_keep_the_angles_increasing() -> None:
    problem = RCMProblem(33)
    increasing = np.linspace(1.0, 89.0, problem.n_var)[None, :]
    swapped = increasing.copy()
    swapped[0, [3, 4]] = swapped[0, [4, 3]]
    _, G = _evaluate("rcm33", np.vstack([increasing, swapped]))
    assert G is not None
    assert np.all(G[0] < 0.0)
    assert G[1, 3] > 0.0 and np.sum(G[1] > 0.0) == 1


# CRE problems that share their formulas with an RE problem (which adds the violation as an objective) and with
# a classic constrained problem.
_RE_COUNTERPARTS = {
    "cre21": "re31",
    "cre22": "re32",
    "cre23": "re33",
    "cre24": "re35",
    "cre25": "re36",
    "cre31": "re41",
    "cre32": "re42",
    "cre51": "re61",
}


@pytest.mark.parametrize("key", sorted(_RE_COUNTERPARTS))
def test_cre_problems_are_the_re_problems_with_their_constraints(key: str) -> None:
    X = _uniform(key, 200, seed=3)
    F, G = _evaluate(key, X)
    re = make_problem_selection(_RE_COUNTERPARTS[key]).instantiate()
    assert np.array_equal(re.xl, CREProblem(key).xl) and np.array_equal(re.xu, CREProblem(key).xu)
    out = {"F": np.empty((X.shape[0], re.n_obj))}
    re.evaluate(X, out)
    assert G is not None
    # RE42 guards its powers and divisions; inside the bounds the guards do not change the values.
    np.testing.assert_allclose(out["F"][:, :-1], F, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(out["F"][:, -1], np.maximum(G, 0.0).sum(axis=1), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("cre", "rcm", "classic"), [("cre21", "rcm03", None), ("cre23", "rcm05", None), ("cre51", "rcm11", "water")])
def test_overlapping_cre_and_rcm_problems_are_the_same_problem(cre: str, rcm: str, classic: str | None) -> None:
    X = _uniform(cre, 200, seed=4)
    F, G = _evaluate(cre, X)
    for other in (rcm, classic):
        if other is None:
            continue
        problem = make_problem_selection(other).instantiate()
        assert np.array_equal(problem.xl, CREProblem(cre).xl) and np.array_equal(problem.xu, CREProblem(cre).xu)
        F_other, G_other = _evaluate(other, X)
        np.testing.assert_array_equal(F_other, F)
        assert G is not None and G_other is not None
        np.testing.assert_array_equal(G_other, G)


def test_cre31_is_the_car_side_impact_problem() -> None:
    X = _uniform("cre31", 200, seed=5)
    np.testing.assert_array_equal(_evaluate("cre31", X)[0], _evaluate("car_side_impact", X)[0])


def test_cre22_and_rcm04_differ_only_in_the_buckling_load() -> None:
    # Both are the welded beam; RCM04's buckling load uses x3^2 + x4^6 (the suite's code), CRE22's x3^2 x4^6.
    X = _uniform("cre22", 50, seed=6)
    F_cre, G_cre = _evaluate("cre22", X)
    F_rcm, G_rcm = _evaluate("rcm04", X)
    assert G_cre is not None and G_rcm is not None
    np.testing.assert_allclose(F_rcm, F_cre, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(G_rcm[:, :3], G_cre[:, :3], rtol=1e-14, atol=1e-12)
    assert np.all(np.abs(G_rcm[:, 3] - G_cre[:, 3]) > 1e-6 * np.abs(G_cre[:, 3]))
