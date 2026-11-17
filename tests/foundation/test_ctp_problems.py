"""CTP1-CTP7 (Deb, Pratap and Meyarivan, 2001): definitions, parameters, bounds and the distance function g.

The expected values come from the paper's formulas, evaluated row by row with :mod:`math` and the distance function
in ZDT4's cosine form, ``g = 1 + 10 (n - 1) + sum_{i>=2} (x_i^2 - 10 cos(4 pi x_i))``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vamos import optimize
from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.foundation.problem.ctp import PAPER_PARAMETERS, CTPProblem, ctp1_parameters, rastrigin_g
from vamos.foundation.problem.registry import make_problem_selection
from vamos.problems import CTP

# The paper's parameters (theta, a, b, c, d, e) of CTP2-CTP7, from its section on the generator.
_PAPER = {
    2: (-0.2 * math.pi, 0.2, 10.0, 1, 6.0, 1.0),
    3: (-0.2 * math.pi, 0.1, 10.0, 1, 0.5, 1.0),
    4: (-0.2 * math.pi, 0.75, 10.0, 1, 0.5, 1.0),
    5: (-0.2 * math.pi, 0.1, 10.0, 2, 0.5, 1.0),
    6: (0.1 * math.pi, 40.0, 0.5, 1, 2.0, -2.0),
    7: (-0.05 * math.pi, 40.0, 5.0, 1, 6.0, 0.0),
}


def _g(x: list[float]) -> float:
    return 1.0 + 10.0 * (len(x) - 1) + sum(xi * xi - 10.0 * math.cos(4.0 * math.pi * xi) for xi in x[1:])


def _paper_values(k: int, x: list[float], a_j: tuple[float, ...] = (), b_j: tuple[float, ...] = ()) -> tuple[list[float], list[float]]:
    """F and G (= -c) of CTP``k`` at ``x``, as the paper writes them."""
    f1, g = x[0], _g(x)
    if k == 1:
        f2 = g * math.exp(-f1 / g)
        return [f1, f2], [-(f2 - a * math.exp(-b * f1)) for a, b in zip(a_j, b_j)]
    theta, a, b, c, d, e = _PAPER[k]
    f2 = g * (1.0 - f1 / g)
    lhs = math.cos(theta) * (f2 - e) - math.sin(theta) * f1
    rhs = a * abs(math.sin(b * math.pi * (math.sin(theta) * (f2 - e) + math.cos(theta) * f1) ** c)) ** d
    return [f1, f2], [-(lhs - rhs)]


_POINTS = [[0.5, 0.0, 0.0, 0.0, 0.0], [0.3, 0.05, 0.0, 0.0, 0.01], [0.9, 0.02, 0.0, -0.01, 0.0]]
# F and G at _POINTS, computed once with _paper_values.
_EXPECTED: dict[int, list[tuple[list[float], list[float]]]] = {
    1: [
        ([0.5, 0.6065306597126334], [0.0481686636358607, 0.02182314318980627]),
        ([0.3, 2.705836197375014], [-1.9762546708794668, -2.0392888390671686]),
        ([0.9, 0.7305103024286763], [-0.2033071256311828, -0.17210347390733827]),
    ],
    2: [
        ([0.5, 0.5], [0.11061587424546339]),
        ([0.3, 2.6912830431057455], [-1.3457801130328475]),
        ([0.9, 0.49352137556890907], [-0.09021314629077205]),
    ],
    3: [
        ([0.5, 0.5], [0.13302325200213763]),
        ([0.3, 2.6912830431057455], [-1.4446610894513838]),
        ([0.9, 0.49352137556890907], [-0.034110064125947304]),
    ],
    4: [
        ([0.5, 0.5], [0.2786712282479906]),
        ([0.3, 2.6912830431057455], [-0.7949782218053438]),
        ([0.9, 0.49352137556890907], [0.5193444510257695]),
    ],
    5: [
        ([0.5, 0.5], [0.1718534512559201]),
        ([0.3, 2.6912830431057455], [-1.449929379499986]),
        ([0.9, 0.49352137556890907], [-0.019387283342946857]),
    ],
    6: [
        ([0.5, 0.5], [32.00445713276677]),
        ([0.3, 2.6912830431057455], [2.170633568194498]),
        ([0.9, 0.49352137556890907], [10.166480002614454]),
    ],
    7: [
        ([0.5, 0.5], [-0.5638237836038166]),
        ([0.3, 2.6912830431057455], [22.457527847979673]),
        ([0.9, 0.49352137556890907], [-0.6267343069243536]),
    ],
}


def _evaluate(problem: CTPProblem, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    F, G = evaluate_population_with_constraints(problem, X)
    assert G is not None
    return np.asarray(F), np.asarray(G)


@pytest.mark.parametrize("k", range(1, 8))
def test_ctp_matches_the_values_of_the_papers_formulas_at_fixed_points(k: int) -> None:
    problem = make_problem_selection(f"ctp{k}").instantiate()
    F, G = _evaluate(problem, np.asarray(_POINTS))

    np.testing.assert_allclose(F, [row[0] for row in _EXPECTED[k]], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(G, [row[1] for row in _EXPECTED[k]], rtol=1e-12, atol=1e-14)
    a_j, b_j = ctp1_parameters(2)
    for x, (f, g) in zip(_POINTS, _EXPECTED[k]):
        paper_f, paper_g = _paper_values(k, x, a_j, b_j)
        np.testing.assert_allclose(paper_f, f, rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(paper_g, g, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("k", range(1, 8))
def test_ctp_agrees_with_the_papers_formulas_at_random_points(k: int) -> None:
    problem = CTP(k)
    rng = np.random.default_rng(k)
    X = problem.xl + rng.random((200, problem.n_var)) * (problem.xu - problem.xl)
    # Half of them near g = 1, where the constraints change sign.
    X[:100, 1:] *= 1e-3
    F, G = _evaluate(problem, X)

    expected = [_paper_values(k, list(x), problem.a_j, problem.b_j) for x in X]
    np.testing.assert_allclose(F, [row[0] for row in expected], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(G, [row[1] for row in expected], rtol=1e-10, atol=1e-10)
    assert np.any(np.all(G <= 0.0, axis=1)) and np.any(np.any(G > 0.0, axis=1))


@pytest.mark.parametrize("k", range(1, 8))
def test_ctp_evaluates_a_batch_as_its_rows(k: int) -> None:
    problem = CTP(k, n_var=7)
    X = problem.xl + np.random.default_rng(10 + k).random((64, 7)) * (problem.xu - problem.xl)
    F, G = _evaluate(problem, X)

    for i, x in enumerate(X):
        row_F, row_G = _evaluate(problem, x[None, :])
        np.testing.assert_array_equal(row_F[0], F[i])
        np.testing.assert_array_equal(row_G[0], G[i])


def test_ctp1_parameters_round_to_the_papers_table() -> None:
    # The paper prints, for J = 2, a = (0.858, 0.728) and b = (0.541, 0.295).
    a, b = ctp1_parameters(2)

    assert [round(v, 3) for v in a] == [0.858, 0.728]
    assert [round(v, 3) for v in b] == [0.541, 0.295]
    assert [round(v, 6) for v in a] == [0.858266, 0.728234]
    assert [round(v, 6) for v in b] == [0.541475, 0.295039]
    assert (CTP(1).a_j, CTP(1).b_j) == (a, b)


@pytest.mark.parametrize("J", [1, 2, 3, 5])
def test_ctp1_boundaries_follow_the_papers_procedure(J: int) -> None:
    # Each boundary a_j exp(-b_j f1) starts at (0, a_j), halfway between a_{j-1} and the previous boundary at
    # x_j = j / (J + 1), and passes through the previous boundary there (a_0 = b_0 = 1: the unconstrained front).
    a, b = ctp1_parameters(J)
    previous_a, previous_b = (1.0, *a[:-1]), (1.0, *b[:-1])

    assert len(a) == len(b) == J
    for j in range(J):
        x = (j + 1) / (J + 1)
        y = previous_a[j] * math.exp(-previous_b[j] * x)
        assert a[j] == pytest.approx(0.5 * (previous_a[j] + y), rel=1e-15)
        assert a[j] * math.exp(-b[j] * x) == pytest.approx(y, rel=1e-14)
    assert all(np.diff(a) < 0.0) and all(np.diff(b) < 0.0)


@pytest.mark.parametrize("J", [1, 3, 6])
def test_ctp1_takes_its_number_of_constraints(J: int) -> None:
    problem = CTP(1, n_constraints=J)
    _, G = _evaluate(problem, np.zeros((4, problem.n_var)))

    assert problem.n_constraints == J and G.shape == (4, J)
    assert (problem.a_j, problem.b_j) == ctp1_parameters(J)
    assert make_problem_selection("ctp1").instantiate().n_constraints == 2


@pytest.mark.parametrize("k", range(2, 8))
def test_ctp_generator_parameters_default_to_the_papers(k: int) -> None:
    problem = CTP(k)
    theta, a, b, c, d, e = _PAPER[k]

    assert PAPER_PARAMETERS[k] == _PAPER[k]
    assert (problem.theta, problem.a, problem.b, problem.c, problem.d, problem.e) == (theta, a, b, c, d, e)
    assert isinstance(problem.c, int) and problem.n_constraints == 1


def test_ctp_generator_takes_other_parameters() -> None:
    # CTP2 with CTP4's a, 0.75, is CTP4 with CTP2's d, 6.
    X = CTP(2).xl + np.random.default_rng(3).random((50, 5)) * (CTP(2).xu - CTP(2).xl)
    wider, same = CTP(2, a=0.75), CTP(4, d=6.0)

    for got, expected in zip(_evaluate(wider, X), _evaluate(same, X)):
        np.testing.assert_array_equal(got, expected)
    assert CTP(5, c=2.0).c == 2


@pytest.mark.parametrize(
    ("variant", "options", "message"),
    [
        (0, {}, "variants are 1 to 7"),
        (8, {}, "variants are 1 to 7"),
        (2, {"n_var": 1}, "at least 2"),
        (2, {"n_var": 2.5}, "at least 2"),
        (1, {"n_constraints": 0}, "positive integer"),
        (1, {"n_constraints": 1.5}, "positive integer"),
        (1, {"n_constraints": True}, "positive integer"),
        (1, {"theta": 0.1}, "no parameter theta"),
        (1, {"a": 0.1, "d": 2.0}, "no parameter a, d"),
        (3, {"n_constraints": 2}, "one constraint"),
        (5, {"c": 1.5}, "positive integer"),
        (5, {"c": 0}, "positive integer"),
        (2, {"a": -0.1}, "a must be at least 0"),
        (2, {"d": 0.0}, "d positive"),
        (2, {"theta": math.inf}, "finite"),
        (2, {"e": math.nan}, "finite"),
    ],
)
def test_ctp_rejects_invalid_parameters(variant: int, options: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CTPProblem(variant, **options)


@pytest.mark.parametrize("n_var", [2, 5, 10])
def test_ctp_bounds_x1_in_unit_interval_and_the_others_in_five(n_var: int) -> None:
    for k in range(1, 8):
        problem = make_problem_selection(f"ctp{k}", n_var=n_var).instantiate()

        assert (problem.n_var, problem.n_obj) == (n_var, 2)
        np.testing.assert_array_equal(problem.xl, [0.0] + [-5.0] * (n_var - 1))
        np.testing.assert_array_equal(problem.xu, [1.0] + [5.0] * (n_var - 1))


def test_ctp_registry_has_the_papers_dimensions() -> None:
    for k in range(1, 8):
        selection = make_problem_selection(f"ctp{k}")
        assert (selection.n_var, selection.n_obj) == (5, 2)
        assert selection.spec.option_defaults() == {}
        with pytest.raises(ValueError, match="fixed number of objectives"):
            make_problem_selection(f"ctp{k}", n_obj=3)
        with pytest.raises(ValueError, match="at least 2"):
            make_problem_selection(f"ctp{k}", n_var=1).instantiate()


def test_ctp_distance_function_is_zdt4s_rastrigin() -> None:
    # g = 1 + 10 (n - 1) + sum_{i>=2} (x_i^2 - 10 cos(4 pi x_i)), the g of ZDT4 (Deb, 1999), with minimum 1 at x_i = 0.
    rng = np.random.default_rng(5)
    X = rng.uniform(-5.0, 5.0, (500, 6))
    cosine_form = 1.0 + 10.0 * 5 + np.sum(X[:, 1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * X[:, 1:]), axis=1)

    np.testing.assert_allclose(rastrigin_g(X), cosine_form, rtol=1e-13)
    assert rastrigin_g(np.array([[0.7, 0.0, 0.0, 0.0, 0.0, 0.0]]))[0] == 1.0
    assert np.all(rastrigin_g(X) >= 1.0)
    # ZDT4's local optima: each x_i at a multiple of 1/2 adds x_i^2.
    assert rastrigin_g(np.array([[0.0, 0.5, -1.0, 0.0, 0.0, 0.0]]))[0] == pytest.approx(2.25, rel=1e-14)


@pytest.mark.parametrize("k", range(1, 8))
def test_ctp_with_g_one_is_on_the_unconstrained_front(k: int) -> None:
    problem = CTP(k)
    X = np.zeros((11, 5))
    X[:, 0] = np.linspace(0.0, 1.0, 11)
    F, _ = _evaluate(problem, X)

    expected = np.exp(-X[:, 0]) if k == 1 else 1.0 - X[:, 0]
    np.testing.assert_allclose(F, np.column_stack([X[:, 0], expected]), rtol=0.0, atol=1e-15)


def test_ctp_rejects_a_decision_matrix_of_another_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 5\)"):
        CTP(3).evaluate(np.zeros((2, 4)), {})
    with pytest.raises(ValueError, match=r"shape \(N, 5\)"):
        CTP(1).evaluate(np.zeros(5), {})


@pytest.mark.parametrize("key", ["ctp1", "ctp7"])
def test_ctp_runs_with_nsgaii_at_a_small_budget(key: str) -> None:
    result = optimize(key, algorithm="nsgaii", max_evaluations=36, pop_size=12, seed=7, engine="numpy")

    assert result.F is not None and result.F.shape[1] == 2 and result.F.shape[0] > 0
    G = np.asarray(result.data["G"])
    assert G.shape == (result.F.shape[0], make_problem_selection(key).instantiate().n_constraints)
    assert result.feasibility()["n_solutions"] == result.F.shape[0]
