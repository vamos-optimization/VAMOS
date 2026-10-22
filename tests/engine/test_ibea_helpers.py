import numpy as np
import pytest

from vamos.engine.algorithm.ibea.helpers import (
    compute_indicator_matrix,
    environmental_selection,
    epsilon_indicator,
    hypervolume_indicator,
    ibea_fitness,
)
from vamos.foundation.quality_indicators.hypervolume import hypervolume


def test_epsilon_indicator_matches_jmetal_definition() -> None:
    F = np.array([[0.0, 0.0], [1.0, 2.0]])
    ind = epsilon_indicator(F)

    assert ind.shape == (2, 2)
    assert np.isclose(ind[0, 1], 2.0)
    assert np.isclose(ind[1, 0], -1.0)


def _hypervolume_indicator_reference(F: np.ndarray) -> np.ndarray:
    # Entry (i, j) is I_HD(x_j, x_i) of Zitzler and Kuenzli (2004), as in jMetal's IBEA: the
    # fitness of x_i sums -exp(-I(x_j, x_i) / kappa) over its row.
    ref = np.max(F, axis=0) + 1.0
    indicator = np.zeros((F.shape[0], F.shape[0]), dtype=float)
    for i in range(F.shape[0]):
        for j in range(F.shape[0]):
            if i == j:
                continue
            hv_i, hv_j = hypervolume(F[i : i + 1], ref), hypervolume(F[j : j + 1], ref)
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                indicator[i, j] = hv_i - hv_j  # x_j dominates x_i.
            else:
                indicator[i, j] = hypervolume(np.vstack([F[i], F[j]]), ref) - hv_j  # dominated by x_i only.
    return indicator


def test_hypervolume_indicator_matches_reference_pairwise_definition() -> None:
    F = np.array(
        [
            [0.3, 0.9],
            [0.6, 0.5],
            [0.8, 0.2],
            [0.4, 0.7],
            [0.7, 0.6],  # Dominated by [0.6, 0.5].
        ],
        dtype=float,
    )

    expected = _hypervolume_indicator_reference(F)
    actual = hypervolume_indicator(F)

    np.testing.assert_allclose(actual, expected)


def test_hypervolume_indicator_is_normalized_for_fitness() -> None:
    F = np.random.default_rng(5).random((12, 3))
    rescaled = F * np.array([1.0, 100.0, 1.0e4]) + np.array([0.0, -50.0, 3.0])

    indicator = compute_indicator_matrix(F, "hypervolume")

    np.testing.assert_allclose(compute_indicator_matrix(rescaled, "hypervolume"), indicator, atol=1e-12)
    assert np.max(np.abs(indicator)) == pytest.approx(1.0)


@pytest.mark.parametrize("indicator", ["epsilon", "hypervolume"])
def test_ibea_converges_on_zdt1_with_either_indicator(indicator: str) -> None:
    import dataclasses

    import vamos
    from vamos.engine.algorithm.config import IBEAConfig

    config = dataclasses.replace(IBEAConfig.default(pop_size=20, n_var=10), indicator=indicator, kappa=0.05)
    result = vamos.optimize(
        vamos.make_problem_selection("zdt1", n_var=10).instantiate(),
        algorithm="ibea",
        algorithm_config=config,
        max_evaluations=3000,
        seed=0,
        engine="numpy",
    )
    ref = np.array([1.1, 1.1])
    F = np.asarray(result.F)
    # The true front scores about 0.871; before the hypervolume indicator was reoriented, it scored 0.
    assert hypervolume(F[np.all(F <= ref, axis=1)], ref) > 0.7


@pytest.mark.parametrize("indicator", ["epsilon", "hypervolume"])
def test_selection_removes_the_dominated_solution(indicator: str) -> None:
    F = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, -0.5]])  # Row 1 is dominated by both others.
    _, F_kept, _, _ = environmental_selection(np.arange(3, dtype=float)[:, None], F, None, 2, indicator, 1.0)

    np.testing.assert_array_equal(F_kept, F[[0, 2]])


# (0, 0) and (0.5, -0.5) are mutually non-dominated; (1000, 1000) is dominated by both by a margin
# whose exponential contribution exp(-I / kappa) exceeds the float64 range.
_FAR_DOMINATED = np.array([[0.0, 0.0], [1000.0, 1000.0], [0.5, -0.5]])


@pytest.mark.parametrize("indicator", ["epsilon", "hypervolume"])
def test_selection_removes_a_solution_dominated_beyond_the_exponential_range(indicator: str) -> None:
    X = np.arange(3, dtype=float)[:, None]
    _, F_kept, _, fitness = environmental_selection(X, _FAR_DOMINATED, None, 2, indicator, 1.0)

    assert not any(np.array_equal(row, [1000.0, 1000.0]) for row in F_kept)
    assert np.isfinite(fitness).all()


@pytest.mark.parametrize("indicator", ["epsilon", "hypervolume"])
def test_fitness_ranks_a_solution_dominated_beyond_the_exponential_range_worst(indicator: str) -> None:
    fitness = ibea_fitness(compute_indicator_matrix(_FAR_DOMINATED, indicator), 1.0)

    assert int(np.argmin(fitness)) == 1
    assert np.isfinite(fitness).all()


def _selection_without_rescaling(F: np.ndarray, pop_size: int, indicator: str, kappa: float):
    contrib = np.exp(-compute_indicator_matrix(F, indicator) / kappa)
    np.fill_diagonal(contrib, 0.0)
    fitness = -np.sum(contrib, axis=1)
    active = np.arange(F.shape[0])
    while active.size > pop_size:
        worst_pos = int(np.argmin(fitness[active]))
        delta = contrib[active, active[worst_pos]].copy()
        delta[worst_pos] = 0.0
        fitness[active] += delta
        active = np.delete(active, worst_pos)
    return active, fitness[active]


@pytest.mark.parametrize("indicator", ["epsilon", "hypervolume"])
def test_selection_without_exponential_overflow_is_unchanged(indicator: str) -> None:
    F = np.random.default_rng(11).random((30, 3))
    X = np.arange(30, dtype=float)[:, None]
    assert np.max(-compute_indicator_matrix(F, indicator) / 0.05) < 700.0  # far from the float64 exponential limit
    expected_active, expected_fitness = _selection_without_rescaling(F, 15, indicator, 0.05)

    X_kept, _, _, fitness = environmental_selection(X, F, None, 15, indicator, 0.05)

    assert np.array_equal(X_kept[:, 0].astype(int), expected_active)
    assert np.array_equal(fitness, expected_fitness)


def _exact_selection(F: np.ndarray, G: np.ndarray | None, pop_size: int, kappa: float) -> list[int]:
    """IBEA's removal loop with the epsilon indicator in exact high-precision arithmetic, which never overflows."""
    from decimal import Decimal, localcontext

    from vamos.foundation.constraints.utils import compute_violation, is_feasible

    ind = compute_indicator_matrix(F, "epsilon")
    with localcontext() as ctx:
        # Contributions span exp(-max|I| / kappa) to exp(max|I| / kappa); enough digits keep every term and the
        # penalty's "+ 1" through all additions and removals.
        ctx.prec = int(2.0 * np.max(np.abs(ind)) / kappa / np.log(10.0)) + 60
        n = F.shape[0]
        contrib = [
            [Decimal(0) if i == j else (-Decimal(repr(float(ind[i, j]))) / Decimal(repr(kappa))).exp() for j in range(n)] for i in range(n)
        ]
        fitness = [-sum(row, Decimal(0)) for row in contrib]
        if G is not None and not is_feasible(G).all():
            infeasible = np.flatnonzero(~is_feasible(G))
            _, rank = np.unique(compute_violation(G)[infeasible], return_inverse=True)
            penalty = max(abs(value) for value in fitness) + 1
            for row, r in zip(infeasible, rank):
                fitness[row] -= penalty * (1 + int(r))
        active = list(range(n))
        while len(active) > pop_size:
            worst = min(active, key=lambda k: (fitness[k], k))
            active.remove(worst)
            for k in active:
                fitness[k] += contrib[k][worst]
        return active


@pytest.mark.parametrize(
    "F, G, pop_size, expected",
    [
        # Shifting every exponent by the same amount underflowed the 25-versus-0 contribution, so 0 and 25 tied
        # once 100 was removed and the dominated 25 survived.
        (np.array([[0.0], [25.0], [100.0]]), None, 1, [0]),
        # A rescaled contribution near exp(600) absorbed the penalty's "+ 1", so the feasible 100 tied with the
        # penalized infeasible 0 and was removed instead.
        (np.array([[100.0], [0.0], [50.0]]), np.array([[-1.0], [1.0], [-1.0]]), 2, [0, 2]),
        # Rows 1 and 2 share the leading exponent 1640, so their float log-sums tie; only the lower-order terms
        # (exp(-1480) versus exp(0)) show that row 2, which row 1 dominates, has the larger sum.
        (np.array([[14.0, -88.0], [96.0, 21.0], [96.0, 95.0]]), None, 2, [0, 1]),
    ],
)
def test_selection_beyond_the_exponential_range_matches_exact_arithmetic(F, G, pop_size, expected) -> None:
    X = np.arange(F.shape[0], dtype=float)[:, None]
    X_kept, _, _, fitness = environmental_selection(X, F, G, pop_size, "epsilon", 0.05)

    assert sorted(X_kept[:, 0].astype(int).tolist()) == expected == sorted(_exact_selection(F, G, pop_size, 0.05))
    assert np.isfinite(fitness).all()


@pytest.mark.parametrize("seed", range(6))
def test_wide_objective_ranges_select_like_exact_arithmetic(seed: int) -> None:
    rng = np.random.default_rng(seed)
    # Objective ranges up to 50 give exponents up to 1000 with kappa=0.05, beyond float64's exp range.
    F = rng.random((10, 2)) * rng.choice([1.0, 10.0, 50.0], size=2)
    G = (rng.random((10, 1)) - 0.7) if seed % 2 else None
    X = np.arange(10, dtype=float)[:, None]

    X_kept, _, _, _ = environmental_selection(X, F, G, 4, "epsilon", 0.05)

    assert sorted(X_kept[:, 0].astype(int).tolist()) == sorted(_exact_selection(F, G, 4, 0.05))


@pytest.mark.parametrize("seed", range(4))
def test_tied_leading_exponents_select_like_exact_arithmetic(seed: int) -> None:
    # Integer objectives make many contributions share exactly the same exponent.
    rng = np.random.default_rng(100 + seed)
    F = rng.integers(-40, 41, size=(8, 2)).astype(float)
    X = np.arange(8, dtype=float)[:, None]

    X_kept, _, _, _ = environmental_selection(X, F, None, 3, "epsilon", 0.05)

    assert sorted(X_kept[:, 0].astype(int).tolist()) == sorted(_exact_selection(F, None, 3, 0.05))
