"""Deterministic regression tests for real-valued crossover operators."""

import numpy as np
from numpy.testing import assert_allclose

from vamos.engine.operators.impl.real import (
    ArithmeticCrossover,
    BLXAlphaCrossover,
    DifferentialCrossover,
    PCXCrossover,
    SBXCrossover,
    SPXCrossover,
    UNDXCrossover,
)

PARENTS = np.array(
    [
        [[-1.0, 0.0, 1.5], [1.0, 2.5, 3.0]],
        [[-2.5, 1.0, 0.5], [2.0, -1.5, 4.0]],
        [[0.5, -0.5, 2.0], [-0.5, 0.25, -2.0]],
    ],
    dtype=float,
)

POPULATION = np.array(
    [
        [-1.0, 0.0, 1.5],
        [1.0, 2.5, 3.0],
        [-2.5, 1.0, 0.5],
        [2.0, -1.5, 4.0],
        [0.5, -0.5, 2.0],
    ],
    dtype=float,
)

LOWER = np.array([-5.0, -5.0, -5.0])
UPPER = np.array([5.0, 5.0, 5.0])


def test_sbx_crossover_matches_reference_output():
    operator = SBXCrossover(prob_crossover=1.0, eta=15.0, lower=LOWER, upper=UPPER)
    rng = np.random.default_rng(2)
    offspring = operator(PARENTS, rng)
    expected = np.array(
        [
            [[-0.99099557, 0.0, 1.5], [0.99099557, 2.5, 3.0]],
            [[2.04398263, -1.73268703, 4.05049481], [-2.543982, 1.23268703, 0.44941752]],
            [[0.5, -0.5, 1.95449037], [-0.5, 0.25, -1.95449037]],
        ]
    )
    assert_allclose(offspring, expected)


def test_blx_alpha_crossover_matches_reference_output():
    operator = BLXAlphaCrossover(alpha=0.35, prob_crossover=1.0, lower=LOWER, upper=UPPER)
    rng = np.random.default_rng(3)
    offspring = operator(PARENTS, rng)
    expected = np.array(
        [
            [[0.2793509226, -0.4749532705, 2.0794736976], [-0.2358647306, 1.6188939286, 2.8564863576]],
            [[-0.4102575692, -1.6961096128, 3.6457340509], [3.2404444995, -1.1671450541, 3.1338558821]],
            [[-0.6567575661, -0.2636840571, 0.1138332418], [0.3335671943, -0.389281045, -3.3898674321]],
        ]
    )
    assert_allclose(offspring, expected)


def test_blx_alpha_crossover_can_leave_bounds_without_pipeline_repair():
    operator = BLXAlphaCrossover(
        alpha=1.0,
        prob_crossover=1.0,
        lower=np.array([0.0]),
        upper=np.array([1.0]),
    )
    rng = np.random.default_rng(0)
    offspring = operator(np.array([[[0.0], [1.0]]], dtype=float), rng)
    assert np.any((offspring < 0.0) | (offspring > 1.0))


def test_arithmetic_crossover():
    operator = ArithmeticCrossover(prob_crossover=1.0)
    rng = np.random.default_rng(4)
    offspring = operator(PARENTS, rng)
    expected = np.array(
        [
            [[0.83832795, 2.29790994, 2.87874596], [-0.83832795, 0.20209006, 1.62125404]],
            [[-0.73310124, 0.01838958, 1.87425459], [0.23310124, -0.51838958, 2.62574541]],
            [[-0.12351342, -0.03236494, -0.49405366], [0.12351342, -0.21763506, 0.49405366]],
        ]
    )
    assert_allclose(offspring, expected)


def test_differential_crossover_matches_reference_output():
    operator = DifferentialCrossover(F=0.6, CR=0.8, lower=LOWER, upper=UPPER)
    rng = np.random.default_rng(5)
    trial = operator(POPULATION, rng)
    expected = np.array(
        [
            [-1.9, 0.6, 0.3],
            [1.9, 1.9, 3.0],
            [-4.3, 1.0, 0.5],
            [3.8, -2.4, 4.9],
            [-2.2, 1.0, -0.1],
        ]
    )
    assert_allclose(trial, expected)


def test_differential_crossover_can_leave_bounds_without_pipeline_repair():
    operator = DifferentialCrossover(
        F=1.0,
        CR=1.0,
        lower=np.array([0.0]),
        upper=np.array([1.0]),
    )
    rng = np.random.default_rng(0)
    trial = operator(np.array([[0.9], [1.0], [0.0]], dtype=float), rng)
    assert np.any((trial < 0.0) | (trial > 1.0))


def _three_parents(n_var: int = 10, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.2, 0.8, size=(3, n_var))


def _perpendicular(vectors: np.ndarray, direction: np.ndarray) -> np.ndarray:
    unit = direction / np.linalg.norm(direction)
    return vectors - np.outer(vectors @ unit, unit)


def test_pcx_offspring_follow_deb_distribution():
    """PCX of Deb, Anand and Joshi (2002), as pymoo and Deb's C code implement it."""
    n_var, n_groups = 10, 20_000
    parents = _three_parents(n_var)
    lower, upper = np.full(n_var, -10.0), np.full(n_var, 10.0)
    operator = PCXCrossover(sigma_eta=0.1, sigma_zeta=0.1, lower=lower, upper=upper)
    offspring = operator(np.repeat(parents[None], n_groups, axis=0), np.random.default_rng(3))
    assert offspring.shape == (n_groups, 3, n_var)
    centroid = parents.mean(axis=0)
    for j in range(3):
        # Child j is centred on its own index parent.
        d = centroid - parents[j]
        step = offspring[:, j, :] - parents[j]
        along = step @ d / (d @ d)
        assert abs(float(along.mean())) < 0.005, f"child {j}: mean component along d {along.mean():+.4f}"
        assert abs(float(along.std()) / 0.1 - 1.0) < 0.05
        # Orthogonal noise spans all n - 1 directions with scale sigma_eta * D, where D is the
        # mean distance of the other parents to the line through the index parent along d.
        others = np.delete(parents, j, axis=0) - parents[j]
        mean_distance = float(np.linalg.norm(_perpendicular(others, d), axis=1).mean())
        orth = _perpendicular(step, d)
        expected = (n_var - 1) * (0.1 * mean_distance) ** 2
        assert abs(float((orth**2).sum(axis=1).mean()) / expected - 1.0) < 0.05
        assert np.linalg.matrix_rank(step[:200] - step[:200].mean(axis=0)) == n_var


def test_spx_preserves_parent_covariance():
    """SPX of Tsutsui, Yamamura and Higuchi (1999): uniform in the simplex expanded by sqrt(m + 1)."""
    n_var, n_groups = 10, 20_000
    parents = _three_parents(n_var, seed=11)
    lower, upper = np.full(n_var, -10.0), np.full(n_var, 10.0)
    operator = SPXCrossover(lower=lower, upper=upper)
    offspring = operator(np.repeat(parents[None], n_groups, axis=0), np.random.default_rng(2)).reshape(-1, n_var)
    centroid = parents.mean(axis=0)
    parent_trace = np.trace(np.cov(parents.T, bias=True))
    ratio = np.trace(np.cov(offspring.T, bias=True)) / parent_trace
    assert abs(float(ratio) - 1.0) < 0.05, f"children/parents covariance trace ratio {ratio:.3f}"
    assert np.abs(offspring.mean(axis=0) - centroid).max() < 0.01
    # Every child lies in the plane of the parents.
    basis = parents[1:] - parents[0]
    residual = offspring - parents[0] - (offspring - parents[0]) @ np.linalg.pinv(basis) @ basis
    assert np.abs(residual).max() < 1e-9


def test_undx_matches_reference_moments():
    """UNDX of Ono and Kobayashi (1997) and Kita, Ono and Kobayashi (1999)."""
    n_var, n_groups, alpha, beta = 10, 20_000, 0.5, 0.35
    p1, p2, p3 = _three_parents(n_var, seed=5)
    lower, upper = np.full(n_var, -10.0), np.full(n_var, 10.0)
    operator = UNDXCrossover(prob_crossover=1.0, zeta=alpha, eta=beta, lower=lower, upper=upper)
    middle, d = 0.5 * (p1 + p2), p2 - p1
    offspring = operator(np.repeat(np.stack([p1, p2, p3])[None], n_groups, axis=0), np.random.default_rng(4))
    step = offspring.reshape(-1, n_var) - middle
    along = step @ d / (d @ d)
    assert abs(float(along.mean())) < 0.01
    assert abs(float(along.std()) / alpha - 1.0) < 0.05, f"std along d {along.std():.3f}|d|, expected {alpha}|d|"
    distance = float(np.linalg.norm(_perpendicular((p3 - middle)[None], d)))
    expected = distance**2 * (n_var - 1) * beta**2 / n_var
    orth = _perpendicular(step, d)
    assert abs(float((orth**2).sum(axis=1).mean()) / expected - 1.0) < 0.05
    # The two children of a group are symmetric about the middle point.
    assert_allclose(offspring[:, 0, :] + offspring[:, 1, :], np.broadcast_to(2 * middle, (n_groups, n_var)), atol=1e-12)

    # With the third parent on the line p1-p2 there is no orthogonal component.
    p3_on_line = middle + 0.3 * d
    on_line = operator(np.repeat(np.stack([p1, p2, p3_on_line])[None], 500, axis=0), np.random.default_rng(5))
    orth_on_line = _perpendicular(on_line.reshape(-1, n_var) - middle, d)
    assert np.abs(orth_on_line).max() < 1e-12 * np.linalg.norm(d) * 10
