"""Deterministic regression tests for real-valued mutation operators."""

import numpy as np
from numpy.testing import assert_allclose

from vamos.engine.operators.impl.real import (
    GaussianMutation,
    NonUniformMutation,
    PolynomialMutation,
    UniformMutation,
    UniformResetMutation,
    VariationWorkspace,
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


def test_polynomial_mutation_matches_reference_output():
    operator = PolynomialMutation(prob_mutation=0.7, eta=12.0, lower=LOWER, upper=UPPER)
    rng = np.random.default_rng(7)
    mutated = operator(POPULATION, rng)
    expected = np.array(
        [
            [-0.9133301035, 0.0, 1.5],
            [1.212905412, 4.6180787172, 3.0],
            [-3.3035316854, 1.0, 0.5],
            [0.1621495702, -1.4767732, 3.9463135163],
            [1.7901325422, -0.2726208852, 2.0217916808],
        ]
    )
    assert_allclose(mutated, expected)


def test_gaussian_mutation_matches_reference_output():
    operator = GaussianMutation(
        prob_mutation=0.6,
        sigma=np.array([0.1, 0.2, 0.05]),
        lower=LOWER,
        upper=UPPER,
    )
    rng = np.random.default_rng(8)
    mutated = operator(POPULATION, rng)
    # sigma is a fraction of the range of 10: the steps are 10 times those of an absolute sigma.
    expected = np.array(
        [
            [-0.3896488542, 0.0, 2.2200083627],
            [1.0, 2.5, 3.1811692966],
            [-2.241889733, -2.2788959249, 0.6800776161],
            [1.8815023005, -1.9794956984, 3.922349169],
            [0.7189717051, -4.1327913229, 2.0],
        ]
    )
    assert_allclose(mutated, expected)


def test_uniform_reset_mutation_resamples_within_bounds():
    operator = UniformResetMutation(prob_mutation=0.5, lower=LOWER, upper=UPPER)
    rng = np.random.default_rng(9)
    mutated = operator(POPULATION, rng)
    expected = np.array(
        [
            [-1.0, -1.8439651002, 1.5],
            [1.0, 2.5, 3.0],
            [-2.5, 1.0, 4.8623854928],
            [3.82862567, 4.1281005178, 2.082069141],
            [0.5449687726, -0.5, 2.0],
        ]
    )
    assert_allclose(mutated, expected)


def test_non_uniform_mutation_with_workspace_matches_reference_output():
    operator = NonUniformMutation(
        prob_mutation=0.6,
        perturbation=0.8,
        lower=LOWER,
        upper=UPPER,
        workspace=VariationWorkspace(),
    )
    rng = np.random.default_rng(10)
    mutated = operator(POPULATION, rng)
    # Each step is a fraction of the distance to the chosen bound, so every value stays in [-5, 5].
    expected = np.array(
        [
            [-1.0, 4.2750265253, 1.5],
            [-4.1638916368, 2.7336781097, 4.5477711327],
            [-2.5, 1.0, 0.6357633431],
            [2.0, -1.5, -2.4580983455],
            [2.2736075188, -0.5, 2.0],
        ]
    )
    assert_allclose(mutated, expected)


def test_uniform_mutation_can_leave_bounds_without_pipeline_repair():
    operator = UniformMutation(
        prob_mutation=1.0,
        perturb=1.0,
        lower=np.zeros(4, dtype=float),
        upper=np.ones(4, dtype=float),
        rng=np.random.default_rng(0),
    )
    mutated = operator(np.full(4, 0.95, dtype=float))
    assert np.any((mutated < 0.0) | (mutated > 1.0))


def test_non_uniform_mutation_stays_inside_bounds():
    # Michalewicz's step towards the chosen bound is a fraction of the distance to it.
    lower, upper = np.zeros(4), np.ones(4)
    for x in (0.5, 0.9):
        population = np.full((5000, 4), x)
        for workspace in (None, VariationWorkspace()):
            operator = NonUniformMutation(prob_mutation=1.0, perturbation=0.5, lower=lower, upper=upper, workspace=workspace)
            mutated = operator(population.copy(), np.random.default_rng(1))
            # The full-range step with a random sign left [0, 1] for 25% of the genes at 0.5 and 41% at 0.9.
            assert np.all((mutated >= 0.0) & (mutated <= 1.0)), (x, float(np.mean((mutated < 0.0) | (mutated > 1.0))))
            assert np.all(mutated != x)


def test_gaussian_and_cauchy_steps_scale_with_variable_range():
    from vamos.engine.operators.impl.real import CauchyMutation

    def median_step(operator_type, upper, **params):
        lower = np.zeros(3)
        centre = np.tile(upper / 2.0, (4000, 1))
        operator = operator_type(prob_mutation=1.0, lower=lower, upper=upper, **params)
        return float(np.median(np.abs(operator(centre.copy(), np.random.default_rng(2)) - centre)))

    for operator_type, params in ((GaussianMutation, {"sigma": 0.1}), (CauchyMutation, {"gamma": 0.1})):
        small = median_step(operator_type, np.ones(3), **params)
        large = median_step(operator_type, np.full(3, 1000.0), **params)
        # sigma and gamma are fractions of the range, as in pymoo; the absolute steps were the same for both.
        assert 900.0 < large / small < 1100.0, (operator_type.__name__, small, large)


def test_gaussian_mutation_has_default_sigma():
    from vamos.engine.operators.impl.registry import get_operator_registry

    operator = get_operator_registry().get("gaussian")(prob_mutation=0.5, lower=LOWER, upper=UPPER)
    assert float(operator.sigma) == 0.1
