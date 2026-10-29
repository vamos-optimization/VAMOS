import numpy as np
import pytest

from vamos.engine.operators.impl.real.mutation import LinkedPolynomialMutation, PowerLawMutation, UniformMutation


def _recover_polynomial_draws(x, y, lower, upper, eta):
    """Invert the bounded polynomial mutation of Deb (as in jMetal) to recover each gene's ``u``."""
    span = upper - lower
    delta1 = (x - lower) / span
    delta2 = (upper - x) / span
    deltaq = (y - x) / span
    power = eta + 1.0
    u = np.empty_like(x)
    low = deltaq <= 0.0
    a = (1.0 - delta1[low]) ** power
    u[low] = ((deltaq[low] + 1.0) ** power - a) / (2.0 * (1.0 - a))
    b = (1.0 - delta2[~low]) ** power
    u[~low] = ((2.0 - b) - (1.0 - deltaq[~low]) ** power) / (2.0 * (1.0 - b))
    return u


def test_uniform_mutation_handles_vector_input():
    rng = np.random.default_rng(3)
    lower = np.zeros(4)
    upper = np.ones(4)
    mut = UniformMutation(prob_mutation=1.0, perturb=0.2, lower=lower, upper=upper, rng=rng)
    x = np.full(4, 0.5)
    y = mut(x)
    assert y.shape == x.shape


def test_linked_polynomial_mutation_shares_one_draw_across_a_vector():
    rng = np.random.default_rng(4)
    lower = np.zeros(3)
    upper = np.ones(3)
    mut = LinkedPolynomialMutation(prob_mutation=1.0, eta=20.0, lower=lower, upper=upper, rng=rng)
    x = np.array([0.2, 0.4, 0.6])
    y = mut(x)
    assert y.shape == x.shape
    u = _recover_polynomial_draws(x, y, lower, upper, 20.0)
    # Every mutated variable of the solution uses the same random draw.
    assert np.allclose(u, u[0], atol=1e-8)


def test_uniform_mutation_handles_population_input():
    rng = np.random.default_rng(5)
    lower = np.zeros(3)
    upper = np.ones(3)
    mut = UniformMutation(prob_mutation=1.0, perturb=0.3, lower=lower, upper=upper, rng=rng)
    x = np.full((2, 3), 0.5)
    y = mut(x)
    assert y.shape == x.shape


def test_linked_polynomial_mutation_draws_one_delta_per_individual():
    rng = np.random.default_rng(6)
    n_ind, n_var, eta = 400, 4, 5.0
    lower = np.array([0.0, -1.0, 2.0, 0.0])
    upper = np.array([1.0, 1.0, 5.0, 10.0])
    # Distinct rows that reach close to both bounds of every variable.
    x = lower + (upper - lower) * rng.uniform(0.01, 0.99, size=(n_ind, n_var))
    mut = LinkedPolynomialMutation(prob_mutation=1.0, eta=eta, lower=lower, upper=upper)
    y = mut(x, rng)

    # The operator itself keeps every child inside the bounds, before any repair.
    assert np.all((y >= lower) & (y <= upper))
    step = (y - x) / (upper - lower)
    assert np.unique(np.round(step[:, 0], 12)).size == n_ind

    # One draw per individual, shared by all its variables, and different between individuals.
    u = _recover_polynomial_draws(x, y, lower, upper, eta)
    assert np.allclose(u, u[:, :1], atol=1e-7)
    assert np.unique(np.round(u[:, 0], 6)).size > 0.95 * n_ind
    assert 0.4 < float(np.mean(u[:, 0] <= 0.5)) < 0.6


def test_power_law_mutation_stays_inside_bounds():
    rng = np.random.default_rng(7)
    n = 20_000
    lower = np.zeros(1)
    upper = np.ones(1)
    for index in (0.25, 1.0, 4.0, 10.0):
        mut = PowerLawMutation(prob_mutation=1.0, index=index, lower=lower, upper=upper)
        y = mut(np.full((n, 1), 0.5), rng)
        assert np.all((y >= 0.0) & (y <= 1.0)), f"index={index}: {np.mean((y < 0.0) | (y > 1.0)):.1%} out of bounds"
        at_bounds = float(np.mean((y == 0.0) | (y == 1.0)))
        assert at_bounds < 1e-3, f"index={index}: {at_bounds:.1%} of the genes end on a bound"


@pytest.mark.parametrize("x0", [0.1, 0.3, 0.5, 0.9])
def test_power_law_mutation_is_unbiased_and_shrinks_with_index(x0):
    rng = np.random.default_rng(8)
    n = 200_000
    lower = np.array([-2.0])
    upper = np.array([3.0])
    x = np.full((n, 1), lower + x0 * (upper - lower))
    mean_steps = []
    for index in (0.25, 1.0, 4.0):
        y = PowerLawMutation(prob_mutation=1.0, index=index, lower=lower, upper=upper)(x, rng)
        step = (y - x) / (upper - lower)
        # The step towards a bound is scaled by the distance to that bound, so the expected
        # displacement is zero in the interior of the range.
        assert abs(float(np.mean(step))) < 0.005
        mean_steps.append(float(np.mean(np.abs(step))))
    # A larger index concentrates the strength s = u**index near 0.
    assert mean_steps[0] > mean_steps[1] > mean_steps[2]
