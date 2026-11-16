import numpy as np

from vamos.engine.operators.impl.binary import (
    bit_flip_mutation,
    one_point_crossover,
    random_binary_population,
    segment_inversion_mutation,
)
from vamos.engine.operators.impl.integer import (
    boundary_integer_mutation,
    gaussian_integer_mutation,
    random_integer_population,
    random_reset_mutation,
    uniform_integer_crossover,
)
from vamos.engine.operators.impl.mixed import mixed_crossover, mixed_initialize, mixed_mutation
from vamos.engine.operators.impl.permutation import alternating_edges_crossover, two_opt_mutation


def test_binary_operators_shape_and_values():
    rng = np.random.default_rng(0)
    X = random_binary_population(5, 8, rng)
    assert X.shape == (5, 8)
    assert set(np.unique(X)).issubset({0, 1})

    parents = np.vstack([X, X]).reshape(-1, 2, 8)
    children = one_point_crossover(parents.reshape(-1, 8), prob=1.0, rng=rng)
    assert children.shape == parents.reshape(-1, 8).shape

    pre = children.copy()
    bit_flip_mutation(children, prob=1.0, rng=rng)
    assert not np.array_equal(children, pre)
    assert set(np.unique(children)).issubset({0, 1})


def test_binary_segment_inversion_mutation_flips_segments():
    rng = np.random.default_rng(4)
    X = random_binary_population(6, 10, rng)
    before = X.copy()
    segment_inversion_mutation(X, prob=1.0, rng=rng)
    assert X.shape == before.shape
    assert np.any(X != before)
    assert set(np.unique(X)).issubset({0, 1})


def test_integer_operators_respect_bounds():
    rng = np.random.default_rng(1)
    lower = np.array([0, 5, 10])
    upper = np.array([3, 7, 12])
    X = random_integer_population(4, 3, lower, upper, rng)
    assert X.shape == (4, 3)
    assert np.all(X >= lower) and np.all(X <= upper)

    parents = np.vstack([X, X]).reshape(-1, 2, 3)
    children = uniform_integer_crossover(parents.reshape(-1, 3), prob=1.0, rng=rng)
    assert children.shape == parents.reshape(-1, 3).shape

    random_reset_mutation(children, prob=1.0, lower=lower, upper=upper, rng=rng)
    assert np.all(children >= lower) and np.all(children <= upper)


def test_integer_gaussian_and_boundary_mutations_respect_bounds():
    rng = np.random.default_rng(5)
    lower = np.array([0, 2, 5], dtype=np.int32)
    upper = np.array([4, 6, 9], dtype=np.int32)
    X = random_integer_population(5, 3, lower, upper, rng)

    gaussian_integer_mutation(X, prob=1.0, sigma=1.25, lower=lower, upper=upper, rng=rng)
    assert np.all(X >= lower) and np.all(X <= upper)

    boundary_integer_mutation(X, prob=1.0, lower=lower, upper=upper, rng=rng)
    assert np.all(X >= lower) and np.all(X <= upper)
    assert np.all((X == lower) | (X == upper))


def test_mixed_initialize_and_variation():
    rng = np.random.default_rng(2)
    spec = {
        "real_idx": np.array([0, 1]),
        "int_idx": np.array([2]),
        "cat_idx": np.array([3]),
        "real_lower": np.array([0.0, -1.0]),
        "real_upper": np.array([1.0, 1.0]),
        "int_lower": np.array([0]),
        "int_upper": np.array([3]),
        "cat_cardinality": np.array([4]),
    }
    X = mixed_initialize(6, 4, spec, rng)
    assert X.shape == (6, 4)
    # Real within bounds
    assert np.all(X[:, spec["real_idx"]] >= spec["real_lower"])
    assert np.all(X[:, spec["real_idx"]] <= spec["real_upper"])
    # Int within bounds
    assert np.all(X[:, spec["int_idx"]] >= spec["int_lower"])
    assert np.all(X[:, spec["int_idx"]] <= spec["int_upper"])
    # Categories within range
    assert np.all((X[:, spec["cat_idx"]] >= 0) & (X[:, spec["cat_idx"]] < spec["cat_cardinality"]))

    parents = np.vstack([X[:4], X[:4]]).reshape(-1, 2, 4)
    children = mixed_crossover(parents.reshape(-1, 4), prob=1.0, spec=spec, rng=rng)
    assert children.shape == parents.reshape(-1, 4).shape
    mixed_mutation(children, prob=1.0, spec=spec, rng=rng)
    assert np.all(children[:, spec["real_idx"]] >= spec["real_lower"])
    assert np.all(children[:, spec["real_idx"]] <= spec["real_upper"])
    assert np.all(children[:, spec["int_idx"]] >= spec["int_lower"])
    assert np.all(children[:, spec["int_idx"]] <= spec["int_upper"])
    assert np.all((children[:, spec["cat_idx"]] >= 0) & (children[:, spec["cat_idx"]] < spec["cat_cardinality"]))


def test_mixed_with_permutation_segment():
    rng = np.random.default_rng(3)
    perm_idx = np.array([0, 1, 2, 3, 4])
    spec = {
        "perm_idx": perm_idx,
        "int_idx": np.array([5]),
        "real_idx": np.array([6, 7]),
        "real_lower": np.array([-1.0, 0.0]),
        "real_upper": np.array([1.0, 2.0]),
        "int_lower": np.array([1]),
        "int_upper": np.array([3]),
        "perm_crossover": "ox",
        "perm_mutation": "swap",
    }
    n_var = 8
    X = mixed_initialize(6, n_var, spec, rng)
    assert X.shape == (6, n_var)

    expected = set(range(perm_idx.size))
    for row in X[:, perm_idx]:
        assert set(row.astype(int)) == expected

    parents = np.vstack([X[:4], X[:4]]).reshape(-1, 2, n_var)
    children = mixed_crossover(parents.reshape(-1, n_var), prob=1.0, spec=spec, rng=rng)
    assert children.shape == parents.reshape(-1, n_var).shape

    for row in children[:, perm_idx]:
        assert set(row.astype(int)) == expected

    mixed_mutation(children, prob=1.0, spec=spec, rng=rng)
    for row in children[:, perm_idx]:
        assert set(row.astype(int)) == expected
    assert np.all(children[:, spec["int_idx"]] >= spec["int_lower"])
    assert np.all(children[:, spec["int_idx"]] <= spec["int_upper"])


def test_permutation_aex_and_two_opt_keep_valid_permutations():
    rng = np.random.default_rng(6)
    n_var = 8
    parents = np.vstack([rng.permutation(n_var) for _ in range(6)]).astype(np.int32, copy=False)
    children = alternating_edges_crossover(parents, prob=1.0, rng=rng)
    assert children.shape == parents.shape
    expected = np.arange(n_var)
    assert all(np.array_equal(np.sort(row), expected) for row in children)

    two_opt_mutation(children, prob=1.0, rng=rng)
    assert children.shape == parents.shape
    assert all(np.array_equal(np.sort(row), expected) for row in children)


def test_mixed_segment_specific_probabilities_override_global_probability():
    rng = np.random.default_rng(7)
    perm_idx = np.array([0, 1, 2, 3], dtype=int)
    int_idx = np.array([4, 5], dtype=int)
    spec = {
        "perm_idx": perm_idx,
        "int_idx": int_idx,
        "int_lower": np.array([0, 0], dtype=int),
        "int_upper": np.array([9, 9], dtype=int),
        "perm_crossover": "ox",
        "perm_mutation": "swap",
        "perm_crossover_prob": 0.0,
        "int_crossover_prob": 1.0,
        "perm_mutation_prob": 0.0,
        "int_mutation_prob": 1.0,
        "int_mutation": "reset",
    }
    parents = np.array(
        [
            [0, 1, 2, 3, 1, 8],
            [3, 2, 1, 0, 7, 2],
            [1, 0, 3, 2, 2, 6],
            [2, 3, 0, 1, 8, 1],
        ],
        dtype=float,
    )
    children = mixed_crossover(parents, prob=0.0, spec=spec, rng=rng)
    assert np.array_equal(children[:, perm_idx], parents[:, perm_idx])
    assert np.any(children[:, int_idx] != parents[:, int_idx])

    before_mutation = children.copy()
    mixed_mutation(children, prob=0.0, spec=spec, rng=rng)
    assert np.array_equal(children[:, perm_idx], before_mutation[:, perm_idx])
    assert np.any(children[:, int_idx] != before_mutation[:, int_idx])
    assert np.all(children[:, int_idx] >= spec["int_lower"])
    assert np.all(children[:, int_idx] <= spec["int_upper"])


def test_mixed_integer_segment_supports_arithmetic_crossover_and_creep_mutation():
    rng = np.random.default_rng(11)
    spec = {
        "perm_idx": np.array([0, 1, 2], dtype=int),
        "int_idx": np.array([3, 4], dtype=int),
        "int_lower": np.array([0, 0], dtype=int),
        "int_upper": np.array([10, 10], dtype=int),
        "perm_crossover": "ox",
        "perm_mutation": "swap",
        "perm_crossover_prob": 0.0,
        "int_crossover_prob": 1.0,
        "int_crossover": "arithmetic",
        "perm_mutation_prob": 0.0,
        "int_mutation_prob": 1.0,
        "int_mutation": "creep",
        "int_mutation_step": 2,
    }
    parents = np.array(
        [
            [0, 1, 2, 2, 8],
            [2, 1, 0, 8, 2],
        ],
        dtype=float,
    )
    children = mixed_crossover(parents, prob=1.0, spec=spec, rng=rng)
    assert np.array_equal(children[:, :3], parents[:, :3])
    # Complementary rounded children of the arithmetic crossover, between the parents' values.
    assert np.array_equal(children[0, 3:] + children[1, 3:], parents[0, 3:] + parents[1, 3:])
    assert np.all((children[:, 3:] >= 2.0) & (children[:, 3:] <= 8.0))

    before_mutation = children.copy()
    mixed_mutation(children, prob=1.0, spec=spec, rng=rng)
    assert np.array_equal(children[:, :3], before_mutation[:, :3])
    deltas = np.abs(children[:, 3:] - before_mutation[:, 3:])
    assert np.array_equal(deltas, np.full_like(deltas, 2.0))


def test_integer_sbx_is_unbiased_away_from_bounds():
    from vamos.engine.operators.impl.integer import integer_sbx_crossover

    # Parents 3 and 4 in [0, 10]: SBX spreads the children symmetrically around 3.5, far from both bounds.
    parents = np.tile(np.array([[3.0], [4.0]]), (20_000, 1))
    children = integer_sbx_crossover(
        parents, prob=1.0, eta=20.0, lower=np.array([0.0]), upper=np.array([10.0]), rng=np.random.default_rng(0)
    )
    assert np.all(children == np.rint(children))
    # Truncating with int() moved the mean to 3.25.
    assert abs(float(children.mean()) - 3.5) < 0.02, float(children.mean())


def test_integer_arithmetic_crossover_children_are_complementary():
    from vamos.engine.operators.impl.integer import arithmetic_integer_crossover

    p1, p2 = np.array([1, 2, 0, 9]), np.array([3, 6, 7, 1])
    parents = np.tile(np.stack([p1, p2]), (500, 1))
    children = arithmetic_integer_crossover(parents, prob=1.0, rng=np.random.default_rng(1), lower=np.zeros(4), upper=np.full(4, 10))
    first, second = children[0::2], children[1::2]
    # lambda p1 + (1 - lambda) p2 and (1 - lambda) p1 + lambda p2, rounded: they add up to the parents' sum.
    np.testing.assert_array_equal(first + second, np.tile(p1 + p2, (500, 1)))
    assert np.all((first >= np.minimum(p1, p2)) & (first <= np.maximum(p1, p2)))
    assert np.mean(np.all(first == second, axis=1)) < 0.05


def test_mixed_real_crossover_children_differ():
    spec = {
        "real_idx": np.array([0, 1, 2]),
        "int_idx": np.array([3]),
        "real_lower": np.zeros(3),
        "real_upper": np.ones(3),
        "int_lower": np.array([0]),
        "int_upper": np.array([5]),
    }
    rng = np.random.default_rng(3)
    parents = np.column_stack([rng.random((400, 3)), rng.integers(0, 6, size=400)]).astype(float)
    for custom in ({}, {"real_crossover": "arithmetic", "real_crossover_prob": 1.0}):
        children = mixed_crossover(parents, prob=1.0, spec={**spec, **custom}, rng=rng)
        real = children[:, spec["real_idx"]]
        # The parents' mean in both children halved the variance of the real part at every mating.
        assert np.mean(np.all(real[0::2] == real[1::2], axis=1)) < 0.05
        np.testing.assert_allclose(real[0::2] + real[1::2], parents[0::2, :3] + parents[1::2, :3])
    # "mean" keeps the parents' mean in both children.
    children = mixed_crossover(parents, prob=1.0, spec={**spec, "real_crossover": "mean", "real_crossover_prob": 1.0}, rng=rng)
    np.testing.assert_allclose(children[0::2, :3], children[1::2, :3])
