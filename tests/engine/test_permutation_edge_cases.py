from __future__ import annotations

import numpy as np

from vamos.engine.operators.impl._permutation_common import ensure_distinct_indices
from vamos.engine.operators.impl._permutation_crossovers import order_crossover
from vamos.engine.operators.impl._permutation_mutations import insert_mutation, inversion_mutation, scramble_mutation, swap_mutation


class _FixedRng:
    def __init__(self, integers: list[int]) -> None:
        self._integers = list(integers)

    def random(self, size=None):  # noqa: ANN001
        if size is None:
            return 0.0
        return np.zeros(size, dtype=float)

    def integers(self, low, high=None, size=None):  # noqa: ANN001
        if size is not None:
            return np.zeros(size, dtype=int)
        return self._integers.pop(0)

    def shuffle(self, values: np.ndarray) -> None:
        values[:] = values[::-1]


def test_ensure_distinct_indices_noops_when_only_one_index_exists() -> None:
    idx = np.array([[0, 0]], dtype=int)

    ensure_distinct_indices(idx, upper=1, rng=np.random.default_rng(0))

    np.testing.assert_array_equal(idx, np.array([[0, 0]], dtype=int))


def test_order_crossover_noops_for_single_gene_permutation() -> None:
    parents = np.array([[0], [0]], dtype=int)

    offspring = order_crossover(parents, prob=1.0, rng=np.random.default_rng(0))

    np.testing.assert_array_equal(offspring, parents)
    assert offspring is not parents


def test_swap_mutation_noops_for_single_gene_permutation() -> None:
    X = np.array([[0], [0]], dtype=int)

    swap_mutation(X, prob=1.0, rng=np.random.default_rng(0))

    np.testing.assert_array_equal(X, np.array([[0], [0]], dtype=int))


def test_scramble_mutation_zero_segment_limit_means_unbounded() -> None:
    X = np.arange(60, dtype=int)[None, :]

    scramble_mutation(X, prob=1.0, rng=_FixedRng([0, 58]), max_segment_length=0)

    np.testing.assert_array_equal(X[0], np.arange(59, -1, -1, dtype=int))


def test_inversion_mutation_can_include_last_gene() -> None:
    X = np.arange(5, dtype=int)[None, :]

    inversion_mutation(X, prob=1.0, rng=_FixedRng([0, 3]))

    np.testing.assert_array_equal(X[0], np.array([4, 3, 2, 1, 0], dtype=int))


def test_insert_mutation_moves_genes_in_both_directions() -> None:
    rng = np.random.default_rng(3)
    seen = set()
    for _ in range(300):
        X = np.arange(3, dtype=int)[None, :]
        insert_mutation(X, prob=1.0, rng=rng)
        seen.add(tuple(X[0].tolist()))

    # Moving a gene to the left (2 before 0 and 1) is as reachable as moving it to the right.
    assert (2, 0, 1) in seen
    assert seen == {(1, 0, 2), (0, 2, 1), (1, 2, 0), (2, 0, 1)}


def test_scramble_mutation_changes_selected_segment() -> None:
    rng = np.random.default_rng(5)
    X = np.tile(np.arange(12, dtype=int), (4000, 1))

    scramble_mutation(X, prob=1.0, rng=rng)

    unchanged = np.all(X == np.arange(12), axis=1).mean()
    # A segment of two or more genes still can be shuffled into its own order
    # (about 11.4 % with 12 genes; jMetal: 11.6 %), but never a one-gene segment.
    assert unchanged < 0.15
