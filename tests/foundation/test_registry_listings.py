"""The public registry listings resolve: every listed algorithm, and operators for every encoding."""

from __future__ import annotations

import pytest

from vamos.algorithms import available_algorithms, available_crossover_methods, available_mutation_methods, resolve_algorithm


@pytest.mark.parametrize("name", available_algorithms())
def test_every_listed_algorithm_resolves(name: str) -> None:
    assert callable(resolve_algorithm(name))


@pytest.mark.parametrize("encoding", ["real", "binary", "permutation", "integer", "mixed"])
def test_every_encoding_offers_a_crossover_and_a_mutation(encoding: str) -> None:
    assert available_crossover_methods(encoding)
    assert available_mutation_methods(encoding)
