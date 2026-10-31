"""RVEA reference-vector partitions derived from the population size."""

from __future__ import annotations

from math import comb

import pytest

from vamos.engine.algorithm.config.rvea import rvea_partitions_for_population
from vamos.foundation.exceptions import ConfigurationError


@pytest.mark.parametrize(
    ("pop_size", "n_obj", "partitions"),
    [(100, 2, 99), (13, 2, 12), (2, 2, 1), (91, 3, 12), (105, 3, 13), (3, 3, 1), (210, 5, 6)],
)
def test_partitions_match_population(pop_size: int, n_obj: int, partitions: int) -> None:
    assert rvea_partitions_for_population(pop_size, n_obj) == partitions
    assert comb(partitions + n_obj - 1, n_obj - 1) == pop_size


def test_population_without_lattice_names_nearest_sizes() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        rvea_partitions_for_population(100, 3)

    assert "pop_size=100" in excinfo.value.message
    assert excinfo.value.suggestion is not None
    assert "91 (n_partitions=12)" in excinfo.value.suggestion
    assert "105 (n_partitions=13)" in excinfo.value.suggestion


@pytest.mark.parametrize(("pop_size", "n_obj"), [(0, 2), (10, 1)])
def test_invalid_inputs_are_rejected(pop_size: int, n_obj: int) -> None:
    with pytest.raises(ConfigurationError):
        rvea_partitions_for_population(pop_size, n_obj)
