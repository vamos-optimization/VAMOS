"""``pop_size`` is a run setting: the grammars do not tune it, and the caller gives it with the assignment.

MOEA/D, NSGA-III and RVEA derive their populations from that fixed ``pop_size`` and the number of
objectives, as they derived them from the sampled one: MOEA/D runs the largest population, at most
``pop_size``, that has weight vectors, RVEA the largest complete Das-Dennis lattice with at most
``pop_size`` directions, and NSGA-III keeps ``pop_size`` with the largest lattice that fits in it.
"""

from __future__ import annotations

import pytest

from vamos.engine.algorithm.components.weight_vectors import largest_weighted_population
from vamos.engine.reference_directions import largest_lattice_divisions, lattice_size
from vamos.engine.tuning import config_from_assignment
from vamos.engine.tuning.grammar import builtin_parameter_space, builtin_parameter_space_names

REFERENCE_SPACES = sorted(name for name in builtin_parameter_space_names() if name.split("_")[0] in {"moead", "nsgaiii", "rvea"})


@pytest.mark.parametrize("name", REFERENCE_SPACES)
@pytest.mark.parametrize("n_obj", [2, 3, 5])
@pytest.mark.parametrize("pop_size", [20, 60, 100, 150])
def test_reference_direction_algorithms_derive_their_population_from_the_fixed_size(name: str, n_obj: int, pop_size: int) -> None:
    space = builtin_parameter_space(name)
    config = config_from_assignment(name, {**space.default_values(), "pop_size": pop_size, "n_obj": n_obj})

    if space.algorithm == "moead":
        assert config.pop_size == (largest_weighted_population(pop_size, n_obj) if n_obj > 2 else pop_size)
    elif space.algorithm == "rvea":
        divisions = largest_lattice_divisions(pop_size, n_obj)
        assert (config.pop_size, config.n_partitions) == (lattice_size(n_obj, divisions), divisions)
    else:
        assert config.pop_size == pop_size


@pytest.mark.parametrize("name", sorted(builtin_parameter_space_names()))
def test_the_fixed_population_reaches_every_algorithm(name: str) -> None:
    space = builtin_parameter_space(name)
    config = config_from_assignment(name, {**space.default_values(), "pop_size": 48, "n_obj": 2})

    assert config.pop_size == 48


@pytest.mark.parametrize("name", ["nsgaii", "spea2", "smpso", "moead", "rvea"])
def test_an_assignment_without_a_population_runs_the_default_size(name: str) -> None:
    space = builtin_parameter_space(name)
    config = config_from_assignment(name, {**space.default_values(), "n_obj": 2})

    assert config.pop_size == 100
