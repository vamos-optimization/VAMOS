"""Configurations of a parameter space: derivations of the grammar, and their three serializations.

A configuration holds exactly the active parameters of the tree, each inside its domain, so the
canonical document, the flat mapping and the command line (VAMOS's names or Evolver's) carry the same
information and each one reads back into the same configuration.
"""

from __future__ import annotations

import numpy as np
import pytest

from vamos.engine.tuning.grammar import (
    CONFIGURATION_SCHEMA,
    ParameterSpaceError,
    configuration,
    configuration_from_command_line,
    configuration_from_document,
    default_configuration,
    parameter_space_from_document,
)

from .test_parameter_space_schema import small_space

SPACE = parameter_space_from_document(small_space())


def test_the_default_configuration_holds_the_defaults_of_the_active_branch() -> None:
    values = default_configuration(SPACE).to_values()

    assert list(values) == [
        "offspring_ratio",
        "selection",
        "tournament_size",
        "crossover",
        "crossover_probability",
        "sbx_distribution_index",
        "mutation",
        "mutation_probability_factor",
        "polynomial_mutation_distribution_index",
        "keep_duplicates",
    ]
    assert values["sbx_distribution_index"] == 20.0 and values["tournament_size"] == 2
    assert "crossover_alpha" not in values


def test_a_configuration_is_ordered_depth_first_whatever_the_order_given() -> None:
    values = default_configuration(SPACE).to_values()

    assert configuration(SPACE, dict(reversed(list(values.items())))).to_values() == values


def test_numbers_take_the_type_of_their_parameter() -> None:
    values = {**default_configuration(SPACE).to_values(), "sbx_distribution_index": 20, "tournament_size": np.int64(3)}
    built = configuration(SPACE, values).to_values()

    assert built["sbx_distribution_index"] == 20.0 and isinstance(built["sbx_distribution_index"], float)
    assert built["tournament_size"] == 3 and type(built["tournament_size"]) is int


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"crossover_alpha": 0.5}, "inactive"),
        ({"population_size": 100}, "not parameters of"),
        ({"sbx_distribution_index": 50.0}, "outside the domain"),
        ({"selection": "boltzmann"}, "outside the domain"),
        ({"keep_duplicates": 1}, "outside the domain"),
        ({"tournament_size": 2.0}, "outside the domain"),
    ],
)
def test_an_invalid_configuration_is_rejected(change: dict[str, object], message: str) -> None:
    with pytest.raises(ParameterSpaceError, match=message) as raised:
        configuration(SPACE, {**default_configuration(SPACE).to_values(), **change})
    assert raised.value.rule == "configuration"


def test_an_active_parameter_without_a_value_is_rejected() -> None:
    values = default_configuration(SPACE).to_values()
    del values["tournament_size"]

    with pytest.raises(ParameterSpaceError, match="tournament_size: the parameter is active and has no value"):
        configuration(SPACE, values)


def test_switching_a_production_needs_the_parameters_it_activates() -> None:
    values = {**default_configuration(SPACE).to_values(), "crossover": "blx_alpha"}
    del values["sbx_distribution_index"]

    with pytest.raises(ParameterSpaceError, match="crossover_alpha"):
        configuration(SPACE, values)
    assert configuration(SPACE, {**values, "crossover_alpha": 0.25}).to_values()["crossover_alpha"] == 0.25


def test_the_three_serializations_read_back_into_the_same_configuration() -> None:
    rng = np.random.default_rng(11)
    for _ in range(40):
        built = configuration(SPACE, SPACE.sample(rng))
        document = built.to_document()

        assert document["schema"] == CONFIGURATION_SCHEMA and document["space"] == SPACE.identity.to_document()
        assert configuration_from_document(SPACE, document) == built
        assert configuration(SPACE, built.to_values()) == built
        assert configuration_from_command_line(SPACE, built.to_command_line()) == built
        assert configuration_from_command_line(SPACE, built.to_command_line(evolver=True), evolver=True) == built


def test_the_command_lines_use_vamos_names_or_evolvers() -> None:
    default = default_configuration(SPACE)

    assert default.to_command_line() == (
        "--offspring_ratio 1.0 --selection tournament --tournament_size 2 --crossover sbx --crossover_probability 0.9 "
        "--sbx_distribution_index 20.0 --mutation polynomial --mutation_probability_factor 1.0 "
        "--polynomial_mutation_distribution_index 20.0 --keep_duplicates false"
    )
    # Evolver's names and values where Evolver has them; VAMOS's canonical names for the rest.
    assert default.to_command_line(evolver=True) == (
        "--offspring_ratio 1.0 --selection tournament --selectionTournamentSize 2 --crossover SBX --crossoverProbability 0.9 "
        "--sbxDistributionIndex 20.0 --mutation polynomial --mutationProbabilityFactor 1.0 "
        "--polynomialMutationDistributionIndex 20.0 --keep_duplicates false"
    )


def test_a_document_of_another_space_is_rejected() -> None:
    document = default_configuration(SPACE).to_document()
    document["space"] = {**document["space"], "sha256": "0" * 64}

    with pytest.raises(ParameterSpaceError, match="space"):
        configuration_from_document(SPACE, document)


@pytest.mark.parametrize("text", ["--crossover", "crossover sbx", "--crossover sbx --crossover blx_alpha"])
def test_a_malformed_command_line_is_rejected(text: str) -> None:
    with pytest.raises(ParameterSpaceError):
        configuration_from_command_line(SPACE, text)


def test_sampling_draws_only_the_active_branch() -> None:
    rng = np.random.default_rng(2)
    draws = [SPACE.sample(rng) for _ in range(200)]

    for draw in draws:
        assert configuration(SPACE, draw).to_values() == draw
        assert ("sbx_distribution_index" in draw) == (draw["crossover"] == "sbx")
        assert ("tournament_size" in draw) == (draw["selection"] == "tournament")
        assert 0.6 <= float(draw["crossover_probability"]) <= 1.0
    assert {draw["crossover"] for draw in draws} == {"sbx", "blx_alpha"}
    assert {draw["keep_duplicates"] for draw in draws} == {True, False}
    # Seeded draws repeat.
    assert [SPACE.sample(np.random.default_rng(5)) for _ in range(2)] == [SPACE.sample(np.random.default_rng(5))] * 2
