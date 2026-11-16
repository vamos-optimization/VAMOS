"""The tuning spaces use Evolver's parameter names in snake_case, with the earlier names as aliases.

The equivalence table (``vamos.engine.tuning.racing.parameter_names``) gives, for each parameter, the
name the configuration bridges read (the spaces' name up to VAMOS 1.2.1), the canonical name and
Evolver's camelCase name, so that assignments can be exchanged with Evolver in both directions.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from vamos.algorithms import MOEADConfig, SMSEMOAConfig
from vamos.engine.tuning import config_from_assignment
from vamos.engine.tuning.racing import bridge_spaces
from vamos.engine.tuning.racing.config_space import AlgorithmConfigSpace
from vamos.engine.tuning.racing.param_space import Categorical
from vamos.engine.tuning.racing.parameter_names import (
    PARAMETER_NAMES,
    VALUE_NAMES,
    bridge_assignment,
    canonical_assignment,
    from_evolver,
    to_evolver,
)

ROOT = Path(__file__).resolve().parents[2]
SPACES: dict[str, Callable[[], AlgorithmConfigSpace]] = {
    name[len("build_") : -len("_config_space")]: getattr(bridge_spaces, name)
    for name in dir(bridge_spaces)
    if name.startswith("build_") and name.endswith("_config_space")
}
EARLIER_NAMES = {vamos for vamos, canonical, _ in PARAMETER_NAMES if vamos != canonical}
EARLIER_VALUES = {(parameter, vamos) for parameter, rows in VALUE_NAMES.items() for vamos, canonical, _ in rows if vamos != canonical}

# Evolver's defaultConfigurations/MOEADDoubleDefault.txt.
MOEAD_DOUBLE_DEFAULT = (
    "--neighborhoodSize 20 --maximumNumberOfReplacedSolutions 2 --aggregationFunction penaltyBoundaryIntersection "
    "--normalizeObjectives false --pbiTheta 5.0 --algorithmResult population --createInitialSolutions default "
    "--subProblemIdGenerator randomPermutationCycle --variation crossoverAndMutationVariation --crossoverProbability 0.9 "
    "--crossoverRepairStrategy bounds --mutation polynomial --mutationProbabilityFactor 1.0 --mutationRepairStrategy bounds "
    "--polynomialMutationDistributionIndex 20.0 --crossover SBX --sbxDistributionIndex 20.0 "
    "--selection populationAndNeighborhoodMatingPoolSelection --neighborhoodSelectionProbability 0.9"
)


def _parse(line: str) -> dict[str, object]:
    tokens = line.split()
    values: dict[str, object] = {}
    for name, raw in zip(tokens[::2], tokens[1::2]):
        try:
            values[name[2:]] = float(raw) if "." in raw else int(raw)
        except ValueError:
            values[name[2:]] = raw
    return values


def test_the_table_covers_every_parameter_of_the_request() -> None:
    rows = {vamos: (canonical, evolver) for vamos, canonical, evolver in PARAMETER_NAMES}

    assert rows["crossover_eta"] == ("sbx_distribution_index", "sbxDistributionIndex")
    assert rows["mutation_eta"] == ("polynomial_mutation_distribution_index", "polynomialMutationDistributionIndex")
    assert rows["neighbor_size"] == ("neighborhood_size", "neighborhoodSize")
    assert rows["delta"] == ("neighborhood_selection_probability", "neighborhoodSelectionProbability")
    assert rows["replace_limit"] == ("maximum_number_of_replaced_solutions", "maximumNumberOfReplacedSolutions")
    assert rows["de_cr"] == ("de_crossover_rate", "CR") and rows["de_f"] == ("de_scale_factor", "F")
    assert rows["selection_pressure"] == ("tournament_size", "selectionTournamentSize")
    assert ("undx", "undc", "UNDC") in VALUE_NAMES["crossover"]


@pytest.mark.parametrize("name", sorted(SPACES))
def test_every_space_uses_the_canonical_names(name: str) -> None:
    params = SPACES[name]().to_param_space().params

    assert not EARLIER_NAMES & set(params), sorted(EARLIER_NAMES & set(params))
    for param_name, param in params.items():
        if isinstance(param, Categorical):
            earlier = {(param_name, value) for value in param.choices} & {(p, v) for p, v in EARLIER_VALUES}
            assert not earlier, earlier


@pytest.mark.parametrize("name", sorted(SPACES))
def test_an_assignment_with_the_earlier_names_builds_the_same_configuration(name: str) -> None:
    space = SPACES[name]()
    for seed in range(15):
        draw = space.sample(np.random.default_rng(seed))
        assert config_from_assignment(name, draw) == config_from_assignment(name, bridge_assignment(draw))


def test_earlier_values_are_still_accepted() -> None:
    base = {"pop_size": 20, "crossover": "sbx", "crossover_probability": 0.9, "sbx_distribution_index": 20.0}
    base |= {"selection": "random", "reference_point_offset": 1.0}

    earlier = config_from_assignment("smsemoa", {**base, "mutation": "pm", "mutation_eta": 20.0})
    canonical = config_from_assignment("smsemoa", {**base, "mutation": "polynomial", "polynomial_mutation_distribution_index": 20.0})

    assert isinstance(earlier, SMSEMOAConfig) and isinstance(canonical, SMSEMOAConfig)
    assert earlier.mutation == ("pm", {"eta": 20.0})
    assert canonical.mutation == ("polynomial", {"eta": 20.0})


@pytest.mark.parametrize("name", ["nsgaii", "moead", "smsemoa"])
def test_assignments_round_trip_through_evolvers_names(name: str) -> None:
    draw = SPACES[name]().sample(np.random.default_rng(3))

    assert from_evolver(to_evolver(draw)) == canonical_assignment(draw)


def test_an_evolver_configuration_reads_into_vamos() -> None:
    assignment = {"pop_size": 100, **from_evolver(_parse(MOEAD_DOUBLE_DEFAULT))}
    config = config_from_assignment("moead", assignment)

    assert isinstance(config, MOEADConfig)
    assert config.crossover == ("sbx", {"prob": 0.9, "eta": 20.0})
    assert config.aggregation == ("pbi", {"theta": 5.0})
    assert config.mutation == ("polynomial", {"eta": 20.0}) and config.mutation_prob_factor == 1.0
    assert (config.neighbor_size, config.delta, config.replace_limit) == (20, 0.9, 2)


def test_a_vamos_assignment_writes_evolvers_names() -> None:
    evolver = to_evolver({"crossover": "undx", "mutation": "pm", "mutation_eta": 20.0, "aggregation": "tchebycheff", "de_cr": 1.0})

    assert evolver == {
        "crossover": "UNDC",
        "mutation": "polynomial",
        "polynomialMutationDistributionIndex": 20.0,
        "aggregationFunction": "tschebyscheff",
        "CR": 1.0,
    }


def test_the_tuning_guide_lists_the_table() -> None:
    text = (ROOT / "docs" / "topics" / "tuning.md").read_text(encoding="utf-8")
    rows = set(re.findall(r"^\| `([a-z_]+)` \| `([a-z_]+)` \| (?:`(\w+)`|—) \|$", text, flags=re.MULTILINE))

    expected = {(vamos, canonical, evolver or "") for vamos, canonical, evolver in PARAMETER_NAMES if vamos != canonical or evolver}
    assert expected <= rows


def test_no_value_is_renamed_to_another_operators_name() -> None:
    # A renamed value that is also an operator's own name is translated back whatever the encoding:
    # "random_reset", an integer mutation, became the real-coded "uniform_reset" and broke integer runs.
    from typing import get_args

    from vamos.engine.algorithm.config.types import CrossoverName, MutationName

    operators = {"crossover": set(get_args(CrossoverName)), "mutation": set(get_args(MutationName))}
    for parameter, names in operators.items():
        clashes = [(vamos, canonical) for vamos, canonical, _ in VALUE_NAMES[parameter] if canonical != vamos and canonical in names]
        assert clashes == [], parameter
    assert bridge_assignment({"mutation": "random_reset"})["mutation"] == "random_reset"
