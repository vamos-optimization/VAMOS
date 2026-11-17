"""SMS-EMOA's tuning spaces can express its default mating selection and reference point."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from vamos.algorithms import SMSEMOAConfig
from vamos.engine.tuning import config_from_assignment
from vamos.engine.tuning.racing.bridge_spaces import (
    build_smsemoa_binary_config_space,
    build_smsemoa_config_space,
    build_smsemoa_constrained_config_space,
    build_smsemoa_integer_config_space,
    build_smsemoa_mixed_config_space,
    build_smsemoa_permutation_config_space,
)
from vamos.engine.tuning.racing.config_space import AlgorithmConfigSpace
from vamos.engine.tuning.racing.param_space import Categorical, Real

SPACES: tuple[Callable[[], AlgorithmConfigSpace], ...] = (
    build_smsemoa_config_space,
    build_smsemoa_constrained_config_space,
    build_smsemoa_mixed_config_space,
    build_smsemoa_binary_config_space,
    build_smsemoa_integer_config_space,
    build_smsemoa_permutation_config_space,
)

# A real-coded assignment with SMS-EMOA's default operators.
DEFAULT_OPERATORS = {
    "pop_size": 100,
    "crossover": "sbx",
    "crossover_prob": 0.9,
    "crossover_eta": 20.0,
    "mutation": "pm",
    "mutation_prob": 1.0 / 30,
    "mutation_eta": 20.0,
}


@pytest.mark.parametrize("builder", SPACES, ids=lambda builder: builder.__name__)
def test_every_space_offers_random_selection_and_the_reference_point_offset(builder: Callable[[], AlgorithmConfigSpace]) -> None:
    params = builder().to_param_space().params

    selection = params["selection"]
    assert isinstance(selection, Categorical)
    assert set(selection.choices) == {"random", "tournament"}
    offset = params["reference_point_offset"]
    assert isinstance(offset, Real)
    # Both the default offset (1.0) and the one the bridge always used (0.1) are reachable.
    assert offset.low <= 0.1 and offset.high >= 1.0


@pytest.mark.parametrize("builder", SPACES, ids=lambda builder: builder.__name__)
def test_the_tournament_size_is_active_only_with_a_tournament(builder: Callable[[], AlgorithmConfigSpace]) -> None:
    space = builder()
    draws = [space.sample(np.random.default_rng(seed)) for seed in range(60)]

    assert {draw["selection"] for draw in draws} == {"random", "tournament"}
    for draw in draws:
        assert ("tournament_size" in draw) == (draw["selection"] == "tournament")
        assert "reference_point_offset" in draw


def test_an_assignment_reproduces_the_default_selection_and_reference_point() -> None:
    default = SMSEMOAConfig.default(n_var=30)

    config = config_from_assignment("smsemoa", {**DEFAULT_OPERATORS, "selection": "random", "reference_point_offset": 1.0})

    assert isinstance(config, SMSEMOAConfig)
    assert config.selection == default.selection == ("random", {})
    assert config.reference_point == default.reference_point


def test_a_tournament_assignment_uses_its_size_and_offset() -> None:
    config = config_from_assignment(
        "smsemoa",
        {**DEFAULT_OPERATORS, "selection": "tournament", "selection_pressure": 4, "reference_point_offset": 0.5},
    )

    assert isinstance(config, SMSEMOAConfig)
    assert config.selection == ("tournament", {"size": 4})
    assert config.reference_point == {"vector": None, "offset": 0.5, "adaptive": True}


def test_assignments_without_the_new_parameters_keep_their_meaning() -> None:
    """Assignments saved from earlier tuning runs name only the tournament size."""
    config = config_from_assignment("smsemoa", {**DEFAULT_OPERATORS, "selection_pressure": 3})

    assert isinstance(config, SMSEMOAConfig)
    assert config.selection == ("tournament", {"size": 3})
    assert config.reference_point == {"vector": None, "offset": 0.1, "adaptive": True}
