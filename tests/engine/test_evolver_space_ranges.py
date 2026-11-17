"""MOEA/D's and SMS-EMOA's tuning spaces offer what Evolver's spaces offer for these algorithms.

Evolver's ``parameterSpaces/MOEADDouble.yaml`` and ``SMSEMOADouble.yaml``: MOEA/D's neighbourhood
size in [5, 50], its neighbourhood selection probability in [0, 1], PBI's theta up to 200, the modified
Tchebycheff aggregation, linked polynomial mutation, a mutation probability factor (probability =
factor / n) and, as VAMOS's other real-coded spaces, a repair and an initializer. Assignments saved from
the earlier spaces keep building the configurations they built.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.algorithms import MOEADConfig, SMSEMOAConfig
from vamos.engine.tuning import config_from_assignment
from vamos.engine.tuning.racing.bridge_spaces import (
    build_moead_binary_config_space,
    build_moead_config_space,
    build_moead_constrained_config_space,
    build_moead_integer_config_space,
    build_moead_mixed_config_space,
    build_moead_permutation_config_space,
    build_smsemoa_config_space,
    build_smsemoa_constrained_config_space,
)
from vamos.engine.tuning.racing.config_space import AlgorithmConfigSpace
from vamos.engine.tuning.racing.param_space import Categorical, Int, Real
from vamos.engine.tuning.racing.parameter_names import canonical_parameter_name

MOEAD_SPACES: dict[str, Callable[[], AlgorithmConfigSpace]] = {
    "moead": build_moead_config_space,
    "moead_constrained": build_moead_constrained_config_space,
    "moead_mixed": build_moead_mixed_config_space,
    "moead_binary": build_moead_binary_config_space,
    "moead_integer": build_moead_integer_config_space,
    "moead_permutation": build_moead_permutation_config_space,
}
MOEAD_REAL = ("moead", "moead_constrained")
SMSEMOA_REAL: dict[str, Callable[[], AlgorithmConfigSpace]] = {
    "smsemoa": build_smsemoa_config_space,
    "smsemoa_constrained": build_smsemoa_constrained_config_space,
}

MOEAD_BASE = {"pop_size": 100, "neighbor_size": 20, "delta": 0.9, "replace_limit": 2}


def _params(builder: Callable[[], AlgorithmConfigSpace]) -> dict[str, Any]:
    return dict(builder().to_param_space().params)


@pytest.mark.parametrize("name", sorted(MOEAD_SPACES))
def test_every_moead_space_has_evolvers_neighbourhood_and_aggregation_ranges(name: str) -> None:
    params = _params(MOEAD_SPACES[name])

    neighbor_size, delta = params["neighborhood_size"], params["neighborhood_selection_probability"]
    theta = params["pbi_theta"]
    assert isinstance(neighbor_size, Int) and (neighbor_size.low, neighbor_size.high) == (5, 50)
    assert isinstance(delta, Real) and (delta.low, delta.high) == (0.0, 1.0)
    assert isinstance(theta, Real) and (theta.low, theta.high) == (1.0, 200.0)
    aggregation = params["aggregation_function"]
    assert isinstance(aggregation, Categorical)
    assert set(aggregation.choices) == {"tchebycheff", "weighted_sum", "pbi", "modified_tchebycheff"}
    assert "mtch_rho" in params


@pytest.mark.parametrize("name", MOEAD_REAL)
def test_real_coded_moead_spaces_have_evolvers_mutation_repair_and_initializer(name: str) -> None:
    params = _params(MOEAD_SPACES[name])

    assert "linked_polynomial" in params["mutation"].choices
    assert "mutation_prob" not in params
    factor = params["mutation_probability_factor"]
    assert isinstance(factor, Real) and factor.low <= 1.0 <= factor.high
    assert set(params["repair"].choices) == {"auto", "clip", "reflect", "random"}
    assert set(params["initial_solutions"].choices) >= {"random", "lhs", "scatter"}


@pytest.mark.parametrize("name", MOEAD_REAL)
def test_the_polynomial_index_is_active_only_for_the_polynomial_mutations(name: str) -> None:
    space = MOEAD_SPACES[name]()
    draws = [space.sample(np.random.default_rng(seed)) for seed in range(80)]

    assert {"polynomial", "linked_polynomial"} <= {draw["mutation"] for draw in draws}
    for draw in draws:
        # Each polynomial mutation has its own index (one parent per parameter).
        assert ("polynomial_mutation_distribution_index" in draw) == (draw["mutation"] == "polynomial")
        assert ("linked_polynomial_mutation_distribution_index" in draw) == (draw["mutation"] == "linked_polynomial")


@pytest.mark.parametrize("name", sorted(MOEAD_SPACES))
def test_every_draw_of_a_moead_space_builds_a_configuration(name: str) -> None:
    space = MOEAD_SPACES[name]()
    for seed in range(30):
        config = config_from_assignment(name, space.sample(np.random.default_rng(seed)))
        assert isinstance(config, MOEADConfig)


def test_the_bridge_builds_evolvers_moead_options() -> None:
    config = config_from_assignment(
        "moead",
        {
            **MOEAD_BASE,
            "crossover": "de",
            "de_cr": 1.0,
            "de_f": 0.5,
            "mutation": "linked_polynomial",
            "mutation_prob_factor": 1.5,
            "mutation_eta": 30.0,
            "aggregation": "modified_tchebycheff",
            "mtch_rho": 0.01,
            "initializer": "lhs",
            "repair": "reflect",
        },
    )

    assert isinstance(config, MOEADConfig)
    assert config.mutation == ("linked_polynomial", {"eta": 30.0})
    assert config.mutation_prob_factor == 1.5
    assert config.aggregation == ("modified_tchebycheff", {"rho": 0.01})
    assert config.initializer is not None and config.initializer.get("type") == "lhs"
    assert config.repair == ("reflect", {})


@pytest.mark.parametrize(
    ("assignment", "crossover", "aggregation"),
    [
        # Evolver's MOEADDEDefault.
        ({"crossover": "de", "de_cr": 1.0, "de_f": 0.5, "aggregation": "tchebycheff"}, ("de", {"cr": 1.0, "f": 0.5}), ("tchebycheff", {})),
        # Evolver's MOEADDoubleDefault.
        (
            {"crossover": "sbx", "crossover_prob": 0.9, "crossover_eta": 20.0, "aggregation": "pbi", "pbi_theta": 5.0},
            ("sbx", {"prob": 0.9, "eta": 20.0}),
            ("pbi", {"theta": 5.0}),
        ),
    ],
    ids=["MOEADDEDefault", "MOEADDoubleDefault"],
)
def test_evolvers_default_moead_configurations_are_points_of_the_space(
    assignment: dict[str, Any], crossover: tuple[str, dict[str, float]], aggregation: tuple[str, dict[str, float]]
) -> None:
    full = {**MOEAD_BASE, **assignment, "mutation": "polynomial", "mutation_prob_factor": 1.0, "mutation_eta": 20.0}
    params = _params(build_moead_config_space)
    for key, value in full.items():
        if key == "pop_size":  # a run setting, not a parameter of the space
            continue
        param = params[canonical_parameter_name(key)]
        if isinstance(param, Categorical):
            assert value in param.choices, key
        else:
            assert param.low <= value <= param.high, key

    config = config_from_assignment("moead", full)

    assert isinstance(config, MOEADConfig)
    assert (config.crossover, config.aggregation) == (crossover, aggregation)
    assert config.mutation == ("polynomial", {"eta": 20.0})
    assert config.mutation_prob_factor == 1.0
    assert (config.neighbor_size, config.delta, config.replace_limit) == (20, 0.9, 2)


def test_a_moead_assignment_from_the_earlier_space_builds_the_same_configuration() -> None:
    earlier = {**MOEAD_BASE, "crossover": "sbx", "crossover_prob": 0.9, "crossover_eta": 20.0}
    earlier |= {"mutation": "polynomial", "mutation_prob": 0.05, "mutation_eta": 20.0, "aggregation": "pbi", "pbi_theta": 5.0}

    config = config_from_assignment("moead", earlier)

    expected = (
        MOEADConfig.builder()
        .pop_size(100)
        .neighbor_size(20)
        .delta(0.9)
        .replace_limit(2)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob=0.05, eta=20.0)
        .aggregation("pbi", theta=5.0)
        .build()
    )
    assert config == expected


@pytest.mark.parametrize("name", sorted(SMSEMOA_REAL))
def test_real_coded_smsemoa_spaces_have_a_mutation_probability_factor(name: str) -> None:
    params = _params(SMSEMOA_REAL[name])

    assert "mutation_prob" not in params
    factor = params["mutation_probability_factor"]
    assert isinstance(factor, Real) and factor.low <= 1.0 <= factor.high


def test_the_smsemoa_bridge_builds_the_factor_and_keeps_the_earlier_absolute_probability() -> None:
    base = {"pop_size": 100, "crossover": "sbx", "crossover_prob": 0.9, "crossover_eta": 20.0, "mutation": "pm", "mutation_eta": 20.0}
    base |= {"selection": "random", "reference_point_offset": 1.0}

    with_factor = config_from_assignment("smsemoa", {**base, "mutation_prob_factor": 1.0})
    earlier = config_from_assignment("smsemoa", {**base, "mutation_prob": 0.05})

    assert isinstance(with_factor, SMSEMOAConfig) and isinstance(earlier, SMSEMOAConfig)
    assert with_factor.mutation_prob_factor == 1.0 and with_factor.mutation == ("pm", {"eta": 20.0})
    assert earlier.mutation_prob_factor is None and earlier.mutation == ("pm", {"prob": 0.05, "eta": 20.0})


def test_a_tuned_moead_configuration_with_the_new_options_runs() -> None:
    config = config_from_assignment(
        "moead",
        {
            **MOEAD_BASE,
            "pop_size": 20,
            "neighbor_size": 50,
            "delta": 0.0,
            "crossover": "sbx",
            "crossover_prob": 0.9,
            "crossover_eta": 20.0,
            "mutation": "linked_polynomial",
            "mutation_prob_factor": 1.0,
            "mutation_eta": 20.0,
            "aggregation": "modified_tchebycheff",
            "mtch_rho": 0.001,
            "initializer": "scatter",
            "repair": "random",
        },
    )

    result = vamos.optimize("zdt1", algorithm="moead", algorithm_config=config, max_evaluations=60, n_var=6, seed=2, engine="numpy")

    assert result.data["evaluations"] == 60
    resolved = result.explain_defaults()["resolved_spec"]
    assert resolved["algorithm"]["config"]["mutation_prob_factor"] == 1.0
    operators = resolved["operators"]
    assert (operators["repair"]["component_id"], operators["initializer"]["component_id"]) == (
        "vamos.operator:random@1",
        "vamos.operator:scatter@1",
    )
