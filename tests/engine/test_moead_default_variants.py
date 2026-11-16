"""MOEADConfig.default: VAMOS's default, and jMetal's and Evolver's two default MOEA/D configurations as variants.

Evolver's ``defaultConfigurations/MOEADDEDefault.txt`` and ``MOEADDoubleDefault.txt``, and jMetal's
``MOEADDEBuilder`` and ``MOEADBuilder`` (``MOEADDEDefaultConfigurationExample`` and
``MOEADDefaultConfigurationExample``):

- MOEA/D-DE (Li and Zhang, 2009): DE rand/1/bin with CR = 1.0 and F = 0.5, Tchebycheff;
- MOEA/D with SBX: SBX with p = 0.9 and eta = 20, PBI with theta = 5.

Both use polynomial mutation with p = 1/n (a factor of 1.0) and eta = 20, a neighbourhood of 20,
a probability of 0.9 of mating in the neighbourhood, and at most 2 replaced solutions. Without a
variant, VAMOS's default combines MOEA/D-DE's operators with PBI (theta = 5) and stays until 2.0.
"""

from __future__ import annotations

import dataclasses

import pytest

from vamos.engine.algorithm.builders import build_moead_algorithm
from vamos.engine.algorithm.config import MOEADConfig
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem

N_VAR = 30
MUTATION = ("pm", {"prob": 1.0 / N_VAR, "eta": 20.0})
NEIGHBOURHOOD = {"neighbor_size": 20, "delta": 0.9, "replace_limit": 2, "batch_size": 1}
VARIANTS = {
    "de": {"crossover": ("de", {"cr": 1.0, "f": 0.5}), "aggregation": ("tchebycheff", {})},
    "sbx": {"crossover": ("sbx", {"prob": 0.9, "eta": 20.0}), "aggregation": ("pbi", {"theta": 5.0})},
}


def _fields(config: MOEADConfig) -> dict[str, object]:
    return {
        "crossover": config.crossover,
        "aggregation": config.aggregation,
        "neighbor_size": config.neighbor_size,
        "delta": config.delta,
        "replace_limit": config.replace_limit,
        "batch_size": config.batch_size,
    }


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_each_variant_is_its_jmetal_default_configuration(variant: str) -> None:
    config = MOEADConfig.default(pop_size=100, n_var=N_VAR, n_obj=2, variant=variant)

    assert _fields(config) == {**VARIANTS[variant], **NEIGHBOURHOOD}
    assert config.mutation == MUTATION


def test_without_a_variant_the_default_keeps_de_with_pbi() -> None:
    config = MOEADConfig.default(pop_size=100, n_var=N_VAR, n_obj=2)

    assert _fields(config) == {"crossover": ("de", {"cr": 1.0, "f": 0.5}), "aggregation": ("pbi", {"theta": 5.0}), **NEIGHBOURHOOD}
    assert config.mutation == MUTATION
    # MOEA/D-DE differs from it only in the aggregation.
    de = MOEADConfig.default(pop_size=100, n_var=N_VAR, n_obj=2, variant="de")
    assert config == dataclasses.replace(de, aggregation=("pbi", {"theta": 5.0}))


def test_the_variant_is_a_keyword_argument() -> None:
    with pytest.raises(TypeError):
        MOEADConfig.default(100, N_VAR, 2, None, "sbx")


def test_an_unknown_variant_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="'de', 'sbx'"):
        MOEADConfig.default(pop_size=100, n_var=N_VAR, n_obj=2, variant="pbi")


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_a_discrete_encoding_keeps_its_operators_and_takes_the_variants_aggregation(variant: str) -> None:
    config = MOEADConfig.default(pop_size=20, n_var=16, n_obj=2, encoding="binary", variant=variant)

    assert config.crossover[0] == "uniform"
    assert config.mutation[0] == "bitflip"
    assert config.aggregation == VARIANTS[variant]["aggregation"]


def test_the_cli_builder_aggregates_as_the_default() -> None:
    # The `vamos` CLI builds MOEA/D with build_moead_algorithm when no aggregation is given.
    _, config = build_moead_algorithm(kernel=NumPyKernel(), problem=ZDT1Problem(n_var=6), pop_size=10, moead_variation=None)

    assert isinstance(config, MOEADConfig)
    assert config.aggregation == ("pbi", {"theta": 5.0})
