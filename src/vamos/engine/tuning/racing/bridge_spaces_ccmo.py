"""
CCMO configuration space builder for tuning.

CCMO (Tian et al., 2021) runs the GA variation, with a two-parent crossover and a mutation, or
the DE variation, with ``de_cr`` and ``de_f``: the crossover parameter chooses between them. The
paper's configuration is a population of 100, SBX with probability 1.0 and distribution index 20,
and polynomial mutation with probability ``1/n`` and distribution index 20; the DE variation of
the authors' code uses CR = 1 and F = 0.5.
"""

from __future__ import annotations

from .bridge_space_parts_discrete import PAIRWISE_REAL_CROSSOVERS_MEDIUM, real_operator_part_medium
from .config_space import AlgorithmConfigSpace, SpacePart, compose_config_space
from .param_space import Condition, ConditionalBlock, Int, ParamType, Real

# The GA variation mates parents two by two; "de" selects the DE variation.
CCMO_CROSSOVERS: tuple[str, ...] = (*PAIRWISE_REAL_CROSSOVERS_MEDIUM, "de")


def _core_part() -> SpacePart:
    params: list[ParamType] = [Int("pop_size", 20, 200, log=True, role="population")]
    return params, [], []


def _real_operator_part() -> SpacePart:
    params, conditionals, conditions = real_operator_part_medium(crossovers=CCMO_CROSSOVERS)
    conditionals = [
        *conditionals,
        ConditionalBlock("crossover", "de", [Real("de_cr", 0.0, 1.0, role="operator_rate"), Real("de_f", 0.0, 1.0, role="operator_rate")]),
    ]
    # The DE variation recombines every group: it has no crossover probability.
    return params, conditionals, [*conditions, Condition("crossover_prob", "cfg['crossover'] != 'de'")]


def build_ccmo_config_space() -> AlgorithmConfigSpace:
    """The real-coded space of CCMO: the population size and the GA or DE variation."""
    return compose_config_space("ccmo", _core_part(), _real_operator_part())


__all__ = ["CCMO_CROSSOVERS", "build_ccmo_config_space"]
