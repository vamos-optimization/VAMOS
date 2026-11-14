"""
CMOEA-MS configuration space builder for tuning.

CMOEA-MS (Tian et al., 2022) runs the GA variation, with a two-parent crossover that gives two
children and a mutation, or the DE variation, with ``de_cr``, ``de_f`` and ``de_variant`` (the
binomial or the exponential crossover): the crossover parameter chooses between them. Its own
parameter is ``feasible_ratio_threshold``, the paper's ``lambda``, the ratio of feasible solutions
above which its fitness is that of stage B. The configuration of the paper, of PlatEMO's
implementation and of jMetal's port is a population of 100, SBX with probability 1.0 and
distribution index 20, polynomial mutation with probability ``1/n`` and distribution index 20,
and ``lambda = 0.5``; their DE variation uses CR = 1 and F = 0.5 with the binomial crossover.
"""

from __future__ import annotations

from .bridge_space_parts_discrete import PAIRWISE_REAL_CROSSOVERS_MEDIUM, real_operator_part_medium
from .config_space import AlgorithmConfigSpace, SpacePart, compose_config_space
from .param_space import Categorical, Condition, ConditionalBlock, Int, ParamType, Real

# The GA variation mates parents two by two and keeps both children; "de" selects the DE variation.
CMOEAMS_CROSSOVERS: tuple[str, ...] = (*PAIRWISE_REAL_CROSSOVERS_MEDIUM, "de")


def _core_part() -> SpacePart:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True, role="population"),
        Real("feasible_ratio_threshold", 0.1, 0.9, role="adaptive"),
    ]
    return params, [], []


def _real_operator_part() -> SpacePart:
    params, conditionals, conditions = real_operator_part_medium(crossovers=CMOEAMS_CROSSOVERS)
    differential: list[ParamType] = [
        Real("de_cr", 0.0, 1.0, role="operator_rate"),
        Real("de_f", 0.1, 1.0, role="operator_rate"),
        Categorical("de_variant", ["rand_1_bin", "rand_1_exp"], role="operator"),
    ]
    conditionals = [*conditionals, ConditionalBlock("crossover", "de", differential)]
    # The DE variation recombines every solution: it has no crossover probability.
    return params, conditionals, [*conditions, Condition("crossover_prob", "cfg['crossover'] != 'de'")]


def build_cmoeams_config_space() -> AlgorithmConfigSpace:
    """The real-coded space of CMOEA-MS: the population size, ``lambda`` and the GA or DE variation."""
    return compose_config_space("cmoeams", _core_part(), _real_operator_part())


__all__ = ["CMOEAMS_CROSSOVERS", "build_cmoeams_config_space"]
