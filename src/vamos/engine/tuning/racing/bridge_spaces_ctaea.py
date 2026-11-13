"""
C-TAEA configuration space builder for tuning.

C-TAEA (Li et al., 2019) mates its parents two by two and keeps the first child, so its space
offers the two-parent real-coded crossovers with their probability, and the mutation,
initializer and repair of the other real-coded spaces. ``pop_size`` is the requested number of
weight vectors, from which C-TAEA generates its population. The paper's configuration, which
jMetal's port also uses, is 100 requested vectors, SBX with probability 1.0 and distribution index
20, and polynomial mutation with probability ``1/n`` and distribution index 20.
"""

from __future__ import annotations

from .bridge_space_parts_discrete import PAIRWISE_REAL_CROSSOVERS_MEDIUM, real_operator_part_medium
from .config_space import AlgorithmConfigSpace, SpacePart, compose_config_space
from .param_space import Int, ParamType


def _core_part() -> SpacePart:
    params: list[ParamType] = [Int("pop_size", 20, 200, log=True, role="population")]
    return params, [], []


def build_ctaea_config_space() -> AlgorithmConfigSpace:
    """The real-coded space of C-TAEA: the requested population size and a two-parent GA variation."""
    return compose_config_space("ctaea", _core_part(), real_operator_part_medium(crossovers=PAIRWISE_REAL_CROSSOVERS_MEDIUM))


__all__ = ["build_ctaea_config_space"]
