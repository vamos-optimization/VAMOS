"""
ToP configuration space builder for tuning.

ToP (Liu and Wang, 2019) creates every trial vector with its own differential evolution, so its
space has no crossover choice: it offers the crossover rate ``de_cr`` and the scale factor
``de_f``, the mutation applied to the trial vectors and the initializer and repair of the other
real-coded spaces, and the three thresholds of the switch between its phases. The configuration
of PlatEMO's implementation, which jMetal's port also uses, is a population of 100, CR = 1.0 and F
= 0.5, polynomial mutation with probability ``1/n`` and distribution index 20, and the thresholds
0.2 for ``delta``, 1/3 for ``Pf`` and 90% of the budget.
"""

from __future__ import annotations

from .bridge_space_parts_discrete import real_operator_part_medium
from .config_space import AlgorithmConfigSpace, SpacePart, compose_config_space
from .param_space import Int, ParamType, Real


def _core_part() -> SpacePart:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True, role="population"),
        Real("delta_threshold", 0.05, 0.5, role="adaptive"),
        Real("feasible_ratio_threshold", 0.1, 0.9, role="adaptive"),
        Real("phase_one_evaluation_fraction", 0.5, 1.0, role="adaptive"),
    ]
    return params, [], []


def _variation_part() -> SpacePart:
    params, conditionals, conditions = real_operator_part_medium(crossovers=("de",))
    # The variation is DE only: no crossover choice, probability or crossover parameters.
    params = [param for param in params if param.name not in {"crossover", "crossover_prob"}]
    conditionals = [block for block in conditionals if block.parent_name != "crossover"]
    params.extend([Real("de_cr", 0.0, 1.0, role="operator_rate"), Real("de_f", 0.1, 1.0, role="operator_rate")])
    return params, conditionals, conditions


def build_top_config_space() -> AlgorithmConfigSpace:
    """The real-coded space of ToP: the population size, the switch thresholds, CR, F and the mutation."""
    return compose_config_space("top", _core_part(), _variation_part())


__all__ = ["build_top_config_space"]
