"""
MOEA/D configuration space builders for tuning.
"""

from __future__ import annotations

from .bridge_space_parts_constraints import subproblem_update_part
from .bridge_space_parts_discrete import (
    binary_operator_part_full,
    external_archive_part,
    integer_operator_part_full,
    mixed_operator_part,
    permutation_operator_part_full,
)
from .config_space import AlgorithmConfigSpace, SpacePart, compose_config_space
from .param_space import Categorical, Condition, ConditionalBlock, Int, ParamType, Real

# ---------------------------------------------------------------------------
# Core part (shared by ALL MOEA/D encoding variants)
# ---------------------------------------------------------------------------


def _core_part() -> SpacePart:
    # Evolver's ranges (parameterSpaces/MOEADDouble.yaml): neighborhoodSize [5, 50],
    # neighborhoodSelectionProbability [0, 1], maximumNumberOfReplacedSolutions [1, 5].
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True, role="population"),
        Int("neighbor_size", 5, 50, role="population"),
        Real("delta", 0.0, 1.0, role="operator_rate"),
        Int("replace_limit", 1, 5, role="population"),
    ]
    arch_params, arch_conds, arch_conditions = external_archive_part()
    return [*params, *arch_params], arch_conds, arch_conditions


def _aggregation_part() -> SpacePart:
    """Aggregation params, with PBI's theta up to 200 as in Evolver; the usual value, 5, lies at the low end, so theta is log-uniform."""
    conditionals: list[ConditionalBlock] = [
        ConditionalBlock("aggregation", "pbi", [Real("pbi_theta", 1.0, 200.0, log=True, role="operator_rate")]),
        ConditionalBlock("aggregation", "modified_tchebycheff", [Real("mtch_rho", 0.0001, 0.1, role="operator_rate")]),
    ]
    choices = ["tchebycheff", "weighted_sum", "pbi", "modified_tchebycheff"]
    return [Categorical("aggregation", choices, role="structural")], conditionals, []


# ---------------------------------------------------------------------------
# Encoding-specific operator parts
# ---------------------------------------------------------------------------


def _real_operator_part() -> SpacePart:
    # The mutation probability is a factor of 1/n_var, as in NSGA-II's space and Evolver's mutationProbabilityFactor.
    params: list[ParamType] = [
        Categorical("initializer", ["random", "lhs", "scatter"], role="structural"),
        Categorical("crossover", ["sbx", "de"], role="operator"),
        Categorical("mutation", ["polynomial", "linked_polynomial", "gaussian", "non_uniform", "cauchy", "uniform"], role="operator"),
        Real("mutation_prob_factor", 0.25, 3.0, role="operator_rate"),
        Real("mutation_eta", 5.0, 40.0, role="operator_rate"),
        Categorical("repair", ["clip", "reflect", "random"], role="structural"),
    ]
    conditionals = [
        ConditionalBlock(
            "crossover",
            "sbx",
            [Real("crossover_prob", 0.6, 1.0, role="operator_rate"), Real("crossover_eta", 10.0, 40.0, role="operator_rate")],
        ),
        ConditionalBlock("crossover", "de", [Real("de_cr", 0.0, 1.0, role="operator_rate"), Real("de_f", 0.0, 1.0, role="operator_rate")]),
        ConditionalBlock("mutation", "non_uniform", [Real("nonuniform_perturbation", 0.05, 0.5, role="operator_rate")]),
        ConditionalBlock("mutation", "gaussian", [Real("gaussian_sigma", 0.001, 0.5, role="operator_rate")]),
        ConditionalBlock("mutation", "cauchy", [Real("cauchy_gamma", 0.001, 0.5, role="operator_rate")]),
        ConditionalBlock("mutation", "uniform", [Real("uniform_perturb", 0.01, 0.5, role="operator_rate")]),
    ]
    return params, conditionals, [Condition("mutation_eta", "cfg['mutation'] == 'polynomial' or cfg['mutation'] == 'linked_polynomial'")]


def _permutation_operator_part() -> SpacePart:
    return permutation_operator_part_full()


def _mixed_operator_part() -> SpacePart:
    return mixed_operator_part()


def _binary_operator_part() -> SpacePart:
    return binary_operator_part_full()


def _integer_operator_part() -> SpacePart:
    return integer_operator_part_full()


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_moead_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("moead", _core_part(), _aggregation_part(), _real_operator_part())


def build_moead_constrained_config_space() -> AlgorithmConfigSpace:
    """The real-coded space with the subproblem update criterion, as Evolver's ``MOEADDoubleConstrained``."""
    return compose_config_space("moead_constrained", _core_part(), _aggregation_part(), _real_operator_part(), subproblem_update_part())


def build_moead_permutation_config_space() -> AlgorithmConfigSpace:
    return compose_config_space(
        "moead_permutation",
        _core_part(),
        _aggregation_part(),
        _permutation_operator_part(),
    )


def build_moead_mixed_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("moead_mixed", _core_part(), _aggregation_part(), _mixed_operator_part())


def build_moead_binary_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("moead_binary", _core_part(), _aggregation_part(), _binary_operator_part())


def build_moead_integer_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("moead_integer", _core_part(), _aggregation_part(), _integer_operator_part())


__all__ = [
    "build_moead_config_space",
    "build_moead_constrained_config_space",
    "build_moead_permutation_config_space",
    "build_moead_mixed_config_space",
    "build_moead_binary_config_space",
    "build_moead_integer_config_space",
]
