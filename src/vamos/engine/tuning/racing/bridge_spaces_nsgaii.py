"""
NSGA-II configuration space builders for tuning.
"""

from __future__ import annotations

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
# Core part (shared by ALL NSGA-II encoding variants)
# ---------------------------------------------------------------------------


def _core_part() -> SpacePart:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True, role="population"),
        Categorical("offspring_ratio", [0, 0.25, 0.5, 0.75, 1.0], role="population"),
        Categorical("selection", ["tournament", "random", "boltzmann", "ranking", "sus"], role="structural"),
    ]
    arch_params, arch_conds, arch_conditions = external_archive_part()
    conditionals = [
        *arch_conds,
        ConditionalBlock("selection", "tournament", [Int("selection_size", 2, 10, role="structural")]),
    ]
    return [*params, *arch_params], conditionals, arch_conditions


# ---------------------------------------------------------------------------
# Encoding-specific operator parts
# ---------------------------------------------------------------------------


def _real_operator_part() -> SpacePart:
    params: list[ParamType] = [
        Categorical("initializer", ["random", "lhs", "scatter", "sobol", "halton", "obl"], role="structural"),
        Categorical(
            "crossover",
            [
                "sbx",
                "blx_alpha",
                "blx_alpha_beta",
                "arithmetic",
                "whole_arithmetic",
                "laplace",
                "fuzzy",
                "pcx",
                "undx",
                "simplex",
            ],
            role="operator",
        ),
        Real("crossover_prob", 0.6, 1.0, role="operator_rate"),
        Categorical(
            "mutation",
            [
                "polynomial",
                "linked_polynomial",
                "non_uniform",
                "gaussian",
                "uniform_reset",
                "cauchy",
                "uniform",
                "levy_flight",
                "power_law",
            ],
            role="operator",
        ),
        Real("mutation_prob_factor", 0.25, 3.0, role="operator_rate"),
        Real("mutation_eta", 5.0, 40.0, role="operator_rate"),
        Categorical("repair", ["clip", "reflect", "random", "round", "wrap", "midpoint"], role="structural"),
    ]
    conditionals = [
        ConditionalBlock("crossover", "sbx", [Real("crossover_eta", 5.0, 40.0)]),
        ConditionalBlock(
            "crossover",
            "blx_alpha",
            [Real("crossover_alpha", 0.0, 1.0)],
        ),
        ConditionalBlock(
            "crossover",
            "blx_alpha_beta",
            [Real("blxab_alpha", 0.0, 1.0), Real("blxab_beta", 0.0, 1.0)],
        ),
        ConditionalBlock(
            "crossover",
            "whole_arithmetic",
            [Real("wa_alpha", 0.0, 1.0)],
        ),
        ConditionalBlock(
            "crossover",
            "laplace",
            [Real("laplace_a", -1.0, 1.0), Real("laplace_b", 0.01, 2.0)],
        ),
        ConditionalBlock(
            "crossover",
            "fuzzy",
            [Real("fuzzy_d", 0.0, 2.0)],
        ),
        ConditionalBlock(
            "crossover",
            "pcx",
            [Real("pcx_sigma_eta", 0.01, 0.5), Real("pcx_sigma_zeta", 0.01, 0.5)],
        ),
        ConditionalBlock(
            "crossover",
            "undx",
            [Real("undx_zeta", 0.1, 1.0), Real("undx_eta", 0.1, 1.0)],
        ),
        ConditionalBlock("crossover", "simplex", [Real("simplex_epsilon", 0.1, 1.0)]),
        ConditionalBlock("mutation", "non_uniform", [Real("nonuniform_perturbation", 0.05, 0.5)]),
        ConditionalBlock("mutation", "gaussian", [Real("gaussian_sigma", 0.001, 0.5)]),
        ConditionalBlock("mutation", "cauchy", [Real("cauchy_gamma", 0.001, 0.5)]),
        ConditionalBlock("mutation", "uniform", [Real("uniform_perturb", 0.01, 0.5)]),
        ConditionalBlock(
            "mutation",
            "levy_flight",
            [Real("levy_beta", 0.5, 2.0), Real("levy_scale", 0.001, 0.1)],
        ),
        ConditionalBlock(
            "mutation",
            "power_law",
            [Real("power_index", 0.5, 5.0)],
        ),
        ConditionalBlock(
            "initializer",
            "scatter",
            [Categorical("scatter_base_size_factor", [0.1, 0.2, 0.3, 0.5, 0.75, 1.0])],
        ),
    ]
    return params, conditionals, [Condition("mutation_eta", "cfg['mutation'] == 'polynomial' or cfg['mutation'] == 'linked_polynomial'")]


def _permutation_operator_part() -> SpacePart:
    return permutation_operator_part_full(mutation_prob_param="mutation_prob_factor", mutation_prob_bounds=(0.25, 3.0))


def _mixed_operator_part() -> SpacePart:
    return mixed_operator_part(
        mutation_prob_param="mutation_prob_factor",
        mutation_prob_bounds=(0.25, 3.0),
    )


def _binary_operator_part() -> SpacePart:
    return binary_operator_part_full(mutation_prob_param="mutation_prob_factor", mutation_prob_bounds=(0.25, 3.0))


def _integer_operator_part() -> SpacePart:
    return integer_operator_part_full(mutation_prob_param="mutation_prob_factor", mutation_prob_bounds=(0.25, 3.0))


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_nsgaii_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("nsgaii", _core_part(), _real_operator_part())


def build_nsgaii_permutation_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("nsgaii_permutation", _core_part(), _permutation_operator_part())


def build_nsgaii_mixed_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("nsgaii_mixed", _core_part(), _mixed_operator_part())


def build_nsgaii_binary_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("nsgaii_binary", _core_part(), _binary_operator_part())


def build_nsgaii_integer_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("nsgaii_integer", _core_part(), _integer_operator_part())


__all__ = [
    "build_nsgaii_config_space",
    "build_nsgaii_permutation_config_space",
    "build_nsgaii_mixed_config_space",
    "build_nsgaii_binary_config_space",
    "build_nsgaii_integer_config_space",
]
