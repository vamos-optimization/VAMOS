"""
MOEA/D configuration space builders for tuning.
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
from .param_space import Categorical, ConditionalBlock, Int, ParamType, Real

# ---------------------------------------------------------------------------
# Core part (shared by ALL MOEA/D encoding variants)
# ---------------------------------------------------------------------------


def _core_part() -> SpacePart:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True, role="population"),
        Int("neighbor_size", 5, 40, role="population"),
        Real("delta", 0.5, 0.95, role="operator_rate"),
        Int("replace_limit", 1, 5, role="population"),
    ]
    arch_params, arch_conds, arch_conditions = external_archive_part()
    return [*params, *arch_params], arch_conds, arch_conditions


def _aggregation_part(*, extended: bool = False) -> SpacePart:
    """Aggregation params.  *extended* adds ``modified_tchebycheff`` (permutation variant)."""
    choices = ["tchebycheff", "weighted_sum", "pbi"]
    if extended:
        choices.append("modified_tchebycheff")
    conditionals: list[ConditionalBlock] = [
        ConditionalBlock("aggregation", "pbi", [Real("pbi_theta", 1.0, 10.0, role="operator_rate")]),
    ]
    if extended:
        conditionals.append(
            ConditionalBlock("aggregation", "modified_tchebycheff", [Real("mtch_rho", 0.0001, 0.1, role="operator_rate")]),
        )
    return [Categorical("aggregation", choices, role="structural")], conditionals, []


# ---------------------------------------------------------------------------
# Encoding-specific operator parts
# ---------------------------------------------------------------------------


def _real_operator_part() -> SpacePart:
    params: list[ParamType] = [
        Categorical("crossover", ["sbx", "de"], role="operator"),
        Categorical("mutation", ["polynomial", "gaussian", "non_uniform", "cauchy", "uniform"], role="operator"),
        Real("mutation_prob", 0.01, 0.5, role="operator_rate"),
    ]
    conditionals = [
        ConditionalBlock("mutation", "polynomial", [Real("mutation_eta", 5.0, 40.0, role="operator_rate")]),
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
    return params, conditionals, []


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


def build_moead_permutation_config_space() -> AlgorithmConfigSpace:
    return compose_config_space(
        "moead_permutation",
        _core_part(),
        _aggregation_part(extended=True),
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
    "build_moead_permutation_config_space",
    "build_moead_mixed_config_space",
    "build_moead_binary_config_space",
    "build_moead_integer_config_space",
]
