"""
RVEA configuration space builders for tuning.
"""

from __future__ import annotations

from vamos.engine.algorithm.components.constraint_handling import PAIRWISE_TECHNIQUES

from .bridge_space_parts_constraints import constraint_handling_part
from .bridge_space_parts_discrete import (
    binary_operator_part_full,
    external_archive_part,
    integer_operator_part_full,
    mixed_operator_part,
    permutation_operator_part_full,
    real_operator_part_medium,
)
from .config_space import AlgorithmConfigSpace, SpacePart, compose_config_space
from .param_space import Int, ParamType, Real

# ---------------------------------------------------------------------------
# Core part (shared by ALL RVEA encoding variants)
# ---------------------------------------------------------------------------


def _core_part() -> SpacePart:
    params: list[ParamType] = [
        Int("n_partitions", 4, 12, role="structural"),
        Real("alpha", 1.0, 4.0, role="operator_rate"),
        Real("adapt_freq", 0.05, 0.3, role="adaptive"),
    ]
    arch_params, arch_conds, arch_conditions = external_archive_part()
    return [*params, *arch_params], arch_conds, arch_conditions


# ---------------------------------------------------------------------------
# Encoding-specific operator parts
# ---------------------------------------------------------------------------


def _real_operator_part() -> SpacePart:
    return real_operator_part_medium()


def _mixed_operator_part() -> SpacePart:
    return mixed_operator_part()


def _permutation_operator_part() -> SpacePart:
    return permutation_operator_part_full()


def _binary_operator_part() -> SpacePart:
    return binary_operator_part_full()


def _integer_operator_part() -> SpacePart:
    return integer_operator_part_full()


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_rvea_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("rvea", _core_part(), _real_operator_part())


def build_rvea_constrained_config_space() -> AlgorithmConfigSpace:
    """The real-coded space with the techniques that judge single solutions, as Evolver's ``RVEADoubleConstrained``."""
    return compose_config_space("rvea_constrained", _core_part(), _real_operator_part(), constraint_handling_part(PAIRWISE_TECHNIQUES))


def build_rvea_mixed_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("rvea_mixed", _core_part(), _mixed_operator_part())


def build_rvea_permutation_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("rvea_permutation", _core_part(), _permutation_operator_part())


def build_rvea_binary_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("rvea_binary", _core_part(), _binary_operator_part())


def build_rvea_integer_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("rvea_integer", _core_part(), _integer_operator_part())


__all__ = [
    "build_rvea_config_space",
    "build_rvea_constrained_config_space",
    "build_rvea_mixed_config_space",
    "build_rvea_permutation_config_space",
    "build_rvea_binary_config_space",
    "build_rvea_integer_config_space",
]
