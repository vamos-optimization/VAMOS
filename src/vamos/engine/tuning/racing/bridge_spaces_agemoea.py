"""
AGE-MOEA configuration space builders for tuning.
"""

from __future__ import annotations

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
from .param_space import Int, ParamType

# ---------------------------------------------------------------------------
# Core part (shared by ALL AGE-MOEA encoding variants)
# ---------------------------------------------------------------------------


def _core_part() -> SpacePart:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True, role="population"),
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


def build_agemoea_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("agemoea", _core_part(), _real_operator_part())


def build_agemoea_constrained_config_space() -> AlgorithmConfigSpace:
    """The real-coded space with the constraint-handling technique, as Evolver's ``AGEMOEADoubleConstrained``."""
    return compose_config_space("agemoea_constrained", _core_part(), _real_operator_part(), constraint_handling_part())


def build_agemoea_mixed_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("agemoea_mixed", _core_part(), _mixed_operator_part())


def build_agemoea_permutation_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("agemoea_permutation", _core_part(), _permutation_operator_part())


def build_agemoea_binary_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("agemoea_binary", _core_part(), _binary_operator_part())


def build_agemoea_integer_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("agemoea_integer", _core_part(), _integer_operator_part())


__all__ = [
    "build_agemoea_config_space",
    "build_agemoea_constrained_config_space",
    "build_agemoea_mixed_config_space",
    "build_agemoea_permutation_config_space",
    "build_agemoea_binary_config_space",
    "build_agemoea_integer_config_space",
]
