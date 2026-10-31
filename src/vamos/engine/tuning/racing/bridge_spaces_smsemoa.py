"""
SMS-EMOA configuration space builders for tuning.
"""

from __future__ import annotations

from .bridge_space_parts_discrete import (
    PAIRWISE_REAL_CROSSOVERS_MEDIUM,
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
# Core part (shared by ALL SMS-EMOA encoding variants)
# ---------------------------------------------------------------------------


def _core_part() -> SpacePart:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True, role="population"),
        Int("selection_pressure", 2, 10, role="structural"),
    ]
    arch_params, arch_conds, arch_conditions = external_archive_part()
    return [*params, *arch_params], arch_conds, arch_conditions


# ---------------------------------------------------------------------------
# Encoding-specific operator parts
# ---------------------------------------------------------------------------


def _real_operator_part() -> SpacePart:
    # Parents are mated two by two, so three-parent crossovers cannot run.
    return real_operator_part_medium(crossovers=PAIRWISE_REAL_CROSSOVERS_MEDIUM)


def _mixed_operator_part() -> SpacePart:
    return mixed_operator_part()


def _binary_operator_part() -> SpacePart:
    return binary_operator_part_full()


def _integer_operator_part() -> SpacePart:
    return integer_operator_part_full()


def _permutation_operator_part() -> SpacePart:
    return permutation_operator_part_full()


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_smsemoa_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("smsemoa", _core_part(), _real_operator_part())


def build_smsemoa_mixed_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("smsemoa_mixed", _core_part(), _mixed_operator_part())


def build_smsemoa_binary_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("smsemoa_binary", _core_part(), _binary_operator_part())


def build_smsemoa_integer_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("smsemoa_integer", _core_part(), _integer_operator_part())


def build_smsemoa_permutation_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("smsemoa_permutation", _core_part(), _permutation_operator_part())


__all__ = [
    "build_smsemoa_config_space",
    "build_smsemoa_mixed_config_space",
    "build_smsemoa_binary_config_space",
    "build_smsemoa_integer_config_space",
    "build_smsemoa_permutation_config_space",
]
