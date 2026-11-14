"""
SMPSO configuration space builders for tuning.

SMPSO uses particle swarm (velocity-based) movement rather than traditional
crossover/mutation. The mixed variant exposes mutation-only tuning.
"""

from __future__ import annotations

from vamos.engine.algorithm.components.constraint_handling import PAIRWISE_TECHNIQUES

from .bridge_space_parts_constraints import constraint_handling_part
from .bridge_space_parts_discrete import external_archive_part
from .config_space import AlgorithmConfigSpace, SpacePart, compose_config_space
from .param_space import Categorical, ConditionalBlock, Int, ParamType, Real

# ---------------------------------------------------------------------------
# SMPSO has a single encoding; no core/operator split needed.
# ---------------------------------------------------------------------------


def _smpso_part() -> SpacePart:
    ext_params, ext_conds, ext_conditions = external_archive_part()
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True, role="population"),
        Int("archive_size", 20, 200, log=True, role="population"),
        Real("inertia", 0.1, 0.9, role="operator_rate"),
        Real("c1", 0.5, 2.5, role="operator_rate"),
        Real("c2", 0.5, 2.5, role="operator_rate"),
        Real("vmax_fraction", 0.1, 1.0, role="operator_rate"),
        Categorical("repair", ["clip", "reflect", "random", "round", "wrap", "midpoint"], role="structural"),
        Real("mutation_prob", 0.01, 0.5, role="operator_rate"),
        Real("mutation_eta", 5.0, 40.0, role="operator_rate"),
        *ext_params,
    ]
    return params, ext_conds, ext_conditions


def _smpso_mixed_part() -> SpacePart:
    ext_params, ext_conds, ext_conditions = external_archive_part()
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True, role="population"),
        Int("archive_size", 20, 200, log=True, role="population"),
        Real("inertia", 0.1, 0.9, role="operator_rate"),
        Real("c1", 0.5, 2.5, role="operator_rate"),
        Real("c2", 0.5, 2.5, role="operator_rate"),
        Real("vmax_fraction", 0.1, 1.0, role="operator_rate"),
        Categorical("mutation", ["mixed"], role="operator"),
        Categorical("perm_mutation", ["swap", "insert", "scramble", "inversion", "displacement", "two_opt"], role="operator"),
        Real("perm_mutation_prob", 0.01, 0.5, role="operator_rate"),
        Categorical("real_mutation", ["gaussian", "uniform_reset", "polynomial"], role="operator"),
        Real("real_mutation_prob", 0.01, 0.5, role="operator_rate"),
        Categorical("int_mutation", ["reset", "creep", "polynomial"], role="operator"),
        Real("int_mutation_prob", 0.01, 0.5, role="operator_rate"),
        Categorical("cat_mutation", ["reset"], role="operator"),
        Real("cat_mutation_prob", 0.01, 0.5, role="operator_rate"),
        *ext_params,
    ]
    conditionals = [
        *ext_conds,
        ConditionalBlock("real_mutation", "gaussian", [Real("real_mutation_sigma_factor", 0.01, 0.5, role="operator_rate")]),
        ConditionalBlock("real_mutation", "polynomial", [Real("real_mutation_eta", 5.0, 40.0, role="operator_rate")]),
        ConditionalBlock("int_mutation", "creep", [Int("int_mutation_step", 1, 5, role="operator_rate")]),
        ConditionalBlock("int_mutation", "polynomial", [Real("int_mutation_eta", 5.0, 40.0, role="operator_rate")]),
    ]
    return params, conditionals, ext_conditions


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_smpso_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("smpso", _smpso_part())


def build_smpso_constrained_config_space() -> AlgorithmConfigSpace:
    """The real-coded space with the techniques that judge single solutions, as Evolver's ``MOPSOConstrained``."""
    return compose_config_space("smpso_constrained", _smpso_part(), constraint_handling_part(PAIRWISE_TECHNIQUES))


def build_smpso_mixed_config_space() -> AlgorithmConfigSpace:
    return compose_config_space("smpso_mixed", _smpso_mixed_part())


__all__ = ["build_smpso_config_space", "build_smpso_mixed_config_space"]
