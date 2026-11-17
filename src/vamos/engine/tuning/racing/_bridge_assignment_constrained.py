"""Assignment -> configuration builders of the algorithms that handle constraints themselves: CCMO, C-TAEA, ToP and CMOEA-MS.

Their configurations have no ``mutation_prob_factor`` field, so the grammars' factor reaches them as
the probability ``"<factor>/n"``.
"""

from __future__ import annotations

from typing import Any

from vamos.engine.algorithm.config import CCMOConfig, CMOEAMSConfig, CTAEAConfig, ToPConfig

from ._bridge_assignment_shared import (
    apply_initializer,
    apply_mutation_prob_factor,
    apply_optional_external_archive,
    apply_optional_repair,
    extend_crossover_params,
    extend_mutation_params,
    run_pop_size,
)


def _crossover_or_de(assignment: dict[str, Any]) -> str:
    """The crossover, or ``de`` for the grammars' ``variation: differential_evolution`` and the earlier ``crossover: de``."""
    if assignment.get("variation") == "differential_evolution":
        return "de"
    return str(assignment.get("crossover", "sbx"))


def build_ccmo_config(assignment: dict[str, Any], *, mixed: bool = False) -> CCMOConfig:
    """CCMO's configuration: the DE variation (``de_cr``, ``de_f``) or the GA variation with ``crossover``.

    A missing crossover probability is 1.0, CCMO's default.
    """
    builder = CCMOConfig.builder()
    pop_size = run_pop_size(assignment)
    builder.pop_size(pop_size)
    apply_initializer(builder, assignment, pop_size)

    cross = _crossover_or_de(assignment)
    if cross == "de":
        builder.crossover("de", cr=float(assignment.get("de_cr", 1.0)), f=float(assignment.get("de_f", 0.5)))
    else:
        cross_params: dict[str, Any] = {"prob": float(assignment.get("crossover_prob", 1.0))}
        extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
        builder.crossover(cross, **cross_params)

    mut = str(assignment.get("mutation", "pm"))
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = assignment["mutation_prob"]
    apply_mutation_prob_factor(builder, assignment, mut_params, factor_field=False)
    extend_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)

    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


def build_ctaea_config(assignment: dict[str, Any], *, mixed: bool = False) -> CTAEAConfig:
    """C-TAEA's configuration: ``pop_size`` is the requested number of weight vectors, and the crossover mates two parents.

    A missing crossover probability is 1.0, C-TAEA's default.
    """
    builder = CTAEAConfig.builder()
    pop_size = run_pop_size(assignment)
    builder.pop_size(pop_size)
    apply_initializer(builder, assignment, pop_size)

    cross = str(assignment.get("crossover", "sbx"))
    cross_params: dict[str, Any] = {"prob": float(assignment.get("crossover_prob", 1.0))}
    extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
    builder.crossover(cross, **cross_params)

    mut = str(assignment.get("mutation", "pm"))
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = assignment["mutation_prob"]
    apply_mutation_prob_factor(builder, assignment, mut_params, factor_field=False)
    extend_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)

    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


def build_top_config(assignment: dict[str, Any], *, mixed: bool = False) -> ToPConfig:
    """ToP's configuration: DE with ``de_cr`` and ``de_f``, the mutation of its trial vectors and the switch thresholds.

    A missing ``de_cr``, ``de_f`` or threshold gets the value of PlatEMO's implementation: CR = 1.0,
    F = 0.5, 0.2 for ``delta``, 1/3 for ``Pf`` and 0.9 of the budget.
    """
    builder = ToPConfig.builder()
    pop_size = run_pop_size(assignment)
    builder.pop_size(pop_size)
    apply_initializer(builder, assignment, pop_size)
    builder.crossover("de", cr=float(assignment.get("de_cr", 1.0)), f=float(assignment.get("de_f", 0.5)))

    mut = str(assignment.get("mutation", "pm"))
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = assignment["mutation_prob"]
    apply_mutation_prob_factor(builder, assignment, mut_params, factor_field=False)
    extend_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)

    builder.delta_threshold(float(assignment.get("delta_threshold", 0.2)))
    builder.feasible_ratio_threshold(float(assignment.get("feasible_ratio_threshold", 1.0 / 3.0)))
    builder.phase_one_evaluation_fraction(float(assignment.get("phase_one_evaluation_fraction", 0.9)))
    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


def build_cmoeams_config(assignment: dict[str, Any], *, mixed: bool = False) -> CMOEAMSConfig:
    """CMOEA-MS's configuration: the DE variation (``de_cr``, ``de_f``, ``de_variant``) or the GA variation with ``crossover``.

    A missing crossover probability is 1.0, a missing ``de_cr``, ``de_f`` or ``de_variant`` CR = 1.0,
    F = 0.5 and ``rand_1_bin``, and a missing ``feasible_ratio_threshold`` 0.5: the values of the
    paper and of PlatEMO's implementation.
    """
    builder = CMOEAMSConfig.builder()
    pop_size = run_pop_size(assignment)
    builder.pop_size(pop_size)
    apply_initializer(builder, assignment, pop_size)

    cross = _crossover_or_de(assignment)
    if cross == "de":
        builder.crossover("de", cr=float(assignment.get("de_cr", 1.0)), f=float(assignment.get("de_f", 0.5)))
        builder.de_variant(str(assignment.get("de_variant", "rand_1_bin")))
    else:
        cross_params: dict[str, Any] = {"prob": float(assignment.get("crossover_prob", 1.0))}
        extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
        builder.crossover(cross, **cross_params)

    mut = str(assignment.get("mutation", "pm"))
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = assignment["mutation_prob"]
    apply_mutation_prob_factor(builder, assignment, mut_params, factor_field=False)
    extend_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)

    builder.feasible_ratio_threshold(float(assignment.get("feasible_ratio_threshold", 0.5)))
    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


__all__ = ["build_ccmo_config", "build_cmoeams_config", "build_ctaea_config", "build_top_config"]
