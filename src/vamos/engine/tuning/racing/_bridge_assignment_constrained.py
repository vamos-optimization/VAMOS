"""Assignment -> configuration builders of the algorithms that handle constraints themselves: CCMO, C-TAEA, ToP, CMOEA-MS and PPS.

Their configurations have no ``mutation_prob_factor`` field, so the grammars' factor reaches them as
the probability ``"<factor>/n"``.
"""

from __future__ import annotations

from typing import Any

from vamos.engine.algorithm.components.weight_vectors import largest_weighted_population
from vamos.engine.algorithm.config import CCMOConfig, CMOEAMSConfig, CTAEAConfig, PPSConfig, ToPConfig

from ._bridge_assignment_shared import (
    apply_initializer,
    apply_mutation_prob_factor,
    apply_optional_external_archive,
    apply_optional_repair,
    crossover_or_de,
    extend_crossover_params,
    extend_mutation_params,
    run_pop_size,
)


def build_ccmo_config(assignment: dict[str, Any], *, mixed: bool = False) -> CCMOConfig:
    """CCMO's configuration: the DE variation (``de_cr``, ``de_f``) or the GA variation with ``crossover``.

    A missing crossover probability is 1.0, CCMO's default.
    """
    builder = CCMOConfig.builder()
    pop_size = run_pop_size(assignment)
    builder.pop_size(pop_size)
    apply_initializer(builder, assignment, pop_size)

    cross = crossover_or_de(assignment)
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

    cross = crossover_or_de(assignment)
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


def build_pps_config(assignment: dict[str, Any], *, mixed: bool = False) -> PPSConfig:
    """PPS's configuration: DE with ``de_cr`` and ``de_f``, the mutation of the new solutions, the neighbourhoods and the stages.

    A missing value gets the paper's: CR = 1.0 and F = 0.5; ``neighbor_size`` 30 (at most
    ``pop_size``), ``delta`` 0.9 and ``replace_limit`` 2; ``switch_threshold`` 1e-3 and
    ``switch_window`` 20; ``feasibility_ratio_threshold`` 0.95, ``tao`` 0.1, ``cp`` 2 and
    ``decay_fraction`` 0.8. With ``n_obj`` in the assignment, the population is the largest up to
    the run's ``pop_size`` that has weight vectors, as MOEA/D's.
    """
    builder = PPSConfig.builder()
    pop_size = run_pop_size(assignment)
    if "n_obj" in assignment:
        pop_size = largest_weighted_population(pop_size, int(assignment["n_obj"]))
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

    builder.neighbor_size(min(int(assignment.get("neighbor_size", 30)), pop_size))
    builder.delta(float(assignment.get("delta", 0.9)))
    builder.replace_limit(int(assignment.get("replace_limit", 2)))
    builder.switch_threshold(float(assignment.get("switch_threshold", 1e-3)))
    builder.switch_window(int(assignment.get("switch_window", 20)))
    builder.feasibility_ratio_threshold(float(assignment.get("feasibility_ratio_threshold", 0.95)))
    builder.tao(float(assignment.get("tao", 0.1)))
    builder.cp(float(assignment.get("cp", 2.0)))
    builder.decay_fraction(float(assignment.get("decay_fraction", 0.8)))
    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


__all__ = ["build_ccmo_config", "build_cmoeams_config", "build_ctaea_config", "build_pps_config", "build_top_config"]
