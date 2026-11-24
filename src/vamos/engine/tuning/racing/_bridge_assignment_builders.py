from __future__ import annotations

import math
from typing import Any

import numpy as np

from vamos.engine.algorithm.components.weight_vectors import largest_weighted_population
from vamos.engine.algorithm.config import (
    AGEMOEAConfig,
    IBEAConfig,
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    RVEAConfig,
    SMPSOConfig,
    SMSEMOAConfig,
    SPEA2Config,
)
from vamos.engine.algorithm.variants import ALGORITHM_VARIANT_GROUPS
from vamos.engine.reference_directions import largest_lattice_divisions, lattice_size

from ._bridge_assignment_constrained import build_ccmo_config, build_cmoeams_config, build_ctaea_config, build_pps_config, build_top_config
from ._bridge_assignment_shared import (
    apply_initializer,
    apply_mutation_prob_factor,
    apply_optional_external_archive,
    apply_optional_repair,
    crossover_or_de,
    extend_crossover_params,
    extend_mutation_params,
    has_mixed_mutation_assignment,
    run_pop_size,
)
from .bridge_space_parts_constraints import apply_constraint_handling, apply_subproblem_update_criterion

NSGAII_NAMES = ALGORITHM_VARIANT_GROUPS["nsgaii"]
MOEAD_NAMES = ALGORITHM_VARIANT_GROUPS["moead"] - {"moead_permutation"}
NSGAIII_NAMES = ALGORITHM_VARIANT_GROUPS["nsgaiii"]
SMSEMOA_NAMES = ALGORITHM_VARIANT_GROUPS["smsemoa"]
SPEA2_NAMES = ALGORITHM_VARIANT_GROUPS["spea2"]
IBEA_NAMES = ALGORITHM_VARIANT_GROUPS["ibea"]
SMPSO_NAMES = ALGORITHM_VARIANT_GROUPS["smpso"]
AGEMOEA_NAMES = ALGORITHM_VARIANT_GROUPS["agemoea"]
RVEA_NAMES = ALGORITHM_VARIANT_GROUPS["rvea"]
CCMO_NAMES = ALGORITHM_VARIANT_GROUPS["ccmo"]
CTAEA_NAMES = ALGORITHM_VARIANT_GROUPS["ctaea"]
TOP_NAMES = ALGORITHM_VARIANT_GROUPS["top"]
CMOEAMS_NAMES = ALGORITHM_VARIANT_GROUPS["cmoeams"]
PPS_NAMES = ALGORITHM_VARIANT_GROUPS["pps"]


def build_nsgaii_config(assignment: dict[str, Any], *, mixed: bool = False) -> NSGAIIConfig:
    builder = NSGAIIConfig.builder()
    pop_size = run_pop_size(assignment)
    builder.pop_size(pop_size)
    apply_initializer(builder, assignment, pop_size)

    raw_offspring_size = assignment.get("offspring_size")
    if raw_offspring_size is not None:
        offspring_size = int(raw_offspring_size)
    else:
        ratio = assignment.get("offspring_ratio", 1.0)
        try:
            ratio_f = float(ratio)
        except (TypeError, ValueError):
            ratio_f = 1.0
        ratio_f = max(0.0, min(1.0, ratio_f))
        offspring_size = int(math.floor(pop_size * ratio_f + 0.5))
    builder.offspring_size(max(1, min(pop_size, int(offspring_size))))

    cross = assignment["crossover"]
    cross_params: dict[str, Any] = {}
    if "crossover_prob" in assignment:
        cross_params["prob"] = float(assignment["crossover_prob"])
    extend_crossover_params(str(cross), assignment, cross_params, mixed=mixed)
    builder.crossover(cross, **cross_params)

    mut = assignment["mutation"]
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = assignment["mutation_prob"]
    apply_mutation_prob_factor(builder, assignment, mut_params, factor_field=True)
    extend_mutation_params(str(mut), assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)

    selection = str(assignment.get("selection", "tournament"))
    if selection == "tournament":
        builder.selection(selection, size=int(assignment.get("selection_size", 2)))
    else:
        builder.selection(selection)

    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    apply_constraint_handling(builder, assignment)
    return builder.build()


def _moead_pop_size(assignment: dict[str, Any]) -> int:
    """The run's ``pop_size``; with ``n_obj`` in the assignment, the largest population up to it that has MOEA/D weights.

    With three or more objectives MOEA/D runs only populations whose weights are a complete
    Das-Dennis lattice or a packaged jMetal file, which the sampled range does not know about.
    """
    pop_size = run_pop_size(assignment)
    if "n_obj" in assignment:
        return largest_weighted_population(pop_size, int(assignment["n_obj"]))
    return pop_size


def _moead_aggregation(builder: Any, assignment: dict[str, Any]) -> None:
    """The aggregation, its parameter, and the normalization of the objectives when it is on.

    Options left at their defaults are left out, so a configuration without them is the one VAMOS
    1.3.0 built, with its task id.
    """
    aggregation = str(assignment["aggregation"])
    params: dict[str, Any] = {}
    if aggregation == "pbi":
        params["theta"] = float(assignment.get("pbi_theta", 5.0))
    elif aggregation == "modified_tchebycheff":
        params["rho"] = float(assignment.get("mtch_rho", 0.001))
    elif aggregation == "augmented_tchebycheff" and float(assignment.get("atch_rho", 1e-4)) != 1e-4:
        params["rho"] = float(assignment["atch_rho"])
    elif aggregation == "inverted_pbi" and float(assignment.get("ipbi_theta", 0.1)) != 0.1:
        params["theta"] = float(assignment["ipbi_theta"])
    if str(assignment.get("normalize_objectives", "false")).strip().lower() == "true":
        params |= {"normalize": True, "epsilon": float(assignment.get("normalization_epsilon", 1e-6))}
    builder.aggregation(aggregation, **params)


def _moead_order(builder: Any, assignment: dict[str, Any]) -> None:
    """The order of the subproblems, where it is not the default random permutation."""
    order = assignment.get("subproblem_order")
    if order is not None and order != "random_permutation":
        builder.subproblem_order(str(order))


def _moead_de(assignment: dict[str, Any]) -> dict[str, Any]:
    """The DE crossover's parameters: ``cr`` and ``f``, and the scheme and base where they are not the defaults."""
    params: dict[str, Any] = {"cr": float(assignment.get("de_cr", 1.0)), "f": float(assignment.get("de_f", 0.5))}
    if assignment.get("de_scheme", "rand_1_bin") != "rand_1_bin":
        params["scheme"] = str(assignment["de_scheme"])
    if assignment.get("de_base", "current") != "current":
        params["base"] = str(assignment["de_base"])
    return params


def build_moead_config(assignment: dict[str, Any], *, mixed: bool = False) -> MOEADConfig:
    builder = MOEADConfig.builder()
    pop_size = _moead_pop_size(assignment)
    builder.pop_size(pop_size)
    # Assignments from spaces without an initializer leave MOEA/D's default.
    if "initializer" in assignment:
        apply_initializer(builder, assignment, pop_size)
    builder.neighbor_size(min(int(assignment["neighbor_size"]), pop_size))
    builder.delta(float(assignment["delta"]))
    builder.replace_limit(int(assignment["replace_limit"]))
    # The grammar's variation (Evolver's MOEADDouble), or the crossover of earlier assignments.
    cross = crossover_or_de(assignment, default="de")
    if cross == "de":
        builder.crossover("de", **_moead_de(assignment))
    else:
        cross_params: dict[str, Any] = {}
        if "crossover_prob" in assignment:
            cross_params["prob"] = float(assignment["crossover_prob"])
        extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
        builder.crossover(cross, **cross_params)

    mut = str(assignment["mutation"])
    mut_params: dict[str, Any] = {}
    # Assignments from the earlier real-coded space give an absolute probability.
    if "mutation_prob" in assignment:
        mut_params["prob"] = float(assignment["mutation_prob"])
    apply_mutation_prob_factor(builder, assignment, mut_params, factor_field=True)
    extend_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)

    _moead_aggregation(builder, assignment)
    _moead_order(builder, assignment)
    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    apply_subproblem_update_criterion(builder, assignment)
    return builder.build()


def build_moead_permutation_config(assignment: dict[str, Any]) -> MOEADConfig:
    builder = MOEADConfig.builder()
    pop_size = _moead_pop_size(assignment)
    builder.pop_size(pop_size)
    builder.neighbor_size(min(int(assignment["neighbor_size"]), pop_size))
    builder.delta(float(assignment["delta"]))
    builder.replace_limit(int(assignment["replace_limit"]))
    builder.crossover(str(assignment["crossover"]), prob=float(assignment.get("crossover_prob", 0.9)))
    builder.mutation(str(assignment["mutation"]), prob=assignment.get("mutation_prob", 0.1))
    _moead_aggregation(builder, assignment)
    _moead_order(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


def build_nsgaiii_config(assignment: dict[str, Any], *, mixed: bool = False) -> NSGAIIIConfig:
    builder = NSGAIIIConfig.builder()
    pop_size = run_pop_size(assignment)
    builder.pop_size(pop_size)
    apply_initializer(builder, assignment, pop_size)
    cross = str(assignment["crossover"])
    cross_params: dict[str, Any] = {}
    if "crossover_prob" in assignment:
        cross_params["prob"] = float(assignment["crossover_prob"])
    extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
    builder.crossover(cross, **cross_params)
    mut = str(assignment["mutation"])
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = float(assignment["mutation_prob"])
    apply_mutation_prob_factor(builder, assignment, mut_params, factor_field=True)
    extend_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)
    # VAMOS 1.5.0: the spaces choose random parents (the default) or a tournament. An assignment
    # of an earlier space names only the tournament size, and still builds that tournament.
    selection = str(assignment.get("selection", "tournament" if "selection_pressure" in assignment else "random"))
    if selection == "tournament":
        builder.selection(selection, size=int(assignment.get("selection_pressure", 2)))
    else:
        builder.selection(selection)
    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    apply_constraint_handling(builder, assignment)
    return builder.build()


def build_smsemoa_config(assignment: dict[str, Any], *, mixed: bool = False) -> SMSEMOAConfig:
    builder = SMSEMOAConfig.builder()
    pop_size = run_pop_size(assignment)
    builder.pop_size(pop_size)
    apply_initializer(builder, assignment, pop_size)
    cross = str(assignment["crossover"])
    cross_params: dict[str, Any] = {}
    if "crossover_prob" in assignment:
        cross_params["prob"] = float(assignment["crossover_prob"])
    extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
    builder.crossover(cross, **cross_params)
    mut = str(assignment["mutation"])
    mut_params: dict[str, Any] = {}
    # Assignments from the earlier real-coded spaces give an absolute probability.
    if "mutation_prob" in assignment:
        mut_params["prob"] = float(assignment["mutation_prob"])
    apply_mutation_prob_factor(builder, assignment, mut_params, factor_field=True)
    extend_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)
    # Assignments from earlier spaces name only the tournament size and used an offset of 0.1.
    selection = str(assignment.get("selection", "tournament"))
    if selection == "tournament":
        builder.selection(selection, size=int(assignment["selection_pressure"]))
    else:
        builder.selection(selection)
    builder.reference_point(offset=float(assignment.get("reference_point_offset", 0.1)), adaptive=True)
    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    apply_constraint_handling(builder, assignment)
    return builder.build()


def build_spea2_config(assignment: dict[str, Any], *, mixed: bool = False) -> SPEA2Config:
    builder = SPEA2Config.builder()
    pop_size = run_pop_size(assignment)
    builder.pop_size(pop_size)
    builder.archive_size(int(assignment.get("archive_size", pop_size)))
    apply_initializer(builder, assignment, pop_size)
    cross = str(assignment["crossover"])
    cross_params: dict[str, Any] = {}
    if "crossover_prob" in assignment:
        cross_params["prob"] = float(assignment["crossover_prob"])
    extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
    builder.crossover(cross, **cross_params)
    mut = str(assignment["mutation"])
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = float(assignment["mutation_prob"])
    apply_mutation_prob_factor(builder, assignment, mut_params, factor_field=True)
    extend_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)
    # SPEA2 mates with its own binary tournament: a ``selection_pressure`` saved from an earlier space has no effect.
    builder.selection("tournament")
    builder.k_neighbors(int(assignment.get("k_neighbors", max(1, int(np.sqrt(pop_size))))))
    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


def build_ibea_config(assignment: dict[str, Any], *, mixed: bool = False) -> IBEAConfig:
    builder = IBEAConfig.builder()
    pop_size = run_pop_size(assignment)
    builder.pop_size(pop_size)
    apply_initializer(builder, assignment, pop_size)
    cross = str(assignment["crossover"])
    cross_params: dict[str, Any] = {}
    if "crossover_prob" in assignment:
        cross_params["prob"] = float(assignment["crossover_prob"])
    extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
    builder.crossover(cross, **cross_params)
    mut = str(assignment["mutation"])
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = float(assignment["mutation_prob"])
    apply_mutation_prob_factor(builder, assignment, mut_params, factor_field=True)
    extend_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)
    builder.selection("tournament", size=int(assignment["selection_pressure"]))
    builder.indicator(str(assignment.get("indicator", "eps")))
    builder.kappa(float(assignment.get("kappa", 0.05)))
    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


def build_smpso_config(assignment: dict[str, Any], *, mixed: bool = False) -> SMPSOConfig:
    builder = SMPSOConfig.builder()
    pop_size = run_pop_size(assignment)
    builder.pop_size(pop_size)
    builder.archive_size(int(assignment.get("archive_size", pop_size)))
    builder.inertia(float(assignment["inertia"]))
    builder.c1(float(assignment["c1"]))
    builder.c2(float(assignment["c2"]))
    builder.vmax_fraction(float(assignment["vmax_fraction"]))
    mut = str(assignment.get("mutation", "pm"))
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = float(assignment["mutation_prob"])
    apply_mutation_prob_factor(builder, assignment, mut_params, factor_field=False)
    if mut in {"pm", "polynomial"}:
        mut_params["eta"] = float(assignment.get("mutation_eta", 20.0))
    elif mixed or has_mixed_mutation_assignment(assignment):
        extend_mutation_params(mut, assignment, mut_params, mixed=True)
    builder.mutation(mut, **mut_params)
    apply_optional_external_archive(builder, assignment, pop_size)
    apply_constraint_handling(builder, assignment)
    return builder.build()


def build_agemoea_config(assignment: dict[str, Any], *, mixed: bool = False) -> AGEMOEAConfig:
    builder = AGEMOEAConfig.builder()
    pop_size = run_pop_size(assignment)
    builder.pop_size(pop_size)
    apply_initializer(builder, assignment, pop_size)

    cross = str(assignment.get("crossover", "sbx"))
    cross_params: dict[str, Any] = {}
    if "crossover_prob" in assignment:
        cross_params["prob"] = float(assignment["crossover_prob"])
    extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
    builder.crossover(cross, **cross_params)

    mut = str(assignment.get("mutation", "pm"))
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = assignment["mutation_prob"]
    apply_mutation_prob_factor(builder, assignment, mut_params, factor_field=True)
    extend_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)

    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    apply_constraint_handling(builder, assignment)
    if "replacement" in assignment:
        builder.variant(assignment["replacement"])
    return builder.build()


def _rvea_population(assignment: dict[str, Any]) -> tuple[int, int]:
    """RVEA's population and partitions: one individual per direction of a complete Das-Dennis lattice.

    The lattice is the largest with at most the run's ``pop_size`` directions (100 without it) for the
    assignment's ``n_obj`` (two objectives without it). An assignment with ``n_partitions``, saved from
    a space of VAMOS 1.2.1, keeps that lattice.
    """
    n_obj = max(2, int(assignment.get("n_obj", 2)))
    if "n_partitions" in assignment:
        n_partitions = int(assignment["n_partitions"])
    else:
        n_partitions = largest_lattice_divisions(run_pop_size(assignment), n_obj)
    return lattice_size(n_obj, n_partitions), n_partitions


def build_rvea_config(assignment: dict[str, Any], *, mixed: bool = False) -> RVEAConfig:
    builder = RVEAConfig.builder()
    pop_size, n_partitions = _rvea_population(assignment)
    builder.pop_size(pop_size)
    builder.n_partitions(n_partitions)
    builder.alpha(float(assignment.get("alpha", 2.0)))
    builder.adapt_freq(float(assignment.get("adapt_freq", 0.1)))
    apply_initializer(builder, assignment, pop_size)

    cross = str(assignment.get("crossover", "sbx"))
    cross_params: dict[str, Any] = {}
    if "crossover_prob" in assignment:
        cross_params["prob"] = float(assignment["crossover_prob"])
    extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
    builder.crossover(cross, **cross_params)

    mut = str(assignment.get("mutation", "pm"))
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = assignment["mutation_prob"]
    apply_mutation_prob_factor(builder, assignment, mut_params, factor_field=True)
    extend_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)

    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    apply_constraint_handling(builder, assignment)
    return builder.build()


def config_builders() -> dict[str, tuple[set[str], Any]]:
    return {
        "nsgaii": (NSGAII_NAMES, build_nsgaii_config),
        "moead": (MOEAD_NAMES, build_moead_config),
        "nsgaiii": (NSGAIII_NAMES, build_nsgaiii_config),
        "smsemoa": (SMSEMOA_NAMES, build_smsemoa_config),
        "spea2": (SPEA2_NAMES, build_spea2_config),
        "ibea": (IBEA_NAMES, build_ibea_config),
        "smpso": (SMPSO_NAMES, build_smpso_config),
        "agemoea": (AGEMOEA_NAMES, build_agemoea_config),
        "rvea": (RVEA_NAMES, build_rvea_config),
        "ccmo": (CCMO_NAMES, build_ccmo_config),
        "ctaea": (CTAEA_NAMES, build_ctaea_config),
        "top": (TOP_NAMES, build_top_config),
        "cmoeams": (CMOEAMS_NAMES, build_cmoeams_config),
        "pps": (PPS_NAMES, build_pps_config),
    }


__all__ = [
    "build_moead_permutation_config",
    "config_builders",
]
