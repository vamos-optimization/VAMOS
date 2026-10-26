from __future__ import annotations

import math
from typing import Any

import numpy as np

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

from ._bridge_assignment_shared import (
    apply_initializer,
    apply_optional_external_archive,
    apply_optional_repair,
    extend_crossover_params,
    extend_mutation_params,
    has_mixed_mutation_assignment,
)

NSGAII_NAMES = ALGORITHM_VARIANT_GROUPS["nsgaii"]
MOEAD_NAMES = ALGORITHM_VARIANT_GROUPS["moead"] - {"moead_permutation"}
NSGAIII_NAMES = ALGORITHM_VARIANT_GROUPS["nsgaiii"]
SMSEMOA_NAMES = ALGORITHM_VARIANT_GROUPS["smsemoa"]
SPEA2_NAMES = ALGORITHM_VARIANT_GROUPS["spea2"]
IBEA_NAMES = ALGORITHM_VARIANT_GROUPS["ibea"]
SMPSO_NAMES = ALGORITHM_VARIANT_GROUPS["smpso"]
AGEMOEA_NAMES = ALGORITHM_VARIANT_GROUPS["agemoea"]
RVEA_NAMES = ALGORITHM_VARIANT_GROUPS["rvea"]


def build_nsgaii_config(assignment: dict[str, Any], *, mixed: bool = False) -> NSGAIIConfig:
    builder = NSGAIIConfig.builder()
    pop_size = int(assignment["pop_size"])
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
    mut_factor = assignment.get("mutation_prob_factor")
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = assignment["mutation_prob"]
    if mut_factor is not None:
        builder.mutation_prob_factor(float(mut_factor))
    extend_mutation_params(str(mut), assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)

    selection = str(assignment.get("selection", "tournament"))
    if selection == "tournament":
        builder.selection(selection, size=int(assignment.get("selection_size", 2)))
    else:
        builder.selection(selection)

    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


def build_moead_config(assignment: dict[str, Any], *, mixed: bool = False) -> MOEADConfig:
    builder = MOEADConfig.builder()
    builder.pop_size(int(assignment["pop_size"]))
    builder.neighbor_size(int(assignment["neighbor_size"]))
    builder.delta(float(assignment["delta"]))
    builder.replace_limit(int(assignment["replace_limit"]))
    cross = str(assignment["crossover"])
    if cross == "de":
        builder.crossover("de", cr=float(assignment.get("de_cr", 1.0)), f=float(assignment.get("de_f", 0.5)))
    else:
        cross_params: dict[str, Any] = {}
        if "crossover_prob" in assignment:
            cross_params["prob"] = float(assignment["crossover_prob"])
        extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
        builder.crossover(cross, **cross_params)

    mut = str(assignment["mutation"])
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = float(assignment["mutation_prob"])
    extend_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)

    aggregation = str(assignment["aggregation"])
    if aggregation == "pbi":
        builder.aggregation("pbi", theta=float(assignment.get("pbi_theta", 5.0)))
    else:
        builder.aggregation(aggregation)
    apply_optional_external_archive(builder, assignment, int(assignment["pop_size"]))
    return builder.build()


def build_moead_permutation_config(assignment: dict[str, Any]) -> MOEADConfig:
    builder = MOEADConfig.builder()
    builder.pop_size(int(assignment["pop_size"]))
    builder.neighbor_size(int(assignment["neighbor_size"]))
    builder.delta(float(assignment["delta"]))
    builder.replace_limit(int(assignment["replace_limit"]))
    builder.crossover(str(assignment["crossover"]), prob=float(assignment.get("crossover_prob", 0.9)))
    builder.mutation(str(assignment["mutation"]), prob=assignment.get("mutation_prob", 0.1))
    aggregation = str(assignment["aggregation"])
    if aggregation == "pbi":
        builder.aggregation("pbi", theta=float(assignment.get("pbi_theta", 5.0)))
    elif aggregation == "modified_tchebycheff":
        builder.aggregation("modified_tchebycheff", rho=float(assignment.get("mtch_rho", 0.001)))
    else:
        builder.aggregation(aggregation)
    apply_optional_external_archive(builder, assignment, int(assignment["pop_size"]))
    return builder.build()


def build_nsgaiii_config(assignment: dict[str, Any], *, mixed: bool = False) -> NSGAIIIConfig:
    builder = NSGAIIIConfig.builder()
    pop_size = int(assignment["pop_size"])
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
    extend_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)
    builder.selection("tournament", size=int(assignment["selection_pressure"]))
    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


def build_smsemoa_config(assignment: dict[str, Any], *, mixed: bool = False) -> SMSEMOAConfig:
    builder = SMSEMOAConfig.builder()
    pop_size = int(assignment["pop_size"])
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
    extend_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)
    builder.selection("tournament", size=int(assignment["selection_pressure"]))
    builder.reference_point(offset=0.1, adaptive=True)
    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


def build_spea2_config(assignment: dict[str, Any], *, mixed: bool = False) -> SPEA2Config:
    builder = SPEA2Config.builder()
    pop_size = int(assignment["pop_size"])
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
    extend_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)
    builder.selection("tournament", size=int(assignment["selection_pressure"]))
    builder.k_neighbors(int(assignment.get("k_neighbors", max(1, int(np.sqrt(pop_size))))))
    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


def build_ibea_config(assignment: dict[str, Any], *, mixed: bool = False) -> IBEAConfig:
    builder = IBEAConfig.builder()
    pop_size = int(assignment["pop_size"])
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
    pop_size = int(assignment["pop_size"])
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
    if mut in {"pm", "polynomial"}:
        mut_params["eta"] = float(assignment.get("mutation_eta", 20.0))
    elif mixed or has_mixed_mutation_assignment(assignment):
        extend_mutation_params(mut, assignment, mut_params, mixed=True)
    builder.mutation(mut, **mut_params)
    apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


def build_agemoea_config(assignment: dict[str, Any], *, mixed: bool = False) -> AGEMOEAConfig:
    builder = AGEMOEAConfig.builder()
    pop_size = int(assignment["pop_size"])
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
    extend_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)

    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


def build_rvea_config(assignment: dict[str, Any], *, mixed: bool = False) -> RVEAConfig:
    builder = RVEAConfig.builder()
    n_partitions = int(assignment.get("n_partitions", 12))
    n_obj = max(2, int(assignment.get("n_obj", 2)))
    pop_size = int(math.comb(n_partitions + n_obj - 1, n_obj - 1))
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
    extend_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)

    apply_optional_repair(builder, assignment)
    apply_optional_external_archive(builder, assignment, pop_size)
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
    }


__all__ = [
    "build_moead_permutation_config",
    "config_builders",
]
