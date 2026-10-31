from __future__ import annotations

import math
from typing import Any

_TUNING_ARCHIVE_PRUNE_POLICIES = {"crowding", "hv", "mc_hv", "knn", "maxmin", "ref_dirs"}
_MIXED_CROSSOVER_ASSIGNMENT_KEYS = (
    "perm_crossover",
    "perm_crossover_prob",
    "real_crossover",
    "real_crossover_prob",
    "int_crossover",
    "int_crossover_prob",
    "int_crossover_eta",
    "cat_crossover",
    "cat_crossover_prob",
)
_MIXED_MUTATION_ASSIGNMENT_KEYS = (
    "perm_mutation",
    "perm_mutation_prob",
    "real_mutation",
    "real_mutation_prob",
    "real_mutation_sigma",
    "real_mutation_sigma_factor",
    "real_mutation_eta",
    "int_mutation",
    "int_mutation_prob",
    "int_mutation_step",
    "int_mutation_eta",
    "cat_mutation",
    "cat_mutation_prob",
)
_MIXED_CROSSOVER_RATE_KEYS = (
    "perm_crossover_prob",
    "real_crossover_prob",
    "int_crossover_prob",
    "cat_crossover_prob",
)
_MIXED_MUTATION_RATE_KEYS = (
    "perm_mutation_prob",
    "real_mutation_prob",
    "int_mutation_prob",
    "cat_mutation_prob",
)


def sanitize_assignment(assignment: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(assignment)
    use_external_archive = bool(sanitized.get("use_external_archive", False))
    archive_unbounded = bool(sanitized.get("archive_unbounded", False))
    if not use_external_archive or archive_unbounded:
        sanitized.pop("archive_prune_policy", None)
    if any(key in sanitized for key in _MIXED_CROSSOVER_ASSIGNMENT_KEYS):
        sanitized.setdefault("crossover", "mixed")
    if any(key in sanitized for key in _MIXED_MUTATION_ASSIGNMENT_KEYS):
        sanitized.setdefault("mutation", "mixed")
    if any(key in sanitized for key in _MIXED_CROSSOVER_RATE_KEYS):
        sanitized.pop("crossover_prob", None)
    if any(key in sanitized for key in _MIXED_MUTATION_RATE_KEYS):
        sanitized.pop("mutation_prob", None)
        sanitized.pop("mutation_prob_factor", None)
    return sanitized


def apply_initializer(builder: Any, assignment: dict[str, Any], pop_size: int) -> None:
    initializer = str(assignment.get("initializer", "random")).strip().lower()
    if initializer == "random":
        builder.initializer("random")
    elif initializer == "lhs":
        builder.initializer("lhs")
    elif initializer == "scatter":
        factor = assignment.get("scatter_base_size_factor", 0.25)
        try:
            factor_f = float(factor)
        except (TypeError, ValueError):
            factor_f = 0.25
        factor_f = max(0.01, min(1.0, factor_f))
        base_size = int(math.floor(pop_size * factor_f + 0.5))
        base_size = max(2, min(pop_size, base_size))
        builder.initializer("scatter", base_size=base_size)
    elif initializer == "sobol":
        builder.initializer("sobol")
    elif initializer == "halton":
        builder.initializer("halton")
    elif initializer in {"obl", "opposition"}:
        builder.initializer("obl")


def apply_optional_external_archive(builder: Any, assignment: dict[str, Any], pop_size: int) -> None:
    use_external_archive = bool(assignment.get("use_external_archive", False))
    if not use_external_archive:
        return
    archive_unbounded = bool(assignment.get("archive_unbounded", False))
    if archive_unbounded:
        builder.external_archive(capacity=None)
        return
    pruning = str(assignment.get("archive_prune_policy", "crowding"))
    if pruning not in _TUNING_ARCHIVE_PRUNE_POLICIES:
        valid = ", ".join(sorted(_TUNING_ARCHIVE_PRUNE_POLICIES))
        raise ValueError(f"Unsupported pruning '{pruning}'. Expected one of: {valid}.")
    builder.external_archive(capacity=pop_size, pruning=pruning)


def apply_optional_repair(builder: Any, assignment: dict[str, Any]) -> None:
    repair = assignment.get("repair")
    if str(assignment.get("crossover", "")).strip().lower() == "blx_alpha" and assignment.get("blx_repair") is not None:
        repair = assignment.get("blx_repair")
    if repair is None:
        return
    repair_name = str(repair).strip().lower()
    if repair_name and repair_name not in {"none", "disabled", "false", "0"}:
        builder.repair(repair_name)


def _extend_real_crossover_params(cross: str, assignment: dict[str, Any], cross_params: dict[str, Any]) -> None:
    if cross == "sbx":
        cross_params["eta"] = float(assignment.get("crossover_eta", 20.0))
    elif cross == "blx_alpha":
        cross_params["alpha"] = float(assignment.get("crossover_alpha", 0.5))
    elif cross == "pcx":
        cross_params["sigma_eta"] = float(assignment.get("pcx_sigma_eta", 0.1))
        cross_params["sigma_zeta"] = float(assignment.get("pcx_sigma_zeta", 0.1))
    elif cross == "undx":
        cross_params["zeta"] = float(assignment.get("undx_zeta", 0.5))
        cross_params["eta"] = float(assignment.get("undx_eta", 0.35))
    elif cross == "simplex":
        cross_params["epsilon"] = float(assignment.get("simplex_epsilon", 2.0))
    elif cross == "blx_alpha_beta":
        cross_params["alpha"] = float(assignment.get("blxab_alpha", 0.75))
        cross_params["beta"] = float(assignment.get("blxab_beta", 0.25))
    elif cross == "whole_arithmetic":
        cross_params["alpha"] = float(assignment.get("wa_alpha", 0.5))
    elif cross == "laplace":
        cross_params["a"] = float(assignment.get("laplace_a", 0.0))
        cross_params["b"] = float(assignment.get("laplace_b", 0.5))
    elif cross == "fuzzy":
        cross_params["d"] = float(assignment.get("fuzzy_d", 0.5))


def _extend_mutation_params(mut: str, assignment: dict[str, Any], mut_params: dict[str, Any]) -> None:
    if mut in {"pm", "polynomial", "linked_polynomial"}:
        mut_params["eta"] = float(assignment.get("mutation_eta", 20.0))
    elif mut == "creep":
        mut_params["step"] = int(assignment.get("creep_step", 1))
    elif mut == "non_uniform":
        mut_params["perturbation"] = float(assignment.get("nonuniform_perturbation", 0.5))
    elif mut == "gaussian":
        mut_params["sigma"] = float(assignment.get("gaussian_sigma", 1.0))
    elif mut == "cauchy":
        mut_params["gamma"] = float(assignment.get("cauchy_gamma", 0.1))
    elif mut == "uniform":
        mut_params["perturb"] = float(assignment.get("uniform_perturb", 0.1))
    elif mut == "levy_flight":
        mut_params["beta"] = float(assignment.get("levy_beta", 1.5))
        mut_params["scale"] = float(assignment.get("levy_scale", 0.01))
    elif mut == "power_law":
        mut_params["index"] = float(assignment.get("power_index", 1.0))


def has_mixed_crossover_assignment(assignment: dict[str, Any]) -> bool:
    return any(key in assignment for key in _MIXED_CROSSOVER_ASSIGNMENT_KEYS)


def has_mixed_mutation_assignment(assignment: dict[str, Any]) -> bool:
    return any(key in assignment for key in _MIXED_MUTATION_ASSIGNMENT_KEYS)


def extend_crossover_params(
    cross: str,
    assignment: dict[str, Any],
    cross_params: dict[str, Any],
    *,
    mixed: bool = False,
) -> None:
    normalized = str(cross).strip().lower()
    if mixed or has_mixed_crossover_assignment(assignment):
        for key in _MIXED_CROSSOVER_ASSIGNMENT_KEYS:
            if key in assignment:
                cross_params[key] = assignment[key]
        return
    _extend_real_crossover_params(normalized, assignment, cross_params)


def extend_mutation_params(
    mut: str,
    assignment: dict[str, Any],
    mut_params: dict[str, Any],
    *,
    mixed: bool = False,
) -> None:
    normalized = str(mut).strip().lower()
    if mixed or has_mixed_mutation_assignment(assignment):
        for key in _MIXED_MUTATION_ASSIGNMENT_KEYS:
            if key in assignment:
                mut_params[key] = assignment[key]
        return
    _extend_mutation_params(normalized, assignment, mut_params)


__all__ = [
    "apply_initializer",
    "apply_optional_external_archive",
    "apply_optional_repair",
    "extend_crossover_params",
    "extend_mutation_params",
    "has_mixed_mutation_assignment",
    "sanitize_assignment",
]
