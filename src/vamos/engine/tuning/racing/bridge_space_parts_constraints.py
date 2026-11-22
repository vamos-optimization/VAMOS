"""How the bridges read the constraint-handling technique and MOEA/D's subproblem update criterion.

The ``<algorithm>_constrained`` grammars offer the technique and its parameters with the names and
ranges of Evolver's spaces (parameter ``constraintHandling`` of ``NSGAIIDoubleConstrained.yaml``),
in snake case; the parameters are scale-free: quantiles, fractions of the run and probabilities. The
grammars of NSGA-II, NSGA-III, SMS-EMOA and AGE-MOEA also offer techniques that Evolver does not
have, ``static_penalty``, ``dynamic_penalty``, ``violation_objective`` and ``shift_penalty``, and NSGA-II's
``two_rankings``, ``infeasibility_driven``, ``adaptive_tradeoff``, ``multiple_constraint_ranking`` and
``ensemble`` too; the coefficients of the first two multiply violations, so they depend on the
problem's scale.
"""

from __future__ import annotations

from typing import Any

# Space parameter -> parameter of the technique, for the techniques that have parameters.
CONSTRAINT_HANDLING_SPACE_PARAMETERS: dict[str, dict[str, str]] = {
    "epsilon_constrained": {
        "epsilon_initial_quantile": "initial_quantile",
        "epsilon_exponent": "exponent",
        "epsilon_decay_fraction": "decay_fraction",
    },
    "stochastic_ranking": {"stochastic_ranking_probability": "probability"},
    "push_pull": {
        "push_pull_change_threshold": "change_threshold",
        "push_pull_window_fraction": "window_fraction",
        "push_pull_decay_fraction": "decay_fraction",
        "push_pull_exponent": "exponent",
        "push_pull_feasibility_threshold": "feasibility_threshold",
        "push_pull_reduction_rate": "reduction_rate",
    },
    "static_penalty": {"static_penalty_coefficient": "coefficient"},
    "dynamic_penalty": {
        "dynamic_penalty_constant": "constant",
        "dynamic_penalty_alpha": "alpha",
        "dynamic_penalty_beta": "beta",
    },
    "shift_penalty": {"shift_penalty_constant": "constant"},
    "infeasibility_driven": {"infeasibility_driven_proportion": "proportion"},
    "ensemble": {
        "ensemble_nds_weight": "nds_weight",
        "ensemble_cdp_weight": "cdp_weight",
        "ensemble_mcr_weight": "mcr_weight",
        "ensemble_dpf_weight": "dpf_weight",
    },
}

# The space parameters of violation_objective's options (Dam and Truong), conditional children of violation_objective:
# the norm, the ratio objective (none, with equal weights, or with the CDF's, whose alpha is a child of cdf), and
# NSGA-II's feasible-first survival.
VIOLATION_OBJECTIVE_SPACE_PARAMETERS: tuple[str, ...] = (
    "violation_objective_norm",
    "violation_objective_ratio",
    "violation_objective_cdf_alpha",
    "violation_objective_feasible_first",
)


def constraint_handling_from_assignment(assignment: dict[str, Any]) -> tuple[str, dict[str, float | str | bool]] | None:
    """The technique of an assignment as ``(name, parameters)``; None if the assignment has none."""
    name = assignment.get("constraint_handling")
    if name is None:
        return None
    technique = str(name)
    mapping = CONSTRAINT_HANDLING_SPACE_PARAMETERS.get(technique, {})
    missing = [key for key in mapping if key not in assignment]
    if missing:
        raise ValueError(f"constraint_handling {technique!r} needs {', '.join(missing)} in the assignment.")
    params: dict[str, float | str | bool] = {parameter: float(assignment[key]) for key, parameter in mapping.items()}
    if technique == "violation_objective":
        if "violation_objective_norm" in assignment:
            params["violation_norm"] = str(assignment["violation_objective_norm"])
        ratio = assignment.get("violation_objective_ratio", "none")
        if ratio in ("equal", "cdf"):
            params["ratio_objective"] = True
        if ratio == "cdf":
            params["cdf_alpha"] = float(assignment["violation_objective_cdf_alpha"])
        if "violation_objective_feasible_first" in assignment:
            params["feasible_first"] = bool(assignment["violation_objective_feasible_first"])
    return technique, params


def apply_constraint_handling(builder: Any, assignment: dict[str, Any]) -> None:
    """Give a configuration builder the technique of an assignment, if it has one."""
    technique = constraint_handling_from_assignment(assignment)
    if technique is not None:
        builder.constraint_handling(technique[0], **technique[1])


# Space parameter -> parameter of ``improved_epsilon``.
SUBPROBLEM_UPDATE_SPACE_PARAMETERS: dict[str, str] = {
    "improved_epsilon_tao": "tao",
    "improved_epsilon_feasibility_ratio_threshold": "feasibility_ratio_threshold",
    "improved_epsilon_decay_fraction": "decay_fraction",
}


def apply_subproblem_update_criterion(builder: Any, assignment: dict[str, Any]) -> None:
    """Give a MOEA/D configuration builder the subproblem update criterion of an assignment, if it has one."""
    name = assignment.get("subproblem_update_criterion")
    if name is None:
        return
    criterion = str(name)
    params: dict[str, float] = {}
    if criterion == "improved_epsilon":
        missing = [key for key in SUBPROBLEM_UPDATE_SPACE_PARAMETERS if key not in assignment]
        if missing:
            raise ValueError(f"subproblem_update_criterion 'improved_epsilon' needs {', '.join(missing)} in the assignment.")
        params = {parameter: float(assignment[key]) for key, parameter in SUBPROBLEM_UPDATE_SPACE_PARAMETERS.items()}
    builder.subproblem_update_criterion(criterion, **params)


__all__ = [
    "CONSTRAINT_HANDLING_SPACE_PARAMETERS",
    "SUBPROBLEM_UPDATE_SPACE_PARAMETERS",
    "VIOLATION_OBJECTIVE_SPACE_PARAMETERS",
    "apply_constraint_handling",
    "apply_subproblem_update_criterion",
    "constraint_handling_from_assignment",
]
