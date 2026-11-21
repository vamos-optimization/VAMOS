"""How the bridges read the constraint-handling technique and MOEA/D's subproblem update criterion.

The ``<algorithm>_constrained`` grammars offer the technique and its parameters with the names and
ranges of Evolver's spaces (parameter ``constraintHandling`` of ``NSGAIIDoubleConstrained.yaml``),
in snake case; the parameters are scale-free: quantiles, fractions of the run and probabilities. The
grammars of NSGA-II, NSGA-III, SMS-EMOA and AGE-MOEA also offer the three techniques that Evolver does
not have, ``static_penalty``, ``dynamic_penalty`` and ``violation_objective``; the coefficients of the
two penalties multiply violations, so they depend on the problem's scale.
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
}


def constraint_handling_from_assignment(assignment: dict[str, Any]) -> tuple[str, dict[str, float]] | None:
    """The technique of an assignment as ``(name, parameters)``; None if the assignment has none."""
    name = assignment.get("constraint_handling")
    if name is None:
        return None
    technique = str(name)
    mapping = CONSTRAINT_HANDLING_SPACE_PARAMETERS.get(technique, {})
    missing = [key for key in mapping if key not in assignment]
    if missing:
        raise ValueError(f"constraint_handling {technique!r} needs {', '.join(missing)} in the assignment.")
    return technique, {parameter: float(assignment[key]) for key, parameter in mapping.items()}


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
    "apply_constraint_handling",
    "apply_subproblem_update_criterion",
    "constraint_handling_from_assignment",
]
