"""The constraint-handling part of the ``<algorithm>_constrained`` tuning spaces.

The technique and its parameters, with the names and ranges of Evolver's spaces (parameter
``constraintHandling`` of ``NSGAIIDoubleConstrained.yaml``), in snake case. The parameters are
scale-free: quantiles, fractions of the run and probabilities.
"""

from __future__ import annotations

from typing import Any

from vamos.engine.algorithm.components.constraint_handling import CONSTRAINT_HANDLING_TECHNIQUES
from vamos.engine.algorithm.moead.update_criteria import SUBPROBLEM_UPDATE_CRITERIA

from .config_space import SpacePart
from .param_space import Categorical, ConditionalBlock, ParamType, Real

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
}


def constraint_handling_part(techniques: tuple[str, ...] = CONSTRAINT_HANDLING_TECHNIQUES) -> SpacePart:
    """``constraint_handling`` with the ``techniques`` (by default the six) and the parameters of those that have them."""
    params: list[ParamType] = [Categorical("constraint_handling", list(techniques), role="structural")]
    blocks = [
        ConditionalBlock(
            "constraint_handling",
            "epsilon_constrained",
            [
                Real("epsilon_initial_quantile", 0.05, 0.95),
                Real("epsilon_exponent", 1.0, 10.0),
                Real("epsilon_decay_fraction", 0.1, 0.95),
            ],
        ),
        ConditionalBlock("constraint_handling", "stochastic_ranking", [Real("stochastic_ranking_probability", 0.0, 0.5)]),
        ConditionalBlock(
            "constraint_handling",
            "push_pull",
            [
                Real("push_pull_change_threshold", 0.0001, 0.05),
                Real("push_pull_window_fraction", 0.01, 0.1),
                Real("push_pull_decay_fraction", 0.1, 0.95),
                Real("push_pull_exponent", 1.0, 10.0),
                Real("push_pull_feasibility_threshold", 0.5, 1.0),
                Real("push_pull_reduction_rate", 0.01, 0.5),
            ],
        ),
    ]
    conditionals = [block for block in blocks if block.parent_value in techniques]
    return params, conditionals, []


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


def subproblem_update_part() -> SpacePart:
    """MOEA/D's ``subproblem_update_criterion``, with the ranges of Evolver's ``MOEADDoubleConstrained``."""
    params: list[ParamType] = [
        Categorical("subproblem_update_criterion", list(SUBPROBLEM_UPDATE_CRITERIA), role="structural"),
    ]
    conditionals = [
        ConditionalBlock(
            "subproblem_update_criterion",
            "improved_epsilon",
            [
                Real("improved_epsilon_tao", 0.01, 0.5),
                Real("improved_epsilon_feasibility_ratio_threshold", 0.5, 1.0),
                Real("improved_epsilon_decay_fraction", 0.1, 0.95),
            ],
        )
    ]
    return params, conditionals, []


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
    "constraint_handling_part",
    "subproblem_update_part",
]
