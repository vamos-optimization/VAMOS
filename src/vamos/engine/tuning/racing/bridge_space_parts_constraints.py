"""The constraint-handling part of the ``<algorithm>_constrained`` tuning spaces.

The technique and its parameters, with the names and ranges of Evolver's spaces (parameter
``constraintHandling`` of ``NSGAIIDoubleConstrained.yaml``), in snake case. The parameters are
scale-free: quantiles, fractions of the run and probabilities.
"""

from __future__ import annotations

from typing import Any

from vamos.engine.algorithm.components.constraint_handling import CONSTRAINT_HANDLING_TECHNIQUES

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


def constraint_handling_part() -> SpacePart:
    """``constraint_handling`` with the six techniques and the conditional parameters of three."""
    params: list[ParamType] = [Categorical("constraint_handling", list(CONSTRAINT_HANDLING_TECHNIQUES), role="structural")]
    conditionals = [
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


__all__ = [
    "CONSTRAINT_HANDLING_SPACE_PARAMETERS",
    "constraint_handling_from_assignment",
    "constraint_handling_part",
]
