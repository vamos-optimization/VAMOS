"""The registry of the constraint-handling techniques: their parameters, the lists of techniques that the
algorithms take, and the validation and construction of a technique from its name and parameters
(see :mod:`.constraint_handling`)."""

from __future__ import annotations

import math
from collections.abc import Mapping

from .constraint_jmetal import ConstrainedDominance, EpsilonConstrained, PushPullSearch, SelfAdaptivePenalty, StochasticRanking
from .constraint_penalties import DynamicPenalty, ShiftPenalty, StaticPenalty, ViolationObjective
from .constraint_technique import ConstraintHandlingTechnique
from .constraint_two_rankings import TwoRankings

#: The parameters of each technique, in the order of jMetal's constructors.
CONSTRAINT_HANDLING_PARAMETERS: dict[str, tuple[str, ...]] = {
    "cdp": (),
    "violated_constraints": (),
    "epsilon_constrained": ("initial_quantile", "exponent", "decay_fraction"),
    "push_pull": (
        "change_threshold",
        "window_fraction",
        "decay_fraction",
        "exponent",
        "feasibility_threshold",
        "reduction_rate",
    ),
    "stochastic_ranking": ("probability",),
    "self_adaptive_penalty": (),
    "static_penalty": ("coefficient",),
    "dynamic_penalty": ("constant", "alpha", "beta"),
    "violation_objective": (),
    "two_rankings": (),
    "shift_penalty": ("constant",),
}

#: The techniques of jMetal's package ``util.constrainthandling``, in the order of Evolver's spaces.
JMETAL_TECHNIQUES: tuple[str, ...] = (
    "cdp",
    "violated_constraints",
    "epsilon_constrained",
    "push_pull",
    "stochastic_ranking",
    "self_adaptive_penalty",
)

#: Every technique: jMetal's, then those that neither jMetal nor Evolver has.
CONSTRAINT_HANDLING_TECHNIQUES: tuple[str, ...] = (
    *JMETAL_TECHNIQUES,
    "static_penalty",
    "dynamic_penalty",
    "violation_objective",
    "two_rankings",
    "shift_penalty",
)

#: The techniques that NSGA-III, SMS-EMOA and AGE-MOEA take: all but those defined for NSGA-II only.
FRONT_TECHNIQUES: tuple[str, ...] = (*JMETAL_TECHNIQUES, "static_penalty", "dynamic_penalty", "violation_objective", "shift_penalty")

# The admissible values of each parameter, as jMetal's constructors check them, (low, high, low closed); the
# parameters of the techniques that jMetal does not have are positive.
_RANGES: dict[str, tuple[float, float, bool]] = {
    "initial_quantile": (0.0, 1.0, True),
    "probability": (0.0, 1.0, True),
    "feasibility_threshold": (0.0, 1.0, True),
    "reduction_rate": (0.0, 1.0, True),
    "decay_fraction": (0.0, 1.0, False),
    "window_fraction": (0.0, 1.0, False),
    "exponent": (0.0, math.inf, False),
    "change_threshold": (0.0, math.inf, True),
    "coefficient": (0.0, math.inf, False),
    "constant": (0.0, math.inf, False),
    "alpha": (0.0, math.inf, False),
    "beta": (0.0, math.inf, False),
}


def normalize_constraint_handling(value: object) -> tuple[str, dict[str, float]]:
    """Validate a technique given as a name or as ``(name, parameters)``; ``None`` is ``cdp``.

    Every parameter of the technique is required, as in jMetal, and must lie in jMetal's range; those of
    the techniques that jMetal does not have must be positive.
    """
    if value is None:
        return ("cdp", {})
    if isinstance(value, str):
        name, raw = value, {}
    elif isinstance(value, (tuple, list)) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], Mapping):
        name, raw = value[0], dict(value[1])
    else:
        raise ValueError(f"constraint_handling must be a technique name or (name, parameters); got {value!r}.")
    if name not in CONSTRAINT_HANDLING_PARAMETERS:
        raise ValueError(f"Unknown constraint-handling technique {name!r}; the techniques are {', '.join(CONSTRAINT_HANDLING_TECHNIQUES)}.")
    expected = CONSTRAINT_HANDLING_PARAMETERS[name]
    unknown = sorted(set(raw) - set(expected))
    missing = [key for key in expected if key not in raw]
    if unknown or missing:
        needs = ", ".join(expected) if expected else "no parameters"
        raise ValueError(f"constraint_handling {name!r} takes {needs}; unknown: {unknown}, missing: {missing}.")
    params: dict[str, float] = {}
    for key in expected:
        number = float(raw[key])
        low, high, low_closed = _RANGES[key]
        if not (number >= low if low_closed else number > low) or number > high or math.isnan(number):
            interval = f"{'[' if low_closed else '('}{low}, {high}]"
            raise ValueError(f"constraint_handling {name!r}: {key} must lie in {interval}; got {number}.")
        params[key] = number
    return (name, params)


def make_constraint_handling(value: object) -> ConstraintHandlingTechnique:
    """A new technique, with no state, from a name or ``(name, parameters)``; ``None`` is ``cdp``."""
    name, params = normalize_constraint_handling(value)
    if name == "cdp":
        return ConstrainedDominance()
    if name == "violated_constraints":
        return ConstrainedDominance(count_violated_constraints=True)
    if name == "epsilon_constrained":
        return EpsilonConstrained(**params)
    if name == "push_pull":
        return PushPullSearch(**params)
    if name == "stochastic_ranking":
        return StochasticRanking(**params)
    if name == "static_penalty":
        return StaticPenalty(**params)
    if name == "dynamic_penalty":
        return DynamicPenalty(**params)
    if name == "violation_objective":
        return ViolationObjective()
    if name == "two_rankings":
        return TwoRankings()
    if name == "shift_penalty":
        return ShiftPenalty(**params)
    return SelfAdaptivePenalty()


__all__ = [
    "CONSTRAINT_HANDLING_PARAMETERS",
    "CONSTRAINT_HANDLING_TECHNIQUES",
    "FRONT_TECHNIQUES",
    "JMETAL_TECHNIQUES",
    "make_constraint_handling",
    "normalize_constraint_handling",
]
