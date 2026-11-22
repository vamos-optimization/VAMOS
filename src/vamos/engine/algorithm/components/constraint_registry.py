"""The registry of the constraint-handling techniques: their parameters, the lists of techniques that the
algorithms take, and the validation and construction of a technique from its name and parameters
(see :mod:`.constraint_handling`)."""

from __future__ import annotations

import math
from collections.abc import Mapping

from .constraint_ensemble import Ensemble, MultipleConstraintRanking
from .constraint_jmetal import ConstrainedDominance, EpsilonConstrained, PushPullSearch, SelfAdaptivePenalty, StochasticRanking
from .constraint_orders import AdaptiveTradeoff, InfeasibilityDriven
from .constraint_penalties import DynamicPenalty, ShiftPenalty, StaticPenalty, ViolationObjective
from .constraint_technique import ConstraintHandlingTechnique, TechniqueParameter
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
    "infeasibility_driven": ("proportion",),
    "adaptive_tradeoff": (),
    "multiple_constraint_ranking": (),
    "ensemble": (),
}

#: The optional parameters of each technique, with their defaults: Dam and Truong's options of
#: ``violation_objective``, whose defaults rank as it did without them.
OPTIONAL_PARAMETERS: dict[str, dict[str, TechniqueParameter]] = {
    "violation_objective": {"violation_norm": "l1", "ratio_objective": False, "cdf_alpha": 0.0, "feasible_first": False},
    "ensemble": {"nds_weight": 0.25, "cdp_weight": 0.25, "mcr_weight": 0.25, "dpf_weight": 0.25},
}

#: The norms of ``violation_objective``'s violation: the sum, the sum of squares and the largest violation.
VIOLATION_NORMS: tuple[str, ...] = ("l1", "l2_squared", "linf")

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
    "infeasibility_driven",
    "adaptive_tradeoff",
    "multiple_constraint_ranking",
    "ensemble",
)

#: The techniques that NSGA-III, SMS-EMOA and AGE-MOEA take: all but those defined for NSGA-II only.
FRONT_TECHNIQUES: tuple[str, ...] = (*JMETAL_TECHNIQUES, "static_penalty", "dynamic_penalty", "violation_objective", "shift_penalty")

# The admissible values of each parameter, as jMetal's constructors check them, (low, high, low closed); the
# parameters of the techniques that jMetal does not have are positive, and IDEA's proportion at most 1.
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
    "proportion": (0.0, 1.0, False),
    "cdf_alpha": (0.0, math.inf, True),
    "nds_weight": (0.0, math.inf, True),
    "cdp_weight": (0.0, math.inf, True),
    "mcr_weight": (0.0, math.inf, True),
    "dpf_weight": (0.0, math.inf, True),
}


def normalize_constraint_handling(value: object) -> tuple[str, dict[str, TechniqueParameter]]:
    """Validate a technique given as a name or as ``(name, parameters)``; ``None`` is ``cdp``.

    Every parameter of the technique is required, as in jMetal, and must lie in jMetal's range; those of
    the techniques that jMetal does not have must be positive. The optional parameters of
    :data:`OPTIONAL_PARAMETERS` keep only the values given, and take their defaults when the technique is built.
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
    expected, optional = CONSTRAINT_HANDLING_PARAMETERS[name], OPTIONAL_PARAMETERS.get(name, {})
    unknown = sorted(set(raw) - set(expected) - set(optional))
    missing = [key for key in expected if key not in raw]
    if unknown or missing:
        needs = ", ".join((*expected, *(f"[{key}]" for key in optional))) or "no parameters"
        raise ValueError(f"constraint_handling {name!r} takes {needs}; unknown: {unknown}, missing: {missing}.")
    params: dict[str, TechniqueParameter] = {key: _number(name, key, raw[key]) for key in expected}
    for key, default in optional.items():
        if key in raw:
            params[key] = _option(name, key, raw[key], default)
    if name == "ensemble":
        weights = [float({**optional, **params}[key]) for key in optional]
        if not all(math.isfinite(weight) for weight in weights) or not any(weight > 0.0 for weight in weights):
            raise ValueError(
                f"constraint_handling {name!r}: the weights are normalized to sum 1, so they are finite and one must be above 0."
            )
    if "cdf_alpha" in params and params.get("ratio_objective") is not True:
        raise ValueError(f"constraint_handling {name!r}: cdf_alpha weighs the ratio objective, so it needs ratio_objective True.")
    return (name, params)


def _number(name: str, key: str, value: object) -> float:
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float)):
        raise ValueError(f"constraint_handling {name!r}: {key} must be a number; got {value!r}.")
    number = float(value)
    low, high, low_closed = _RANGES[key]
    if not (number >= low if low_closed else number > low) or number > high or math.isnan(number):
        interval = f"{'[' if low_closed else '('}{low}, {high}]"
        raise ValueError(f"constraint_handling {name!r}: {key} must lie in {interval}; got {number}.")
    return number


def _option(name: str, key: str, value: object, default: TechniqueParameter) -> TechniqueParameter:
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"constraint_handling {name!r}: {key} must be True or False; got {value!r}.")
        return value
    if isinstance(default, str):
        if value not in VIOLATION_NORMS:
            raise ValueError(f"constraint_handling {name!r}: {key} must be one of {', '.join(VIOLATION_NORMS)}; got {value!r}.")
        return str(value)
    return _number(name, key, value)


def make_constraint_handling(value: object) -> ConstraintHandlingTechnique:
    """A new technique, with no state, from a name or ``(name, parameters)``; ``None`` is ``cdp``."""
    name, given = normalize_constraint_handling(value)
    params = {key: float(number) for key, number in given.items() if key in CONSTRAINT_HANDLING_PARAMETERS[name]}
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
    if name == "multiple_constraint_ranking":
        return MultipleConstraintRanking()
    if name == "ensemble":
        weights = {**OPTIONAL_PARAMETERS[name], **given}
        return Ensemble([float(weights[f"{member}_weight"]) for member in ("nds", "cdp", "mcr", "dpf")])
    if name == "violation_objective":
        options = {**OPTIONAL_PARAMETERS[name], **given}
        return ViolationObjective(
            str(options["violation_norm"]),
            ratio_objective=options["ratio_objective"] is True,
            cdf_alpha=float(options["cdf_alpha"]),
            feasible_first=options["feasible_first"] is True,
        )
    if name == "two_rankings":
        return TwoRankings()
    if name == "shift_penalty":
        return ShiftPenalty(**params)
    if name == "infeasibility_driven":
        return InfeasibilityDriven(**params)
    if name == "adaptive_tradeoff":
        return AdaptiveTradeoff()
    return SelfAdaptivePenalty()


__all__ = [
    "CONSTRAINT_HANDLING_PARAMETERS",
    "CONSTRAINT_HANDLING_TECHNIQUES",
    "FRONT_TECHNIQUES",
    "JMETAL_TECHNIQUES",
    "OPTIONAL_PARAMETERS",
    "VIOLATION_NORMS",
    "make_constraint_handling",
    "normalize_constraint_handling",
]
