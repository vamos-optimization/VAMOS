"""Constraint-handling techniques: how an algorithm ranks solutions that violate constraints.

The six techniques of jMetal's package ``util.constrainthandling``, which Evolver's spaces choose with the
parameter ``constraintHandling`` (:data:`JMETAL_TECHNIQUES`, :mod:`.constraint_jmetal`): ``cdp``, the default
of every VAMOS algorithm with constraint support, ``violated_constraints``, ``epsilon_constrained``,
``push_pull``, ``stochastic_ranking`` and ``self_adaptive_penalty``.

Seven more, which neither jMetal nor Evolver has:

- ``static_penalty``: every objective plus ``coefficient`` times the overall violation.
- ``dynamic_penalty``: Joines and Houck's non-stationary penalty, every objective plus
  ``(constant * k) ** alpha`` times the sum of the violations raised to ``beta``, where ``k`` is the
  generation (:class:`DynamicPenalty`).
- ``violation_objective``: the overall violation as one more objective (Coello, 2000).
- ``two_rankings``: Ma, Wang and Song's ToR, a fitness of the positions by the constrained-domination
  principle and by Pareto dominance (:mod:`.constraint_two_rankings`), one solution per front; NSGA-II only.
- ``shift_penalty``: Ma and Wang's ShiP, which shifts infeasible objectives towards their local feasible
  nadir and adds the violation (:class:`ShiftPenalty`).
- ``infeasibility_driven``: Ray, Singh, Isaacs and Smith's IDEA, which ranks a proportion of the best
  infeasible solutions above the feasible ones; NSGA-II only (:mod:`.constraint_orders`).
- ``adaptive_tradeoff``: Wang, Cai, Zhou and Zeng's ATM, in Li et al.'s version for several objectives,
  whose ranking depends on the feasible proportion of the population; NSGA-II only (:mod:`.constraint_orders`).

The penalties and ``violation_objective``, like ``self_adaptive_penalty``, rank by Pareto dominance over
modified objectives; crowding distance, niching, hypervolume contributions and survival scores use the
objectives.

:mod:`.constraint_technique` holds the base class, :mod:`.constraint_jmetal`, :mod:`.constraint_penalties`,
:mod:`.constraint_two_rankings` and :mod:`.constraint_orders` the techniques, and :mod:`.constraint_registry` their parameters, lists,
validation and construction. This module gives an algorithm its run's technique and the functions it calls
to rank with it, and exports every name of the others.

A technique ranks a population into fronts (:meth:`ConstraintHandlingTechnique.ranks`). Those that change
during a run follow its progress, the evaluations over the budget, which ``update`` receives with the
population about to be ranked, as in jMetal. Constraints follow VAMOS's convention, ``G <= 0`` feasible.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .constrained_ranking import RankArray
from .constraint_jmetal import ConstrainedDominance, EpsilonConstrained, PushPullSearch, SelfAdaptivePenalty, StochasticRanking
from .constraint_orders import AdaptiveTradeoff, InfeasibilityDriven
from .constraint_penalties import DynamicPenalty, ShiftPenalty, StaticPenalty, ViolationObjective, penalized_objectives
from .constraint_registry import (
    CONSTRAINT_HANDLING_PARAMETERS,
    CONSTRAINT_HANDLING_TECHNIQUES,
    FRONT_TECHNIQUES,
    JMETAL_TECHNIQUES,
    make_constraint_handling,
    normalize_constraint_handling,
)
from .constraint_technique import (
    NSGAII_ONLY_TECHNIQUES,
    PAIRWISE_TECHNIQUES,
    ConstraintHandlingTechnique,
    check_progress,
    rejection_reason,
)
from .constraint_two_rankings import TwoRankings

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import KernelBackend


def config_constraint_handling(
    value: object, constraint_mode: str, techniques: tuple[str, ...] = CONSTRAINT_HANDLING_TECHNIQUES
) -> tuple[str, dict[str, float]] | None:
    """The ``constraint_handling`` field of a configuration, validated; None stays None, which is cdp.

    A JSON round trip returns the pair as a list, which this turns back into a tuple. A technique other
    than cdp needs the feasibility constraint mode, and must be one of the ``techniques`` that the
    algorithm takes.
    """
    if value is None:
        return None
    technique = normalize_constraint_handling(value)
    if technique[0] not in techniques:
        reason = rejection_reason(technique[0], techniques)
        raise ValueError(f"constraint_handling {technique[0]!r} {reason}; this algorithm takes {', '.join(techniques)}.")
    if technique[0] != "cdp" and constraint_mode == "none":
        raise ValueError(f"constraint_handling {technique[0]!r} needs constraint_mode 'feasibility'.")
    return technique


def run_technique(
    value: object,
    constraint_mode: str,
    checkpoint: Mapping[str, Any] | None = None,
    techniques: tuple[str, ...] = CONSTRAINT_HANDLING_TECHNIQUES,
    *,
    population_size: int | None = None,
    max_evaluations: int | None = None,
) -> ConstraintHandlingTechnique | None:
    """A run's own technique, None for cdp, with the state that the checkpoint it resumes from saved.

    cdp keeps each algorithm's own feasibility-first survival and selection, so its results do not
    change. Any other technique needs the feasibility constraint mode, and must be one of the
    ``techniques`` that the algorithm takes. Given ``population_size`` and ``max_evaluations``, the
    technique learns them (:meth:`ConstraintHandlingTechnique.bind_run`).
    """
    technique_value = config_constraint_handling(value, constraint_mode, techniques)
    if technique_value is None or technique_value[0] == "cdp":
        return None
    technique = make_constraint_handling(technique_value)
    if population_size is not None and max_evaluations is not None:
        technique.bind_run(int(population_size), int(max_evaluations))
    extra = checkpoint.get("extra") if checkpoint is not None else None
    state = extra.get("constraint_handling") if isinstance(extra, Mapping) else None
    if isinstance(state, Mapping):
        technique.restore(state)
    return technique


def run_progress(n_eval: int, max_evals: int) -> float:
    """The evaluations so far over the budget, in [0, 1]; NaN without a budget.

    An algorithm takes it before it counts the offspring about to be ranked, as jMetal's ``RunProgress``.
    """
    if max_evals <= 0:
        return float("nan")
    return min(1.0, n_eval / max_evals)


def adapted_ranks(
    technique: ConstraintHandlingTechnique,
    kernel: KernelBackend,
    F: NDArray[Any],
    G: NDArray[Any],
    rng: np.random.Generator,
    progress: float,
) -> RankArray:
    """The fronts of the population of a replacement, after the technique adapts to it at ``progress``.

    This is what jMetal's ``ConstraintHandlingReplacement`` does before each replacement.
    """
    technique.update(kernel, F, G, progress)
    return technique.ranks(kernel, F, G, rng)


def preferred(
    technique: ConstraintHandlingTechnique,
    F_a: NDArray[Any],
    G_a: NDArray[Any],
    F_b: NDArray[Any],
    G_b: NDArray[Any],
) -> NDArray[np.bool_]:
    """Whether the technique prefers each row of ``a`` to the same row of ``b``, as its jMetal comparator.

    The row with the lower effective violation wins. Two rows with the same one compare by Pareto
    dominance if the technique treats both as feasible, and also otherwise for cdp and
    violated_constraints; epsilon_constrained and push_pull prefer neither.
    """
    measure_a = technique.effective_violation(G_a)
    measure_b = technique.effective_violation(G_b)
    F_a = np.asarray(F_a, dtype=np.float64)
    F_b = np.asarray(F_b, dtype=np.float64)
    dominates = np.all(F_a <= F_b, axis=1) & np.any(F_a < F_b, axis=1)
    ties = (measure_a == measure_b) & ((measure_a <= 0.0) | technique.dominance_on_ties)
    return np.asarray((measure_a < measure_b) | (ties & dominates), dtype=np.bool_)


__all__ = [
    "AdaptiveTradeoff",
    "CONSTRAINT_HANDLING_PARAMETERS",
    "CONSTRAINT_HANDLING_TECHNIQUES",
    "FRONT_TECHNIQUES",
    "JMETAL_TECHNIQUES",
    "NSGAII_ONLY_TECHNIQUES",
    "PAIRWISE_TECHNIQUES",
    "ConstrainedDominance",
    "ConstraintHandlingTechnique",
    "DynamicPenalty",
    "EpsilonConstrained",
    "InfeasibilityDriven",
    "PushPullSearch",
    "SelfAdaptivePenalty",
    "ShiftPenalty",
    "StaticPenalty",
    "StochasticRanking",
    "TwoRankings",
    "ViolationObjective",
    "adapted_ranks",
    "check_progress",
    "config_constraint_handling",
    "make_constraint_handling",
    "normalize_constraint_handling",
    "penalized_objectives",
    "preferred",
    "run_progress",
    "run_technique",
]
