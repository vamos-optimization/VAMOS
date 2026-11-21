"""The six constraint-handling techniques of jMetal's package ``util.constrainthandling``.

Evolver's spaces choose them with the parameter ``constraintHandling``; :mod:`.constraint_handling` builds
and validates them with the others.

- ``cdp``: Deb's constrained-domination principle over the overall violation. It is the default of
  every VAMOS algorithm with constraint support.
- ``violated_constraints``: the same rule over the number of violated constraints.
- ``epsilon_constrained``: Takahama and Sakai's epsilon-constrained method. Solutions whose violation
  is at most epsilon count as feasible; epsilon starts at a quantile of the violations of the first
  ranked population and decays to 0 with the run's progress.
- ``push_pull``: Fan et al.'s push and pull search. It ignores the constraints until the ideal and
  nadir points settle, and then acts as epsilon-constrained, with an epsilon that also shrinks while
  few solutions are feasible.
- ``stochastic_ranking``: Runarsson and Yao's stochastic ranking.
- ``self_adaptive_penalty``: Woldesenbet, Yen and Tessema's self-adaptive penalty.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .constrained_ranking import (
    RankArray,
    pareto_ranks,
    ranks_by_violation,
    self_adaptive_penalty_objectives,
    stochastic_ranking_order,
    stochastic_ranking_ranks,
    violated_constraints,
    violation,
)
from .constraint_technique import ConstraintHandlingTechnique, check_progress

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import KernelBackend

# Relative tolerance on the progress at the start of push and pull search's window, as in jMetal.
_PROGRESS_TOLERANCE = 1e-12
# Smallest denominator of a relative change of the ideal or nadir point, as in jMetal.
_DELTA = 1e-6


class ConstrainedDominance(ConstraintHandlingTechnique):
    """Deb's rule over the overall violation (``cdp``) or the number of violated constraints."""

    dominance_on_ties = True

    def __init__(self, *, count_violated_constraints: bool = False) -> None:
        self.count_violated_constraints = count_violated_constraints
        self.name = "violated_constraints" if count_violated_constraints else "cdp"

    def effective_violation(self, G: NDArray[Any]) -> NDArray[np.float64]:
        return violated_constraints(G) if self.count_violated_constraints else violation(G)

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        measure = violated_constraints(G) if self.count_violated_constraints else violation(G)
        return ranks_by_violation(kernel, F, measure, objectives_on_ties=True)


def _within_epsilon(measure: NDArray[np.float64], epsilon: float) -> NDArray[np.float64]:
    """A violation, or 0 where it is at most epsilon."""
    return np.where(measure <= epsilon, 0.0, measure)


def _ranks_within_epsilon(kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], epsilon: float) -> RankArray:
    measure = violation(G)
    return ranks_by_violation(kernel, F, np.where(measure <= epsilon, 0.0, measure), objectives_on_ties=False)


class EpsilonConstrained(ConstraintHandlingTechnique):
    """Takahama and Sakai's epsilon-constrained method, as jMetal's ``EpsilonConstrained``."""

    name = "epsilon_constrained"
    needs_progress = True

    def __init__(self, initial_quantile: float, exponent: float, decay_fraction: float) -> None:
        self.initial_quantile = float(initial_quantile)
        self.exponent = float(exponent)
        self.decay_fraction = float(decay_fraction)
        self.started = False
        self.initial_epsilon = 0.0
        self.epsilon = 0.0

    def update(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], progress: float) -> None:
        progress = check_progress(progress, "epsilon_constrained")
        if not self.started:
            violations = np.sort(violation(G))
            if violations.size == 0:
                raise ValueError("epsilon_constrained cannot start from an empty population.")
            self.initial_epsilon = float(violations[int(math.floor(self.initial_quantile * (violations.size - 1)))])
            self.started = True
        self.epsilon = (
            self.initial_epsilon * (1.0 - progress / self.decay_fraction) ** self.exponent if progress < self.decay_fraction else 0.0
        )

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        return _ranks_within_epsilon(kernel, F, G, self.epsilon)

    def effective_violation(self, G: NDArray[Any]) -> NDArray[np.float64]:
        return _within_epsilon(violation(G), self.epsilon)

    def state(self) -> dict[str, Any]:
        return {"started": self.started, "initial_epsilon": self.initial_epsilon, "epsilon": self.epsilon}

    def restore(self, state: Mapping[str, Any]) -> None:
        self.started = bool(state.get("started", False))
        self.initial_epsilon = float(state.get("initial_epsilon", 0.0))
        self.epsilon = float(state.get("epsilon", 0.0))


class PushPullSearch(ConstraintHandlingTechnique):
    """Fan et al.'s push and pull search, as jMetal's ``PushPullSearch``.

    The push phase ranks by Pareto dominance alone. It ends when, over the last ``window_fraction`` of the
    run, no objective of the ideal point or of the nadir point of the non-dominated solutions changes by
    more than ``change_threshold`` relative to its value, or when the progress reaches ``decay_fraction``.
    The pull phase is epsilon-constrained. Epsilon starts at the largest violation of the population; while
    the fraction of feasible solutions is below ``feasibility_threshold``, it shrinks by ``reduction_rate``
    at each update, and otherwise it decays from its start to 0 at ``decay_fraction``, with ``exponent``.
    """

    name = "push_pull"
    needs_progress = True

    def __init__(
        self,
        change_threshold: float,
        window_fraction: float,
        decay_fraction: float,
        exponent: float,
        feasibility_threshold: float,
        reduction_rate: float,
    ) -> None:
        self.change_threshold = float(change_threshold)
        self.window_fraction = float(window_fraction)
        self.decay_fraction = float(decay_fraction)
        self.exponent = float(exponent)
        self.feasibility_threshold = float(feasibility_threshold)
        self.reduction_rate = float(reduction_rate)
        self.pushing = True
        self.history: list[tuple[float, NDArray[np.float64], NDArray[np.float64]]] = []
        self.initial_epsilon = 0.0
        self.epsilon = 0.0

    def update(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], progress: float) -> None:
        progress = check_progress(progress, "push_pull")
        if progress >= self.decay_fraction:
            self.pushing = False
            self.epsilon = 0.0
            return
        if self.pushing:
            F = np.asarray(F, dtype=np.float64)
            nondominated = pareto_ranks(kernel, F) == 0
            self.history.append((progress, F.min(axis=0), F[nondominated].max(axis=0)))
            if not self._settled():
                return
            self.pushing = False
            self.history.clear()
            measure = violation(G)
            self.initial_epsilon = float(measure.max()) if measure.size else 0.0
            self.epsilon = self.initial_epsilon
        feasible_ratio = float(np.count_nonzero(np.all(np.asarray(G) <= 0.0, axis=1))) / max(1, G.shape[0])
        if feasible_ratio < self.feasibility_threshold:
            self.epsilon = (1.0 - self.reduction_rate) * self.epsilon
        else:
            self.epsilon = self.initial_epsilon * (1.0 - progress / self.decay_fraction) ** self.exponent

    def _settled(self) -> bool:
        current = self.history[-1]
        start = current[0] - self.window_fraction + _PROGRESS_TOLERANCE
        if self.history[0][0] > start:
            return False
        while len(self.history) > 2 and self.history[1][0] <= start:
            self.history.pop(0)
        reference = self.history[0]
        change = max(_largest_relative_change(reference[1], current[1]), _largest_relative_change(reference[2], current[2]))
        return change <= self.change_threshold

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        if self.pushing:
            return pareto_ranks(kernel, F)
        return _ranks_within_epsilon(kernel, F, G, self.epsilon)

    def effective_violation(self, G: NDArray[Any]) -> NDArray[np.float64]:
        return _push_pull_effective_violation(self, G)

    def state(self) -> dict[str, Any]:
        return {
            "pushing": self.pushing,
            "history": [(progress, ideal.copy(), nadir.copy()) for progress, ideal, nadir in self.history],
            "initial_epsilon": self.initial_epsilon,
            "epsilon": self.epsilon,
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        self.pushing = bool(state.get("pushing", True))
        self.history = [
            (float(progress), np.asarray(ideal, dtype=np.float64), np.asarray(nadir, dtype=np.float64))
            for progress, ideal, nadir in state.get("history", [])
        ]
        self.initial_epsilon = float(state.get("initial_epsilon", 0.0))
        self.epsilon = float(state.get("epsilon", 0.0))


def _push_pull_effective_violation(technique: PushPullSearch, G: NDArray[Any]) -> NDArray[np.float64]:
    if technique.pushing:
        return np.zeros(np.asarray(G).shape[0], dtype=np.float64)
    return _within_epsilon(violation(G), technique.epsilon)


def _largest_relative_change(previous: NDArray[np.float64], current: NDArray[np.float64]) -> float:
    return float(np.max(np.abs(current - previous) / np.maximum(np.abs(previous), _DELTA)))


class StochasticRanking(ConstraintHandlingTechnique):
    """Runarsson and Yao's stochastic ranking, as jMetal's ``StochasticRanking``.

    Each ranking draws ``n * (n - 1)`` uniforms from the run's generator and uses them in order, one per
    comparison between rows that are not both feasible.
    """

    name = "stochastic_ranking"

    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        n = F.shape[0]
        pareto_rank = pareto_ranks(kernel, F)
        feasible = np.all(np.asarray(G) <= 0.0, axis=1)
        measure = violation(G)
        uniforms = rng.random(n * max(0, n - 1))
        compiled = kernel.selection_kernels()
        if compiled is None:
            order = stochastic_ranking_order(pareto_rank, feasible, measure, uniforms, self.probability)
        else:
            order = compiled.stochastic_ranking_order(pareto_rank, feasible, measure, uniforms, self.probability)
        return stochastic_ranking_ranks(order, pareto_rank, feasible)


class SelfAdaptivePenalty(ConstraintHandlingTechnique):
    """Woldesenbet, Yen and Tessema's self-adaptive penalty, as jMetal's ``SelfAdaptivePenalty``."""

    name = "self_adaptive_penalty"

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        if F.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        return pareto_ranks(kernel, self_adaptive_penalty_objectives(F, G))


__all__ = ["ConstrainedDominance", "EpsilonConstrained", "PushPullSearch", "SelfAdaptivePenalty", "StochasticRanking"]
