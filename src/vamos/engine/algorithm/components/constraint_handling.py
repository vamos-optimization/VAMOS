"""Constraint-handling techniques: how an algorithm ranks solutions that violate constraints.

The six techniques of jMetal's package ``util.constrainthandling``, which Evolver's spaces choose
with the parameter ``constraintHandling``:

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

A technique ranks a population into fronts (:meth:`ConstraintHandlingTechnique.ranks`). Those that
change during a run follow its progress, the evaluations over the budget, which
:meth:`ConstraintHandlingTechnique.update` receives with the population about to be ranked, as
``update(population, progress)`` does in jMetal. Constraints follow VAMOS's convention, ``G <= 0``
feasible.
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

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import KernelBackend

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
}

#: The names of the techniques, in the order of Evolver's spaces.
CONSTRAINT_HANDLING_TECHNIQUES: tuple[str, ...] = (
    "cdp",
    "violated_constraints",
    "epsilon_constrained",
    "push_pull",
    "stochastic_ranking",
    "self_adaptive_penalty",
)

# The admissible values of each parameter, as jMetal's constructors check them: (low, high, low closed).
_RANGES: dict[str, tuple[float, float, bool]] = {
    "initial_quantile": (0.0, 1.0, True),
    "probability": (0.0, 1.0, True),
    "feasibility_threshold": (0.0, 1.0, True),
    "reduction_rate": (0.0, 1.0, True),
    "decay_fraction": (0.0, 1.0, False),
    "window_fraction": (0.0, 1.0, False),
    "exponent": (0.0, math.inf, False),
    "change_threshold": (0.0, math.inf, True),
}

# Relative tolerance on the progress at the start of push and pull search's window, as in jMetal.
_PROGRESS_TOLERANCE = 1e-12
# Smallest denominator of a relative change of the ideal or nadir point, as in jMetal.
_DELTA = 1e-6


def normalize_constraint_handling(value: object) -> tuple[str, dict[str, float]]:
    """Validate a technique given as a name or as ``(name, parameters)``; ``None`` is ``cdp``.

    Every parameter of the technique is required, as in jMetal, and must lie in jMetal's range.
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


def _check_progress(progress: float, name: str) -> float:
    value = float(progress)
    if math.isnan(value):
        raise ValueError(f"{name} changes during the run and needs its progress: give the run an evaluation budget.")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"The progress must lie in [0, 1]; got {value}.")
    return value


class ConstraintHandlingTechnique:
    """A technique that ranks solutions into fronts, taking their constraint values into account."""

    name = ""
    #: Whether :meth:`update` reads the run's progress.
    needs_progress = False

    def update(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], progress: float) -> None:
        """Adapt to the population about to be ranked, at ``progress`` (evaluations over the budget)."""
        return None

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        """The front of each row (0 first)."""
        raise NotImplementedError

    def state(self) -> dict[str, Any]:
        """What :meth:`update` has learnt, for a checkpoint."""
        return {}

    def restore(self, state: Mapping[str, Any]) -> None:
        """Resume from :meth:`state`."""
        return None


class ConstrainedDominance(ConstraintHandlingTechnique):
    """Deb's rule over the overall violation (``cdp``) or the number of violated constraints."""

    def __init__(self, *, count_violated_constraints: bool = False) -> None:
        self.count_violated_constraints = count_violated_constraints
        self.name = "violated_constraints" if count_violated_constraints else "cdp"

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        measure = violated_constraints(G) if self.count_violated_constraints else violation(G)
        return ranks_by_violation(kernel, F, measure, objectives_on_ties=True)


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
        progress = _check_progress(progress, "epsilon_constrained")
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
        progress = _check_progress(progress, "push_pull")
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
    return SelfAdaptivePenalty()


def config_constraint_handling(value: object, constraint_mode: str) -> tuple[str, dict[str, float]] | None:
    """The ``constraint_handling`` field of a configuration, validated; None stays None, which is cdp.

    A JSON round trip returns the pair as a list, which this turns back into a tuple. A technique other
    than cdp needs the feasibility constraint mode.
    """
    if value is None:
        return None
    technique = normalize_constraint_handling(value)
    if technique[0] != "cdp" and constraint_mode == "none":
        raise ValueError(f"constraint_handling {technique[0]!r} needs constraint_mode 'feasibility'.")
    return technique


def run_technique(value: object, constraint_mode: str, checkpoint: Mapping[str, Any] | None = None) -> ConstraintHandlingTechnique | None:
    """A run's own technique, None for cdp, with the state that the checkpoint it resumes from saved.

    cdp keeps each algorithm's own feasibility-first survival and selection, so its results do not
    change. Any other technique needs the feasibility constraint mode.
    """
    technique_value = config_constraint_handling(value, constraint_mode)
    if technique_value is None or technique_value[0] == "cdp":
        return None
    technique = make_constraint_handling(technique_value)
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


__all__ = [
    "CONSTRAINT_HANDLING_PARAMETERS",
    "CONSTRAINT_HANDLING_TECHNIQUES",
    "ConstrainedDominance",
    "ConstraintHandlingTechnique",
    "EpsilonConstrained",
    "PushPullSearch",
    "SelfAdaptivePenalty",
    "StochasticRanking",
    "adapted_ranks",
    "config_constraint_handling",
    "make_constraint_handling",
    "normalize_constraint_handling",
    "run_progress",
    "run_technique",
]
