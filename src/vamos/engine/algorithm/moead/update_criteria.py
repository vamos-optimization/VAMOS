"""MOEA/D's subproblem update criteria: whether a new solution replaces the solution of a subproblem.

The criteria of jMetal's package ``component.catalogue.ea.replacement.subproblemupdate``, which
Evolver's MOEA/D space chooses with the parameter ``subproblemUpdateCriterion``:

- ``feasibility_rules``: Deb's feasibility rules. A feasible solution beats an infeasible one, the less
  violated of two infeasible ones wins, and two feasible ones compare by their aggregation values. It
  is MOEA/D's own rule in VAMOS and the default; two infeasible solutions with the same violation keep
  the current one, where jMetal compares their aggregation values.
- ``improved_epsilon``: Fan et al.'s improved epsilon method (Soft Computing, 2019). Solutions whose
  violations lie within epsilon, or are equal, compare by their aggregation values, and otherwise the
  less violated wins. Epsilon starts at the violation of the 5% most violated solution of the first
  population, shrinks by ``tao`` each generation while fewer than ``feasibility_ratio_threshold`` of the
  population is feasible, grows to the largest violation seen so far times ``1 + tao`` otherwise, and is
  0 from the generation at ``decay_fraction`` of the run.
- ``violation_threshold``: Jan and Khanum's adaptive threshold (Applied Soft Computing, 2013). A
  solution's violation is its number of violated constraints times its total violation. If either
  violation exceeds the threshold, the less violated wins; otherwise the aggregation values decide. The
  threshold, each generation, is the fraction of infeasible solutions times the mean violation.

As in jMetal, a criterion updates once per new solution, before that solution's replacements, and a
generation is ``pop_size`` new solutions. Constraints follow VAMOS's convention, ``G <= 0`` feasible.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: The criteria, in the order of Evolver's space.
SUBPROBLEM_UPDATE_CRITERIA: tuple[str, ...] = ("feasibility_rules", "improved_epsilon", "violation_threshold")

#: The parameters of each criterion.
SUBPROBLEM_UPDATE_PARAMETERS: dict[str, tuple[str, ...]] = {
    "feasibility_rules": (),
    "improved_epsilon": ("tao", "feasibility_ratio_threshold", "decay_fraction"),
    "violation_threshold": (),
}

# The admissible values of each parameter: (low, high, low closed, high closed). jMetal checks tao.
_RANGES: dict[str, tuple[float, float, bool, bool]] = {
    "tao": (0.0, 1.0, False, False),
    "feasibility_ratio_threshold": (0.0, 1.0, True, True),
    "decay_fraction": (0.0, 1.0, False, True),
}


def normalize_subproblem_update_criterion(value: object) -> tuple[str, dict[str, float]] | None:
    """Validate a criterion given as a name or as ``(name, parameters)``; None stays None, the default.

    A JSON round trip returns the pair as a list, which this turns back into a tuple. Every parameter
    of the criterion is required.
    """
    if value is None:
        return None
    if isinstance(value, str):
        name, raw = value, {}
    elif isinstance(value, (tuple, list)) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], Mapping):
        name, raw = value[0], dict(value[1])
    else:
        raise ValueError(f"subproblem_update_criterion must be a criterion name or (name, parameters); got {value!r}.")
    if name not in SUBPROBLEM_UPDATE_PARAMETERS:
        raise ValueError(f"Unknown subproblem update criterion {name!r}; the criteria are {', '.join(SUBPROBLEM_UPDATE_CRITERIA)}.")
    expected = SUBPROBLEM_UPDATE_PARAMETERS[name]
    unknown = sorted(set(raw) - set(expected))
    missing = [key for key in expected if key not in raw]
    if unknown or missing:
        needs = ", ".join(expected) if expected else "no parameters"
        raise ValueError(f"subproblem_update_criterion {name!r} takes {needs}; unknown: {unknown}, missing: {missing}.")
    params: dict[str, float] = {}
    for key in expected:
        number = float(raw[key])
        low, high, low_closed, high_closed = _RANGES[key]
        above = number >= low if low_closed else number > low
        below = number <= high if high_closed else number < high
        if not (above and below):
            interval = f"{'[' if low_closed else '('}{low}, {high}{']' if high_closed else ')'}"
            raise ValueError(f"subproblem_update_criterion {name!r}: {key} must lie in {interval}; got {number}.")
        params[key] = number
    return (name, params)


def _violation(G: NDArray[Any]) -> NDArray[np.float64]:
    return np.asarray(np.sum(np.maximum(np.asarray(G, dtype=np.float64), 0.0), axis=1), dtype=np.float64)


def _feasibility_ratio(G: NDArray[Any]) -> float:
    G = np.asarray(G, dtype=np.float64)
    return float(np.count_nonzero(np.all(G <= 0.0, axis=1))) / G.shape[0]


class SubproblemUpdateCriterion:
    """A criterion that decides whether a new solution replaces the solutions of some subproblems."""

    name = ""

    def update(self, G: NDArray[Any], child_G: NDArray[Any]) -> None:
        """Adapt to the population (constraints ``G``) before the replacements of a new solution."""
        return None

    def replaces(
        self,
        child_G: NDArray[Any],
        child_values: NDArray[np.float64],
        G: NDArray[Any],
        values: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Whether the new solution (constraints ``child_G``) replaces each current solution.

        ``child_values`` and ``values`` are the aggregation values of the new and current solutions in
        the subproblems of the current ones.
        """
        raise NotImplementedError

    def state(self) -> dict[str, Any]:
        """What :meth:`update` has learnt, for a checkpoint."""
        return {}

    def restore(self, state: Mapping[str, Any]) -> None:
        """Resume from :meth:`state`."""
        return None


class ImprovedEpsilonCriterion(SubproblemUpdateCriterion):
    """Fan et al.'s improved epsilon method, as jMetal's ``ImprovedEpsilonCriterion``."""

    name = "improved_epsilon"

    def __init__(self, tao: float, feasibility_ratio_threshold: float, tc: int) -> None:
        if not 0.0 < tao < 1.0:
            raise ValueError(f"tao must lie in (0, 1); got {tao}.")
        if tc <= 0:
            raise ValueError(f"tc must be positive; got {tc}.")
        self.tao = float(tao)
        self.feasibility_ratio_threshold = float(feasibility_ratio_threshold)
        self.tc = int(tc)
        self.epsilon = 0.0
        self.phi_max = 0.0
        self.generation = 0
        self.updates = 0
        self.initialized = False

    def update(self, G: NDArray[Any], child_G: NDArray[Any]) -> None:
        violations = _violation(G)
        if not self.initialized:
            descending = np.sort(violations)[::-1]
            # A population of fewer than 20 solutions has no solution after its 5% most violated ones.
            index = min(math.ceil(0.05 * descending.size), descending.size - 1)
            self.epsilon = float(descending[index])
            self.phi_max = float(descending[0])
            self.initialized = True
        self.phi_max = max(self.phi_max, float(_violation(np.atleast_2d(child_G))[0]))
        self.updates += 1
        if self.updates % violations.size == 0:
            self.generation += 1
            if self.generation >= self.tc:
                self.epsilon = 0.0
            elif _feasibility_ratio(G) < self.feasibility_ratio_threshold:
                self.epsilon = (1.0 - self.tao) * self.epsilon
            else:
                self.epsilon = self.phi_max * (1.0 + self.tao)

    def replaces(
        self,
        child_G: NDArray[Any],
        child_values: NDArray[np.float64],
        G: NDArray[Any],
        values: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        child_violation = float(_violation(np.atleast_2d(child_G))[0])
        violations = _violation(G)
        both_within = (violations < self.epsilon) & (child_violation <= self.epsilon)
        by_value = both_within | (violations == child_violation)
        return np.asarray(np.where(by_value, child_values < values, child_violation < violations), dtype=np.bool_)

    def state(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "phi_max": self.phi_max,
            "generation": self.generation,
            "updates": self.updates,
            "initialized": self.initialized,
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        self.epsilon = float(state.get("epsilon", 0.0))
        self.phi_max = float(state.get("phi_max", 0.0))
        self.generation = int(state.get("generation", 0))
        self.updates = int(state.get("updates", 0))
        self.initialized = bool(state.get("initialized", False))


class ViolationThresholdCriterion(SubproblemUpdateCriterion):
    """Jan and Khanum's adaptive violation threshold, as jMetal's ``ViolationThresholdCriterion``."""

    name = "violation_threshold"

    def __init__(self) -> None:
        self.threshold = 0.0
        self.updates = 0

    @staticmethod
    def _measure(G: NDArray[Any]) -> NDArray[np.float64]:
        G = np.asarray(G, dtype=np.float64)
        return np.asarray(np.count_nonzero(G > 0.0, axis=1) * _violation(G), dtype=np.float64)

    def update(self, G: NDArray[Any], child_G: NDArray[Any]) -> None:
        if self.updates % np.asarray(G).shape[0] == 0:
            self.threshold = (1.0 - _feasibility_ratio(G)) * float(np.mean(self._measure(G)))
        self.updates += 1

    def replaces(
        self,
        child_G: NDArray[Any],
        child_values: NDArray[np.float64],
        G: NDArray[Any],
        values: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        child_measure = float(self._measure(np.atleast_2d(child_G))[0])
        measures = self._measure(G)
        above = (child_measure > self.threshold) | (measures > self.threshold)
        by_violation = above & (child_measure != measures)
        return np.asarray(np.where(by_violation, child_measure < measures, child_values < values), dtype=np.bool_)

    def state(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "updates": self.updates}

    def restore(self, state: Mapping[str, Any]) -> None:
        self.threshold = float(state.get("threshold", 0.0))
        self.updates = int(state.get("updates", 0))


def make_subproblem_update_criterion(value: object, max_evaluations: int, pop_size: int) -> SubproblemUpdateCriterion | None:
    """A run's criterion, None for ``feasibility_rules``, which keeps MOEA/D's own replacement.

    ``improved_epsilon`` reaches epsilon 0 at the generation ``tc``, ``decay_fraction`` of the run's
    generations (the evaluation budget over the population, at least 1) rounded as Java's
    ``Math.round``, and at least 1, as Evolver's ``SubproblemUpdateCriterionParameter``.
    """
    criterion = normalize_subproblem_update_criterion(value)
    if criterion is None or criterion[0] == "feasibility_rules":
        return None
    name, params = criterion
    if name == "violation_threshold":
        return ViolationThresholdCriterion()
    generations = max(1, int(max_evaluations) // max(1, int(pop_size)))
    tc = max(1, math.floor(params["decay_fraction"] * generations + 0.5))
    return ImprovedEpsilonCriterion(params["tao"], params["feasibility_ratio_threshold"], tc)


__all__ = [
    "SUBPROBLEM_UPDATE_CRITERIA",
    "SUBPROBLEM_UPDATE_PARAMETERS",
    "ImprovedEpsilonCriterion",
    "SubproblemUpdateCriterion",
    "ViolationThresholdCriterion",
    "make_subproblem_update_criterion",
    "normalize_subproblem_update_criterion",
]
