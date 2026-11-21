"""The constraint-handling techniques that jMetal and Evolver do not have (see :mod:`.constraint_handling`).

Each ranks by Pareto dominance over modified objectives, as jMetal's self-adaptive penalty does: a static
penalty, Joines and Houck's non-stationary penalty, and the overall violation as one more objective.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .constrained_ranking import RankArray, pareto_ranks, violation
from .constraint_technique import ConstraintHandlingTechnique, check_progress

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import KernelBackend


class StaticPenalty(ConstraintHandlingTechnique):
    """A static penalty: Pareto dominance over every objective plus ``coefficient`` times the overall violation.

    The overall violation is the sum of the positive constraint values, as the other techniques measure
    it. A row that violates no constraint keeps its objectives, so a feasible population ranks by Pareto
    dominance alone.
    """

    name = "static_penalty"

    def __init__(self, coefficient: float) -> None:
        self.coefficient = float(coefficient)

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        return pareto_ranks(kernel, penalized_objectives(F, self.coefficient * violation(G)))


class DynamicPenalty(ConstraintHandlingTechnique):
    """Joines and Houck's non-stationary penalty, as a ranking by Pareto dominance.

    J. A. Joines and C. R. Houck, "On the use of non-stationary penalty functions to solve nonlinear
    constrained optimization problems with GA's", Proc. First IEEE Conference on Evolutionary Computation,
    1994, pp. 579-584, section 3.1: the GA minimizes ``F = f + P(alpha, beta)`` with ``P(alpha, beta) =
    (C * k) ** alpha * SVC(beta)``, where ``k`` is the generation and ``SVC(beta)`` sums each violation
    raised to ``beta``. Each objective becomes ``f_i + (constant * k) ** alpha * SVC``, the same penalty
    for every objective. VAMOS stores every constraint as ``g_j <= 0`` and measures its violation as
    ``max(0, g_j)``, without the tolerance band that the paper gives each constraint.

    ``k`` counts the run's evaluations in populations, and is at least 1: the progress of each
    :meth:`update` (the evaluations before the offspring's, over the budget) times the budget over the
    population size, which :meth:`bind_run` gives. A generational algorithm ranks its initial population
    and its first replacement at ``k = 1`` and its ``k``-th replacement at ``k``; a steady-state one adds
    ``1 / population_size`` per offspring rather than 1.
    """

    name = "dynamic_penalty"
    needs_progress = True

    def __init__(self, constant: float, alpha: float, beta: float) -> None:
        self.constant = float(constant)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.population_size = 0
        self.max_evaluations = 0
        self.generation = 1.0

    def bind_run(self, population_size: int, max_evaluations: int) -> None:
        if population_size < 1 or max_evaluations < 1:
            raise ValueError(f"dynamic_penalty needs a positive population and budget; got {population_size} and {max_evaluations}.")
        self.population_size = int(population_size)
        self.max_evaluations = int(max_evaluations)

    def update(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], progress: float) -> None:
        progress = check_progress(progress, "dynamic_penalty")
        if self.population_size < 1:
            raise ValueError("dynamic_penalty counts generations: give it the run's population and budget with bind_run().")
        self.generation = max(1.0, progress * self.max_evaluations / self.population_size)

    @property
    def weight(self) -> float:
        """``(constant * k) ** alpha``, the multiplier of the violations at the current generation."""
        return float((self.constant * self.generation) ** self.alpha)

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        excess = np.maximum(np.asarray(G, dtype=np.float64), 0.0)
        severity = np.asarray(np.sum(excess**self.beta, axis=1), dtype=np.float64)
        return pareto_ranks(kernel, penalized_objectives(F, self.weight * severity))

    def state(self) -> dict[str, Any]:
        return {"generation": self.generation}

    def restore(self, state: Mapping[str, Any]) -> None:
        self.generation = float(state.get("generation", 1.0))


class ViolationObjective(ConstraintHandlingTechnique):
    """The overall violation as one more objective: Pareto dominance over the objectives and the violation.

    A feasible population, whose violations are all 0, ranks by Pareto dominance over its objectives.
    """

    name = "violation_objective"

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        F = np.asarray(F, dtype=np.float64)
        return pareto_ranks(kernel, np.column_stack([F, violation(G)]))


def penalized_objectives(F: NDArray[Any], penalty: NDArray[np.float64]) -> NDArray[np.float64]:
    """Every objective of each row plus that row's penalty."""
    return np.asarray(np.asarray(F, dtype=np.float64) + np.asarray(penalty, dtype=np.float64)[:, None], dtype=np.float64)


__all__ = ["DynamicPenalty", "StaticPenalty", "ViolationObjective", "penalized_objectives"]
