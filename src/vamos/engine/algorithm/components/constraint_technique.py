"""The base class of the constraint-handling techniques (see :mod:`.constraint_handling`)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .constrained_ranking import RankArray

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import KernelBackend

#: The techniques that judge the violation of a single solution and compare pairs of solutions, which
#: RVEA's survival and SMPSO's personal bests use, as Evolver's ``EffectiveViolation``.
PAIRWISE_TECHNIQUES: tuple[str, ...] = ("cdp", "violated_constraints", "epsilon_constrained", "push_pull")

#: The techniques that their papers define for NSGA-II only. Each orders the population totally, one solution per
#: front, which would take the place of the niching, hypervolume contributions or survival scores of the others.
NSGAII_ONLY_TECHNIQUES: tuple[str, ...] = ("two_rankings", "infeasibility_driven", "adaptive_tradeoff")

#: A parameter of a technique: a number, or a name or a switch for the options of ``violation_objective``.
TechniqueParameter: TypeAlias = float | str | bool


def nsgaii_only(name: str, params: Mapping[str, TechniqueParameter]) -> bool:
    """Whether the technique, with these parameters, is defined for NSGA-II only.

    ``violation_objective`` is, with ``feasible_first``: Dam and Truong's survival orders the population totally.
    """
    return name in NSGAII_ONLY_TECHNIQUES or (name == "violation_objective" and params.get("feasible_first") is True)


def rejection_reason(name: str, techniques: tuple[str, ...], params: Mapping[str, TechniqueParameter] | None = None) -> str:
    """Why an algorithm that takes ``techniques`` rejects the technique ``name`` with ``params``."""
    if nsgaii_only(name, params or {}) and set(PAIRWISE_TECHNIQUES) < set(techniques):
        return "is defined for NSGA-II only: its total order would replace this algorithm's own diversity mechanism"
    return "does not judge the violation of a single solution"


def check_progress(progress: float, name: str) -> float:
    """The run's progress for the technique ``name``, which must be a budget's fraction in [0, 1]."""
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
    #: Whether two rows with the same positive effective violation compare by Pareto dominance.
    dominance_on_ties = False
    #: Whether :meth:`ranks` puts each row in a front of its own, a total order. NSGA-II then keeps whole
    #: fronts only, so it does not compute the crowding distance that would order a front.
    total_order = False

    def bind_run(self, population_size: int, max_evaluations: int) -> None:
        """Learn the run's population size and evaluation budget, before its first :meth:`update`.

        Only ``dynamic_penalty`` reads them, to count generations; the others ignore them.
        """
        return None

    def update(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], progress: float) -> None:
        """Adapt to the population about to be ranked, at ``progress`` (evaluations over the budget)."""
        return None

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        """The front of each row (0 first)."""
        raise NotImplementedError

    def survived(self, F: NDArray[Any], G: NDArray[Any]) -> None:
        """Learn the population that a replacement kept, once after each replacement.

        NSGA-II calls it, the only algorithm that takes the techniques that need it; the others ignore it.
        A technique that keeps what it learns returns it from :meth:`state`, which checkpoints save.
        """
        return None

    def effective_violation(self, G: NDArray[Any]) -> NDArray[np.float64]:
        """The violation of each row as the technique judges it: 0 for a row that it treats as feasible.

        Only the techniques of :data:`PAIRWISE_TECHNIQUES` judge single rows.
        """
        raise ValueError(
            f"constraint_handling {self.name!r} does not judge the violation of a single solution; use {', '.join(PAIRWISE_TECHNIQUES)}."
        )

    def state(self) -> dict[str, Any]:
        """What :meth:`update` has learnt, for a checkpoint."""
        return {}

    def restore(self, state: Mapping[str, Any]) -> None:
        """Resume from :meth:`state`."""
        return None


__all__ = [
    "NSGAII_ONLY_TECHNIQUES",
    "PAIRWISE_TECHNIQUES",
    "ConstraintHandlingTechnique",
    "TechniqueParameter",
    "check_progress",
    "nsgaii_only",
    "rejection_reason",
]
