"""MCR and the ensemble of constraint-handling techniques, for NSGA-II (see :mod:`.constraint_handling`).

Both follow H. Fukumoto and A. Oyama's framework, "A Generic Framework for Incorporating Constraint Handling
Techniques into Multi-Objective Evolutionary Algorithms", EvoApplications 2018, LNCS 10784, pp. 634-649
(Algorithm 1, Fig. 2): the objectives map to a fitness, the fitness to a rank, and a constraint-handling technique
acts on either. For NSGA-II the fitness is A. Vodopija, A. Oyama and B. Filipic's, "Ensemble-Based Constraint
Handling in Multiobjective Optimization", GECCO '19 Companion, pp. 2072-2075 (Algorithm 1, lines 12-14): the
Pareto rank minus the crowding distance, normalized so as to keep the Pareto order (:func:`nsgaii_fitness`).
Neither paper says how. VAMOS scales the finite crowding distances of the population min-max into [0, 1/4], all to
0 when they are equal, and puts the infinite ones of the extremes at 1/2, the top of the scale: the extremes stay
preferred to every other row of their level, as in NSGA-II, and a scale below 1 keeps the Pareto order.

Both techniques order the population totally (``total_order``); rows with the same value keep the order of the
population. They are defined for NSGA-II.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .constrained_ranking import RankArray, pareto_ranks, ranks_by_violation, violation
from .constraint_penalties import DynamicPenalty
from .constraint_technique import ConstraintHandlingTechnique

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import KernelBackend

#: The members of the ensemble, in the order of its weights.
ENSEMBLE_MEMBERS: tuple[str, ...] = ("nds", "cdp", "mcr", "dpf")
# The scale of the normalized crowding distance: the finite distances span [0, 1/4] and the infinite ones are 1/2.
_FINITE_SCALE = 0.25
_INFINITE_SCALE = 0.5


def nsgaii_fitness(kernel: KernelBackend, F: NDArray[Any], levels: RankArray) -> NDArray[np.float64]:
    """The level of each row minus its crowding distance within the level, normalized (see the module)."""
    F = np.asarray(F, dtype=np.float64)
    distance = np.empty(F.shape[0], dtype=np.float64)
    for level in np.unique(levels):
        rows = np.flatnonzero(levels == level)
        distance[rows] = np.inf if rows.size <= 2 else np.asarray(kernel.nsga2_ranking(np.ascontiguousarray(F[rows]))[1])
    return np.asarray(levels, dtype=np.float64) - _scaled(distance)


def _scaled(distance: NDArray[np.float64]) -> NDArray[np.float64]:
    finite = np.isfinite(distance)
    scaled = np.full(distance.size, _INFINITE_SCALE)
    if np.any(finite):
        low, high = distance[finite].min(), distance[finite].max()
        span = high - low
        scaled[finite] = _FINITE_SCALE * (distance[finite] - low) / span if span > 0.0 else 0.0
    return scaled


def competition_ranks(values: NDArray[Any]) -> NDArray[np.int64]:
    """1 plus the number of rows with a smaller value: equal values share a rank (Garcia et al.)."""
    values = np.asarray(values, dtype=np.float64)
    return np.asarray(np.searchsorted(np.sort(values), values, side="left") + 1, dtype=np.int64)


def multiple_constraint_ranking_fitness(kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any]) -> NDArray[np.float64]:
    """Garcia et al.'s MCR (eq. 10): ``R_Nv + sum_j R_phi^j``, plus ``R_f`` if some row is feasible.

    R. P. Garcia, B. S. L. P. de Lima, A. C. de C. Lemonge and B. P. Jacob, "A rank-based constraint handling
    technique for engineering design optimization problems solved by genetic algorithms", Computers and
    Structures 187:77-87, 2017. ``R_Nv`` ranks the number of violated constraints, ``R_phi^j`` the violation of
    constraint ``j``, and ``R_f`` the objective: for several objectives, the NSGA-II fitness of the objectives
    alone (:func:`nsgaii_fitness`), the rank-type use of the framework.
    """
    F = np.asarray(F, dtype=np.float64)
    excess = np.maximum(np.asarray(G, dtype=np.float64), 0.0)
    total = competition_ranks(np.count_nonzero(excess > 0.0, axis=1))
    for column in excess.T:
        total = total + competition_ranks(column)
    if np.any(np.all(excess <= 0.0, axis=1)):
        total = total + objective_ranks(kernel, F)
    return np.asarray(total, dtype=np.float64)


def objective_ranks(kernel: KernelBackend, F: NDArray[Any]) -> NDArray[np.int64]:
    """MCR's ``R_f``: the competition rank of the NSGA-II fitness of the objectives alone.

    Rows with the same objectives share the best fitness among them, so that equal values share a rank, as in
    Garcia et al.; with one objective this is the rank of the objective itself.
    """
    F = np.asarray(F, dtype=np.float64)
    fitness = nsgaii_fitness(kernel, F, pareto_ranks(kernel, F))
    _, groups = np.unique(F, axis=0, return_inverse=True)
    groups = np.asarray(groups).reshape(-1)
    best = np.full(int(groups.max()) + 1 if groups.size else 0, np.inf)
    np.minimum.at(best, groups, fitness)
    return competition_ranks(best[groups])


def _min_max(values: NDArray[np.float64]) -> NDArray[np.float64]:
    span = float(np.ptp(values)) if values.size else 0.0
    return (values - values.min()) / span if span > 0.0 else np.zeros_like(values)


def _order_ranks(values: NDArray[np.float64]) -> RankArray:
    ranks = np.empty(values.size, dtype=np.int64)
    ranks[np.argsort(values, kind="stable")] = np.arange(values.size, dtype=np.int64)
    return ranks


class MultipleConstraintRanking(ConstraintHandlingTechnique):
    """Garcia et al.'s MCR: the best by :func:`multiple_constraint_ranking_fitness`, one row per front."""

    name = "multiple_constraint_ranking"
    total_order = True

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        return _order_ranks(multiple_constraint_ranking_fitness(kernel, F, G))


class Ensemble(ConstraintHandlingTechnique):
    """Vodopija, Oyama and Filipic's ensemble: the best by ``c_e = sum_i w_i c_i`` over NDS, CDP, MCR and DPF.

    Each member's quality is normalized min-max over the population into [0, 1], lower being better, which the
    paper leaves open: NDS and CDP are the NSGA-II fitness with Pareto dominance and with the constrained-domination
    principle, MCR is :func:`multiple_constraint_ranking_fitness`, and DPF is the NDS fitness plus Joines and
    Houck's dynamic penalty with ``dynamic_penalty``'s settings, C = 0.5 and alpha = beta = 2, which the paper does
    not give. The weights, 1/4 each by default, are normalized to sum 1.
    """

    name = "ensemble"
    total_order = True
    needs_progress = True

    def __init__(self, weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25)) -> None:
        values = np.asarray(weights, dtype=np.float64)
        if values.shape != (len(ENSEMBLE_MEMBERS),) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                f"ensemble needs a finite weight of at least 0 for each of {', '.join(ENSEMBLE_MEMBERS)}; got {list(weights)}."
            )
        largest = float(values.max())
        if largest <= 0.0:
            raise ValueError("ensemble needs a weight above 0.")
        # Over the largest first, so that the sum of large finite weights cannot overflow.
        scaled = values / largest
        self.weights = tuple(float(weight) for weight in scaled / scaled.sum())
        self.penalty = DynamicPenalty(constant=0.5, alpha=2.0, beta=2.0)

    def bind_run(self, population_size: int, max_evaluations: int) -> None:
        self.penalty.bind_run(population_size, max_evaluations)

    def update(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], progress: float) -> None:
        self.penalty.update(kernel, F, G, progress)

    def qualities(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any]) -> NDArray[np.float64]:
        """Each member's normalized quality of each row: columns NDS, CDP, MCR and DPF."""
        F = np.asarray(F, dtype=np.float64)
        nds = nsgaii_fitness(kernel, F, pareto_ranks(kernel, F))
        cdp = nsgaii_fitness(kernel, F, ranks_by_violation(kernel, F, violation(G), objectives_on_ties=True))
        mcr = multiple_constraint_ranking_fitness(kernel, F, G)
        excess = np.maximum(np.asarray(G, dtype=np.float64), 0.0)
        dpf = nds + self.penalty.weight * np.sum(excess**self.penalty.beta, axis=1)
        return np.column_stack([_min_max(nds), _min_max(cdp), _min_max(mcr), _min_max(dpf)])

    def ranks(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], rng: np.random.Generator) -> RankArray:
        return _order_ranks(self.qualities(kernel, F, G) @ np.asarray(self.weights))

    def state(self) -> dict[str, Any]:
        return self.penalty.state()

    def restore(self, state: Mapping[str, Any]) -> None:
        self.penalty.restore(state)


__all__ = [
    "ENSEMBLE_MEMBERS",
    "Ensemble",
    "MultipleConstraintRanking",
    "competition_ranks",
    "multiple_constraint_ranking_fitness",
    "nsgaii_fitness",
    "objective_ranks",
]
