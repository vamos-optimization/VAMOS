"""Constraint-aware AGE-MOEA survival and the mating-tournament metrics it produces."""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos.engine.algorithm.components.constraint_handling import ConstraintHandlingTechnique, adapted_ranks
from vamos.foundation.constraints.utils import compute_violation, is_feasible
from vamos.foundation.kernel.backend import KernelBackend

from .geometry import age_survival


def constraint_aware_age_survival(
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any] | None,
    n_survive: int,
    kernel: KernelBackend,
    constraint_mode: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Apply feasibility-first ordering, then AGE geometry among feasible points.

    Returns the survivor indices and their AGE survival scores. Feasible
    survivors are scored among the feasible points only; infeasible survivors
    score 0, because the tournament compares them by violation.
    """
    if G is None or constraint_mode == "none":
        return age_survival(F, n_survive, kernel)

    feasible = is_feasible(G, n=G.shape[0])
    feasible_idx = np.flatnonzero(feasible)
    target = min(int(n_survive), F.shape[0])
    if feasible_idx.size >= target:
        local, local_scores = age_survival(F[feasible_idx], target, kernel)
        return np.asarray(feasible_idx[local], dtype=int), local_scores

    scores = np.zeros(target, dtype=float)
    if feasible_idx.size > 0:
        # Every feasible point survives; score them all, as pymoo's feasibility filter does.
        _, scores[: feasible_idx.size] = age_survival(F[feasible_idx], feasible_idx.size, kernel)
    violation = compute_violation(G, n=G.shape[0])
    infeasible_idx = np.flatnonzero(~feasible)
    needed = target - feasible_idx.size
    order = np.argsort(violation[infeasible_idx], kind="stable")
    fill = infeasible_idx[order[:needed]]
    return np.asarray(np.concatenate([feasible_idx, fill]), dtype=int), scores


def technique_age_survival(
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any],
    n_survive: int,
    kernel: KernelBackend,
    technique: ConstraintHandlingTechnique,
    rng: np.random.Generator,
    progress: float,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """AGE-MOEA's environmental selection over the fronts of a constraint-handling technique.

    The technique adapts to the union of parents and offspring at ``progress`` and ranks it, as in
    Evolver's AGE-MOEA; whole fronts survive and the survival score cuts the last one.
    """
    ranks = adapted_ranks(technique, kernel, F, G, rng, progress)
    return age_survival(F, n_survive, kernel, ranks=ranks)


def technique_selection_metrics(
    kernel: KernelBackend,
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any],
    technique: ConstraintHandlingTechnique,
    rng: np.random.Generator,
    survival_scores: np.ndarray[Any, Any] | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Tournament ranks and tie-break scores with a constraint-handling technique.

    The ranks are the technique's fronts of the population, which VAMOS ranks again as in NSGA-II; rank
    ties are broken by the AGE survival score. Without ``survival_scores`` (the initial population),
    the survival over the technique's fronts scores the whole population.
    """
    ranks = np.asarray(technique.ranks(kernel, F, G, rng), dtype=int)
    if survival_scores is None:
        survivors, survivor_scores = age_survival(F, F.shape[0], kernel, ranks=ranks)
        scores = np.zeros(F.shape[0], dtype=float)
        scores[survivors] = survivor_scores
    else:
        scores = np.asarray(survival_scores, dtype=float)
    return ranks, scores


def selection_metrics(
    kernel: KernelBackend,
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any] | None,
    constraint_mode: str,
    survival_scores: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Build tournament ranks and tie-break scores with feasibility-first ordering.

    Rank ties are broken by the AGE survival score, as in PlatEMO
    (``TournamentSelection(2, N, FrontNo, -CrowdDis)``) and pymoo; infeasible
    points rank after every feasible one and compare by violation.
    """
    ranks, _ = kernel.nsga2_ranking(F)
    scores = np.asarray(survival_scores, dtype=float)
    if G is None or constraint_mode == "none":
        return np.asarray(ranks, dtype=int), scores
    violation = compute_violation(G, n=G.shape[0])
    feasible = is_feasible(G, n=G.shape[0])
    if feasible.any():
        feasible_idx = np.flatnonzero(feasible)
        feasible_ranks, _ = kernel.nsga2_ranking(F[feasible_idx])
        ranks = np.full(F.shape[0], int(feasible_ranks.max(initial=0)) + 1, dtype=int)
        ranks[feasible_idx] = feasible_ranks
        scores = np.where(feasible, scores, -violation)
    else:
        ranks = np.zeros(F.shape[0], dtype=int)
        scores = -violation
    return np.asarray(ranks, dtype=int), np.asarray(scores, dtype=float)


def population_selection_metrics(
    kernel: KernelBackend,
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any] | None,
    constraint_mode: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Score a whole population by running the survival on it, as PlatEMO does at start-up."""
    survivors, survivor_scores = constraint_aware_age_survival(F, G, F.shape[0], kernel, constraint_mode)
    scores = np.zeros(F.shape[0], dtype=float)
    scores[survivors] = survivor_scores
    return selection_metrics(kernel, F, G, constraint_mode, scores)


__all__ = [
    "constraint_aware_age_survival",
    "population_selection_metrics",
    "selection_metrics",
    "technique_age_survival",
    "technique_selection_metrics",
]
