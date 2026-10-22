# algorithm/ibea/helpers.py
"""
Support functions for IBEA.

This module contains the core IBEA selection functions:
- Indicator computation (epsilon, hypervolume)
- Fitness calculation
- Environmental selection
- Constraint handling
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.constraints.utils import compute_violation, is_feasible

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import SelectionKernels


def epsilon_indicator(F: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Compute additive epsilon indicator matrix.

    Parameters
    ----------
    F : np.ndarray[Any, Any]
        Objective values, shape (N, n_obj).

    Returns
    -------
    np.ndarray[Any, Any]
        Epsilon indicator matrix, shape (N, N).
    """
    # Match jMetalPy: epsilon(i, j) = max_k (f_jk - f_ik)
    diff = F[None, :, :] - F[:, None, :]
    return np.asarray(np.max(diff, axis=2), dtype=float)


def hypervolume_indicator(F: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Compute hypervolume indicator matrix.

    Entry ``(i, j)`` is the hypervolume-difference indicator ``I_HD(x_j, x_i)`` of
    Zitzler and Kuenzli (2004), with the epsilon indicator's orientation: the
    fitness of ``x_i`` sums over its row. It is ``H(x_i) - H(x_j)`` when ``x_j``
    dominates ``x_i``, and otherwise the volume dominated by ``x_i`` but not by
    ``x_j``. Volumes use the reference point ``max(F) + 1``.

    Parameters
    ----------
    F : np.ndarray[Any, Any]
        Objective values, shape (N, n_obj).

    Returns
    -------
    np.ndarray[Any, Any]
        Hypervolume indicator matrix, shape (N, N).
    """
    n = F.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    ref = np.max(F, axis=0) + 1.0
    singleton_box = np.maximum(ref - F, 0.0)
    singleton_hv = np.prod(singleton_box, axis=1)
    pair_intersection = np.maximum(ref - np.maximum(F[:, None, :], F[None, :, :]), 0.0)
    intersection_hv = np.prod(pair_intersection, axis=2)
    indicator = singleton_hv[:, None] - intersection_hv
    row, column = F[:, None, :], F[None, :, :]
    column_dominates_row = np.all(column <= row, axis=2) & np.any(column < row, axis=2)
    indicator = np.where(column_dominates_row, singleton_hv[:, None] - singleton_hv[None, :], indicator)
    np.fill_diagonal(indicator, 0.0)
    return cast(np.ndarray[Any, Any], indicator)


def compute_indicator_matrix(F: np.ndarray[Any, Any], indicator: str) -> np.ndarray[Any, Any]:
    """Compute indicator matrix based on selected type.

    Parameters
    ----------
    F : np.ndarray[Any, Any]
        Objective values.
    indicator : str
        Indicator type: "epsilon" or "hypervolume".

    Returns
    -------
    np.ndarray[Any, Any]
        Indicator matrix.

    Notes
    -----
    The hypervolume indicator follows adaptive IBEA (Zitzler and Kuenzli, 2004), as in
    jMetal. Hypervolume differences grow with the product of the objective ranges, so
    the objectives are normalized over ``F`` and the indicator is divided by its largest
    magnitude, which keeps the fitness exponents within ``[-1/kappa, 1/kappa]``. The
    epsilon indicator is used unscaled, as in jMetalPy.
    """
    if indicator == "hypervolume":
        if F.shape[0] == 0:
            return np.empty((0, 0), dtype=float)
        low, high = np.min(F, axis=0), np.max(F, axis=0)
        values = hypervolume_indicator((F - low) / np.where(high > low, high - low, 1.0))
        largest = float(np.max(np.abs(values), initial=0.0))
        return values / largest if largest > 0.0 else values
    return epsilon_indicator(F)


# np.exp overflows float64 above an exponent of about 709.78. Up to _EXP_LIMIT, IBEA sums the
# exponential contributions directly; beyond it, fitness is handled in the log domain.
_EXP_LIMIT = 700.0


def _fitness_exponents(indicator: np.ndarray[Any, Any], kappa: float) -> np.ndarray[Any, Any]:
    """``-indicator / kappa``, with ``-inf`` (no contribution) on the diagonal and for undefined values.

    ``+inf``, from an infinite indicator or a division that overflows, stays the largest contribution.
    """
    with np.errstate(over="ignore"):
        exponent = -np.asarray(indicator, dtype=float) / kappa
    exponent[np.isnan(exponent)] = -np.inf
    np.fill_diagonal(exponent, -np.inf)
    return exponent


def _overflows(exponent: np.ndarray[Any, Any]) -> bool:
    return float(np.max(exponent, initial=-np.inf)) > _EXP_LIMIT


def _log_sum_exp(exponent: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Row-wise ``log(sum(exp(exponent)))``; ``-inf`` for rows without contributions, ``+inf`` with an infinite one."""
    top = np.max(exponent, axis=1, initial=-np.inf)
    shift = np.where(np.isfinite(top), top, 0.0)
    with np.errstate(divide="ignore"):
        return np.asarray(shift + np.log(np.sum(np.exp(exponent - shift[:, None]), axis=1)), dtype=float)


def _penalty_ranks(G: np.ndarray[Any, Any] | None, n: int) -> np.ndarray[Any, Any]:
    """0 for feasible solutions; 1 + the rank of the total violation for infeasible ones."""
    rank = np.zeros(n, dtype=float)
    if G is None:
        return rank
    feasible = is_feasible(G)
    if feasible.all():
        return rank
    _, violation_rank = np.unique(compute_violation(G)[~feasible], return_inverse=True)
    rank[~feasible] = 1.0 + violation_rank
    return rank


def _log_domain_fitness(log_sums: np.ndarray[Any, Any], rank: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Finite fitness in the same order as the penalized ``-exp(log_sums)``, without overflow."""
    fitness = -np.asarray(log_sums, dtype=float)
    finite = np.isfinite(fitness)
    best, worst = float(np.max(fitness, where=finite, initial=0.0)), float(np.min(fitness, where=finite, initial=0.0))
    fitness[fitness == np.inf] = best + 1.0  # no contributions
    fitness[fitness == -np.inf] = worst - 1.0  # an infinite contribution
    span = float(np.max(fitness, initial=0.0) - np.min(fitness, initial=0.0)) + 1.0
    return np.asarray(fitness - span * rank, dtype=float)


def _sum_exceeds(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> bool:
    """Whether ``sum(exp(a))`` exceeds ``sum(exp(b))`` when their float log-sums tie.

    Equal exponents contribute equally, so they cancel; the remaining terms decide, including
    those that a shared leading term absorbed in floating point.
    """
    left, right = Counter(a[a > -np.inf].tolist()), Counter(b[b > -np.inf].tolist())
    rest_left = np.fromiter((left - right).elements(), dtype=float)
    rest_right = np.fromiter((right - left).elements(), dtype=float)
    return bool(_log_sum_exp(rest_left[None, :])[0] > _log_sum_exp(rest_right[None, :])[0])


def _largest_sum(exponent: np.ndarray[Any, Any], log_sums: np.ndarray[Any, Any], candidates: np.ndarray[Any, Any]) -> int:
    """The first candidate row with the largest ``sum(exp(row))``, resolving float ties exactly."""
    tied = candidates[log_sums[candidates] == np.max(log_sums[candidates])]
    largest = int(tied[0])
    for row in tied[1:]:
        if _sum_exceeds(exponent[row], exponent[largest]):
            largest = int(row)
    return largest


def _log_domain_selection(
    exponent: np.ndarray[Any, Any], rank: np.ndarray[Any, Any], pop_size: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """IBEA's removal loop for contributions beyond the float64 range.

    The fitness of an active solution is ``-exp(L)``, with ``L`` the log-sum of its contributions
    from the other active solutions. ``L`` is recomputed after every removal, so a removed
    solution's dominant term no longer hides the others. The worst solution has the highest
    penalty rank and, among those, the largest ``L``: the order that the penalized fitness
    defines. Rows whose ``L`` tie in floating point are compared on the terms they do not share.
    Fitness differences below floating-point resolution decide as in the non-overflow path.
    """
    active: NDArray[np.int_] = np.arange(exponent.shape[0], dtype=int)
    while active.size > pop_size:
        sub = exponent[np.ix_(active, active)]
        candidates = np.flatnonzero(rank[active] == np.max(rank[active]))
        worst_pos = _largest_sum(sub, _log_sum_exp(sub), candidates)
        active = np.delete(active, worst_pos)
    return active, _log_domain_fitness(_log_sum_exp(exponent[np.ix_(active, active)]), rank[active])


def ibea_fitness(indicator: np.ndarray[Any, Any], kappa: float) -> np.ndarray[Any, Any]:
    """Compute IBEA fitness from indicator matrix.

    IBEA fitness is the negative sum of exponential indicator contributions.
    Lower fitness values are worse (more negative); the worst is removed.

    Parameters
    ----------
    indicator : np.ndarray[Any, Any]
        Indicator matrix, shape (N, N).
    kappa : float
        Scaling factor controlling selection pressure.

    Returns
    -------
    np.ndarray[Any, Any]
        Fitness values, shape (N,). When a contribution exceeds the float64 range, the
        values are finite and in the same order, but no longer equal to the sums.
    """
    exponent = _fitness_exponents(indicator, kappa)
    if _overflows(exponent):
        return _log_domain_fitness(_log_sum_exp(exponent), np.zeros(exponent.shape[0]))
    return np.asarray(-np.sum(np.exp(exponent), axis=1), dtype=float)


def apply_constraint_penalty(fitness: np.ndarray[Any, Any], G: np.ndarray[Any, Any] | None) -> np.ndarray[Any, Any]:
    """Lower the fitness of infeasible solutions so that selection removes them first.

    IBEA removes the solution with the lowest fitness. Each infeasible solution
    is lowered by a multiple of a penalty larger than any fitness magnitude, one
    multiple per rank of its total violation. The indicator updates of later
    removals never exceed that penalty, so feasible solutions always outrank
    infeasible ones and infeasible ones are removed in decreasing order of
    violation; the indicator decides only between equal violations.

    Parameters
    ----------
    fitness : np.ndarray[Any, Any]
        Fitness values (negative sums of indicator contributions).
    G : np.ndarray[Any, Any] | None
        Constraint values.

    Returns
    -------
    np.ndarray[Any, Any]
        Penalized fitness values.
    """
    penalized = np.array(fitness, dtype=float)
    rank = _penalty_ranks(G, penalized.shape[0])
    if not rank.any():
        return penalized
    penalty = float(np.max(np.abs(penalized), initial=0.0)) + 1.0
    infeasible = rank > 0.0
    penalized[infeasible] -= penalty * rank[infeasible]
    return penalized


def environmental_selection(
    X: np.ndarray[Any, Any],
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any] | None,
    pop_size: int,
    indicator: str,
    kappa: float,
    selection_kernels: SelectionKernels | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None, np.ndarray[Any, Any]]:
    """Perform IBEA environmental selection.

    Iteratively removes the worst individual until population size is reached.

    Parameters
    ----------
    X : np.ndarray[Any, Any]
        Decision variables, shape (N, n_var).
    F : np.ndarray[Any, Any]
        Objective values, shape (N, n_obj).
    G : np.ndarray[Any, Any] | None
        Constraint values or None.
    pop_size : int
        Target population size.
    indicator : str
        Indicator type.
    kappa : float
        Scaling factor.
    selection_kernels : SelectionKernels | None
        Optional compiled removal loop from the selected kernel backend.

    Returns
    -------
    tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None, np.ndarray[Any, Any]]
        (X_selected, F_selected, G_selected, fitness)
    """
    exponent = _fitness_exponents(compute_indicator_matrix(F, indicator), kappa)
    if _overflows(exponent):
        selected, fitness = _log_domain_selection(exponent, _penalty_ranks(G, F.shape[0]), pop_size)
        return X[selected], F[selected], G[selected] if G is not None else None, fitness
    contrib = np.exp(exponent)
    fitness = np.asarray(-np.sum(contrib, axis=1), dtype=float)
    fitness = apply_constraint_penalty(fitness, G)
    if selection_kernels is not None:
        # Same removal order and per-row accumulation order as the loop below.
        fitness = np.ascontiguousarray(fitness, dtype=np.float64)
        selected = np.asarray(selection_kernels.indicator_removal(fitness, contrib, pop_size), dtype=int)
        return X[selected], F[selected], G[selected] if G is not None else None, fitness[selected]

    active: NDArray[np.int_] = np.arange(X.shape[0], dtype=int)
    while active.size > pop_size:
        active_fitness = fitness[active]
        worst_pos = int(np.argmin(active_fitness))
        worst = int(active[worst_pos])
        delta = contrib[active, worst].copy()
        delta[worst_pos] = 0.0
        fitness[active] += delta
        active = np.delete(active, worst_pos)

    selected = active
    return X[selected], F[selected], G[selected] if G is not None else None, fitness[selected]


def combine_constraints(G: np.ndarray[Any, Any] | None, G_off: np.ndarray[Any, Any] | None) -> np.ndarray[Any, Any] | None:
    """Combine parent and offspring constraints.

    Parameters
    ----------
    G : np.ndarray[Any, Any] | None
        Parent constraints.
    G_off : np.ndarray[Any, Any] | None
        Offspring constraints.

    Returns
    -------
    np.ndarray[Any, Any] | None
        Combined constraints.
    """
    if G is None and G_off is None:
        return None
    if G is None:
        return G_off
    if G_off is None:
        return G
    return np.vstack([G, G_off])


__all__ = [
    "epsilon_indicator",
    "hypervolume_indicator",
    "compute_indicator_matrix",
    "ibea_fitness",
    "apply_constraint_penalty",
    "environmental_selection",
    "combine_constraints",
]
