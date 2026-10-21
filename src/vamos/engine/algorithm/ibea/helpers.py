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


# np.exp overflows float64 above an exponent of about 709.78. Past _EXP_RESCALE_ABOVE every
# contribution is scaled down so that the largest exponent becomes _EXP_RESCALED_MAX, which
# leaves headroom for summing contributions and for the constraint penalty.
_EXP_RESCALE_ABOVE = 700.0
_EXP_RESCALED_MAX = 600.0


def _exponential_contributions(indicator: np.ndarray[Any, Any], kappa: float) -> np.ndarray[Any, Any]:
    """Return ``exp(-indicator / kappa)``, scaled by one common factor when it would overflow.

    IBEA only compares fitness values and adds contributions to them, so scaling every
    contribution by the same positive factor leaves its selections unchanged. Contributions
    far below the largest one may then underflow to zero. Nonfinite indicator values
    contribute zero.
    """
    exponent = -np.asarray(indicator, dtype=float) / kappa
    finite = np.isfinite(exponent)
    largest = float(np.max(exponent, where=finite, initial=-np.inf))
    if largest > _EXP_RESCALE_ABOVE:
        exponent = exponent - (largest - _EXP_RESCALED_MAX)
    return np.exp(exponent, where=finite, out=np.zeros_like(exponent))


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
        Fitness values, shape (N,).
    """
    mat = indicator.copy()
    np.fill_diagonal(mat, np.inf)
    contrib = _exponential_contributions(mat, kappa)
    return np.asarray(-np.sum(contrib, axis=1), dtype=float)


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
    if G is None:
        return np.asarray(fitness, dtype=float)
    feas = is_feasible(G)
    penalized = np.array(fitness, dtype=float)
    if feas.all():
        return penalized
    infeasible = ~feas
    _, violation_rank = np.unique(compute_violation(G)[infeasible], return_inverse=True)
    penalty = float(np.max(np.abs(penalized), initial=0.0)) + 1.0
    penalized[infeasible] -= penalty * (1.0 + violation_rank)
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
    ind = compute_indicator_matrix(F, indicator)
    contrib = _exponential_contributions(ind, kappa)
    np.fill_diagonal(contrib, 0.0)
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
