"""Compiled one-offspring MOEA/D steps for the numba engine.

This module imports numba at import time; :mod:`.steady_state` loads it lazily.

The kernels never draw random numbers. The caller draws every value from the
run-owned ``numpy.random.Generator`` in the order, shape and conditions of the
reference ask/tell path and passes the draws in. The kernels reuse the compiled
DE, SBX and polynomial-mutation arithmetic and the compiled neighbourhood
replacement; they add only the glue between them: parent selection, the clip
repair with ``np.clip`` semantics and the ideal-point update.

Arguments are packed to keep the per-call dispatch cheap. The rows of ``draws``
(shape ``(5, n_var)``) are :data:`RAND`, :data:`PM_MASK`, :data:`PM_DELTA`,
:data:`VAR_DRAWS` and :data:`SWAP`. The rows of ``bounds`` (shape ``(6, n_var)``)
are the clip, polynomial-mutation and SBX lower and upper bounds. ``params``
holds ``f, cr, pm_prob, pm_eta, sbx_prob_var, -(sbx_eta + 1), 1 / (sbx_eta + 1)``
and the aggregation ``theta`` and ``rho``. ``slots`` (int64, length 7) receives the
active subproblem, the two parents, the forced DE index and the SBX pair index
(always 0), and holds the replacement limit and the aggregation id.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.kernel._numba_variation import (
    _sbx_apply,
    _sbx_mark_active,
    differential_crossover_apply,
    polynomial_mutation_apply,
)
from vamos.foundation.kernel.numba_ops import njit

from .neighborhood_numba import update_neighborhood_numba

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

RAND, PM_MASK, PM_DELTA, VAR_DRAWS, SWAP = 0, 1, 2, 3, 4
N_DRAWS = 5
N_SLOTS = 7


@njit(cache=True, error_model="numpy")
def clip_row(row: FloatArray, lower: FloatArray, upper: FloatArray) -> None:
    """``np.clip(row, lower, upper)`` in place, including its NaN and signed-zero rules.

    NumPy computes ``min(max(x, lo), hi)`` with ``max(a, b) = a if isnan(a) or a > b else b``
    and ``min(a, b) = a if isnan(a) or a < b else b``.
    """
    for j in range(row.shape[0]):
        value = row[j]
        if not np.isnan(value) and not value > lower[j]:
            value = lower[j]
        if not np.isnan(value) and not value < upper[j]:
            value = upper[j]
        row[j] = value


@njit(cache=True, error_model="numpy")
def _select_parents(
    X: FloatArray,
    neighbors: IntArray,
    index: int,
    use_neighbors: bool,
    first: int,
    second: int,
    slots: IntArray,
    flags: BoolArray,
    mating: FloatArray,
) -> None:
    # Two distinct draws: the second skips the first, as in MOEAD._sample_parent_pairs.
    if second >= first:
        second += 1
    if use_neighbors:
        first = neighbors[index, first]
        second = neighbors[index, second]
    slots[0] = index
    slots[1] = first
    slots[2] = second
    flags[0] = use_neighbors
    mating[0, 0, :] = X[first]
    mating[0, 1, :] = X[second]


@njit(cache=True, error_model="numpy")
def _repair_and_mutate(child: FloatArray, draws: FloatArray, bounds: FloatArray, params: FloatArray, squares: bool) -> None:
    # Crossover repair, polynomial mutation, then the mutation repair.
    clip_row(child[0], bounds[0], bounds[1])
    polynomial_mutation_apply(
        child, params[2], params[3], bounds[2], bounds[3], draws[PM_MASK : PM_MASK + 1], draws[PM_DELTA : PM_DELTA + 1], squares
    )
    clip_row(child[0], bounds[0], bounds[1])


@njit(cache=True, error_model="numpy")
def de_child(
    X: FloatArray,
    neighbors: IntArray,
    index: int,
    use_neighbors: bool,
    first: int,
    second: int,
    j_forced: int,
    draws: FloatArray,
    bounds: FloatArray,
    params: FloatArray,
    squares: bool,
    slots: IntArray,
    flags: BoolArray,
    mating: FloatArray,
    child: FloatArray,
) -> None:
    """Write the DE child of subproblem ``index`` (repaired and mutated) into ``child``."""
    _select_parents(X, neighbors, index, use_neighbors, first, second, slots, flags, mating)
    mating[0, 2, :] = X[index]
    slots[3] = j_forced
    differential_crossover_apply(mating, params[0], params[1], draws[RAND : RAND + 1], slots[3:4], child)
    _repair_and_mutate(child, draws, bounds, params, squares)


@njit(cache=True, error_model="numpy")
def sbx_parents(
    X: FloatArray,
    neighbors: IntArray,
    index: int,
    use_neighbors: bool,
    first: int,
    second: int,
    crossed: bool,
    use_var_draws: bool,
    draws: FloatArray,
    params: FloatArray,
    slots: IntArray,
    flags: BoolArray,
    mating: FloatArray,
    sbx_active: BoolArray,
) -> bool:
    """Gather the SBX parents and return whether any variable crosses (then the spread and swap draws follow)."""
    _select_parents(X, neighbors, index, use_neighbors, first, second, slots, flags, mating)
    if not crossed:
        return False
    slots[4] = 0
    return bool(_sbx_mark_active(mating, slots[4:5], draws[VAR_DRAWS : VAR_DRAWS + 1], params[4], use_var_draws, sbx_active))


@njit(cache=True, error_model="numpy")
def sbx_child(
    any_active: bool,
    draws: FloatArray,
    bounds: FloatArray,
    params: FloatArray,
    squares: bool,
    slots: IntArray,
    mating: FloatArray,
    sbx_active: BoolArray,
    child: FloatArray,
) -> None:
    """Write the first SBX child (repaired and mutated) into ``child``."""
    if any_active:
        _sbx_apply(
            mating,
            slots[4:5],
            sbx_active,
            draws[RAND : RAND + 1],
            draws[SWAP : SWAP + 1],
            bounds[4],
            bounds[5],
            params[5],
            params[6],
            squares,
        )
    child[0, :] = mating[0, 0, :]
    _repair_and_mutate(child, draws, bounds, params, squares)


@njit(cache=True, error_model="numpy")
def replace_neighbors(
    X: FloatArray,
    F: FloatArray,
    G: FloatArray,
    cv: FloatArray,
    weights: FloatArray,
    ideal: FloatArray,
    children: FloatArray,
    children_f: FloatArray,
    children_g: FloatArray,
    child_cv: float,
    order: IntArray,
    use_constraints: int,
    slots: IntArray,
    params: FloatArray,
    update_ideal: bool,
) -> int:
    """Update the ideal point, then replace neighbours; ``-1`` leaves everything unchanged.

    ``weights`` stacks the plain, zero-safe and unit weights. ``-1`` means ``np.minimum`` could pick between two different bit patterns
    (a NaN, or ``0.0`` against ``-0.0``): the caller then updates the ideal point
    with NumPy and calls again with ``update_ideal=False``.
    """
    child_f = children_f[0]
    if update_ideal:
        for j in range(ideal.shape[0]):
            value = child_f[j]
            best = ideal[j]
            if np.isnan(value) or np.isnan(best):
                return -1
            if value == best and np.signbit(value) != np.signbit(best):
                return -1
        for j in range(ideal.shape[0]):
            if child_f[j] < ideal[j]:
                ideal[j] = child_f[j]
    return update_neighborhood_numba(
        X,
        F,
        G,
        cv,
        weights[0],
        weights[1],
        weights[2],
        ideal,
        children[0],
        child_f,
        children_g[0],
        child_cv,
        order,
        order.shape[0],
        slots[5],
        slots[6],
        params[7],
        params[8],
        use_constraints,
    )


@njit(cache=True, error_model="numpy")
def replace_then_de_child(
    X: FloatArray,
    F: FloatArray,
    G: FloatArray,
    cv: FloatArray,
    weights: FloatArray,
    ideal: FloatArray,
    children: FloatArray,
    children_f: FloatArray,
    children_g: FloatArray,
    child_cv: float,
    order: IntArray,
    use_constraints: int,
    neighbors: IntArray,
    index: int,
    use_neighbors: bool,
    first: int,
    second: int,
    j_forced: int,
    draws: FloatArray,
    bounds: FloatArray,
    params: FloatArray,
    squares: bool,
    slots: IntArray,
    flags: BoolArray,
    mating: FloatArray,
    child: FloatArray,
) -> int:
    """:func:`replace_neighbors` for the evaluated child, then :func:`de_child` for the next one.

    A ``-1`` from the replacement skips both steps.
    """
    status = replace_neighbors(
        X, F, G, cv, weights, ideal, children, children_f, children_g, child_cv, order, use_constraints, slots, params, True
    )
    if status >= 0:
        de_child(X, neighbors, index, use_neighbors, first, second, j_forced, draws, bounds, params, squares, slots, flags, mating, child)
    return status


@njit(cache=True, error_model="numpy")
def replace_then_sbx_parents(
    X: FloatArray,
    F: FloatArray,
    G: FloatArray,
    cv: FloatArray,
    weights: FloatArray,
    ideal: FloatArray,
    children: FloatArray,
    children_f: FloatArray,
    children_g: FloatArray,
    child_cv: float,
    order: IntArray,
    use_constraints: int,
    neighbors: IntArray,
    index: int,
    use_neighbors: bool,
    first: int,
    second: int,
    crossed: bool,
    use_var_draws: bool,
    draws: FloatArray,
    params: FloatArray,
    slots: IntArray,
    flags: BoolArray,
    mating: FloatArray,
    sbx_active: BoolArray,
) -> int:
    """:func:`replace_neighbors` for the evaluated child, then :func:`sbx_parents` for the next one.

    Returns ``-1`` (neither step ran), or ``1``/``0`` for whether any SBX variable crosses.
    """
    status = replace_neighbors(
        X, F, G, cv, weights, ideal, children, children_f, children_g, child_cv, order, use_constraints, slots, params, True
    )
    if status < 0:
        return status
    any_active = sbx_parents(
        X, neighbors, index, use_neighbors, first, second, crossed, use_var_draws, draws, params, slots, flags, mating, sbx_active
    )
    return 1 if any_active else 0


__all__ = [
    "N_DRAWS",
    "N_SLOTS",
    "PM_DELTA",
    "PM_MASK",
    "RAND",
    "SWAP",
    "VAR_DRAWS",
    "clip_row",
    "de_child",
    "replace_neighbors",
    "replace_then_de_child",
    "replace_then_sbx_parents",
    "sbx_child",
    "sbx_parents",
]
