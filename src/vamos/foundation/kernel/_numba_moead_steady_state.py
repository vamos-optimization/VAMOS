"""Compiled one-offspring (steady-state) MOEA/D step.

Exposed through ``KernelBackend.moead_steady_state_kernels`` (see
:mod:`.moead_kernels` for the contract). The algorithm draws every random value
from the run-owned ``numpy.random.Generator`` in the reference order and shape;
these kernels only consume the draws. They reuse the compiled DE, SBX and
polynomial-mutation arithmetic (:mod:`._numba_variation`) and the compiled
neighbourhood replacement (:mod:`._numba_moead_neighborhood`), and add the glue
between them: parent selection, the clip repair with ``np.clip`` semantics and
the ideal-point update.

Arguments are packed to keep the per-call dispatch cheap. The rows of ``draws``
follow the draw order (:data:`VAR_DRAWS`, :data:`RAND`, :data:`SWAP`,
:data:`PM_MASK`, :data:`PM_DELTA`), the rows of ``bounds`` are the clip,
polynomial-mutation and SBX lower and upper bounds, ``params`` follows the
``PARAM_*`` layout and ``slots`` the ``SLOT_*`` layout.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from numpy.typing import NDArray

from ._numba_moead_neighborhood import update_neighborhood_numba
from ._numba_variation import _sbx_apply, _sbx_mark_active, differential_crossover_apply, numpy_power_squares, polynomial_mutation_apply
from .moead_kernels import IDEAL_DECLINED, MOEADStepWorkspace
from .numba_ops import njit

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

# Rows of ``draws``, in draw order: consecutive draws fill consecutive rows with one Generator call.
VAR_DRAWS, RAND, SWAP, PM_MASK, PM_DELTA = 0, 1, 2, 3, 4
N_DRAWS = 5
# Rows of ``bounds``.
CLIP_LOWER, CLIP_UPPER, PM_LOWER, PM_UPPER, SBX_LOWER, SBX_UPPER = 0, 1, 2, 3, 4, 5
# Entries of ``params``; the derived SBX exponents are computed as the reference SBX does.
PARAM_DE_F, PARAM_DE_CR, PARAM_PM_PROB, PARAM_PM_ETA, PARAM_SBX_PROB_VAR = 0, 1, 2, 3, 4
PARAM_SBX_NEG_ETA_PLUS_ONE, PARAM_SBX_INV_ETA_PLUS_ONE, PARAM_THETA, PARAM_RHO, PARAM_SQUARES = 5, 6, 7, 8, 9
N_PARAMS = 10
# Entries of ``slots``: outputs of the last child, then the replacement settings.
SLOT_ACTIVE, SLOT_FIRST, SLOT_SECOND, SLOT_J_RAND, SLOT_PAIR, SLOT_REPLACE_LIMIT, SLOT_AGGREGATION = 0, 1, 2, 3, 4, 5, 6
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
    slots[SLOT_ACTIVE] = index
    slots[SLOT_FIRST] = first
    slots[SLOT_SECOND] = second
    flags[0] = use_neighbors
    mating[0, 0, :] = X[first]
    mating[0, 1, :] = X[second]


@njit(cache=True, error_model="numpy")
def _repair_and_mutate(child: FloatArray, draws: FloatArray, bounds: FloatArray, params: FloatArray) -> None:
    # Crossover repair, polynomial mutation, then the mutation repair.
    clip_row(child[0], bounds[CLIP_LOWER], bounds[CLIP_UPPER])
    polynomial_mutation_apply(
        child,
        params[PARAM_PM_PROB],
        params[PARAM_PM_ETA],
        bounds[PM_LOWER],
        bounds[PM_UPPER],
        draws[PM_MASK : PM_MASK + 1],
        draws[PM_DELTA : PM_DELTA + 1],
        params[PARAM_SQUARES] != 0.0,
    )
    clip_row(child[0], bounds[CLIP_LOWER], bounds[CLIP_UPPER])


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
    slots: IntArray,
    flags: BoolArray,
    mating: FloatArray,
    child: FloatArray,
) -> None:
    """Write the DE child of subproblem ``index`` (repaired and mutated) into ``child``."""
    _select_parents(X, neighbors, index, use_neighbors, first, second, slots, flags, mating)
    mating[0, 2, :] = X[index]
    slots[SLOT_J_RAND] = j_forced
    j_rand = slots[SLOT_J_RAND : SLOT_J_RAND + 1]
    differential_crossover_apply(mating, params[PARAM_DE_F], params[PARAM_DE_CR], draws[RAND : RAND + 1], j_rand, child)
    _repair_and_mutate(child, draws, bounds, params)


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
    slots[SLOT_PAIR] = 0
    pair = slots[SLOT_PAIR : SLOT_PAIR + 1]
    var_draws = draws[VAR_DRAWS : VAR_DRAWS + 1]
    return bool(_sbx_mark_active(mating, pair, var_draws, params[PARAM_SBX_PROB_VAR], use_var_draws, sbx_active))


@njit(cache=True, error_model="numpy")
def sbx_child(
    any_active: bool,
    draws: FloatArray,
    bounds: FloatArray,
    params: FloatArray,
    slots: IntArray,
    mating: FloatArray,
    sbx_active: BoolArray,
    child: FloatArray,
) -> None:
    """Write the first SBX child (repaired and mutated) into ``child``."""
    if any_active:
        _sbx_apply(
            mating,
            slots[SLOT_PAIR : SLOT_PAIR + 1],
            sbx_active,
            draws[RAND : RAND + 1],
            draws[SWAP : SWAP + 1],
            bounds[SBX_LOWER],
            bounds[SBX_UPPER],
            params[PARAM_SBX_NEG_ETA_PLUS_ONE],
            params[PARAM_SBX_INV_ETA_PLUS_ONE],
            params[PARAM_SQUARES] != 0.0,
        )
    child[0, :] = mating[0, 0, :]
    _repair_and_mutate(child, draws, bounds, params)


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
    """Update the ideal point, then replace neighbours; :data:`IDEAL_DECLINED` leaves everything unchanged."""
    child_f = children_f[0]
    if update_ideal:
        for j in range(ideal.shape[0]):
            value = child_f[j]
            best = ideal[j]
            if np.isnan(value) or np.isnan(best):
                return IDEAL_DECLINED
            if value == best and np.signbit(value) != np.signbit(best):
                return IDEAL_DECLINED
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
        slots[SLOT_REPLACE_LIMIT],
        slots[SLOT_AGGREGATION],
        params[PARAM_THETA],
        params[PARAM_RHO],
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
    slots: IntArray,
    flags: BoolArray,
    mating: FloatArray,
    child: FloatArray,
) -> int:
    """:func:`replace_neighbors` for the evaluated child, then :func:`de_child` for the next one."""
    status = replace_neighbors(
        X, F, G, cv, weights, ideal, children, children_f, children_g, child_cv, order, use_constraints, slots, params, True
    )
    if status != IDEAL_DECLINED:
        de_child(X, neighbors, index, use_neighbors, first, second, j_forced, draws, bounds, params, slots, flags, mating, child)
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
    """:func:`replace_neighbors` for the evaluated child, then :func:`sbx_parents` for the next one."""
    status = replace_neighbors(
        X, F, G, cv, weights, ideal, children, children_f, children_g, child_cv, order, use_constraints, slots, params, True
    )
    if status == IDEAL_DECLINED:
        return status
    any_active = sbx_parents(
        X, neighbors, index, use_neighbors, first, second, crossed, use_var_draws, draws, params, slots, flags, mating, sbx_active
    )
    return 1 if any_active else 0


class NumbaMOEADSteadyStateKernels:
    """Compiled one-offspring MOEA/D step exposed through ``KernelBackend.moead_steady_state_kernels``."""

    name = "numba"
    de_child = staticmethod(de_child)
    sbx_parents = staticmethod(sbx_parents)
    sbx_child = staticmethod(sbx_child)
    replace = staticmethod(replace_neighbors)
    replace_then_de_child = staticmethod(replace_then_de_child)
    replace_then_sbx_parents = staticmethod(replace_then_sbx_parents)

    def workspace(self, n_var: int, replace_limit: int, aggregation_id: int) -> MOEADStepWorkspace:
        """Buffers of one run, with the replacement settings stored in ``slots``."""
        draws = np.empty((N_DRAWS, n_var), dtype=np.float64)
        slots = np.zeros(N_SLOTS, dtype=np.int64)
        slots[SLOT_REPLACE_LIMIT] = int(replace_limit)
        slots[SLOT_AGGREGATION] = int(aggregation_id)
        return MOEADStepWorkspace(
            draws=draws,
            var_draws=draws[VAR_DRAWS : VAR_DRAWS + 1],
            rand=draws[RAND : RAND + 1],
            mutation_draws=draws[PM_MASK : PM_DELTA + 1],
            sbx_draws=draws[RAND : PM_DELTA + 1],
            slots=slots,
            active=slots[SLOT_ACTIVE : SLOT_ACTIVE + 1],
            pairs=slots[SLOT_FIRST : SLOT_SECOND + 1].reshape(1, 2),
            flags=np.zeros(1, dtype=np.bool_),
            mating=np.empty((1, 3, n_var), dtype=np.float64),
            sbx_active=np.zeros((1, n_var), dtype=np.bool_),
        )

    def pack_variation(
        self,
        *,
        de_f: float,
        de_cr: float,
        sbx_prob_var: float,
        sbx_eta: float,
        pm_prob: float,
        pm_eta: float,
        theta: float,
        rho: float,
        clip_bounds: tuple[FloatArray, FloatArray],
        mutation_bounds: tuple[FloatArray, FloatArray],
        crossover_bounds: tuple[FloatArray, FloatArray],
    ) -> tuple[FloatArray, FloatArray]:
        """Pack settings the way the kernels read them; SBX exponents are derived as the reference SBX does."""
        params = np.empty(N_PARAMS, dtype=np.float64)
        params[PARAM_DE_F] = float(de_f)
        params[PARAM_DE_CR] = float(de_cr)
        params[PARAM_PM_PROB] = float(pm_prob)
        params[PARAM_PM_ETA] = float(pm_eta)
        params[PARAM_SBX_PROB_VAR] = float(sbx_prob_var)
        params[PARAM_SBX_NEG_ETA_PLUS_ONE] = -(float(sbx_eta) + 1.0)
        params[PARAM_SBX_INV_ETA_PLUS_ONE] = 1.0 / (float(sbx_eta) + 1.0)
        params[PARAM_THETA] = float(theta)
        params[PARAM_RHO] = float(rho)
        params[PARAM_SQUARES] = 1.0 if numpy_power_squares() else 0.0
        rows = (*clip_bounds, *mutation_bounds, *crossover_bounds)
        bounds = np.ascontiguousarray(np.vstack([np.asarray(row, dtype=np.float64) for row in rows]))
        return params, bounds


@cache
def numba_moead_steady_state_kernels() -> NumbaMOEADSteadyStateKernels:
    """Shared compiled MOEA/D step kernels (created on first use, not at import)."""
    return NumbaMOEADSteadyStateKernels()


__all__ = [
    "NumbaMOEADSteadyStateKernels",
    "clip_row",
    "de_child",
    "numba_moead_steady_state_kernels",
    "replace_neighbors",
    "replace_then_de_child",
    "replace_then_sbx_parents",
    "sbx_child",
    "sbx_parents",
]
