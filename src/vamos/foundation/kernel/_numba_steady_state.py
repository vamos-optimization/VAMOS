"""Compiled one-offspring (steady-state) NSGA-II step.

The algorithm draws every random value from the run-owned
``numpy.random.Generator`` in the reference order and shape; these kernels only
consume the draws. Together they reproduce the reference steady-state step
element by element: a binary tournament on rank and crowding, SBX on one mating
pair, clipping, polynomial mutation and clipping again (this module, reusing the
compiled SBX and polynomial-mutation arithmetic), then the insertion of the
evaluated child into the cached fronts and the NSGA-II truncation back to the
population size (:mod:`._numba_steady_survival`).
"""

from __future__ import annotations

from functools import cache
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._numba_selection import binary_tournament_winners_numba
from ._numba_steady_survival import INT_WORK_ROWS, SPLIT_CROWD, nsga2_steady_insert, nsga2_steady_replace
from ._numba_variation import _SBX_EPS, _sbx_apply, _sbx_mark_active, numpy_power_squares, polynomial_mutation_apply
from .numba_ops import njit

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

# Layout of the ``params`` vector of :func:`nsga2_steady_offspring` (see ``pack_variation``).
PARAM_PROB_CROSSOVER = 0
PARAM_NEG_ETA_PLUS_ONE = 1
PARAM_INV_ETA_PLUS_ONE = 2
PARAM_PROB_VAR = 3
PARAM_PROB_MUTATION = 4
PARAM_ETA_MUTATION = 5
PARAM_SQUARES_EXPONENT_TWO = 6
N_PARAMS = 7

# Rows of the ``bounds`` matrix of :func:`nsga2_steady_offspring`.
BOUND_SBX_LOWER = 0
BOUND_SBX_UPPER = 1
BOUND_CLIP_LOWER = 2
BOUND_CLIP_UPPER = 3
BOUND_PM_LOWER = 4
BOUND_PM_UPPER = 5
N_BOUNDS = 6


@njit(cache=True)
def nsga2_steady_tournament(
    ranks: NDArray[np.integer[Any]],
    crowding: FloatArray,
    first: IntArray,
    second: IntArray,
    X: FloatArray,
    winners: IntArray,
    differing: IntArray,
) -> int:
    """Binary tournaments of the reference kernel.

    ``second`` holds draws from ``[0, N - 1)`` and is shifted in place past
    ``first`` like the reference (``second + (second >= first)``). When some
    tournaments tie, they are marked with ``-1`` in ``winners`` and minus the
    number of ties is returned; the caller draws one coin per tie and passes
    them to :func:`nsga2_steady_offspring`. Otherwise the return value is the
    number ``k`` of variables in which the two parents differ by more than the
    SBX threshold, and ``differing[:k]`` holds their stream offsets
    ``1 + j`` (the SBX variable draws follow the pair draw), so the caller can
    tell from the variable draws whether SBX changes the pair.
    """
    for i in range(first.shape[0]):
        if second[i] >= first[i]:
            second[i] += 1
    found = binary_tournament_winners_numba(ranks, crowding, first, second)
    n_ties = 0
    for i in range(found.shape[0]):
        winners[i] = found[i]
        if found[i] < 0:
            n_ties += 1
    if n_ties > 0:
        return -n_ties
    count = 0
    for j in range(X.shape[1]):
        a = X[winners[0], j]
        b = X[winners[1], j]
        # Same test as the compiled SBX for a variable it may change.
        diff = b - a if a < b else a - b
        if diff > _SBX_EPS:
            differing[count] = 1 + j
            count += 1
    return count


@njit(cache=True)
def _clip_rows(values: FloatArray, lower: FloatArray, upper: FloatArray) -> None:
    """In-place ``np.clip(values, lower, upper)`` with NumPy's NaN and signed-zero rules."""
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            # maximum(value, lower) keeps a NaN value and otherwise returns ``value if value > lower else lower``.
            if value == value and not (value > lower[j]):
                value = lower[j]
            if value == value and not (value < upper[j]):
                value = upper[j]
            values[i, j] = value


@njit(cache=True, error_model="numpy")
def nsga2_steady_offspring(
    X: FloatArray,
    winners: IntArray,
    first: IntArray,
    second: IntArray,
    coins: NDArray[np.integer[Any]],
    stream: FloatArray,
    available: int,
    params: FloatArray,
    bounds: FloatArray,
    work: FloatArray,
    active: NDArray[np.bool_],
    child: FloatArray,
) -> int:
    """Write the first child of SBX + clip + polynomial mutation + clip into ``child[0]``.

    ``stream`` holds the uniform doubles drawn after the tournament, in order:
    the SBX pair draw, the SBX variable draws (when ``prob_var < 1``), the SBX
    spread and swap draws (only when a variable changes), then the mutation
    mask and delta grids for both children. Only ``stream[:available]`` has been
    drawn. When more draws are needed the kernel returns their total count
    without writing ``child``; the caller draws the rest and calls it again.
    Returns ``0`` once ``child`` is written.
    """
    n_var = X.shape[1]
    squares_exponent_two = params[PARAM_SQUARES_EXPONENT_TWO] != 0.0
    tie = 0
    for i in range(2):
        if winners[i] < 0:
            winners[i] = first[i] if coins[tie] != 0 else second[i]
            tie += 1
    for j in range(n_var):
        work[0, j] = X[winners[0], j]
        work[1, j] = X[winners[1], j]

    offspring = work.reshape(1, 2, n_var)
    pair_index = np.zeros(1, dtype=np.int64)
    position = 1
    crossed = False
    if stream[0] <= params[PARAM_PROB_CROSSOVER]:
        use_var_draws = params[PARAM_PROB_VAR] < 1.0
        var_draws = stream[1 : 1 + n_var].reshape(1, n_var)
        if use_var_draws:
            position += n_var
        if _sbx_mark_active(offspring, pair_index, var_draws, params[PARAM_PROB_VAR], use_var_draws, active):
            crossed = True
            position += 2 * n_var
    needed = position + 4 * n_var
    if needed > available:
        return needed

    if crossed:
        rand = stream[position - 2 * n_var : position - n_var].reshape(1, n_var)
        swap = stream[position - n_var : position].reshape(1, n_var)
        _sbx_apply(
            offspring,
            pair_index,
            active,
            rand,
            swap,
            bounds[BOUND_SBX_LOWER],
            bounds[BOUND_SBX_UPPER],
            params[PARAM_NEG_ETA_PLUS_ONE],
            params[PARAM_INV_ETA_PLUS_ONE],
            squares_exponent_two,
        )
    _clip_rows(work, bounds[BOUND_CLIP_LOWER], bounds[BOUND_CLIP_UPPER])
    rnd_mask = stream[position : position + 2 * n_var].reshape(2, n_var)
    rnd_delta = stream[position + 2 * n_var : position + 4 * n_var].reshape(2, n_var)
    polynomial_mutation_apply(
        work,
        params[PARAM_PROB_MUTATION],
        params[PARAM_ETA_MUTATION],
        bounds[BOUND_PM_LOWER],
        bounds[BOUND_PM_UPPER],
        rnd_mask,
        rnd_delta,
        squares_exponent_two,
    )
    _clip_rows(work, bounds[BOUND_CLIP_LOWER], bounds[BOUND_CLIP_UPPER])
    for j in range(n_var):
        child[0, j] = work[0, j]
    return 0


class NumbaSteadyStateKernels:
    """Compiled steady-state NSGA-II step exposed through ``KernelBackend.nsga2_steady_state_kernels``."""

    name = "numba"
    tournament = staticmethod(nsga2_steady_tournament)
    offspring = staticmethod(nsga2_steady_offspring)
    insert = staticmethod(nsga2_steady_insert)
    replace = staticmethod(nsga2_steady_replace)

    def survival_workspace(self, pop_size: int, n_obj: int) -> tuple[IntArray, FloatArray, FloatArray]:
        """Workspaces of ``insert``/``replace`` and the view holding the truncated front's crowding."""
        int_work = np.empty((INT_WORK_ROWS + n_obj, pop_size + 1), dtype=np.int64)
        float_work = np.empty((2, pop_size + 1), dtype=np.float64)
        return int_work, float_work, float_work[SPLIT_CROWD]

    def pack_variation(
        self,
        *,
        prob_crossover: float,
        eta_crossover: float,
        prob_var: float,
        prob_mutation: float,
        eta_mutation: float,
        crossover_bounds: tuple[FloatArray, FloatArray],
        repair_bounds: tuple[FloatArray, FloatArray],
        mutation_bounds: tuple[FloatArray, FloatArray],
    ) -> tuple[FloatArray, FloatArray]:
        """Pack operator parameters and bounds the way :func:`nsga2_steady_offspring` reads them.

        The derived exponents are computed exactly as the reference SBX does.
        """
        params = np.empty(N_PARAMS, dtype=np.float64)
        params[PARAM_PROB_CROSSOVER] = float(prob_crossover)
        params[PARAM_NEG_ETA_PLUS_ONE] = -(float(eta_crossover) + 1.0)
        params[PARAM_INV_ETA_PLUS_ONE] = 1.0 / (float(eta_crossover) + 1.0)
        params[PARAM_PROB_VAR] = float(prob_var)
        params[PARAM_PROB_MUTATION] = float(prob_mutation)
        params[PARAM_ETA_MUTATION] = float(eta_mutation)
        params[PARAM_SQUARES_EXPONENT_TWO] = 1.0 if numpy_power_squares() else 0.0
        rows = (*crossover_bounds, *repair_bounds, *mutation_bounds)
        bounds = np.ascontiguousarray(np.vstack([np.asarray(row, dtype=np.float64) for row in rows]))
        if bounds.shape[0] != N_BOUNDS:
            raise ValueError("pack_variation expects lower and upper bounds for crossover, repair and mutation.")
        return params, bounds


@cache
def numba_steady_state_kernels() -> NumbaSteadyStateKernels:
    """Shared compiled steady-state kernels (created on first use, not at import)."""
    return NumbaSteadyStateKernels()


__all__ = [
    "NumbaSteadyStateKernels",
    "nsga2_steady_offspring",
    "nsga2_steady_tournament",
    "numba_steady_state_kernels",
]
