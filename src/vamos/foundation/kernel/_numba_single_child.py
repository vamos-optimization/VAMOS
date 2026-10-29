"""Compiled single-child SBX and polynomial mutation for steady-state loops.

A steady-state step mates one pair with SBX, keeps the first child, clips it to
the bounds, applies polynomial mutation and clips again. The reference is the
operator chain ``SBXCrossover`` -> ``ClampRepair`` -> ``PolynomialMutation`` ->
``ClampRepair`` with compiled SBX and mutation arithmetic. This module fuses the
chain into two compiled calls; the random draws stay in Python and follow the
reference order, shape and conditions exactly.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._numba_variation import _sbx_apply, _sbx_mark_active, numpy_power_squares, polynomial_mutation_apply
from .numba_ops import njit

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
IndexArray: TypeAlias = NDArray[np.intp]


@njit(cache=True)
def npy_clip(value: float, lower: float, upper: float) -> float:
    """``np.clip`` for one float: ``min(max(value, lower), upper)`` with NumPy's NaN and signed-zero rules."""
    if value == value and not value > lower:
        value = lower
    if value == value and not value < upper:
        value = upper
    return value


@njit(cache=True)
def _load_pair_and_mark(
    X: FloatArray,
    first: int,
    second: int,
    pair: FloatArray,
    pair_index: IndexArray,
    crossover: bool,
    var_draws: FloatArray,
    prob_var: float,
    use_var_draws: bool,
    active: BoolArray,
) -> bool:
    """Copy the two parents into ``pair`` and report whether SBX changes any variable."""
    for j in range(X.shape[1]):
        pair[0, 0, j] = X[first, j]
        pair[0, 1, j] = X[second, j]
    if not crossover:
        return False
    return _sbx_mark_active(pair, pair_index, var_draws, prob_var, use_var_draws, active)


@njit(cache=True)
def _finish_child(
    pair: FloatArray,
    pair_index: IndexArray,
    apply_sbx: bool,
    active: BoolArray,
    draws: FloatArray,
    bounds: FloatArray,
    params: FloatArray,
    squares_exponent_two: bool,
    out: FloatArray,
) -> None:
    """SBX (when it changes the pair), clip the first child, mutate it and clip again into ``out``.

    ``draws`` stacks the SBX spread and swap grids and the mutation mask and
    delta grids; ``bounds`` stacks the SBX, mutation and clip lower/upper rows;
    ``params`` holds ``-(eta_c + 1)``, ``1 / (eta_c + 1)``, the mutation
    probability and ``eta_m``.
    """
    if apply_sbx:
        _sbx_apply(pair, pair_index, active, draws[1], draws[2], bounds[0], bounds[1], params[0], params[1], squares_exponent_two)
    clip_lower = bounds[4]
    clip_upper = bounds[5]
    n_var = pair.shape[2]
    for j in range(n_var):
        out[0, j] = npy_clip(pair[0, 0, j], clip_lower[j], clip_upper[j])
    polynomial_mutation_apply(out, params[2], params[3], bounds[2], bounds[3], draws[3], draws[4], squares_exponent_two)
    for j in range(n_var):
        out[0, j] = npy_clip(out[0, j], clip_lower[j], clip_upper[j])


class NumbaSingleChildVariation:
    """One steady-state child from rows ``first`` and ``second`` of a float64 population.

    Draws, in the reference order: the SBX pair coin; the variable-selection
    grid when the pair mates and ``prob_var < 1``; the spread and swap grids
    only when SBX changes a variable; then the mutation mask and delta grids.
    Consecutive draws are taken in one call, which consumes the generator
    exactly like the separate reference calls.
    """

    def __init__(
        self,
        *,
        sbx_prob: float,
        sbx_eta: float,
        sbx_prob_var: float,
        sbx_lower: FloatArray,
        sbx_upper: FloatArray,
        pm_prob: float,
        pm_eta: float,
        pm_lower: FloatArray,
        pm_upper: FloatArray,
        clip_lower: FloatArray,
        clip_upper: FloatArray,
    ) -> None:
        self._bounds = np.ascontiguousarray(np.stack((sbx_lower, sbx_upper, pm_lower, pm_upper, clip_lower, clip_upper)), dtype=np.float64)
        n_var = self._bounds.shape[1]
        eta_plus_one = float(sbx_eta) + 1.0
        self._params = np.array([-eta_plus_one, 1.0 / eta_plus_one, float(pm_prob), float(pm_eta)], dtype=np.float64)
        self._sbx_prob = float(sbx_prob)
        self._prob_var = float(sbx_prob_var)
        self._use_var_draws = self._prob_var < 1.0
        # A uniform draw in [0, 1) always passes the pair coin when the probability is at least one.
        self._always_mates = self._sbx_prob >= 1.0
        self._squares = numpy_power_squares()
        self._pair_index = np.zeros(1, dtype=np.intp)
        self._pair = np.empty((1, 2, n_var), dtype=np.float64)
        self._active = np.empty((1, n_var), dtype=np.bool_)
        self._coin_and_var = np.empty(1 + n_var, dtype=np.float64)
        self._coin = self._coin_and_var[:1]
        self._var_draws = self._coin_and_var[1:].reshape(1, n_var)
        self._first_draws = self._coin_and_var if self._use_var_draws else self._coin
        # Rows: unused, SBX spread, SBX swap, mutation mask, mutation delta.
        self._draws = np.empty((5, 1, n_var), dtype=np.float64)
        self._sbx_and_mutation_draws = self._draws[1:]
        self._mutation_draws = self._draws[3:]

    def __call__(self, X: FloatArray, first: int, second: int, out: FloatArray, rng: np.random.Generator) -> None:
        if self._always_mates:
            rng.random(out=self._first_draws)
            crossover = True
        else:
            rng.random(out=self._coin)
            crossover = bool(self._coin[0] <= self._sbx_prob)
            if crossover and self._use_var_draws:
                rng.random(out=self._var_draws)
        apply_sbx = _load_pair_and_mark(
            X, first, second, self._pair, self._pair_index, crossover, self._var_draws, self._prob_var, self._use_var_draws, self._active
        )
        rng.random(out=self._sbx_and_mutation_draws if apply_sbx else self._mutation_draws)
        _finish_child(self._pair, self._pair_index, apply_sbx, self._active, self._draws, self._bounds, self._params, self._squares, out)


__all__ = ["NumbaSingleChildVariation", "npy_clip"]
