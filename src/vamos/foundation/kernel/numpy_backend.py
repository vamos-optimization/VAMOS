"""NumPy kernel backend.

Performance-sensitive: keep operations vectorized and avoid Python loops where possible.
Assumes F is float64 of shape (N, M), X is float64 of shape (N, n_var).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, TypeAlias, cast, overload

import numpy as np
from numpy.typing import NDArray

from .backend import DecisionArray, FloatArray, IndexArray, KernelBackend, NumericArray, RealArray, VariationKernels

BoolArray: TypeAlias = NDArray[np.bool_]
from .operator_primitives import polynomial_mutation_population, sbx_crossover_pairs

_DOMINANCE_TENSOR_BUDGET = 2_000_000
_TOURNAMENT_KEY_BUDGET = 2_000_000


def _dominance_rows(F_rows: RealArray, F_all: RealArray) -> BoolArray:
    """Compute the dominance relation for ``F_rows`` against ``F_all``."""
    less_equal = F_rows[:, None, :] <= F_all[None, :, :]
    strictly_less = F_rows[:, None, :] < F_all[None, :, :]
    return cast(
        BoolArray,
        np.logical_and(
            np.all(less_equal, axis=2),
            np.any(strictly_less, axis=2),
        ),
    )


def _fast_non_dominated_sort_dense(F: RealArray) -> tuple[list[list[int]], IndexArray]:
    """Dense vectorized fast non-dominated sort for moderate population sizes."""
    N = F.shape[0]
    less_equal = F[:, None, :] <= F[None, :, :]
    strictly_less = F[:, None, :] < F[None, :, :]
    dom_matrix = np.logical_and(
        np.all(less_equal, axis=2),
        np.any(strictly_less, axis=2),
    )

    dominated_count = dom_matrix.sum(axis=0).astype(np.int64)
    rank = np.empty(N, dtype=int)
    fronts: list[list[int]] = []

    current = np.flatnonzero(dominated_count == 0)
    level = 0
    while current.size > 0:
        fronts.append(current.tolist())
        rank[current] = level
        dom_contrib = dom_matrix[current].sum(axis=0)
        dominated_count -= dom_contrib
        dominated_count[current] = -1
        dom_matrix[current] = False
        level += 1
        current = np.flatnonzero(dominated_count == 0)

    return fronts, rank


def _fast_non_dominated_sort_blocked(F: RealArray) -> tuple[list[list[int]], IndexArray]:
    """Memory-safe fast non-dominated sort using blocked dominance sweeps."""
    N, M = F.shape
    rank = np.empty(N, dtype=int)
    fronts: list[list[int]] = []
    block_rows = max(1, _DOMINANCE_TENSOR_BUDGET // max(1, N * M))

    dominated_count = np.zeros(N, dtype=np.int64)
    for start in range(0, N, block_rows):
        stop = min(N, start + block_rows)
        dom_block = _dominance_rows(F[start:stop], F)
        local_rows = np.arange(start, stop, dtype=int) - start
        dom_block[local_rows, np.arange(start, stop, dtype=int)] = False
        dominated_count += dom_block.sum(axis=0, dtype=np.int64)

    current = np.flatnonzero(dominated_count == 0)
    level = 0
    while current.size > 0:
        fronts.append(current.tolist())
        rank[current] = level
        dominated_count[current] = -1

        dom_contrib = np.zeros(N, dtype=np.int64)
        current_start = 0
        while current_start < current.size:
            current_stop = min(current.size, current_start + block_rows)
            current_block = current[current_start:current_stop]
            dom_block = _dominance_rows(F[current_block], F)
            dom_block[:, current_block] = False
            dom_contrib += dom_block.sum(axis=0, dtype=np.int64)
            current_start = current_stop

        dominated_count -= dom_contrib
        dominated_count[current] = -1
        level += 1
        current = np.flatnonzero(dominated_count == 0)

    return fronts, rank


def _fast_non_dominated_sort(F: RealArray) -> tuple[list[list[int]], IndexArray]:
    """Classic O(N^2) fast non-dominated sort.

    Parameters
    ----------
    F
        Objective matrix with shape ``(N, M)``.

    Returns
    -------
    tuple[list[list[int]], IndexArray]
        Front membership lists and the per-solution front-rank array.
    """
    N = F.shape[0]
    if N == 0:
        return [], np.empty(0, dtype=int)

    if F.shape[0] * F.shape[0] * F.shape[1] <= _DOMINANCE_TENSOR_BUDGET:
        return _fast_non_dominated_sort_dense(F)
    return _fast_non_dominated_sort_blocked(F)


def _compute_crowding(F: RealArray, fronts: list[list[int]]) -> FloatArray:
    """
    Standard crowding-distance computation.
    crowding: array of length N.
    """
    N = F.shape[0]
    crowding = np.zeros(N)

    for front in fronts:
        if len(front) == 0:
            continue
        front_arr = np.asarray(front, dtype=int)
        if front_arr.size == 1:
            crowding[front_arr[0]] = np.inf
            continue

        fvals = F[front_arr]  # shape (k, n_obj)
        n_obj = fvals.shape[1]
        d = np.zeros(front_arr.size, dtype=float)

        for m in range(n_obj):
            order = np.argsort(fvals[:, m], kind="mergesort")
            sorted_vals = fvals[order, m]

            d[order[0]] = np.inf
            d[order[-1]] = np.inf

            span = sorted_vals[-1] - sorted_vals[0]
            if span <= 0.0:
                continue

            contrib = np.zeros_like(sorted_vals)
            contrib[1:-1] = (sorted_vals[2:] - sorted_vals[:-2]) / span
            d[order[1:-1]] += contrib[1:-1]

        crowding[front_arr] = d

    return crowding


def _select_nsga2(fronts: list[list[int]], crowding: RealArray, pop_size: int) -> IndexArray:
    """
    NSGA-II elitist selection based on fronts + crowding.
    """
    selected: list[int] = []
    for front in fronts:
        if len(front) == 0:
            continue
        front_arr = np.asarray(front, dtype=int)
        if len(selected) + front_arr.size <= pop_size:
            selected.extend(front_arr.tolist())
        else:
            rem = pop_size - len(selected)
            order = np.argsort(crowding[front_arr])[::-1]
            selected.extend(front_arr[order[:rem]].tolist())
            break
    return np.array(selected, dtype=int)


def _as_float(value: object | None, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return float(default)


def _sample_tournament_candidates(
    rng: np.random.Generator,
    *,
    n_candidates: int,
    n_parents: int,
    pressure: int,
) -> NDArray[np.int64]:
    take = n_candidates if pressure >= n_candidates else pressure
    kth = n_candidates - 1 if take == n_candidates else take - 1
    if n_parents * n_candidates <= _TOURNAMENT_KEY_BUDGET:
        keys = rng.random((n_parents, n_candidates))
        selected = np.argpartition(keys, kth=kth, axis=1)
        return selected[:, :take].astype(np.int64, copy=False)

    rows_per_chunk = max(1, _TOURNAMENT_KEY_BUDGET // max(1, n_candidates))
    candidates = np.empty((n_parents, take), dtype=np.int64)
    for start in range(0, n_parents, rows_per_chunk):
        stop = min(n_parents, start + rows_per_chunk)
        keys = rng.random((stop - start, n_candidates))
        selected = np.argpartition(keys, kth=kth, axis=1)
        candidates[start:stop] = selected[:, :take]
    return candidates


def _as_numeric_keys(values: NumericArray) -> NDArray[np.int64] | NDArray[np.float64]:
    """Pass integer and floating sort keys to the kernel without changing their values."""
    arr = np.asarray(values)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64, copy=False)
    return arr.astype(np.float64, copy=False)


def _tournament_winners_numpy(
    ranks: NumericArray,
    crowding: RealArray,
    candidates: NDArray[np.integer[Any]],
    tie_break: FloatArray,
    row_index: NDArray[np.int64],
) -> IndexArray:
    candidate_ranks = ranks[candidates]
    best_rank = candidate_ranks.min(axis=1, keepdims=True)
    best_rank_mask = candidate_ranks == best_rank

    candidate_crowding = np.nan_to_num(crowding[candidates], nan=-np.inf)
    filtered_crowding = np.where(best_rank_mask, candidate_crowding, -np.inf)
    best_crowding = filtered_crowding.max(axis=1, keepdims=True)
    # A NaN crowding becomes -inf, the value worse ranks are filtered to: only the best rank may tie.
    best_mask = (filtered_crowding == best_crowding) & best_rank_mask

    tie_keys = np.where(best_mask, tie_break, np.inf)
    winner_pos = np.argmin(tie_keys, axis=1)
    return np.asarray(candidates[row_index, winner_pos], dtype=int)


def _sbx_crossover_flat(
    X_parents: RealArray,
    params: Mapping[str, object],
    rng: np.random.Generator,
    xl: float | RealArray,
    xu: float | RealArray,
    *,
    kernels: VariationKernels | None,
) -> FloatArray:
    """Pair consecutive parent rows and apply SBX (shared by the kernel backends)."""
    Np, D = X_parents.shape
    if Np == 0:
        return np.empty_like(X_parents)
    # Handle odd parent count by duplicating the last parent
    if Np % 2 != 0:
        X_parents = np.vstack([X_parents, X_parents[-1:]])
        Np += 1
    lower, upper = NumPyKernel._normalize_bounds(xl, xu, D)
    pairs = X_parents.reshape(Np // 2, 2, D)
    offspring = sbx_crossover_pairs(
        pairs,
        rng=rng,
        lower=lower,
        upper=upper,
        prob_crossover=_as_float(params.get("prob"), 0.9),
        eta=_as_float(params.get("eta"), 20.0),
        inplace=False,
        kernels=kernels,
    )
    return offspring.reshape(Np, D)


class NumPyKernel(KernelBackend):
    """
    Backend with pure NumPy implementations of the NSGA-II kernels.
    """

    name = "numpy"

    def __init__(self) -> None:
        self._row_index = np.empty(0, dtype=np.int64)
        self._score_buffer: FloatArray | None = None

    def _ensure_row_index(self, length: int) -> NDArray[np.int64]:
        if self._row_index.shape[0] != length:
            self._row_index = np.arange(length, dtype=np.int64)
        return self._row_index

    def _ensure_score_buffer(self, shape: tuple[int, ...]) -> FloatArray:
        if self._score_buffer is None or self._score_buffer.shape != shape:
            self._score_buffer = np.empty(shape, dtype=np.float64)
        return self._score_buffer

    @staticmethod
    def _normalize_bounds(
        xl: float | RealArray,
        xu: float | RealArray,
        n_var: int,
    ) -> tuple[FloatArray, FloatArray]:
        lower = np.asarray(xl, dtype=float)
        upper = np.asarray(xu, dtype=float)
        if lower.ndim == 0 or (lower.ndim == 1 and lower.shape[0] == 1 and n_var > 1):
            lower = np.full(n_var, float(lower.reshape(-1)[0]))
        if upper.ndim == 0 or (upper.ndim == 1 and upper.shape[0] == 1 and n_var > 1):
            upper = np.full(n_var, float(upper.reshape(-1)[0]))
        return lower, upper

    def capabilities(self) -> Iterable[str]:
        return ("cpu",)

    def quality_indicators(self) -> Iterable[str]:
        return ("hypervolume",)

    def nsga2_ranking(self, F: RealArray) -> tuple[IndexArray, FloatArray]:
        fronts, ranks = _fast_non_dominated_sort(F)
        crowding = _compute_crowding(F, fronts)
        return ranks, crowding

    def tournament_selection(
        self,
        ranks: NumericArray,
        crowding: RealArray,
        pressure: int,
        rng: np.random.Generator,
        n_parents: int,
    ) -> IndexArray:
        """
        Standard binary/m-ary tournament:
        smallest rank wins; break ties with higher crowding.
        """
        N = ranks.shape[0]
        if pressure <= 0:
            raise ValueError("pressure must be a positive integer")
        if n_parents <= 0 or N == 0:
            return np.empty(0, dtype=int)
        if pressure > N:
            raise ValueError("pressure cannot exceed population size for tournament selection without replacement")

        if pressure == 1:
            return rng.integers(0, N, size=n_parents, dtype=int)

        if pressure == 2:
            first = rng.integers(0, N, size=n_parents, dtype=int)
            second = rng.integers(0, N - 1, size=n_parents, dtype=int)
            second = second + (second >= first)

            rank_first = ranks[first]
            rank_second = ranks[second]
            crowd_first = np.nan_to_num(crowding[first], nan=-np.inf)
            crowd_second = np.nan_to_num(crowding[second], nan=-np.inf)

            pick_first = rank_first < rank_second
            pick_second = rank_second < rank_first
            unresolved = ~(pick_first | pick_second)

            crowd_first_better = crowd_first > crowd_second
            crowd_second_better = crowd_second > crowd_first
            pick_first = pick_first | (unresolved & crowd_first_better)
            pick_second = pick_second | (unresolved & crowd_second_better)

            ties = ~(pick_first | pick_second)
            winners = np.where(pick_first, first, second)
            if np.any(ties):
                tie_coin = rng.integers(0, 2, size=int(np.sum(ties)), dtype=int).astype(bool)
                winners[ties] = np.where(tie_coin, first[ties], second[ties])
            return winners.astype(int, copy=False)

        candidates = _sample_tournament_candidates(
            rng,
            n_candidates=N,
            n_parents=n_parents,
            pressure=pressure,
        )
        row_index = self._ensure_row_index(n_parents)
        tie_break = rng.random((candidates.shape[0], candidates.shape[1]))
        return _tournament_winners_numpy(
            _as_numeric_keys(ranks),
            np.asarray(crowding, dtype=np.float64),
            candidates,
            tie_break,
            row_index,
        )

    def sbx_crossover(
        self,
        X_parents: RealArray,
        params: Mapping[str, object],
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> FloatArray:
        return _sbx_crossover_flat(X_parents, params, rng, xl, xu, kernels=None)

    def polynomial_mutation(
        self,
        X: RealArray,
        params: Mapping[str, object],
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> None:
        if X.size == 0:
            return
        n_var = X.shape[1]
        lower, upper = self._normalize_bounds(xl, xu, n_var)
        mutated = polynomial_mutation_population(
            X,
            rng=rng,
            lower=lower,
            upper=upper,
            prob_mutation=_as_float(params.get("prob"), 0.1),
            eta=_as_float(params.get("eta"), 20.0),
            inplace=False,
        )
        X[:] = mutated

    @overload
    def nsga2_survival(
        self,
        X: DecisionArray,
        F: RealArray,
        X_off: DecisionArray,
        F_off: RealArray,
        pop_size: int,
        return_indices: Literal[False] = False,
    ) -> tuple[DecisionArray, RealArray]: ...

    @overload
    def nsga2_survival(
        self,
        X: DecisionArray,
        F: RealArray,
        X_off: DecisionArray,
        F_off: RealArray,
        pop_size: int,
        return_indices: Literal[True],
    ) -> tuple[DecisionArray, RealArray, IndexArray]: ...

    def nsga2_survival(
        self,
        X: DecisionArray,
        F: RealArray,
        X_off: DecisionArray,
        F_off: RealArray,
        pop_size: int,
        return_indices: bool = False,
    ) -> tuple[DecisionArray, RealArray] | tuple[DecisionArray, RealArray, IndexArray]:
        """
        NSGA-II elitism: merge parents + offspring, re-rank, and select.
        """
        X_comb = np.vstack([X, X_off])
        F_comb = np.vstack([F, F_off])
        fronts, _ = _fast_non_dominated_sort(F_comb)
        crowding = _compute_crowding(F_comb, fronts)
        sel = _select_nsga2(fronts, crowding, pop_size)
        if return_indices:
            return X_comb[sel], F_comb[sel], sel
        return X_comb[sel], F_comb[sel]

    def hypervolume(self, points: FloatArray, reference_point: FloatArray) -> float:
        from vamos.foundation.quality_indicators.hypervolume import hypervolume as hv_fn

        return hv_fn(points, reference_point)
