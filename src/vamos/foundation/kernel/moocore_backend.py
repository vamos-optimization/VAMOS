from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, cast, overload

import moocore
import numpy as np

from vamos.foundation.quality_indicators.hypervolume import hypervolume as hv_fn

from .backend import DecisionArray, FloatArray, IndexArray, KernelBackend, NumericArray, RealArray
from .numpy_backend import NumPyKernel as _NumPyKernel
from .numpy_backend import _compute_crowding

IntArray = IndexArray


def _fronts_from_ranks(ranks: IntArray) -> list[list[int]]:
    if ranks.size == 0:
        return []
    unique_ranks = np.unique(ranks)
    return [np.flatnonzero(ranks == r).tolist() for r in unique_ranks]


class MooCoreKernel(KernelBackend):
    """MooCore backend: MooCore's Pareto ranking in ranking and buffered NSGA-II survival.

    Crowding distances, tournaments and variation use the NumPy reference, so the
    trajectories match the NumPy kernel.
    """

    name = "moocore"

    def __init__(self) -> None:
        self._numpy_ops = cast(Any, _NumPyKernel)()
        self._X_buffer: DecisionArray | None = None
        self._F_buffer: FloatArray | None = None
        self._keep_buffer: IntArray | None = None

    def capabilities(self) -> Iterable[str]:
        return ("c_backend",)

    def quality_indicators(self) -> Iterable[str]:
        return ("hypervolume",)

    def nsga2_ranking(self, F: RealArray) -> tuple[IndexArray, FloatArray]:
        ranks = cast(
            IndexArray,
            moocore.pareto_rank(np.asarray(F, dtype=np.float64, order="C")),
        )
        fronts = _fronts_from_ranks(ranks)
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
        return cast(IndexArray, self._numpy_ops.tournament_selection(ranks, crowding, pressure, rng, n_parents))

    def sbx_crossover(
        self,
        X_parents: RealArray,
        params: Mapping[str, object],
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> FloatArray:
        return cast(FloatArray, self._numpy_ops.sbx_crossover(X_parents, params, rng, xl, xu))

    def polynomial_mutation(
        self,
        X: DecisionArray,
        params: Mapping[str, object],
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> None:
        self._numpy_ops.polynomial_mutation(X, params, rng, xl, xu)

    def _ensure_buffers(self, total: int, n_var: int, n_obj: int, dtype: Any) -> None:
        dtype = np.dtype(dtype)
        if self._X_buffer is None or self._X_buffer.shape[0] < total or self._X_buffer.shape[1] != n_var or self._X_buffer.dtype != dtype:
            self._X_buffer = np.empty((total, n_var), dtype=dtype, order="C")
        if self._F_buffer is None or self._F_buffer.shape[0] < total or self._F_buffer.shape[1] != n_obj:
            self._F_buffer = np.empty((total, n_obj), dtype=np.float64, order="C")

    def _combine(
        self,
        X_a: DecisionArray,
        F_a: RealArray,
        X_b: DecisionArray,
        F_b: RealArray,
    ) -> tuple[DecisionArray, FloatArray, int]:
        n_a = X_a.shape[0]
        n_b = X_b.shape[0]
        total = n_a + n_b
        if total == 0:
            raise ValueError("Cannot combine empty populations.")
        n_var = X_a.shape[1] if n_a > 0 else X_b.shape[1]
        n_obj = F_a.shape[1] if F_a.size else F_b.shape[1]
        dtype = X_a.dtype if n_a > 0 else X_b.dtype
        self._ensure_buffers(total, n_var, n_obj, dtype)
        assert self._X_buffer is not None
        assert self._F_buffer is not None
        X_view = self._X_buffer[:total]
        F_view = self._F_buffer[:total]
        if n_a > 0:
            np.copyto(X_view[:n_a], X_a)
            np.copyto(F_view[:n_a], F_a, casting="unsafe")
        if n_b > 0:
            np.copyto(X_view[n_a:], X_b)
            np.copyto(F_view[n_a:], F_b, casting="unsafe")
        return X_view, F_view, total

    def _ensure_keep_buffer(self, size: int) -> IntArray:
        if self._keep_buffer is None or self._keep_buffer.shape[0] < size:
            self._keep_buffer = np.empty(size, dtype=np.int64)
        return self._keep_buffer[:size]

    @staticmethod
    def _crowding_single(front: RealArray) -> FloatArray:
        if front.shape[0] == 0:
            return np.empty(0, dtype=float)
        fronts = [list(range(front.shape[0]))]
        return _compute_crowding(front, fronts)

    def _select_partial_front(self, F_comb: RealArray, front_idx: IndexArray, remaining: int) -> IntArray:
        """The ``remaining`` members of the splitting front with the largest crowding distance, as NumPy picks them."""
        crowded = self._crowding_single(F_comb[front_idx])
        order = np.argsort(crowded)[::-1][:remaining]
        return front_idx[order]

    def _survive(
        self,
        X_a: DecisionArray,
        F_a: RealArray,
        X_b: DecisionArray,
        F_b: RealArray,
        target_size: int,
        return_indices: bool = False,
    ) -> tuple[DecisionArray, FloatArray] | tuple[DecisionArray, FloatArray, IntArray]:
        X_comb, F_comb, total = self._combine(X_a, F_a, X_b, F_b)
        ranks = cast(IndexArray, moocore.pareto_rank(F_comb))
        order = np.argsort(ranks, kind="mergesort")
        keep = self._ensure_keep_buffer(target_size)
        taken = 0
        idx = 0
        while taken < target_size and idx < total:
            rank_value = ranks[order[idx]]
            start = idx
            while idx < total and ranks[order[idx]] == rank_value:
                idx += 1
            front_idx = order[start:idx]
            if taken + front_idx.size <= target_size:
                keep[taken : taken + front_idx.size] = front_idx
                taken += front_idx.size
            else:
                remaining = target_size - taken
                selected = self._select_partial_front(F_comb, front_idx, remaining)
                keep[taken : taken + selected.size] = selected
                taken += selected.size
        sel = keep[:target_size].copy()
        # Fancy indexing copies: the caller owns the survivors, as with the NumPy and Numba kernels.
        if return_indices:
            return X_comb[sel], F_comb[sel], sel
        return X_comb[sel], F_comb[sel]

    @overload
    def nsga2_survival(
        self,
        X: DecisionArray,
        F: RealArray,
        X_off: DecisionArray,
        F_off: RealArray,
        pop_size: int,
        return_indices: Literal[False] = False,
    ) -> tuple[DecisionArray, FloatArray]: ...

    @overload
    def nsga2_survival(
        self,
        X: DecisionArray,
        F: RealArray,
        X_off: DecisionArray,
        F_off: RealArray,
        pop_size: int,
        return_indices: Literal[True],
    ) -> tuple[DecisionArray, FloatArray, IntArray]: ...

    def nsga2_survival(
        self,
        X: DecisionArray,
        F: RealArray,
        X_off: DecisionArray,
        F_off: RealArray,
        pop_size: int,
        return_indices: bool = False,
    ) -> tuple[DecisionArray, FloatArray] | tuple[DecisionArray, FloatArray, IntArray]:
        return self._survive(X, F, X_off, F_off, pop_size, return_indices=return_indices)

    def hypervolume(self, points: RealArray, reference_point: RealArray) -> float:
        return hv_fn(points, reference_point)
