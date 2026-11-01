from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import moocore
import numpy as np
from numpy.typing import NDArray

try:  # Optional JIT acceleration for tournament selection
    from numba import njit
except ImportError:  # pragma: no cover - optional dependency
    njit = None

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


def _tournament_winners_numpy(
    ranks: IntArray,
    crowding: FloatArray,
    candidates: IntArray,
) -> IntArray:
    n_rows = candidates.shape[0]
    winners = np.empty(n_rows, dtype=np.int64)
    for i in range(n_rows):
        best_idx = candidates[i, 0]
        best_rank = ranks[best_idx]
        best_crowd = crowding[best_idx]
        for j in range(1, candidates.shape[1]):
            idx = candidates[i, j]
            r = ranks[idx]
            if r < best_rank or (r == best_rank and crowding[idx] > best_crowd):
                best_rank = r
                best_crowd = crowding[idx]
                best_idx = idx
        winners[i] = best_idx
    return winners


if njit is not None:  # pragma: no cover - optional speed-up
    if TYPE_CHECKING:

        def _tournament_winners_numba(
            ranks: IntArray,
            crowding: FloatArray,
            candidates: IntArray,
        ) -> IntArray: ...

    else:

        @njit(cache=True)
        def _tournament_winners_numba(
            ranks: IntArray,
            crowding: FloatArray,
            candidates: IntArray,
        ) -> IntArray:
            n_rows = candidates.shape[0]
            n_cols = candidates.shape[1]
            winners = np.empty(n_rows, dtype=np.int64)
            for i in range(n_rows):
                best_idx = candidates[i, 0]
                best_rank = ranks[best_idx]
                best_crowd = crowding[best_idx]
                for j in range(1, n_cols):
                    idx = candidates[i, j]
                    r = ranks[idx]
                    if r < best_rank or (r == best_rank and crowding[idx] > best_crowd):
                        best_rank = r
                        best_crowd = crowding[idx]
                        best_idx = idx
                winners[i] = best_idx
            return winners

else:

    def _tournament_winners_numba(ranks: IntArray, crowding: FloatArray, candidates: IntArray) -> IntArray:
        return _tournament_winners_numpy(ranks, crowding, candidates)


class MooCoreKernel(KernelBackend):
    """
    Consolidated MooCore backend with buffered survival, adaptive HV/crowding,
    optional numba tournament selection, and incremental archive maintenance.
    """

    name = "moocore"

    def __init__(self) -> None:
        self._numpy_ops = cast(Any, _NumPyKernel)()
        self._X_buffer: DecisionArray | None = None
        self._F_buffer: FloatArray | None = None
        self._keep_buffer: IntArray | None = None
        self._archive_manager: _IncrementalArchive | None = None
        self._stats: dict[str, int] = {"buffer_resizes": 0}

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

    def update_archive(
        self,
        archive_X: DecisionArray | None,
        archive_F: FloatArray | None,
        population_X: DecisionArray,
        population_F: FloatArray,
        archive_size: int,
    ) -> tuple[DecisionArray | None, FloatArray | None]:
        if archive_size <= 0:
            return archive_X, archive_F
        if (
            self._archive_manager is None
            or self._archive_manager.capacity != archive_size
            or self._archive_manager._X.shape[1] != population_X.shape[1]
        ):
            self._archive_manager = _IncrementalArchive(
                self, archive_size, population_X.shape[1], population_F.shape[1], population_X.dtype
            )
            if archive_X is not None and archive_X.size:
                assert archive_F is not None
                self._archive_manager.bootstrap(archive_X, archive_F)
        elif archive_X is not None and archive_X.size and self._archive_manager.is_empty():
            assert archive_F is not None
            self._archive_manager.bootstrap(archive_X, archive_F)

        updated_X, updated_F = self._archive_manager.update(population_X, population_F)
        return updated_X, updated_F

    def hypervolume(self, points: RealArray, reference_point: RealArray) -> float:
        return hv_fn(points, reference_point)


class _IncrementalArchive:
    """
    Lightweight incremental archive that inserts candidates one-by-one,
    removing dominated points and trimming by leveraging the kernel's
    diversity selector when capacity is exceeded.
    """

    def __init__(
        self,
        kernel: MooCoreKernel,
        capacity: int,
        n_var: int,
        n_obj: int,
        dtype: Any,
    ) -> None:
        self.kernel = kernel
        self.capacity = int(capacity)
        self._dtype = np.dtype(dtype)
        self._X: DecisionArray = np.empty((0, n_var), dtype=self._dtype)
        self._F: FloatArray = np.empty((0, n_obj), dtype=float)

    def is_empty(self) -> bool:
        return self._X.size == 0

    def bootstrap(self, archive_X: DecisionArray, archive_F: FloatArray) -> None:
        self._X = np.asarray(archive_X, dtype=self._dtype, order="C").copy()
        self._F = np.asarray(archive_F, dtype=float, order="C").copy()

    def _dominates(self, candidates: RealArray, f: RealArray) -> NDArray[np.bool_]:
        if candidates.size == 0:
            return np.zeros(0, dtype=bool)
        less_equal = candidates <= f
        strictly_less = candidates < f
        dominates = np.all(less_equal, axis=1) & np.any(strictly_less, axis=1)
        return cast(NDArray[np.bool_], dominates)

    def _trim(self) -> None:
        if self._F.shape[0] <= self.capacity:
            return
        idx = np.arange(self._F.shape[0], dtype=int)
        selected = self.kernel._select_partial_front(self._F, idx, self.capacity)
        self._X = self._X[selected]
        self._F = self._F[selected]

    def update(self, pop_X: DecisionArray, pop_F: FloatArray) -> tuple[DecisionArray, FloatArray]:
        """
        Vectorized archive maintenance: merge, filter non-dominated, then trim.
        Avoids per-individual Python loops and repeated vstack allocations.
        """
        if pop_X.size == 0:
            return self._X, self._F

        pop_X = np.asarray(pop_X, dtype=self._dtype, order="C")
        pop_F = np.asarray(pop_F, dtype=float, order="C")

        if self._X.size == 0:
            X_comb = pop_X
            F_comb = pop_F
        else:
            X_comb = np.vstack((self._X, pop_X))
            F_comb = np.vstack((self._F, pop_F))

        F_comb = np.asarray(F_comb, dtype=float, order="C")
        nondom_mask = cast(NDArray[np.bool_], moocore.is_nondominated(F_comb))
        X_nd = X_comb[nondom_mask]
        F_nd = F_comb[nondom_mask]

        if F_nd.shape[0] > self.capacity:
            idx = np.arange(F_nd.shape[0], dtype=int)
            selected = self.kernel._select_partial_front(F_nd, idx, self.capacity)
            X_nd = X_nd[selected]
            F_nd = F_nd[selected]

        self._X = X_nd
        self._F = F_nd
        return self._X, self._F

    def contents(self) -> tuple[DecisionArray, FloatArray]:
        return self._X, self._F
