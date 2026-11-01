from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, overload

import numpy as np

from ._numba_moead_steady_state import numba_moead_steady_state_kernels
from ._numba_ranking import compute_crowding_numba, fast_non_dominated_sort_ranks, nsga2_survival_split
from ._numba_selection import binary_tournament_winners_numba
from ._numba_steady_state import numba_steady_state_kernels
from ._numba_survival import numba_selection_kernels
from ._numba_variation import numpy_power_squares, polynomial_mutation_apply, verified_variation_kernels
from .backend import (
    DecisionArray,
    FloatArray,
    IndexArray,
    KernelBackend,
    NSGA2SteadyStateKernels,
    NumericArray,
    RealArray,
    SelectionKernels,
    VariationKernels,
)
from .moead_kernels import MOEADSteadyStateKernels
from .numba_ops import tournament_winners_numba
from .numpy_backend import NumPyKernel as _NumPyKernel
from .numpy_backend import _as_float, _as_numeric_keys, _sample_tournament_candidates, _sbx_crossover_flat


def _nsga2_survivors(F: FloatArray, pop_size: int) -> IndexArray:
    """NSGA-II elitist selection with the reference ordering of ``_select_nsga2``.

    Complete fronts are taken in rank order with ascending indices; the splitting
    front is ordered by ``np.argsort`` of its crowding distances, reversed, exactly
    like the NumPy reference. Crowding is computed only for that front because no
    other front's crowding affects the selection.
    """
    ranks = fast_non_dominated_sort_ranks(F)
    complete, splitting, crowding = nsga2_survival_split(F, ranks, pop_size)
    if splitting.size == 0:
        return np.asarray(complete, dtype=np.int_)
    order = np.argsort(crowding)[::-1]
    remaining = pop_size - complete.shape[0]
    return np.asarray(np.concatenate((complete, splitting[order[:remaining]])), dtype=np.int_)


class NumbaKernel(KernelBackend):
    """Numba-accelerated kernel backend for ranking, selection, survival, and mutation."""

    name = "numba"

    def __init__(self) -> None:
        self._numpy_ops = _NumPyKernel()
        self._X_buffer: DecisionArray | None = None
        self._F_buffer: FloatArray | None = None
        self._mutation_mask: FloatArray | None = None
        self._mutation_delta: FloatArray | None = None

    def _ensure_buffers(self, total: int, n_var: int, n_obj: int, dtype: Any) -> None:
        dtype = np.dtype(dtype)
        if self._X_buffer is None or self._X_buffer.shape[0] < total or self._X_buffer.shape[1] != n_var or self._X_buffer.dtype != dtype:
            self._X_buffer = np.empty((total, n_var), dtype=dtype, order="C")
        if self._F_buffer is None or self._F_buffer.shape[0] < total or self._F_buffer.shape[1] != n_obj:
            self._F_buffer = np.empty((total, n_obj), dtype=np.float64, order="C")

    def _combine(
        self,
        X: DecisionArray,
        F: RealArray,
        X_off: DecisionArray,
        F_off: RealArray,
    ) -> tuple[DecisionArray, FloatArray]:
        total = X.shape[0] + X_off.shape[0]
        self._ensure_buffers(total, X.shape[1], F.shape[1], X.dtype)
        assert self._X_buffer is not None
        assert self._F_buffer is not None
        X_view = self._X_buffer[:total]
        F_view = self._F_buffer[:total]
        split = X.shape[0]
        np.copyto(X_view[:split], X)
        np.copyto(F_view[:split], F, casting="unsafe")
        np.copyto(X_view[split:], X_off)
        np.copyto(F_view[split:], F_off, casting="unsafe")
        return X_view, F_view

    def _ensure_mutation_buffers(self, n_ind: int, n_var: int) -> None:
        if self._mutation_mask is None or self._mutation_mask.shape[0] < n_ind or self._mutation_mask.shape[1] != n_var:
            self._mutation_mask = np.empty((n_ind, n_var), dtype=np.float64, order="C")
        if self._mutation_delta is None or self._mutation_delta.shape[0] < n_ind or self._mutation_delta.shape[1] != n_var:
            self._mutation_delta = np.empty((n_ind, n_var), dtype=np.float64, order="C")

    def _rank_and_crowding(self, F: RealArray) -> tuple[IndexArray, FloatArray]:
        F_arr = np.asarray(F, dtype=np.float64, order="C")
        if not np.isfinite(F_arr).all():
            ranks, crowding = self._numpy_ops.nsga2_ranking(F_arr)
            return ranks.astype(np.int64, copy=False), crowding
        ranks = fast_non_dominated_sort_ranks(F_arr)
        crowding = compute_crowding_numba(F_arr, ranks)
        return ranks.astype(np.int64, copy=False), crowding

    def capabilities(self) -> Iterable[str]:
        return ("numba",)

    def quality_indicators(self) -> Iterable[str]:
        return ("hypervolume",)

    def variation_kernels(self) -> VariationKernels | None:
        return verified_variation_kernels()

    def selection_kernels(self) -> SelectionKernels:
        return numba_selection_kernels()

    def nsga2_steady_state_kernels(self) -> NSGA2SteadyStateKernels | None:
        if self.variation_kernels() is None:
            # The step embeds compiled SBX and polynomial mutation: keep the reference path.
            return None
        return numba_steady_state_kernels()

    def moead_steady_state_kernels(self) -> MOEADSteadyStateKernels | None:
        if self.variation_kernels() is None:
            # The step embeds compiled DE, SBX and polynomial mutation: keep the reference path.
            return None
        return numba_moead_steady_state_kernels()

    def nsga2_ranking(self, F: RealArray) -> tuple[IndexArray, FloatArray]:
        return self._rank_and_crowding(F)

    def tournament_selection(
        self,
        ranks: NumericArray,
        crowding: RealArray,
        pressure: int,
        rng: np.random.Generator,
        n_parents: int,
    ) -> IndexArray:
        n_candidates = ranks.shape[0]
        if pressure <= 0:
            raise ValueError("pressure must be a positive integer")
        if n_parents <= 0 or n_candidates == 0:
            return np.empty(0, dtype=int)
        if pressure > n_candidates:
            raise ValueError("pressure cannot exceed population size for tournament selection without replacement")

        if pressure == 1:
            return rng.integers(0, n_candidates, size=n_parents, dtype=int)

        if pressure == 2:
            # Same draws as NumPyKernel: two index vectors, then one coin per full tie.
            first = rng.integers(0, n_candidates, size=n_parents, dtype=int)
            second = rng.integers(0, n_candidates - 1, size=n_parents, dtype=int)
            second = second + (second >= first)
            winners = binary_tournament_winners_numba(
                _as_numeric_keys(ranks),
                np.asarray(crowding, dtype=np.float64),
                first,
                second,
            )
            ties = winners < 0
            if np.any(ties):
                tie_coin = rng.integers(0, 2, size=int(np.count_nonzero(ties)), dtype=int).astype(bool)
                winners[ties] = np.where(tie_coin, first[ties], second[ties])
            return winners.astype(int, copy=False)

        candidates = _sample_tournament_candidates(
            rng,
            n_candidates=n_candidates,
            n_parents=n_parents,
            pressure=pressure,
        )
        tie_break_keys = rng.random(candidates.shape)
        return tournament_winners_numba(
            _as_numeric_keys(ranks),
            np.asarray(crowding, dtype=np.float64),
            np.asarray(candidates, dtype=np.int64),
            np.asarray(tie_break_keys, dtype=np.float64),
        )

    def sbx_crossover(
        self,
        X_parents: RealArray,
        params: Mapping[str, object],
        rng: np.random.Generator | None,
        xl: float,
        xu: float,
    ) -> FloatArray:
        if rng is None:
            rng = np.random.default_rng()
        return _sbx_crossover_flat(X_parents, params, rng, xl, xu, kernels=self.variation_kernels())

    def polynomial_mutation(
        self,
        X: RealArray,
        params: Mapping[str, object],
        rng: np.random.Generator | None,
        xl: float,
        xu: float,
    ) -> None:
        if X.size == 0:
            return
        if rng is None:
            rng = np.random.default_rng()
        if self.variation_kernels() is None:
            # Compiled pow differs from np.power on this platform: keep the exact reference.
            self._numpy_ops.polynomial_mutation(X, params, rng, xl, xu)
            return
        n_ind, n_var = X.shape
        lower, upper = self._numpy_ops._normalize_bounds(xl, xu, n_var)
        self._ensure_mutation_buffers(n_ind, n_var)
        assert self._mutation_mask is not None
        assert self._mutation_delta is not None
        rnd_mask = self._mutation_mask[:n_ind, :n_var]
        rnd_delta = self._mutation_delta[:n_ind, :n_var]
        rng.random(out=rnd_mask)
        rng.random(out=rnd_delta)
        polynomial_mutation_apply(
            X,
            _as_float(params.get("prob"), 0.1),
            _as_float(params.get("eta"), 20.0),
            lower,
            upper,
            rnd_mask,
            rnd_delta,
            numpy_power_squares(),
        )

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
        X_comb, F_comb = self._combine(X, F, X_off, F_off)
        if not np.isfinite(F_comb).all():
            if return_indices:
                return self._numpy_ops.nsga2_survival(X, F, X_off, F_off, pop_size, return_indices=True)
            return self._numpy_ops.nsga2_survival(X, F, X_off, F_off, pop_size, return_indices=False)
        sel = _nsga2_survivors(F_comb, pop_size)
        # Fancy indexing returns arrays owned by the caller; the combine buffers stay private.
        X_sel = X_comb[sel]
        F_sel = F_comb[sel]
        if return_indices:
            return X_sel, F_sel, sel
        return X_sel, F_sel

    def hypervolume(self, points: FloatArray, reference_point: FloatArray) -> float:
        from vamos.foundation.quality_indicators.hypervolume import hypervolume as hv_fn

        return hv_fn(points, reference_point)
