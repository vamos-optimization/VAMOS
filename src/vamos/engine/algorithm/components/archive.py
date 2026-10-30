from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.components.archive_core import DeduplicateIn, _BaseArchive
from vamos.engine.archive.pruning_selectors import select_maxmin_subset, select_ref_dirs_subset
from vamos.foundation.kernel.numpy_backend import _compute_crowding

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import SelectionKernels

try:  # pragma: no cover - optional dependency
    import moocore as _moocore
except ImportError:  # pragma: no cover - optional dependency
    _moocore = None

_HV_FALLBACK_WARNED = False


def _single_front_crowding(F: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    if F.shape[0] == 0:
        return np.empty(0, dtype=float)
    return _compute_crowding(F, [list(range(F.shape[0]))])


def _hv_contributions(F: np.ndarray[Any, Any], ref: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Compute HV contributions while preserving this module's MooCore test hook."""
    global _HV_FALLBACK_WARNED
    if F.shape[0] == 0:
        return np.empty(0, dtype=float)
    if _moocore is not None:
        return np.asarray(_moocore.hv_contributions(F, ref=ref), dtype=float)
    if not _HV_FALLBACK_WARNED:
        warnings.warn(
            "Hypervolume contributions requested but 'moocore' is not installed; falling back to crowding distance.",
            UserWarning,
            stacklevel=2,
        )
        _HV_FALLBACK_WARNED = True
    return _single_front_crowding(F)


class CrowdingDistanceArchive(_BaseArchive):
    """Bounded archive with NSGA-II-style iterative crowding truncation."""

    def __init__(
        self,
        capacity: int | None,
        n_var: int,
        n_obj: int,
        dtype: Any,
        *,
        truncate_size: int | None = None,
        objective_tolerance: float = 0.0,
        deduplicate_in: DeduplicateIn = "objective",
        decision_tolerance: float = 1e-32,
        n_con: int | None = None,
        initial_capacity: int = 256,
        selection_kernels: SelectionKernels | None = None,
    ) -> None:
        super().__init__(
            capacity,
            n_var,
            n_obj,
            dtype,
            truncate_size=truncate_size,
            objective_tolerance=objective_tolerance,
            deduplicate_in=deduplicate_in,
            decision_tolerance=decision_tolerance,
            n_con=n_con,
            initial_capacity=initial_capacity,
        )
        self.selection_kernels = selection_kernels

    def _select_subset(
        self,
        F: np.ndarray[Any, Any],
        target_size: int,
        G: np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        _ = G
        if self.selection_kernels is not None and F.shape[0] > target_size and bool(np.isfinite(F).all()):
            # Same crowding values and first-argmin removals as the loop below.
            return np.asarray(self.selection_kernels.iterative_crowding_truncation(F, target_size), dtype=int)
        keep: NDArray[np.int_] = np.arange(F.shape[0], dtype=int)
        while keep.size > target_size:
            crowd = _single_front_crowding(F[keep])
            worst_local = int(np.argmin(crowd))
            keep = np.delete(keep, worst_local)
        return keep


class HypervolumeArchive(_BaseArchive):
    """Bounded archive with iterative hypervolume-contribution truncation."""

    def __init__(
        self,
        capacity: int,
        n_var: int,
        n_obj: int,
        dtype: Any,
        *,
        ref_offset: float = 1.0,
        ref_point: np.ndarray[Any, Any] | list[float] | None = None,
        truncate_size: int | None = None,
        objective_tolerance: float = 0.0,
        deduplicate_in: DeduplicateIn = "objective",
        decision_tolerance: float = 1e-32,
        n_con: int | None = None,
    ) -> None:
        super().__init__(
            capacity,
            n_var,
            n_obj,
            dtype,
            truncate_size=truncate_size,
            objective_tolerance=objective_tolerance,
            deduplicate_in=deduplicate_in,
            decision_tolerance=decision_tolerance,
            n_con=n_con,
        )
        self._ref_offset = float(ref_offset)
        self._global_worst: np.ndarray[Any, Any] | None = None
        self._fixed_ref: np.ndarray[Any, Any] | None = None
        if ref_point is not None:
            ref = np.asarray(ref_point, dtype=float)
            if ref.ndim != 1 or ref.shape[0] != self._n_obj:
                raise ValueError(f"hv_ref_point must be 1D with length {self._n_obj}, got shape {ref.shape}.")
            self._fixed_ref = ref.copy()

    def _stable_ref(self, F: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        current_max = np.max(F, axis=0)
        if self._global_worst is None:
            self._global_worst = current_max.copy()
        else:
            np.maximum(self._global_worst, current_max, out=self._global_worst)
        return self._global_worst + self._ref_offset

    def _reference(self, F: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if self._fixed_ref is not None:
            return self._fixed_ref
        return self._stable_ref(F)

    def _select_subset(
        self,
        F: np.ndarray[Any, Any],
        target_size: int,
        G: np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        _ = G
        keep: NDArray[np.int_] = np.arange(F.shape[0], dtype=int)
        while keep.size > target_size:
            F_keep = F[keep]
            ref = self._reference(F_keep)
            contrib = _hv_contributions(F_keep, ref)
            worst_local = int(np.argmin(contrib))
            keep = np.delete(keep, worst_local)
        return keep


class SPEA2Archive(_BaseArchive):
    """
    Bounded archive with SPEA2-style truncation.

    Uses strength raw fitness for convergence ranking and distance-based
    truncation on the splitting front for diversity.
    """

    def __init__(
        self,
        capacity: int,
        n_var: int,
        n_obj: int,
        dtype: Any,
        *,
        truncate_size: int | None = None,
        objective_tolerance: float = 0.0,
        deduplicate_in: DeduplicateIn = "objective",
        decision_tolerance: float = 1e-32,
        n_con: int | None = None,
        constraint_mode: str = "feasibility",
    ) -> None:
        super().__init__(
            capacity,
            n_var,
            n_obj,
            dtype,
            truncate_size=truncate_size,
            objective_tolerance=objective_tolerance,
            deduplicate_in=deduplicate_in,
            decision_tolerance=decision_tolerance,
            n_con=n_con,
        )
        self._constraint_mode = str(constraint_mode or "none")

    def _select_subset(
        self,
        F: np.ndarray[Any, Any],
        target_size: int,
        G: np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        _ = G
        from vamos.engine.algorithm.spea2.helpers import truncate_by_distance

        n = int(F.shape[0])
        if n <= target_size:
            return np.arange(n, dtype=int)

        dist_matrix = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
        return truncate_by_distance(dist_matrix, target_size)


class MaxMinArchive(_BaseArchive):
    """Bounded archive with greedy max-min distance truncation."""

    def _select_subset(
        self,
        F: np.ndarray[Any, Any],
        target_size: int,
        G: np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        _ = G
        return select_maxmin_subset(F, target_size)


class ReferenceDirectionsArchive(_BaseArchive):
    """Bounded archive with NSGA-III-style reference-direction niching."""

    def __init__(
        self,
        capacity: int,
        n_var: int,
        n_obj: int,
        dtype: Any,
        *,
        rng_seed: int = 0,
        truncate_size: int | None = None,
        objective_tolerance: float = 0.0,
        deduplicate_in: DeduplicateIn = "objective",
        decision_tolerance: float = 1e-32,
        n_con: int | None = None,
    ) -> None:
        super().__init__(
            capacity,
            n_var,
            n_obj,
            dtype,
            truncate_size=truncate_size,
            objective_tolerance=objective_tolerance,
            deduplicate_in=deduplicate_in,
            decision_tolerance=decision_tolerance,
            n_con=n_con,
        )
        self._rng = np.random.default_rng(rng_seed)

    def _select_subset(
        self,
        F: np.ndarray[Any, Any],
        target_size: int,
        G: np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        _ = G
        return select_ref_dirs_subset(F, target_size, self._rng)


class UnboundedArchive(_BaseArchive):
    """Archive that keeps all non-dominated solutions without size limit."""

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        dtype: Any,
        *,
        objective_tolerance: float = 0.0,
        deduplicate_in: DeduplicateIn = "objective",
        decision_tolerance: float = 1e-32,
        n_con: int | None = None,
        initial_capacity: int = 256,
    ) -> None:
        super().__init__(
            None,
            n_var,
            n_obj,
            dtype,
            objective_tolerance=objective_tolerance,
            deduplicate_in=deduplicate_in,
            decision_tolerance=decision_tolerance,
            n_con=n_con,
            initial_capacity=initial_capacity,
        )


__all__ = [
    "HypervolumeArchive",
    "CrowdingDistanceArchive",
    "SPEA2Archive",
    "MaxMinArchive",
    "ReferenceDirectionsArchive",
    "UnboundedArchive",
    "_single_front_crowding",
    "_hv_contributions",
]
