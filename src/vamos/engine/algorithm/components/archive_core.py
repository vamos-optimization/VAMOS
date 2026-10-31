from __future__ import annotations

from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

try:  # pragma: no cover - optional dependency
    import moocore as _moocore
except ImportError:  # pragma: no cover - optional dependency
    _moocore = None

from vamos.foundation.constraints.utils import compute_violation, is_feasible
from vamos.foundation.quality_indicators.pareto import pareto_filter

DeduplicateIn = Literal["objective", "decision", "both"]
FloatArray: TypeAlias = NDArray[np.float64]
DecisionArray: TypeAlias = NDArray[np.generic]
IndexArray: TypeAlias = NDArray[np.intp]
BoolArray: TypeAlias = NDArray[np.bool_]


def _nondominated_mask(F: FloatArray) -> BoolArray:
    if F.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if _moocore is not None:
        return np.asarray(_moocore.is_nondominated(F), dtype=bool)
    _, nd_idx = pareto_filter(F, return_indices=True)
    mask = np.zeros(F.shape[0], dtype=bool)
    mask[nd_idx] = True
    return mask


def _unique_rows_with_tolerance_bucket(values: FloatArray, tol: float) -> IndexArray:
    n = int(values.shape[0])
    if n <= 1 or tol < 0.0:
        return np.arange(n, dtype=int)
    if tol == 0.0:
        _, unique_idx = np.unique(values, axis=0, return_index=True)
        unique_idx.sort()
        return np.asarray(unique_idx, dtype=int)

    first = np.asarray(values[:, 0], dtype=float)
    order = np.argsort(first, kind="mergesort")
    if np.all(np.diff(first[order]) > tol):
        return np.arange(n, dtype=int)

    buckets: dict[int, list[int]] = {}
    keep_idx: list[int] = []
    for i in range(n):
        fi = first[i]
        if not np.isfinite(fi):
            keep_idx.append(i)
            continue

        bucket = int(np.floor(fi / tol))
        row = values[i]
        is_dup = False
        for b in (bucket - 1, bucket, bucket + 1):
            for k in buckets.get(b, ()):
                if np.all(np.abs(row - values[k]) <= tol):
                    is_dup = True
                    break
            if is_dup:
                break

        if not is_dup:
            keep_idx.append(i)
            buckets.setdefault(bucket, []).append(i)

    return np.asarray(keep_idx, dtype=int)


def _unique_rows_with_tolerance(values: FloatArray, tol: float) -> IndexArray:
    n = int(values.shape[0])
    if n <= 1 or tol < 0.0:
        return np.arange(n, dtype=int)
    if tol == 0.0:
        _, unique_idx = np.unique(values, axis=0, return_index=True)
        unique_idx.sort()
        return np.asarray(unique_idx, dtype=int)

    first = np.asarray(values[:, 0], dtype=float)
    order = np.argsort(first, kind="mergesort")
    if np.all(np.diff(first[order]) > tol):
        return np.arange(n, dtype=int)

    finite_rows = np.all(np.isfinite(values), axis=1)
    finite_idx = np.flatnonzero(finite_rows)
    if finite_idx.size <= 32:
        return _unique_rows_with_tolerance_bucket(values, tol)

    finite_values = np.ascontiguousarray(values[finite_idx], dtype=float)
    tree = cKDTree(finite_values)
    removed = np.zeros(finite_idx.shape[0], dtype=bool)
    local_lookup = np.full(n, -1, dtype=int)
    local_lookup[finite_idx] = np.arange(finite_idx.shape[0], dtype=int)

    keep_idx: list[int] = []
    for orig_idx in range(n):
        local_idx = int(local_lookup[orig_idx])
        if local_idx < 0:
            keep_idx.append(orig_idx)
            continue
        if removed[local_idx]:
            continue
        keep_idx.append(orig_idx)
        neighbors = tree.query_ball_point(finite_values[local_idx], r=tol, p=np.inf)
        for neighbor_local in neighbors:
            if finite_idx[neighbor_local] > orig_idx:
                removed[neighbor_local] = True

    return np.asarray(keep_idx, dtype=int)


def _subset_arrays(
    X: DecisionArray,
    F: FloatArray,
    G: FloatArray | None,
    idx: IndexArray,
) -> tuple[DecisionArray, FloatArray, FloatArray | None]:
    if idx.size == 0:
        return X[:0], F[:0], G[:0] if G is not None else None
    return X[idx], F[idx], G[idx] if G is not None else None


class _BaseArchive:
    """
    Base class for bounded external archives.

    Update is batch-based: merge existing + incoming, feasibility filter,
    non-dominated extraction, deduplication, then (if needed) truncation.

    Deduplication drops a row whose objective (or decision) vector is within the
    Chebyshev distance ``objective_tolerance`` (or ``decision_tolerance``) of a kept
    row. The default objective tolerance is 0, exact equality as in jMetal, so the
    archive does not depend on the scale of the objectives.
    """

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
    ) -> None:
        self.capacity = None if capacity is None else int(capacity)
        self.truncate_size: int | None
        if self.capacity is not None:
            if self.capacity <= 0:
                raise ValueError("archive capacity must be positive.")
            self.truncate_size = self.capacity if truncate_size is None else int(truncate_size)
            if self.truncate_size <= 0:
                raise ValueError("truncate_size must be positive.")
            if self.truncate_size > self.capacity:
                raise ValueError("truncate_size must be <= capacity.")
            storage_rows = max(self.capacity + 1, self.truncate_size + 1)
        else:
            if truncate_size is not None:
                raise ValueError("truncate_size requires a finite archive capacity.")
            self.truncate_size = None
            storage_rows = max(1, int(initial_capacity))

        if objective_tolerance < 0.0:
            raise ValueError("objective_tolerance must be >= 0.")
        if decision_tolerance < 0.0:
            raise ValueError("decision_tolerance must be >= 0.")
        if deduplicate_in not in {"objective", "decision", "both"}:
            raise ValueError("deduplicate_in must be 'objective', 'decision', or 'both'.")

        self._dtype = np.dtype(dtype)
        self._n_var = int(n_var)
        self._n_obj = int(n_obj)
        self._objective_tolerance = float(objective_tolerance)
        self._decision_tolerance = float(decision_tolerance)
        self._deduplicate_in = deduplicate_in
        self._n_con = int(n_con) if n_con is not None else None
        if self._n_con is not None and self._n_con <= 0:
            raise ValueError("n_con must be positive when provided.")

        self._X = np.empty((storage_rows, self._n_var), dtype=self._dtype)
        self._F = np.empty((storage_rows, self._n_obj), dtype=float)
        self._G = np.empty((storage_rows, self._n_con), dtype=float) if self._n_con is not None else None
        self._size = 0

    def update(
        self,
        population_X: DecisionArray,
        population_F: NDArray[np.generic],
        population_G: NDArray[np.generic] | None = None,
    ) -> tuple[DecisionArray, FloatArray]:
        pop_X = np.asarray(population_X, dtype=self._dtype, order="C")
        pop_F = np.asarray(population_F, dtype=float, order="C")
        if pop_X.ndim != 2:
            raise ValueError("population_X must be a 2D array.")
        if pop_F.ndim != 2:
            raise ValueError("population_F must be a 2D array.")
        if pop_X.shape[0] != pop_F.shape[0]:
            raise ValueError("population_X and population_F must have the same number of rows.")
        if pop_X.shape[1] != self._n_var:
            raise ValueError(f"population_X has {pop_X.shape[1]} columns, expected {self._n_var}.")
        if pop_F.shape[1] != self._n_obj:
            raise ValueError(f"population_F has {pop_F.shape[1]} columns, expected {self._n_obj}.")
        if pop_X.shape[0] == 0:
            return self._snapshot()

        pop_G: FloatArray | None = None
        if population_G is not None:
            pop_G = np.asarray(population_G, dtype=float, order="C")
            if pop_G.ndim != 2:
                raise ValueError("population_G must be a 2D array when provided.")
            if pop_G.shape[0] != pop_X.shape[0]:
                raise ValueError("population_G must have the same number of rows as population_X.")

        X_comb, F_comb, G_comb = self._merge_with_existing(pop_X, pop_F, pop_G)
        X_work, F_work, G_work = self._apply_feasibility_filter(X_comb, F_comb, G_comb)
        if F_work.shape[0] == 0:
            self._replace_contents(X_work, F_work, G_work)
            return self._snapshot()

        nd_mask = _nondominated_mask(F_work)
        X_nd = X_work[nd_mask]
        F_nd = F_work[nd_mask]
        G_nd = G_work[nd_mask] if G_work is not None else None
        X_nd, F_nd, G_nd = self._dedupe(X_nd, F_nd, G_nd)

        if self.capacity is not None and F_nd.shape[0] > self.capacity:
            assert self.truncate_size is not None
            keep_idx = np.asarray(self._select_subset(F_nd, self.truncate_size, G_nd), dtype=int)
            if keep_idx.ndim != 1:
                raise ValueError("_select_subset() must return a 1D index array.")
            if keep_idx.size != self.truncate_size:
                raise ValueError(f"_select_subset() returned {keep_idx.size} indices, expected {self.truncate_size}.")
            X_nd, F_nd, G_nd = _subset_arrays(X_nd, F_nd, G_nd, keep_idx)

        self._replace_contents(X_nd, F_nd, G_nd)
        return self._snapshot()

    def contents(self) -> tuple[DecisionArray, FloatArray]:
        return self._snapshot()

    def _snapshot(self) -> tuple[DecisionArray, FloatArray]:
        return self._X[: self._size].copy(), self._F[: self._size].copy()

    def _merge_with_existing(
        self,
        pop_X: DecisionArray,
        pop_F: FloatArray,
        pop_G: FloatArray | None,
    ) -> tuple[DecisionArray, FloatArray, FloatArray | None]:
        if self._size == 0:
            return pop_X, pop_F, pop_G

        total = self._size + pop_X.shape[0]
        X_comb = np.empty((total, self._n_var), dtype=self._dtype)
        F_comb = np.empty((total, self._n_obj), dtype=float)

        X_comb[: self._size] = self._X[: self._size]
        F_comb[: self._size] = self._F[: self._size]
        X_comb[self._size :] = pop_X
        F_comb[self._size :] = pop_F

        if self._G is None and pop_G is None:
            return X_comb, F_comb, None

        if pop_G is not None:
            n_con = pop_G.shape[1]
        else:
            assert self._G is not None
            n_con = self._G.shape[1]
        if self._G is not None and self._G.shape[1] != n_con:
            raise ValueError("Constraint column count mismatch with existing archive contents.")
        if pop_G is not None and pop_G.shape[1] != n_con:
            raise ValueError("population_G has an incompatible number of constraint columns.")

        G_comb = np.zeros((total, n_con), dtype=float)
        if self._G is not None and self._size > 0:
            G_comb[: self._size] = self._G[: self._size]
        if pop_G is not None:
            G_comb[self._size :] = pop_G
        return X_comb, F_comb, G_comb

    @staticmethod
    def _apply_feasibility_filter(
        X: DecisionArray,
        F: FloatArray,
        G: FloatArray | None,
    ) -> tuple[DecisionArray, FloatArray, FloatArray | None]:
        if G is None or G.shape[0] == 0:
            return X, F, G

        feas = is_feasible(G, n=G.shape[0])
        if feas.any():
            keep = feas
            return X[keep], F[keep], G[keep]

        cv = compute_violation(G, n=G.shape[0])
        min_cv = float(np.min(cv))
        keep = cv <= (min_cv + 1e-12)
        return X[keep], F[keep], G[keep]

    def _dedupe(
        self,
        X: DecisionArray,
        F: FloatArray,
        G: FloatArray | None,
    ) -> tuple[DecisionArray, FloatArray, FloatArray | None]:
        mode = self._deduplicate_in
        out_X, out_F, out_G = X, F, G

        if mode in {"objective", "both"}:
            keep_obj = _unique_rows_with_tolerance(out_F, self._objective_tolerance)
            out_X, out_F, out_G = _subset_arrays(out_X, out_F, out_G, keep_obj)

        if mode in {"decision", "both"}:
            keep_dec = _unique_rows_with_tolerance(np.asarray(out_X, dtype=float), self._decision_tolerance)
            out_X, out_F, out_G = _subset_arrays(out_X, out_F, out_G, keep_dec)

        return out_X, out_F, out_G

    def _replace_contents(
        self,
        X: DecisionArray,
        F: FloatArray,
        G: FloatArray | None,
    ) -> None:
        new_size = int(F.shape[0])
        self._ensure_storage(new_size)

        self._size = new_size
        if new_size:
            np.copyto(self._X[:new_size], X)
            np.copyto(self._F[:new_size], F)

        if G is None:
            self._G = None
            return

        self._ensure_constraint_storage(G.shape[1], new_size)
        assert self._G is not None
        if new_size:
            np.copyto(self._G[:new_size], G)

    def _ensure_storage(self, min_rows: int) -> None:
        if self._X.shape[0] >= min_rows:
            return
        new_rows = max(min_rows, int(self._X.shape[0] * 1.5) + 1)
        new_X = np.empty((new_rows, self._n_var), dtype=self._dtype)
        new_F = np.empty((new_rows, self._n_obj), dtype=float)
        if self._size:
            new_X[: self._size] = self._X[: self._size]
            new_F[: self._size] = self._F[: self._size]
        self._X = new_X
        self._F = new_F

        if self._G is not None:
            n_con = int(self._G.shape[1])
            new_G = np.empty((new_rows, n_con), dtype=float)
            if self._size:
                new_G[: self._size] = self._G[: self._size]
            self._G = new_G

    def _ensure_constraint_storage(self, n_con: int, min_rows: int) -> None:
        if n_con <= 0:
            raise ValueError("n_con must be positive.")
        if self._G is not None and self._G.shape[1] != n_con:
            raise ValueError("Constraint column count mismatch.")
        if self._G is None:
            rows = max(self._X.shape[0], min_rows)
            self._G = np.empty((rows, n_con), dtype=float)
            return
        if self._G.shape[0] >= min_rows:
            return
        new_rows = max(min_rows, int(self._G.shape[0] * 1.5) + 1)
        new_G = np.empty((new_rows, n_con), dtype=float)
        if self._size:
            new_G[: self._size] = self._G[: self._size]
        self._G = new_G

    def _select_subset(
        self,
        F: FloatArray,
        target_size: int,
        G: FloatArray | None = None,
    ) -> IndexArray:
        raise NotImplementedError


__all__ = ["DeduplicateIn", "_BaseArchive"]
