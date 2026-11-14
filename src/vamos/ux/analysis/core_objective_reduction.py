from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class ObjectiveReductionConfig:
    """Configuration for objective reduction."""

    method: str = "correlation"
    target_dim: int | None = None
    corr_threshold: float = 0.99
    angle_threshold_deg: float = 5.0
    keep_mandatory: tuple[int, ...] = ()


class ObjectiveReducer:
    """Learns a subset of objective indices using correlation or angular diversity."""

    def __init__(self, config: ObjectiveReductionConfig):
        self.config = config
        self.selected_indices_: NDArray[np.int_] | None = None
        self.removed_indices_: NDArray[np.int_] | None = None
        self._n_obj_fit: int | None = None

    def fit(self, F: NDArray[Any]) -> ObjectiveReducer:
        F = self._validate_input(F)
        n_obj = F.shape[1]
        self._n_obj_fit = n_obj
        if n_obj <= 1:
            self.selected_indices_ = np.arange(n_obj, dtype=int)
            self.removed_indices_ = np.array([], dtype=int)
            return self

        method = self.config.method.lower()
        if method == "correlation":
            selected = self._fit_correlation(F)
        elif method == "angle":
            selected = self._fit_angle(F)
        elif method == "hybrid":
            selected_corr = self._fit_correlation(F)
            selected = self._fit_angle(F, initial_indices=selected_corr)
        else:
            raise ValueError(f"Unsupported objective reduction method '{self.config.method}'.")

        all_idx = set(range(n_obj))
        selected_set = set(selected)
        self.selected_indices_ = np.array(sorted(selected_set), dtype=int)
        self.removed_indices_ = np.array(sorted(all_idx - selected_set), dtype=int)
        return self

    def transform(self, F: NDArray[Any]) -> NDArray[Any]:
        if self.selected_indices_ is None:
            raise RuntimeError("ObjectiveReducer.transform() called before fit().")
        F = np.asarray(F)
        if F.ndim != 2:
            raise ValueError("Input F must be 2-dimensional.")
        if self._n_obj_fit is not None and F.shape[1] < self._n_obj_fit:
            raise ValueError("Input F has fewer objectives than were seen during fit.")
        return F[:, self.selected_indices_]

    def fit_transform(self, F: NDArray[Any]) -> NDArray[Any]:
        return self.fit(F).transform(F)

    def _validate_input(self, F: NDArray[Any]) -> NDArray[np.float64]:
        F = np.asarray(F, dtype=float)
        if F.ndim != 2:
            raise ValueError("F must be a 2-dimensional array.")
        if F.shape[0] == 0 or F.shape[1] == 0:
            raise ValueError("F must have at least one sample and one objective.")
        return F

    def _fit_correlation(self, F: NDArray[np.float64]) -> list[int]:
        cfg = self.config
        keep_mand = set(cfg.keep_mandatory)
        n_obj = F.shape[1]
        stds = np.std(F, axis=0)
        epsilon = 1e-12
        remaining: list[int] = []
        removed: set[int] = set()
        for idx in range(n_obj):
            if stds[idx] < epsilon and idx not in keep_mand:
                removed.add(idx)
            else:
                remaining.append(idx)

        if len(remaining) <= 1:
            return remaining

        remaining = self._correlation_greedy(F, remaining, stds, keep_mand, cfg.corr_threshold)
        if cfg.target_dim is not None and len(remaining) > cfg.target_dim:
            remaining = self._truncate_to_target_dim(F, remaining, keep_mand, cfg.target_dim)
        return remaining

    def _correlation_greedy(
        self,
        F: NDArray[np.float64],
        indices: list[int],
        stds: NDArray[np.float64],
        keep_mand: set[int],
        threshold: float,
    ) -> list[int]:
        remaining = list(indices)
        while True:
            sub = F[:, remaining]
            centered = sub - sub.mean(axis=0, keepdims=True)
            denom = np.std(centered, axis=0, ddof=0)
            denom[denom == 0] = 1.0
            corr = (centered.T @ centered) / (centered.shape[0] * denom[np.newaxis, :] * denom[:, np.newaxis])
            np.clip(corr, -1.0, 1.0, out=corr)
            np.fill_diagonal(corr, 0.0)
            # Consider positive correlation only; negative correlation is treated as diverse.
            max_idx = np.unravel_index(np.argmax(corr), corr.shape)
            max_val = corr[max_idx]
            if max_val < threshold:
                break
            i_local, j_local = max_idx
            i_global = remaining[i_local]
            j_global = remaining[j_local]
            if i_global in keep_mand and j_global in keep_mand:
                break
            drop = self._choose_drop(i_global, j_global, stds, keep_mand)
            remaining.remove(drop)
            if len(remaining) <= 1:
                break
        return remaining

    def _choose_drop(self, i: int, j: int, stds: NDArray[np.float64], keep_mand: set[int]) -> int:
        if i in keep_mand:
            return j
        if j in keep_mand:
            return i
        var_i = stds[i] ** 2
        var_j = stds[j] ** 2
        if var_i > var_j:
            return j
        if var_j > var_i:
            return i
        return max(i, j)

    def _truncate_to_target_dim(self, F: NDArray[np.float64], remaining: list[int], keep_mand: set[int], target_dim: int) -> list[int]:
        if target_dim >= len(remaining):
            return remaining
        mandatory = [idx for idx in remaining if idx in keep_mand]
        if len(mandatory) >= target_dim:
            return sorted(mandatory)
        capacity = target_dim - len(mandatory)
        # Variance on each objective's own range, so that a change of units does not change the choice.
        spans = np.ptp(F, axis=0)
        variances = np.var(F, axis=0) / np.where(spans > 0, spans, 1.0) ** 2
        candidates = [(idx, variances[idx]) for idx in remaining if idx not in keep_mand]
        candidates.sort(key=lambda pair: (-pair[1], pair[0]))
        selected = mandatory + [idx for idx, _ in candidates[:capacity]]
        return sorted(selected)

    def _fit_angle(self, F: NDArray[np.float64], initial_indices: Iterable[int] | None = None) -> list[int]:
        cfg = self.config
        keep_mand = set(cfg.keep_mandatory)
        indices = list(initial_indices) if initial_indices is not None else list(range(F.shape[1]))
        variances = np.var(F, axis=0)
        remaining: list[int] = []
        zero_norm: list[int] = []
        normalized_vectors: dict[int, NDArray[np.float64]] = {}
        for idx in indices:
            vec = F[:, idx] - F[:, idx].mean()
            norm = np.linalg.norm(vec)
            if norm == 0.0:
                zero_norm.append(idx)
                normalized_vectors[idx] = np.zeros_like(vec)
            else:
                normalized_vectors[idx] = vec / norm
            if norm == 0.0 and idx not in keep_mand:
                continue
            remaining.append(idx)
        if not remaining:
            return list(keep_mand) if keep_mand else indices

        selected: list[int] = []
        if keep_mand:
            selected.extend(sorted(set(remaining) & keep_mand))
        if not selected:
            first = max(remaining, key=lambda idx: variances[idx])
            selected.append(first)

        while True:
            if cfg.target_dim is not None and len(selected) >= cfg.target_dim:
                break
            candidates = [idx for idx in remaining if idx not in selected]
            if not candidates:
                break
            best_idx = None
            best_angle = -np.inf
            for cand in candidates:
                min_angle = self._min_angle_to_set(cand, selected, normalized_vectors)
                if min_angle > best_angle:
                    best_angle = min_angle
                    best_idx = cand
            if best_idx is None:
                break
            if best_angle < cfg.angle_threshold_deg:
                break
            selected.append(best_idx)
        return sorted(selected)

    def _min_angle_to_set(self, cand: int, selected: list[int], normalized_vectors: dict[int, NDArray[np.float64]]) -> float:
        angles = []
        v_cand = normalized_vectors[cand]
        for sel in selected:
            v_sel = normalized_vectors[sel]
            cos_val = float(np.dot(v_cand, v_sel))
            cos_val = max(-1.0, min(1.0, cos_val))
            angle = np.degrees(np.arccos(cos_val))
            angles.append(angle)
        return float(np.min(angles)) if angles else 180.0


def reduce_objectives(
    F: NDArray[Any],
    method: str = "correlation",
    target_dim: int | None = None,
    corr_threshold: float = 0.99,
    angle_threshold_deg: float = 5.0,
    keep_mandatory: tuple[int, ...] = (),
) -> tuple[NDArray[Any], NDArray[np.int_]]:
    """
    Convenience wrapper for ObjectiveReducer.

    With ``target_dim``, the correlation method keeps the ``keep_mandatory`` objectives and then
    those with the largest variance relative to their range, so the choice does not depend on units.

    Returns:
        F_reduced: objective matrix with selected objectives.
        selected_indices: indices kept.
    """
    cfg = ObjectiveReductionConfig(
        method=method,
        target_dim=target_dim,
        corr_threshold=corr_threshold,
        angle_threshold_deg=angle_threshold_deg,
        keep_mandatory=keep_mandatory,
    )
    reducer = ObjectiveReducer(cfg)
    F_reduced = reducer.fit_transform(F)
    if reducer.selected_indices_ is None:
        raise RuntimeError("ObjectiveReducer did not set selected indices after fit.")
    return F_reduced, reducer.selected_indices_
