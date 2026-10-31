"""Ranking helpers for ``OptimizationResult``."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.components.results import feasible_result_indices
from vamos.foundation.exceptions import NoSolutionsError, ResultSelectionError
from vamos.foundation.quality_indicators.pareto import pareto_filter

BestMethod: TypeAlias = Literal["balanced_sum", "knee", "min_f1", "min_f2", "balanced"]
RankingSource: TypeAlias = Literal["auto", "result", "archive", "population"]
RankingMethod: TypeAlias = Literal[
    "balanced_sum",
    "knee",
    "min_f1",
    "min_f2",
    "balanced",
    "weighted_sum",
    "crowding",
    "farthest",
    "knn",
    "reference_directions",
    "kmeans",
    "angle",
    "hv_greedy",
]

_BEST_METHODS = {"balanced_sum", "knee", "min_f1", "min_f2", "balanced"}
_RANKING_METHODS = _BEST_METHODS | {
    "weighted_sum",
    "crowding",
    "farthest",
    "knn",
    "reference_directions",
    "kmeans",
    "angle",
    "hv_greedy",
}
_RANKING_SOURCES = {"auto", "result", "archive", "population"}
_KNEE_ALIAS_WARNING = (
    "method='knee' is an alias of 'balanced_sum' (the smallest sum of normalized objectives), not a knee point; "
    "it will be removed in VAMOS 2.0. Use method='balanced_sum' for the same selection, or "
    "vamos.ux.api.knee_point_scores for a geometric knee."
)


def normalize_best_method(method: BestMethod | str) -> BestMethod:
    key = str(method).strip().lower()
    if key not in _BEST_METHODS:
        raise ResultSelectionError(f"Unknown method '{method}'. Use: balanced_sum, knee, min_f1, min_f2, balanced.")
    return key  # type: ignore[return-value]


def normalize_ranking_method(method: RankingMethod | str) -> RankingMethod:
    key = str(method).strip().lower()
    if key not in _RANKING_METHODS:
        raise ResultSelectionError(
            "Unknown method. Use: balanced_sum, knee, min_f1, min_f2, balanced, weighted_sum, "
            "crowding, farthest, knn, reference_directions, kmeans, angle, hv_greedy"
        )
    return key  # type: ignore[return-value]


def warn_if_knee_alias(method: str) -> None:
    """Warn the caller of an ``OptimizationResult`` method that ``'knee'`` names the normalized-sum rule."""
    if str(method).strip().lower() == "knee":
        warnings.warn(_KNEE_ALIAS_WARNING, FutureWarning, stacklevel=3)


def normalize_ranking_source(source: RankingSource | str) -> RankingSource:
    key = str(source).strip().lower()
    if key not in _RANKING_SOURCES:
        raise ResultSelectionError("source must be one of: auto, result, archive, population")
    return key  # type: ignore[return-value]


def selection_candidates(F: NDArray[np.float64], G: object, *, feasible_only: bool) -> NDArray[np.int_]:
    """Rows of ``F`` that a selection helper ranks.

    With ``feasible_only`` and constraint values ``G`` row-aligned with ``F``, these
    are the feasible rows or, when none is feasible, the least violated ones, the
    rule algorithms use for their non-dominated results. Otherwise every row.
    """
    rows = np.arange(F.shape[0], dtype=int)
    if not feasible_only or G is None:
        return rows
    G_arr = np.asarray(G, dtype=float)
    if G_arr.ndim != 2 or G_arr.shape[0] != F.shape[0]:
        return rows
    return feasible_result_indices(F, G_arr, constraint_mode="feasibility")


def top_k(
    result: Any,
    *,
    k: int,
    source: RankingSource | str,
    method: RankingMethod | str,
    nondominated_only: bool,
    weights: NDArray[np.float64] | None,
    feasible_only: bool = True,
) -> dict[str, Any]:
    if k <= 0:
        raise ResultSelectionError("k must be a positive integer.")
    key = normalize_ranking_method(method)
    src = resolve_ranking_source(result, source)
    F_src, X_src, G_src = _extract_source(result, src)
    idx_src = selection_candidates(F_src, G_src, feasible_only=feasible_only)
    if nondominated_only:
        front = pareto_filter(F_src[idx_src], return_indices=True)
        if front is None:
            raise NoSolutionsError("No solutions available after Pareto filtering.")
        F_rank, local_idx = front
        idx_rank = idx_src[np.asarray(local_idx, dtype=int)]
        if F_rank.size == 0:
            raise NoSolutionsError("No solutions available after Pareto filtering.")
    else:
        F_rank = F_src[idx_src]
        idx_rank = idx_src

    if key in {"crowding", "farthest", "knn", "reference_directions", "kmeans", "angle", "hv_greedy"}:
        return _top_k_subset_method(F_src, X_src, F_rank, idx_rank, k=int(k), source=src, method=key)

    scores = _ranking_scores(F_rank, method=key, weights=weights)
    order = np.argsort(scores)[: min(int(k), scores.shape[0])]
    selected_idx = idx_rank[order]
    return {
        "X": X_src[selected_idx] if X_src is not None else None,
        "F": F_src[selected_idx],
        "indices": np.asarray(selected_idx, dtype=int),
        "scores": np.asarray(scores[order], dtype=float),
        "source": src,
        "method": key,
    }


def top_k_report(
    result: Any,
    *,
    k: int,
    source: RankingSource | str,
    method: RankingMethod | str,
    nondominated_only: bool,
    weights: NDArray[np.float64] | None,
    feasible_only: bool = True,
) -> list[dict[str, Any]]:
    top = top_k(
        result,
        k=k,
        source=source,
        method=method,
        nondominated_only=nondominated_only,
        weights=weights,
        feasible_only=feasible_only,
    )
    F = np.asarray(top["F"], dtype=float)
    X = top["X"]
    idx = np.asarray(top["indices"], dtype=int)
    scores = np.asarray(top["scores"], dtype=float)
    rows: list[dict[str, Any]] = []
    X_arr = None if X is None else np.asarray(X)
    for rank in range(F.shape[0]):
        row: dict[str, Any] = {
            "rank": rank + 1,
            "index": int(idx[rank]),
            "score": float(scores[rank]),
            "source": top["source"],
            "method": top["method"],
        }
        for j in range(F.shape[1]):
            row[f"f{j + 1}"] = float(F[rank, j])
        if X_arr is not None:
            for j in range(X_arr.shape[1]):
                value = X_arr[rank, j]
                row[f"x{j + 1}"] = value.item() if hasattr(value, "item") else value
        rows.append(row)
    return rows


def resolve_ranking_source(result: Any, source: RankingSource | str) -> RankingSource:
    """The concrete source: ``'auto'`` is the archive when the result has one, else the result."""
    src = normalize_ranking_source(source)
    if src != "auto":
        return src
    return "archive" if isinstance(result.data.get("archive"), Mapping) else "result"


def _extract_source(
    result: Any,
    src: RankingSource,
) -> tuple[NDArray[np.float64], NDArray[np.generic] | None, object]:
    if src == "result":
        F = result.F
        X = result.X
        G = result.data.get("G")
    else:
        payload = result.data.get("archive" if src == "archive" else "population")
        if not isinstance(payload, Mapping):
            raise ResultSelectionError(f"{src.title()} data is not available in this result.")
        F = payload.get("F")
        X = payload.get("X")
        G = payload.get("G")
    if F is None:
        raise NoSolutionsError(f"No objective values available for source='{src}'.")
    F_arr = np.asarray(F, dtype=float)
    if F_arr.ndim != 2 or F_arr.shape[0] == 0:
        raise NoSolutionsError(f"Empty or invalid objective array for source='{src}'.")
    return F_arr, None if X is None else np.asarray(X), G


def _normalize_front(F: NDArray[np.float64]) -> NDArray[np.float64]:
    normalized = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-12)
    return np.asarray(normalized, dtype=float)


def _ranking_scores(
    F: NDArray[np.float64],
    *,
    method: RankingMethod | str,
    weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    key = normalize_ranking_method(method)
    if key in {"balanced_sum", "knee"}:
        return np.asarray(_normalize_front(F).sum(axis=1), dtype=float)
    if key == "min_f1":
        return np.asarray(F[:, 0], dtype=float)
    if key == "min_f2":
        if F.shape[1] < 2:
            raise ResultSelectionError(f"'min_f2' requires at least 2 objectives, but this set has {F.shape[1]}.")
        return np.asarray(F[:, 1], dtype=float)
    if key == "balanced":
        return np.asarray(_normalize_front(F).max(axis=1), dtype=float)
    if key == "weighted_sum":
        F_norm = _normalize_front(F)
        if weights is None:
            w: NDArray[np.float64] = np.full(F.shape[1], 1.0 / F.shape[1], dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if w.ndim != 1 or w.shape[0] != F.shape[1]:
                raise ResultSelectionError(f"weights must be a 1D array with length {F.shape[1]}.")
            if np.any(w < 0.0):
                raise ResultSelectionError("weights must be non-negative.")
            total = float(w.sum())
            if total <= 0.0:
                raise ResultSelectionError("weights must sum to a positive value.")
            w = w / total
        return np.asarray(F_norm @ w, dtype=float)
    raise AssertionError(f"Unhandled ranking method '{key}'.")


def _top_k_subset_method(
    F_src: NDArray[np.float64],
    X_src: NDArray[np.generic] | None,
    F_rank: NDArray[np.float64],
    idx_rank: NDArray[np.int_],
    *,
    k: int,
    source: str,
    method: str,
) -> dict[str, Any]:
    from vamos.engine.algorithm.components.subset_selection import (
        _hv_contributions,
        _single_front_crowding,
        select_top_k_angle,
        select_top_k_crowding,
        select_top_k_farthest,
        select_top_k_hv_greedy,
        select_top_k_kmeans,
        select_top_k_knn,
        select_top_k_reference_directions,
    )
    from vamos.engine.algorithm.components.utils import normalize_objectives

    k_eff = min(int(k), F_rank.shape[0])
    if method == "crowding":
        selected_local = select_top_k_crowding(F_rank, k_eff)
        selected_scores = np.asarray(_single_front_crowding(F_rank)[selected_local], dtype=float)
    elif method in {"farthest", "knn"}:
        selector = select_top_k_farthest if method == "farthest" else select_top_k_knn
        selected_local = selector(F_rank, k_eff)
        selected_scores = _min_pairwise_distance(F_rank[selected_local])
    elif method in {"reference_directions", "kmeans"}:
        selector = select_top_k_reference_directions if method == "reference_directions" else select_top_k_kmeans
        selected_local = selector(F_rank, k_eff)
        selected_scores = np.linalg.norm(normalize_objectives(F_rank[selected_local]), axis=1)
    elif method == "angle":
        selected_local = select_top_k_angle(F_rank, k_eff)
        selected_scores = _minimum_angles(F_rank[selected_local])
    else:
        selected_local = select_top_k_hv_greedy(F_rank, k_eff)
        selected_scores = _hv_contributions(F_rank[selected_local], np.max(F_rank, axis=0) + 1.0)

    selected_idx = idx_rank[selected_local]
    return {
        "X": X_src[selected_idx] if X_src is not None else None,
        "F": F_src[selected_idx],
        "indices": np.asarray(selected_idx, dtype=int),
        "scores": np.asarray(selected_scores, dtype=float),
        "source": str(source).strip().lower(),
        "method": method,
    }


def _min_pairwise_distance(points: NDArray[np.float64]) -> NDArray[np.float64]:
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    return np.asarray(np.min(dists, axis=1), dtype=float)


def _minimum_angles(points: NDArray[np.float64]) -> NDArray[np.float64]:
    ideal = points.min(axis=0)
    vectors = points - ideal
    norms = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-30)
    unit = vectors / norms
    cos_sim = unit @ unit.T
    np.clip(cos_sim, -1.0, 1.0, out=cos_sim)
    angles = np.arccos(cos_sim)
    np.fill_diagonal(angles, np.inf)
    return np.asarray(np.min(angles, axis=1), dtype=float)


__all__ = [
    "BestMethod",
    "RankingMethod",
    "RankingSource",
    "normalize_best_method",
    "normalize_ranking_method",
    "normalize_ranking_source",
    "resolve_ranking_source",
    "selection_candidates",
    "top_k",
    "top_k_report",
    "warn_if_knee_alias",
]
