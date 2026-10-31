from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vamos.ux.analysis.mcdm import (
    MCDMResult,
    knee_point_scores,
    reference_point_scores,
    tchebycheff_scores,
    topsis_scores,
    weighted_sum_scores,
)
from vamos.ux.studio.data import FrontRecord, normalize_objectives

SolutionDecoder = Callable[[NDArray[Any]], dict[str, Any]]


def default_decoder(var: NDArray[Any]) -> dict[str, Any]:
    return {f"x{i}": float(val) for i, val in enumerate(var)}


def build_decoder(_problem_name: str, metadata: dict[str, Any] | None = None) -> SolutionDecoder:
    # Placeholder for richer problem-specific decoding; metadata can be extended later.
    return default_decoder


def _mcdm_results(
    F: NDArray[np.float64],
    *,
    weights: NDArray[np.float64] | None,
    reference_point: NDArray[np.float64] | None,
    methods: Sequence[str],
) -> dict[str, MCDMResult]:
    results: dict[str, MCDMResult] = {}
    if weights is None:
        weights = np.ones(F.shape[1]) / F.shape[1]
    for method in methods:
        if method == "weighted_sum":
            results[method] = weighted_sum_scores(F, weights)
        elif method in {"tchebycheff", "tchebychev"}:
            results[method] = tchebycheff_scores(F, weights, reference_point)
        elif method == "knee":
            try:
                results[method] = knee_point_scores(F)
            except ValueError:
                continue
        elif method == "reference":
            if reference_point is not None:
                results[method] = reference_point_scores(F, reference_point)
        elif method == "topsis":
            results[method] = topsis_scores(F, weights)
        else:
            continue
    return results


def compute_mcdm_scores(
    F: NDArray[np.float64],
    *,
    weights: NDArray[np.float64] | None = None,
    reference_point: NDArray[np.float64] | None = None,
    methods: Sequence[str] = ("weighted_sum", "tchebycheff", "knee"),
) -> dict[str, NDArray[np.float64]]:
    results = _mcdm_results(F, weights=weights, reference_point=reference_point, methods=methods)
    return {method: result.scores for method, result in results.items()}


@dataclass
class DecisionView:
    front: FrontRecord
    normalized_F: NDArray[np.float64]
    decoded_X: list[dict[str, Any]] | None
    constraints: NDArray[Any] | None
    mcdm_scores: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    # The index each method prefers, which can break a tie of its lowest score.
    mcdm_best: dict[str, int] = field(default_factory=dict)


def build_decision_view(
    front: FrontRecord,
    *,
    decoder: SolutionDecoder | None = None,
    weights: NDArray[np.float64] | None = None,
    reference_point: NDArray[np.float64] | None = None,
    methods: Sequence[str] = ("weighted_sum", "tchebycheff", "knee"),
) -> DecisionView:
    F = front.points_F
    norm_F = normalize_objectives(F)
    decoded = None
    if front.points_X is not None:
        dec = decoder or build_decoder(front.problem_name, front.extra.get("metadata", {}))
        decoded = [dec(x) for x in front.points_X]
    results = _mcdm_results(norm_F, weights=weights, reference_point=reference_point, methods=methods)
    return DecisionView(
        front=front,
        normalized_F=norm_F,
        decoded_X=decoded,
        constraints=front.constraints,
        mcdm_scores={method: result.scores for method, result in results.items()},
        mcdm_best={method: result.best_index for method, result in results.items()},
    )


def rank_by_score(view: DecisionView, method: str) -> NDArray[np.intp]:
    """Order the solutions by ``method``'s score, lowest first.

    The method's preferred solution comes first even when it ties with others,
    as the no-knee fallback of the knee point does on a linear front. Other ties
    keep the order of the front.
    """
    scores = view.mcdm_scores.get(method)
    if scores is None:
        raise ValueError(f"No scores for method '{method}'.")
    order = np.argsort(scores, kind="stable")
    best = view.mcdm_best.get(method)
    if best is not None:
        order = np.concatenate(([best], order[order != best])).astype(np.intp)
    return order


def feasible_indices(view: DecisionView, max_violation: float = 0.0) -> NDArray[np.intp]:
    if view.constraints is None:
        return np.arange(view.front.points_F.shape[0])
    violations = np.maximum(view.constraints, 0.0).sum(axis=1)
    return np.nonzero(violations <= max_violation)[0]


def filter_by_objective_ranges(view: DecisionView, ranges: Sequence[tuple[float | None, float | None]]) -> NDArray[np.intp]:
    F = view.front.points_F
    if len(ranges) != F.shape[1]:
        raise ValueError("ranges length must equal number of objectives.")
    mask = np.ones(F.shape[0], dtype=bool)
    for idx, (lo, hi) in enumerate(ranges):
        if lo is not None:
            mask &= F[:, idx] >= lo
        if hi is not None:
            mask &= F[:, idx] <= hi
    return np.nonzero(mask)[0]
