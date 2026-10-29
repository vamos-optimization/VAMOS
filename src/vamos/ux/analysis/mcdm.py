from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.quality_indicators.pareto import pareto_filter


@dataclass
class MCDMResult:
    scores: NDArray[np.float64]
    best_index: int
    best_point: NDArray[np.float64]


def _validate_front(F: NDArray[np.float64]) -> NDArray[np.float64]:
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] == 0 or F.shape[1] == 0:
        raise ValueError("F must be a 2D array with at least one point and one objective.")
    return F


def _validate_weights(weights: NDArray[np.float64], n_obj: int) -> NDArray[np.float64]:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.shape[0] != n_obj:
        raise ValueError("weights must be 1D with length equal to number of objectives.")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    if np.allclose(w.sum(), 0):
        raise ValueError("weights must not sum to zero.")
    return np.asarray(w / w.sum(), dtype=float)


def weighted_sum_scores(F: NDArray[np.float64], weights: NDArray[np.float64]) -> MCDMResult:
    F = _validate_front(F)
    w = _validate_weights(weights, F.shape[1])
    scores = F @ w
    best_idx = int(np.argmin(scores))
    return MCDMResult(scores=scores, best_index=best_idx, best_point=F[best_idx].copy())


def tchebycheff_scores(
    F: NDArray[np.float64],
    weights: NDArray[np.float64],
    reference: NDArray[np.float64] | None = None,
) -> MCDMResult:
    F = _validate_front(F)
    w = _validate_weights(weights, F.shape[1])
    if reference is None:
        reference = np.min(F, axis=0)
    reference = np.asarray(reference, dtype=float)
    if reference.shape != (F.shape[1],):
        raise ValueError("reference must have shape (n_obj,).")
    diff = np.abs(F - reference)
    scores = np.max(w * diff, axis=1)
    best_idx = int(np.argmin(scores))
    return MCDMResult(scores=scores, best_index=best_idx, best_point=F[best_idx].copy())


def reference_point_scores(F: NDArray[np.float64], reference: NDArray[np.float64]) -> MCDMResult:
    F = _validate_front(F)
    ref = np.asarray(reference, dtype=float)
    if ref.shape != (F.shape[1],):
        raise ValueError("reference must have shape (n_obj,).")
    diff = F - ref
    scores = np.linalg.norm(diff, axis=1)
    best_idx = int(np.argmin(scores))
    return MCDMResult(scores=scores, best_index=best_idx, best_point=F[best_idx].copy())


def knee_point_scores(F: NDArray[np.float64]) -> MCDMResult:
    """
    Score a two-objective front by its knee.

    The knee is the non-dominated point farthest from the chord between the two
    extreme points of the non-dominated set, on the side of the ideal point
    (Das, 1999; Deb and Gupta, 2011). Each non-dominated point scores minus its
    signed distance to that chord, positive towards the ideal point, so the knee
    has the lowest score. Dominated points score ``inf``.

    A concave or linear front has no point on the ideal side of the chord, and
    therefore no knee. The function then emits a ``UserWarning`` and the lowest
    score, and so ``best_index``, falls on an extreme point of the front.

    Distances are measured in the units of ``F``; normalize the objectives
    first when their scales differ.
    """
    F = _validate_front(F)
    if F.shape[1] != 2:
        raise ValueError("knee_point_scores currently supports only 2D fronts.")
    scores = np.full(F.shape[0], np.inf, dtype=float)
    _, front_idx = pareto_filter(F, return_indices=True)
    front = F[front_idx]
    order = np.argsort(front[:, 0], kind="stable")
    p_start = front[order[0]]
    chord = front[order[-1]] - p_start
    norm = float(np.linalg.norm(chord))
    if norm == 0.0:
        # One distinct non-dominated point: it is the only candidate.
        scores[front_idx] = 0.0
    else:
        # The chord runs from the smallest f1 to the smallest f2; this unit
        # normal points from it towards the ideal point.
        normal = np.array([chord[1], -chord[0]]) / norm
        signed = (front - p_start) @ normal
        signed[np.abs(signed) <= 1e-12 * norm] = 0.0
        scores[front_idx] = 0.0 - signed  # 0.0 - x avoids a -0.0 score at the extremes
        if not np.any(signed > 0.0):
            warnings.warn(
                "The front has no knee: no non-dominated point lies on the ideal side of the chord "
                "between its extreme points (concave or linear front).",
                UserWarning,
                stacklevel=2,
            )
    best_idx = int(np.argmin(scores))
    return MCDMResult(scores=scores, best_index=best_idx, best_point=F[best_idx].copy())


def topsis_scores(F: NDArray[np.float64], weights: NDArray[np.float64]) -> MCDMResult:
    """
    Calculate TOPSIS scores (closeness coefficient to ideal solution).
    Scores are relative closeness C_i = S_minus / (S_plus + S_minus).
    For MCDMResult compatibility where 'lower score is better' is often assumed,
    we return 1 - C_i as the score (so minimizing score maximizes closeness).
    BUT wait, existing methods return different things.
      - weighted_sum: weighted sum (minimize)
      - tchebycheff: max deviation (minimize)
      - reference_point: distance (minimize)
      - knee_point: negative distance (minimize)

    So consistently, we want to MINIMIZE the returned 'scores'.
    TOPSIS 'C' is in [0, 1], where 1 is best.
    So we will return scores = 1.0 - C.
    """
    F = _validate_front(F)
    w = _validate_weights(weights, F.shape[1])

    # 1. Vector Normalization
    norm = np.linalg.norm(F, axis=0)
    # Handle zero norm (constant objective 0)
    norm = np.where(norm == 0, 1.0, norm)
    F_norm = F / norm

    # 2. Weighted Normalized Decision Matrix
    V = F_norm * w

    # 3. Determine Ideal (Min) and Anti-Ideal (Max) for minimization problems
    # VAMOS assumes minimization convention for F
    A_best = np.min(V, axis=0)
    A_worst = np.max(V, axis=0)

    # 4. Calculate Separation Measures
    S_plus = np.linalg.norm(V - A_best, axis=1)
    S_minus = np.linalg.norm(V - A_worst, axis=1)

    # 5. Calculate Relative Closeness
    denom = S_plus + S_minus
    # Handle division by zero (if point coincides with both ideal and anti-ideal? Impossible unless F is single point)
    denom = np.where(denom == 0, 1e-10, denom)

    C = S_minus / denom

    # We want to maximize C, so we minimize (1 - C)
    scores = 1.0 - C
    best_idx = int(np.argmin(scores))

    return MCDMResult(scores=scores, best_index=best_idx, best_point=F[best_idx].copy())
