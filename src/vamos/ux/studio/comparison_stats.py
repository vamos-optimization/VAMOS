"""Panel-free statistics for comparing algorithms in VAMOS Studio.

Runs of different algorithms are independent, so per-problem comparisons use
the two-sided Mann-Whitney rank-sum test and the Vargha-Delaney A12 effect
size. Across problems, the median of each algorithm on each problem forms a
blocked design for the Friedman test, Demsar's average ranks, the Nemenyi
critical difference and Holm's post-hoc test against a pivot algorithm.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats  # type: ignore[import-untyped]

from vamos.engine.tuning.racing.stats import vargha_delaney_a12
from vamos.ux.analysis.stats import nemenyi_cd

Correction = Literal["holm", "bh", "none"]
CORRECTIONS: dict[str, str] = {"holm": "Holm", "bh": "Benjamini-Hochberg", "none": "No correction"}

# Vargha and Delaney (2000) thresholds on |A12 - 0.5|.
_A12_THRESHOLDS = ((0.06, "negligible"), (0.14, "small"), (0.21, "medium"))


def holm(p_values: ArrayLike) -> NDArray[np.float64]:
    """Holm's step-down adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    result = np.empty(m)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def benjamini_hochberg(p_values: ArrayLike) -> NDArray[np.float64]:
    """Benjamini-Hochberg false-discovery-rate adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    result = np.empty(m)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def adjust(p_values: ArrayLike, correction: str) -> NDArray[np.float64]:
    if correction == "holm":
        return holm(p_values)
    if correction == "bh":
        return benjamini_hochberg(p_values)
    return np.asarray(p_values, dtype=float)


def rank_sum_p(a: ArrayLike, b: ArrayLike) -> float:
    """Two-sided Mann-Whitney p-value; 1 when the samples cannot differ."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0 or np.ptp(np.concatenate([x, y])) == 0.0:
        return 1.0
    return float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)


def a12(a: ArrayLike, b: ArrayLike, *, higher_is_better: bool) -> float:
    """Probability that a run of ``a`` beats a run of ``b`` (ties count one half)."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(vargha_delaney_a12(x, y) if higher_is_better else vargha_delaney_a12(-x, -y))


def a12_magnitude(value: float) -> str:
    distance = abs(value - 0.5)
    for threshold, label in _A12_THRESHOLDS:
        if distance < threshold:
            return label
    return "large"


def verdict(p_adjusted: float, effect: float, alpha: float) -> str:
    """``+`` when the contender is significantly better, ``-`` when worse, ``=`` otherwise."""
    if p_adjusted >= alpha or effect == 0.5:
        return "="
    return "+" if effect > 0.5 else "-"


def average_ranks(matrix: NDArray[np.float64], *, higher_is_better: bool) -> NDArray[np.float64]:
    """Average rank of every column over the rows; rank 1 is best, ties share their mean rank."""
    data = np.asarray(matrix, dtype=float)
    oriented = -data if higher_is_better else data
    ranks = np.vstack([stats.rankdata(row) for row in oriented])
    return np.asarray(ranks.mean(axis=0), dtype=float)


def friedman_p(matrix: NDArray[np.float64]) -> tuple[float, float] | None:
    """Friedman statistic and p-value, or ``None`` with fewer than 3 algorithms or 2 problems."""
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2 or data.shape[1] < 3 or data.shape[0] < 2:
        return None
    if np.all(np.ptp(data, axis=1) == 0.0):
        return 0.0, 1.0
    result = stats.friedmanchisquare(*[data[:, j] for j in range(data.shape[1])])
    return float(result.statistic), float(result.pvalue)


def cd_groups(ranks: ArrayLike, cd: float) -> list[tuple[int, int]]:
    """Maximal groups, as index ranges into the sorted ranks, whose ranks differ by less than ``cd``."""
    ordered = sorted(float(rank) for rank in np.asarray(ranks, dtype=float).ravel())
    groups: list[tuple[int, int]] = []
    for start in range(len(ordered)):
        end = start
        while end + 1 < len(ordered) and ordered[end + 1] - ordered[start] < cd:
            end += 1
        if end > start and not any(s <= start and end <= e for s, e in groups):
            groups.append((start, end))
    return groups


def holm_against_pivot(ranks: ArrayLike, pivot: int, n: int) -> list[tuple[float, float, float]]:
    """``(z, p, Holm-adjusted p)`` of every algorithm's average rank against the pivot's."""
    r = np.asarray(ranks, dtype=float)
    k = r.size
    se = math.sqrt(k * (k + 1) / (6.0 * n))
    z = (r - r[pivot]) / se
    p = 2.0 * stats.norm.sf(np.abs(z))
    others = [index for index in range(k) if index != pivot]
    adjusted = np.ones(k)
    if others:
        adjusted[others] = holm(p[others])
    return [(float(z[i]), float(p[i]) if i != pivot else 1.0, float(adjusted[i])) for i in range(k)]


def attainment_surface_2d(fronts: Sequence[NDArray[np.float64]], level: int) -> NDArray[np.float64]:
    """Staircase of the points attained by at least ``level`` of the 2-objective runs (minimization)."""
    runs = [np.atleast_2d(np.asarray(front, dtype=float)) for front in fronts if np.size(front)]
    if not runs or not 1 <= level <= len(runs):
        return np.empty((0, 2))
    thresholds = np.unique(np.concatenate([front[:, 0] for front in runs]))
    best = np.full((len(runs), thresholds.size), np.inf)
    for row, front in enumerate(runs):
        order = np.argsort(front[:, 0])
        xs, running = front[order, 0], np.minimum.accumulate(front[order, 1])
        positions = np.searchsorted(xs, thresholds, side="right") - 1
        valid = positions >= 0
        best[row, valid] = running[positions[valid]]
    attained = np.sort(best, axis=0)[level - 1]
    with np.errstate(invalid="ignore"):  # inf - inf before the first attained threshold
        keep = np.isfinite(attained) & np.concatenate([[True], np.diff(attained) < 0])
    return np.column_stack([thresholds[keep], attained[keep]])


__all__ = [
    "CORRECTIONS",
    "Correction",
    "a12",
    "a12_magnitude",
    "adjust",
    "attainment_surface_2d",
    "average_ranks",
    "benjamini_hochberg",
    "cd_groups",
    "friedman_p",
    "holm",
    "holm_against_pivot",
    "nemenyi_cd",
    "rank_sum_p",
    "verdict",
]
