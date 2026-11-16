from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover
    from .state import ConfigState

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def build_score_matrix(
    configs: Sequence[ConfigState],
) -> tuple[FloatArray, list[int]]:
    """
    Build a score matrix from the given configs, trimmed to the minimum history
    length across alive configs.

    Returns:
        scores: np.ndarray of shape (n_configs, n_blocks)
            Each row corresponds to one config, each column to one "block"
            (instance x seed combination). We assume that scores are aligned
            in time: the t-th element of each config's scores corresponds to
            the same block.
        alive_indices: list[int]
            Indices in the original `configs` sequence corresponding to the
            rows in the returned score matrix (i.e., only alive configs with
            at least one score).
    """
    rows: list[FloatArray] = []
    alive_indices: list[int] = []

    for idx, state in enumerate(configs):
        if not state.alive or not state.scores:
            continue
        rows.append(np.asarray(state.scores, dtype=float))
        alive_indices.append(idx)

    if not rows:
        return np.empty((0, 0), dtype=float), []

    min_len = min(row.shape[0] for row in rows)
    if min_len == 0:
        return np.empty((0, 0), dtype=float), []

    trimmed_rows = [row[:min_len] for row in rows]
    scores = np.vstack(trimmed_rows)

    return scores, alive_indices


def _z_critical(alpha: float) -> float:
    """
    Return the two-sided critical z-value for the given significance level alpha.

    If scipy is available, use scipy.stats.norm.ppf. Otherwise, support a small
    set of typical alpha values with precomputed constants.
    """
    try:
        from scipy.stats import norm  # type: ignore[import-untyped]
    except Exception:
        if abs(alpha - 0.05) < 1e-8:
            return 1.96
        if abs(alpha - 0.10) < 1e-8:
            return 1.64
        if abs(alpha - 0.01) < 1e-8:
            return 2.58
        return 1.96

    return float(norm.ppf(1.0 - alpha / 2.0))


def _t_critical(alpha: float, df: int) -> float:
    """
    Two-sided critical t-value for the given alpha and degrees of freedom.

    Falls back to z critical when scipy is unavailable.
    """
    try:
        from scipy.stats import t
    except Exception:
        return _z_critical(alpha)

    return float(t.ppf(1.0 - alpha / 2.0, df))


def _get_p_value(t_stat: float, df: int) -> float:
    """
    Two-sided p-value for a given t-statistic and degrees of freedom.
    """
    try:
        from scipy.stats import t

        # Survival function (1 - cdf) for the absolute t-stat * 2 for two-sided
        return float(t.sf(abs(t_stat), df) * 2)
    except Exception:
        # Fallback to Normal distribution if scipy is missing
        from math import erf, sqrt

        # z-test p-value
        # 1 - erf(z / sqrt(2)) is 2 * (1 - Phi(z)) for one-sided?
        # standard normal cdf Phi(z) = 0.5 * (1 + erf(z/sqrt(2)))
        # p = 2 * (1 - Phi(|z|)) = 2 * (1 - 0.5 * (1 + erf(|z|/sqrt(2))))
        #   = 2 * 0.5 * (1 - erf(|z|/sqrt(2))) = 1 - erf(|z|/sqrt(2))
        return 1.0 - erf(abs(t_stat) / sqrt(2.0))


def select_configs_by_paired_test(
    scores: FloatArray,
    maximize: bool,
    alpha: float,
    *,
    aggregator: Callable[[list[float]], float] | None = None,
) -> BoolArray:
    """
    Perform pairwise t-tests against the best configuration with Holm-Bonferroni correction.

    Returns:
        keep: boolean array where True means the config is kept (not eliminated).
    """
    n_configs, n_blocks = scores.shape
    keep = np.ones(n_configs, dtype=bool)

    if n_configs <= 1 or n_blocks <= 1:
        return keep

    if aggregator is None:
        agg_scores = scores.mean(axis=1)
    else:
        agg_scores = np.asarray([float(aggregator(row.tolist())) for row in scores], dtype=float)

    best_idx = int(np.argmax(agg_scores)) if maximize else int(np.argmin(agg_scores))
    best_scores = scores[best_idx, :]

    # Store tuples of (p_value, config_index)
    comparisons = []

    for i in range(n_configs):
        if i == best_idx:
            continue

        cfg_scores = scores[i, :]
        diffs = best_scores - cfg_scores if maximize else cfg_scores - best_scores

        mean_diff = float(diffs.mean())

        # If mean difference is negative, the candidate is actually *better* (or equal)
        # than our chosen 'best' in this sample (could happen if aggregator != mean).
        # In that case, we definitely keep it.
        # Even if aggregator == mean, float precision might cause slight mismatches,
        # but generally mean_diff >= 0 if i != best_idx for mean aggregator.
        # If mean_diff <= 0, we treat it as "not significantly worse".
        if mean_diff <= 0:
            continue

        sd_diff = float(diffs.std(ddof=1)) if n_blocks > 1 else 0.0

        if sd_diff <= 1e-12:
            # Deterministically worse
            # p-value ~ 0
            comparisons.append((0.0, i))
            continue

        t_stat = mean_diff / (sd_diff / math.sqrt(n_blocks))
        p_val = _get_p_value(t_stat, df=n_blocks - 1)
        comparisons.append((p_val, i))

    if not comparisons:
        return keep

    # Holm-Bonferroni Step-Down Procedure
    # 1. Sort p-values from smallest to largest
    comparisons.sort(key=lambda x: x[0])

    m = len(comparisons)  # Total number of hypotheses (candidates vs best)

    for k, (p_val, idx) in enumerate(comparisons):
        # Rank k is 0-based here, so it corresponds to k=1..m in literature
        # Adjusted alpha = alpha / (m - k)
        adj_alpha = alpha / (m - k)

        if p_val < adj_alpha:
            # Reject H0 -> Significantly worse -> Eliminate
            keep[idx] = False
        else:
            # Fail to reject H0 -> Stop eliminating
            # Since p-values are sorted, all subsequent p-values are larger
            # and denominators (m-k) are smaller, so they will also pass.
            break

    return keep


def _wilcoxon_signed_rank(differences: FloatArray) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test, as R's ``wilcox.test(exact=FALSE)``.

    Zero differences are dropped; the normal approximation corrects for ties and continuity.
    Returns the two-sided p-value and the signed-rank statistic of the remaining differences: the
    sum of the ranks of the positive ones minus the sum of the ranks of the negative ones. Its sign
    says which way the test leans, and it is zero when the two sums balance.
    """
    from scipy.stats import norm, rankdata

    d = differences[differences != 0.0]
    n = int(d.size)
    if n == 0:
        return 1.0, 0.0
    ranks = rankdata(np.abs(d))
    signed_rank = float(ranks[d > 0.0].sum()) - float(ranks[d < 0.0].sum())
    _, counts = np.unique(ranks, return_counts=True)
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - float(np.sum(counts**3 - counts)) / 48.0
    if variance <= 0.0:
        return 1.0, signed_rank
    # Half the statistic is the sum of the positive ranks minus its mean, n * (n + 1) / 4.
    centred = 0.5 * signed_rank
    z = (centred - 0.5 * math.copysign(1.0, centred)) / math.sqrt(variance) if centred else 0.0
    return float(min(1.0, 2.0 * norm.sf(abs(z)))), signed_rank


def select_configs_by_frace(scores: FloatArray, maximize: bool, alpha: float) -> BoolArray:
    """Configurations that survive one F-race test (Birattari et al., 2002), as irace runs it.

    ``scores`` has one row per configuration and one column per block (instance and seed). Each
    block ranks the configurations, with average ranks for ties, so blocks on different scales
    weigh the same. With two configurations, a paired Wilcoxon signed-rank test decides: when its
    p-value is below ``alpha``, it eliminates the configuration that the signed ranks of the nonzero
    differences find worse, and blocks where the two tie count for neither. With more,
    a Friedman test on the ranks comes first; when its p-value is below ``alpha``, Conover's
    post-hoc comparison keeps the configuration with the best rank sum and every configuration
    whose rank sum is within the critical difference of it (Conover, *Practical Nonparametric
    Statistics*, 1999, pp. 369-371; irace's ``aux.friedman``).

    Returns a boolean mask that is true for the configurations to keep.
    """
    from scipy.stats import chi2, rankdata, t

    n_configs, n_blocks = scores.shape
    keep = np.ones(n_configs, dtype=bool)
    if n_configs <= 1 or n_blocks <= 1:
        return keep
    # Lower is better from here on.
    oriented = -scores if maximize else scores
    if n_configs == 2:
        p_value, signed_rank = _wilcoxon_signed_rank(oriented[0] - oriented[1])
        # A positive statistic means the first configuration scored higher, that is worse.
        if p_value < alpha and signed_rank != 0.0:
            keep[0 if signed_rank > 0.0 else 1] = False
        return keep

    ranks = np.vstack([rankdata(oriented[:, block]) for block in range(n_blocks)])  # blocks x configurations
    rank_sums = ranks.sum(axis=0)
    ties = sum(float(np.sum(counts**3 - counts)) for counts in (np.unique(row, return_counts=True)[1] for row in ranks))
    expected = n_blocks * (n_configs + 1) / 2.0
    denominator = n_blocks * n_configs * (n_configs + 1) - ties / (n_configs - 1)
    if denominator <= 0.0:
        return keep
    statistic = 12.0 * float(np.sum((rank_sums - expected) ** 2)) / denominator
    if not float(chi2.sf(statistic, n_configs - 1)) < alpha:
        return keep

    df = (n_blocks - 1) * (n_configs - 1)
    spread = 2.0 * (n_blocks * float(np.sum(ranks**2)) - float(np.sum(rank_sums**2))) / df
    critical = float(t.ppf(1.0 - alpha / 2.0, df)) * math.sqrt(max(spread, 0.0))
    best = int(np.argmin(rank_sums))
    keep = rank_sums - rank_sums[best] <= critical
    keep[best] = True
    return np.asarray(keep, dtype=bool)


def vargha_delaney_a12(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Compute Vargha-Delaney A12 effect size.

    A12 = P(a > b) + 0.5 * P(a == b)

    Interpretation:
      0.5  = no effect
      >0.56 = small effect
      >0.64 = medium effect
      >0.71 = large effect

    Returns a value in [0, 1].
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.5
    more = 0.0
    equal = 0.0
    for ai in a:
        for bi in b:
            if ai > bi:
                more += 1.0
            elif ai == bi:
                equal += 1.0
    return (more + 0.5 * equal) / (m * n)


def a12_magnitude(a12: float) -> str:
    """Return the effect size magnitude for an A12 value."""
    d = abs(a12 - 0.5)
    if d < 0.06:
        return "negligible"
    if d < 0.14:
        return "small"
    if d < 0.21:
        return "medium"
    return "large"


__all__ = [
    "a12_magnitude",
    "build_score_matrix",
    "select_configs_by_frace",
    "select_configs_by_paired_test",
    "vargha_delaney_a12",
]
