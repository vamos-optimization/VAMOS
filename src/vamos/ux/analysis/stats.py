from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats as spstats  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Two-tailed Nemenyi critical values q_alpha for k = 2..10 algorithms (Demsar, 2006,
# Table 5): the studentized range quantile with infinite degrees of freedom over sqrt(2).
_NEMENYI_Q: dict[float, tuple[float, ...]] = {
    0.05: (1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164),
    0.10: (1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920),
}


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise ImportError(
            "Statistics plotting requires matplotlib. Install with `pip install vamos[analysis]` or `pip install matplotlib`."
        ) from exc


@dataclass
class FriedmanResult:
    statistic: float
    p_value: float
    ranks: NDArray[np.float64]
    avg_ranks: NDArray[np.float64]


@dataclass
class WilcoxonResult:
    """A paired signed-rank test; ``p_adjusted`` is Holm's over all the pairs of the call."""

    algo_i: str
    algo_j: str
    statistic: float
    p_value: float
    p_adjusted: float = math.nan
    significant: bool = False


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order.

    A NaN p-value (a test that could not be computed) stays NaN and does not count as a
    hypothesis.
    """
    ordered = sorted((p, index) for index, p in enumerate(p_values) if not math.isnan(p))
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    for rank, (p, index) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[index] = running
    return adjusted


def compute_ranks(scores: NDArray[np.float64], higher_is_better: bool = True) -> NDArray[np.float64]:
    """
    Compute per-problem ranks for algorithms.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be 2-dimensional (n_problems, n_algorithms).")
    n_problems, n_algos = scores.shape
    ranks = np.empty_like(scores)
    for p in range(n_problems):
        vals = scores[p]
        if not higher_is_better:
            vals = -vals
        ranks[p] = spstats.rankdata(-vals, method="average")  # -vals so best gets rank 1
    return ranks


def friedman_test(scores: NDArray[np.float64], higher_is_better: bool = True) -> FriedmanResult:
    """
    Run Friedman test using ranks across problems.

    ``scores`` needs at least 2 problems (rows), at least 3 algorithms (columns) and no NaN.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be 2-dimensional (n_problems, n_algorithms).")
    if scores.shape[1] < 3:
        raise ValueError(f"The Friedman test needs at least 3 algorithms; got {scores.shape[1]}. Use pairwise_wilcoxon for two.")
    if scores.shape[0] < 2:
        raise ValueError(f"The Friedman test needs at least 2 problems; got {scores.shape[0]}.")
    if np.isnan(scores).any():
        raise ValueError("The Friedman test cannot rank NaN scores; drop or impute them first.")
    ranks = compute_ranks(scores, higher_is_better=higher_is_better)
    avg_ranks = np.mean(ranks, axis=0)
    # scipy expects columns; we run on raw scores as in classic use
    stat, p = spstats.friedmanchisquare(*[scores[:, j] if higher_is_better else -scores[:, j] for j in range(scores.shape[1])])
    return FriedmanResult(statistic=stat, p_value=p, ranks=ranks, avg_ranks=avg_ranks)


def pairwise_wilcoxon(
    scores: NDArray[np.float64],
    algo_names: Sequence[str],
    higher_is_better: bool = True,
    alpha: float = 0.05,
) -> list[WilcoxonResult]:
    """
    Pairwise Wilcoxon signed-rank tests across algorithms.

    ``p_value`` is each test's raw p-value; ``p_adjusted`` corrects it with Holm's method over
    all the pairs, and ``significant`` is ``p_adjusted <= alpha``.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be 2-dimensional (n_problems, n_algorithms).")
    n_algos = scores.shape[1]
    if len(algo_names) != n_algos:
        raise ValueError("algo_names length must match number of algorithms.")
    results: list[WilcoxonResult] = []
    for i in range(n_algos):
        for j in range(i + 1, n_algos):
            a = scores[:, i]
            b = scores[:, j]
            if not higher_is_better:
                a = -a
                b = -b
            differences = _paired_differences(a, b)
            if np.all(differences == 0.0):
                # Equal on every problem, including both infeasible everywhere: no evidence of a
                # difference. SciPy 1.12 and earlier raise on an all-zero sample under Pratt's
                # zero method, and later versions divide by zero.
                stat, p = 0.0, 1.0
            else:
                stat, p = spstats.wilcoxon(differences, zero_method="pratt", alternative="two-sided")
            results.append(WilcoxonResult(algo_i=algo_names[i], algo_j=algo_names[j], statistic=stat, p_value=p))
    for item, adjusted in zip(results, holm_adjust([float(item.p_value) for item in results]), strict=True):
        item.p_adjusted = adjusted
        item.significant = bool(adjusted <= alpha)
    return results


def _paired_differences(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``a - b`` with equal pairs, a shared infinite penalty included, as ties.

    ``inf - inf`` is NaN, which would drop the pair from the signed-rank test
    instead of counting it as the tie it is. An infinity against a finite value
    stays an infinite difference, the largest rank.
    """
    with np.errstate(invalid="ignore"):
        return np.asarray(np.where(a == b, 0.0, a - b), dtype=float)


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float | None:
    """Nemenyi critical difference of average ranks for ``k`` algorithms on ``n`` problems.

    ``q_alpha * sqrt(k (k + 1) / (6 n))``, where ``q_alpha`` is the two-tailed critical
    value for ``k`` algorithms: Demsar (2006), Table 5, for alpha 0.05 and 0.10 with
    k <= 10, and otherwise the studentized range quantile with infinite degrees of
    freedom divided by sqrt(2). ``None`` with fewer than two algorithms or no problem.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly between 0 and 1; got {alpha}.")
    if k < 2 or n < 1:
        return None
    table = _NEMENYI_Q.get(alpha)
    if table is not None and k - 2 < len(table):
        q = table[k - 2]
    else:
        q = float(spstats.studentized_range.ppf(1.0 - alpha, k, np.inf)) / math.sqrt(2.0)
    return q * math.sqrt(k * (k + 1) / (6.0 * n))


def plot_critical_distance(
    avg_ranks: NDArray[np.float64],
    algo_names: Sequence[str],
    alpha: float = 0.05,
    n_problems: int | None = None,
    higher_is_better: bool = True,
    ax: Axes | None = None,
    show: bool = False,
) -> object:
    """
    Plot average ranks and, when ``n_problems`` is given, the Nemenyi critical difference.

    The bar length is :func:`nemenyi_cd` for ``len(algo_names)`` algorithms at ``alpha``,
    which must lie strictly between 0 and 1.
    """
    plt = _require_matplotlib()
    avg_ranks = np.asarray(avg_ranks, dtype=float)
    if avg_ranks.ndim != 1:
        raise ValueError("avg_ranks must be 1-dimensional.")
    if len(algo_names) != avg_ranks.shape[0]:
        raise ValueError("algo_names length must match avg_ranks length.")
    k = avg_ranks.shape[0]
    cd = nemenyi_cd(k, n_problems, alpha) if n_problems is not None else None
    ax = ax or plt.figure().add_subplot(111)
    # avg_ranks from compute_ranks always encode rank 1 = best regardless of
    # higher_is_better; no inversion needed for the CD diagram (lower x = better).
    ranks = avg_ranks
    y = np.zeros_like(ranks)
    ax.scatter(ranks, y, c="black")
    for idx, name in enumerate(algo_names):
        ax.text(ranks[idx], 0.02, name, rotation=45, ha="center", va="bottom")
    ax.set_yticks([])
    ax.set_xlabel("Average Rank (lower is better)")
    if cd is not None:
        xmin = ranks.min()
        ax.hlines(0.1, xmin, xmin + cd, colors="black")
        ax.vlines([xmin, xmin + cd], 0.08, 0.12, colors="black")
        ax.text(xmin + cd / 2, 0.13, f"CD={cd:.2f}", ha="center")
    if show:
        plt.show()
    return ax
