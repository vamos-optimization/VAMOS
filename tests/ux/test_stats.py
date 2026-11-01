import math
import warnings

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from scipy.stats import studentized_range

from vamos.ux.analysis.stats import (
    FriedmanResult,
    compute_ranks,
    friedman_test,
    nemenyi_cd,
    pairwise_wilcoxon,
    plot_critical_distance,
)


def test_compute_ranks_higher_and_lower():
    # Arrange
    scores = np.array([[1.0, 2.0, 3.0], [0.5, 0.2, 0.1]])

    # Act
    ranks_high = compute_ranks(scores, higher_is_better=True)
    ranks_low = compute_ranks(scores, higher_is_better=False)

    # Assert
    assert ranks_high[0].tolist() == [3.0, 2.0, 1.0]
    assert ranks_low[0].tolist() == [1.0, 2.0, 3.0]


def test_friedman_test_detects_difference():
    # Arrange
    scores = np.array([[0.9, 0.5, 0.4], [0.8, 0.4, 0.3], [0.85, 0.45, 0.35]])

    # Act
    result = friedman_test(scores, higher_is_better=True)

    # Assert
    assert isinstance(result, FriedmanResult)
    assert result.statistic > 0
    assert result.p_value < 0.1
    assert result.avg_ranks[0] < result.avg_ranks[1]


def test_pairwise_wilcoxon_pair_count_and_order():
    # Arrange
    scores = np.array([[1.0, 0.5, 0.6], [1.1, 0.4, 0.65], [0.9, 0.45, 0.7]])
    algo_names = ["A", "B", "C"]

    # Act
    results = pairwise_wilcoxon(scores, algo_names, higher_is_better=True)

    # Assert
    assert len(results) == 3
    p_ab = next(res.p_value for res in results if res.algo_i == "A" and res.algo_j == "B")
    p_bc = next(res.p_value for res in results if res.algo_i == "B" and res.algo_j == "C")
    assert p_ab <= p_bc


def test_pairwise_wilcoxon_ties_shared_infinite_penalties() -> None:
    # IGD+ of +inf marks a problem where the algorithm found no feasible solution.
    # 14 finite pairs with mixed signs, so a dropped pair and a tied pair give different p-values.
    a = [0.10, 0.20, 0.35, 0.30, 0.15, 0.25, 0.48, 0.35, 0.40, 0.12, 0.22, 0.43, 0.18, 0.28]
    b = [0.20, 0.40, 0.30, 0.60, 0.30, 0.50, 0.40, 0.70, 0.80, 0.24, 0.44, 0.40, 0.36, 0.56]
    finite = [list(pair) for pair in zip(a, b)]
    both_failed = np.array([[math.inf, math.inf], *finite])
    one_failed = np.array([[0.5, math.inf], *finite])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (shared,) = pairwise_wilcoxon(both_failed, ["A", "B"], higher_is_better=False)
        (worst,) = pairwise_wilcoxon(one_failed, ["A", "B"], higher_is_better=False)

    # Both failing is a tie, as two equal finite values would be.
    (tie,) = pairwise_wilcoxon(np.array([[1.0, 1.0], *finite]), ["A", "B"], higher_is_better=False)
    assert math.isfinite(shared.p_value)
    assert (shared.statistic, shared.p_value) == (tie.statistic, tie.p_value)
    # Failing alone keeps the worst rank, as the largest finite gap would.
    (largest_gap,) = pairwise_wilcoxon(np.array([[0.5, 100.0], *finite]), ["A", "B"], higher_is_better=False)
    assert (worst.statistic, worst.p_value) == (largest_gap.statistic, largest_gap.p_value)
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]


def scipy_112_wilcoxon(real):
    """``scipy.stats.wilcoxon`` as SciPy 1.12 and earlier behave on an all-zero sample.

    ``_morestats.wilcoxon`` in 1.12 switches an exact test with zeros to the normal
    approximation and then raises for ``zero_method`` "wilcox" or "pratt" when every
    difference is zero; ``pyproject.toml`` allows ``scipy>=1.10``.
    """

    def wilcoxon(d, *args, zero_method="wilcox", **kwargs):
        values = np.asarray(d, dtype=float)
        if zero_method in ("wilcox", "pratt") and values.size and np.all(values == 0):
            raise ValueError("zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.")
        return real(d, *args, zero_method=zero_method, **kwargs)

    return wilcoxon


@pytest.mark.parametrize("shared", [[0.3, 0.5, 0.2, 0.9], [math.inf] * 4])
def test_pairwise_wilcoxon_identical_algorithms_are_not_different(monkeypatch: pytest.MonkeyPatch, shared: list[float]) -> None:
    import vamos.ux.analysis.stats as stats_module

    # A and B are equal on every problem: equal values, or both infeasible (+inf) everywhere.
    monkeypatch.setattr(stats_module.spstats, "wilcoxon", scipy_112_wilcoxon(stats_module.spstats.wilcoxon))
    other = [0.1, 0.25, 0.4, 0.6]
    scores = np.array([[value, value, extra] for value, extra in zip(shared, other)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = pairwise_wilcoxon(scores, ["A", "B", "C"], higher_is_better=False)

    identical = next(result for result in results if (result.algo_i, result.algo_j) == ("A", "B"))
    assert (identical.statistic, identical.p_value) == (0.0, 1.0)
    assert len(results) == 3
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]


def test_plot_critical_distance_smoke():
    # Arrange
    avg_ranks = np.array([1.2, 2.5, 2.3])
    algo_names = ["A", "B", "C"]

    # Act
    ax = plot_critical_distance(avg_ranks, algo_names, n_problems=10, show=False)

    # Assert
    assert hasattr(ax, "scatter")


# Demsar (2006), "Statistical comparisons of classifiers over multiple data sets",
# JMLR 7:1-30, Table 5: two-tailed Nemenyi critical values q_alpha for k = 2..10.
_DEMSAR_Q = {
    0.05: (1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164),
    0.10: (1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920),
}


def _drawn_critical_difference(ax: Axes) -> float:
    """Length of the horizontal CD bar that the diagram draws."""
    bars = [
        segment
        for collection in ax.collections
        if isinstance(collection, LineCollection)
        for segment in collection.get_segments()
        if segment[0][1] == segment[-1][1]
    ]
    assert len(bars) == 1
    return float(bars[0][-1][0] - bars[0][0][0])


def _drawn_q(k: int, alpha: float, n_problems: int) -> float:
    ranks = np.linspace(1.0, float(k), k)
    ax = plot_critical_distance(ranks, [f"A{i}" for i in range(k)], alpha=alpha, n_problems=n_problems, show=False)
    assert isinstance(ax, Axes)
    try:
        return _drawn_critical_difference(ax) / math.sqrt(k * (k + 1) / (6.0 * n_problems))
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize("alpha", [0.05, 0.10])
@pytest.mark.parametrize("k", range(2, 11))
def test_plot_critical_distance_matches_demsar_table(k: int, alpha: float) -> None:
    q = _drawn_q(k, alpha, n_problems=10)

    assert q == pytest.approx(_DEMSAR_Q[alpha][k - 2], abs=1e-3)


@pytest.mark.parametrize("k", range(11, 21))
def test_plot_critical_distance_uses_studentized_range_beyond_the_table(k: int) -> None:
    expected = float(studentized_range.ppf(0.99, k, np.inf)) / math.sqrt(2.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        q = _drawn_q(k, 0.01, n_problems=10)

    assert q == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.05, 1.5, float("nan")])
def test_critical_difference_rejects_alpha_outside_the_unit_interval(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        nemenyi_cd(4, 10, alpha)
    with pytest.raises(ValueError, match="alpha"):
        plot_critical_distance(np.array([1.0, 2.0]), ["A", "B"], alpha=alpha, n_problems=10, show=False)


def test_nemenyi_cd_needs_two_algorithms_and_a_problem() -> None:
    assert nemenyi_cd(1, 10) is None
    assert nemenyi_cd(3, 0) is None
    assert nemenyi_cd(3, 10) == pytest.approx(2.343 * math.sqrt(12 / 60))


def test_pairwise_wilcoxon_reports_holm_adjusted_p_values() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(size=(12, 1))
    scores = np.hstack([base, base + 0.3 + 0.05 * rng.normal(size=(12, 1)), base + 0.01 * rng.normal(size=(12, 1))])

    strict = pairwise_wilcoxon(scores, ["a", "b", "c"], alpha=1e-9)
    loose = pairwise_wilcoxon(scores, ["a", "b", "c"], alpha=0.05)

    raw = np.array([item.p_value for item in loose])
    order = np.argsort(raw)
    expected = np.empty(3)
    expected[order] = np.minimum(1.0, np.maximum.accumulate((3 - np.arange(3)) * raw[order]))
    np.testing.assert_allclose([item.p_adjusted for item in loose], expected)
    assert [item.p_value for item in strict] == list(raw)
    assert [item.significant for item in loose] == [bool(p <= 0.05) for p in expected]
    assert not any(item.significant for item in strict)
    assert any(item.significant for item in loose)


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        (np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]]), "at least 3 algorithms"),
        (np.array([[1.0, 2.0, 3.0]]), "at least 2 problems"),
        (np.array([[1.0, 2.0, 3.0], [2.0, np.nan, 1.0], [3.0, 1.0, 2.0]]), "NaN"),
    ],
)
def test_friedman_test_validates_shape_and_nan(scores: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        friedman_test(scores)
