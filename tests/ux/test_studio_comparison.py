"""Tests for the statistics and the study comparison of VAMOS Studio."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from vamos.study_artifacts import StudySpec, create_study
from vamos.ux.studio.comparison import (
    HV,
    IGD_PLUS,
    compare_with_pivot,
    igd_plus,
    load_comparison_data,
    median_run,
    rank_algorithms,
    verdict_counts,
)
from vamos.ux.studio.comparison_export import comparison_html, comparison_latex, ranking_latex, saes_csv, saes_metrics_csv
from vamos.ux.studio.comparison_stats import (
    a12,
    a12_magnitude,
    adjust,
    attainment_surface_2d,
    average_ranks,
    benjamini_hochberg,
    cd_groups,
    friedman_p,
    holm,
    holm_against_pivot,
    nemenyi_cd,
    rank_sum_p,
    verdict,
)


def test_holm_and_benjamini_hochberg_adjust_in_input_order() -> None:
    p = [0.01, 0.04, 0.03, 0.005]

    assert holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert adjust(p, "none") == pytest.approx(p)
    assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])


def test_rank_sum_and_effect_size() -> None:
    better, worse = [5.0, 6.0, 7.0, 8.0, 9.0], [0.0, 1.0, 2.0, 3.0, 4.0]

    assert rank_sum_p(better, worse) == pytest.approx(2.0 / math.comb(10, 5))
    assert rank_sum_p([1.0, 1.0], [1.0, 1.0]) == 1.0
    assert a12(better, worse, higher_is_better=True) == 1.0
    assert a12(better, worse, higher_is_better=False) == 0.0
    assert [a12_magnitude(value) for value in (0.52, 0.6, 0.68, 0.9, 0.1)] == ["negligible", "small", "medium", "large", "large"]
    assert (verdict(0.01, 0.8, 0.05), verdict(0.01, 0.2, 0.05), verdict(0.2, 0.9, 0.05)) == ("+", "-", "=")


def test_ranks_friedman_and_critical_difference() -> None:
    matrix = np.array([[0.9, 0.8, 0.1], [0.7, 0.9, 0.2], [0.6, 0.5, 0.3], [0.9, 0.9, 0.1]])

    ranks = average_ranks(matrix, higher_is_better=True)
    statistic, p = friedman_p(matrix)  # type: ignore[misc]

    assert ranks == pytest.approx([1.375, 1.625, 3.0])
    assert average_ranks(np.array([[1.0, 1.0, 0.0]]), higher_is_better=True) == pytest.approx([1.5, 1.5, 3.0])
    assert p == pytest.approx(stats.friedmanchisquare(matrix[:, 0], matrix[:, 1], matrix[:, 2]).pvalue)
    assert friedman_p(matrix[:, :2]) is None
    assert nemenyi_cd(3, 10) == pytest.approx(2.343 * math.sqrt(12 / 60))
    assert cd_groups([1.0, 1.5, 3.0], 1.0) == [(0, 1)]
    assert cd_groups([1.0, 1.4, 1.8], 1.0) == [(0, 2)]
    against = holm_against_pivot([1.0, 2.0, 3.0], pivot=0, n=10)
    assert against[0][1:] == (1.0, 1.0)
    assert against[2][2] >= against[2][1]


def test_attainment_surfaces_of_two_runs() -> None:
    first = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    second = np.array([[0.5, 1.5], [1.5, 0.5]])

    best = attainment_surface_2d([first, second], 1)
    worst = attainment_surface_2d([first, second], 2)

    assert best.tolist() == [[0.0, 2.0], [0.5, 1.5], [1.0, 1.0], [1.5, 0.5], [2.0, 0.0]]
    assert worst.tolist() == [[0.5, 2.0], [1.0, 1.5], [1.5, 1.0], [2.0, 0.5]]
    assert attainment_surface_2d([first], 2).shape == (0, 2)


def test_igd_plus_counts_only_shortfalls() -> None:
    reference = np.array([[0.0, 1.0], [1.0, 0.0]])

    assert igd_plus(reference, reference) == 0.0
    assert igd_plus(np.array([[0.0, 0.0]]), reference) == 0.0
    assert igd_plus(np.array([[1.0, 1.0]]), reference) == pytest.approx(1.0)
    assert math.isnan(igd_plus(np.empty((0, 2)), reference))


@pytest.fixture(scope="module")
def small_study(tmp_path_factory: pytest.TempPathFactory) -> Path:
    spec = StudySpec(
        problems=["zdt1", "zdt2"],
        algorithms=["nsgaii", "spea2"],
        seeds=[0, 1, 2],
        pop_size=8,
        max_evaluations=80,
        engine="numpy",
        on_error="continue",
    )
    return create_study(spec, output=tmp_path_factory.mktemp("compare") / "study").run().root


def test_comparison_of_a_study(small_study: Path) -> None:
    data = load_comparison_data(small_study)

    comparisons = compare_with_pivot(data, HV, "NSGA-II", correction="holm", alpha=0.05)
    ranking = rank_algorithms(data, HV, "NSGA-II")

    assert data.indicators == (HV, IGD_PLUS)
    assert data.algorithms == ("NSGA-II", "SPEA2")
    assert [(item.problem, item.contender) for item in comparisons] == [("zdt1", "SPEA2"), ("zdt2", "SPEA2")]
    assert sum(verdict_counts(comparisons)["SPEA2"].values()) == 2
    assert all(item.verdict == "=" for item in comparisons)  # Three runs cannot reach significance.
    assert sum(ranking.ranks) == pytest.approx(3.0)  # Ranks 1..k always sum to k(k+1)/2.
    assert any("problem" in warning for warning in ranking.warnings)
    assert median_run(data, HV, "zdt1", "SPEA2") is not None
    assert all(len(values) == 3 for values in data.values[IGD_PLUS].values())
    assert set(data.runtimes_ms) == {(p, a) for p in data.problems for a in data.algorithms}


def test_comparison_exports(small_study: Path) -> None:
    data = load_comparison_data(small_study)
    comparisons = compare_with_pivot(data, IGD_PLUS, "SPEA2", correction="bh", alpha=0.05)

    latex = comparison_latex(data, IGD_PLUS, comparisons, "SPEA2", correction="bh", alpha=0.05)
    ranking = ranking_latex(rank_algorithms(data, IGD_PLUS, "SPEA2"), "SPEA2", IGD_PLUS)

    assert r"\begin{tabular}{lrr}" in latex and "lower is better" in latex and "Benjamini-Hochberg" in latex
    assert latex.index("NSGA-II") < latex.index("SPEA2 \\\\")  # The pivot is the last column.
    assert r"\textbf{" in latex
    assert "Average rank" in ranking
    assert "(pivot)" in comparison_html(data, IGD_PLUS, comparisons, "SPEA2")
    assert saes_csv(data).splitlines()[0] == "Algorithm,Problem,MetricName,ExecutionId,MetricValue"
    assert len(saes_csv(data).splitlines()) == 1 + 2 * 12
    assert saes_metrics_csv(data).splitlines() == ["MetricName,Maximize", "HV,True", "IGD+,False"]
