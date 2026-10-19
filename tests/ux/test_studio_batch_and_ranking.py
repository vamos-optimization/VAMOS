"""Tests for the Panel-free batch hypervolume summaries and MCDM rankings of VAMOS Studio."""

from __future__ import annotations

import numpy as np
import pytest

from vamos.ux.studio.batch import (
    BatchFailure,
    BatchRun,
    csv_text,
    latex_escape,
    normalized_hypervolumes,
    run_from_result,
    run_rows,
    summary_latex,
    summary_rows,
)
from vamos.ux.studio.data import FrontRecord
from vamos.ux.studio.ranking import front_summary_rows, problems_in, rank_fronts


def _run(algorithm: str, seed: int, points: list[list[float]], G: list[list[float]] | None = None) -> BatchRun:
    return BatchRun(
        problem="p",
        algorithm=algorithm,
        seed=seed,
        F=np.asarray(points, dtype=float),
        G=None if G is None else np.asarray(G, dtype=float),
    )


def test_hypervolume_uses_the_batch_normalization_and_reference_point() -> None:
    runs = [_run("A", 0, [[0.0, 1.0], [1.0, 0.0]]), _run("B", 0, [[0.5, 0.5]])]

    hv = normalized_hypervolumes(runs)

    # A: boxes [0,1.1]x[1,1.1] and [1,1.1]x[0,1.1] overlap in [1,1.1]^2.
    assert hv[0] == pytest.approx(1.1 * 0.1 + 0.1 * 1.1 - 0.1 * 0.1)
    assert hv[1] == pytest.approx(0.6 * 0.6)


def test_hypervolume_ignores_infeasible_solutions() -> None:
    runs = [
        _run("A", 0, [[0.0, 1.0], [1.0, 0.0]]),
        _run("B", 0, [[0.0, 0.0], [0.5, 0.5]], G=[[1.0], [-1.0]]),
    ]

    hv = normalized_hypervolumes(runs)

    assert runs[1].feasible_F.tolist() == [[0.5, 0.5]]
    assert hv[1] == pytest.approx(0.6 * 0.6)


def test_a_run_without_feasible_solutions_scores_zero() -> None:
    runs = [_run("A", 0, [[0.0, 1.0], [1.0, 0.0]]), _run("B", 0, [[0.5, 0.5]], G=[[2.0]])]

    assert normalized_hypervolumes(runs)[1] == 0.0


def test_summary_marks_the_best_mean_and_counts_failures() -> None:
    runs = [_run("A", 0, [[0.0]]), _run("A", 1, [[0.0]]), _run("B", 0, [[0.0]])]
    failures = [BatchFailure("p", "B", 1, "boom")]

    rows = summary_rows(runs, [0.5, 0.7, 0.4], failures)

    by_algorithm = {row["Algorithm"]: row for row in rows}
    assert by_algorithm["A"]["HV mean"] == pytest.approx(0.6)
    assert by_algorithm["A"]["HV std"] == pytest.approx(np.std([0.5, 0.7], ddof=1))
    assert by_algorithm["A"]["Best"] == "yes"
    assert by_algorithm["B"]["Best"] == ""
    assert by_algorithm["B"]["Failed"] == 1


def test_run_from_result_requires_objectives_and_keeps_constraints() -> None:
    run = run_from_result("p", "A", 3, {"F": [[1.0, 2.0]], "G": [[0.0]]})

    assert run.F.shape == (1, 2)
    assert run.G is not None and run.G.shape == (1, 1)
    with pytest.raises(RuntimeError, match="no objectives"):
        run_from_result("p", "A", 3, {})


def test_exports_are_plain_text_tables() -> None:
    runs = [_run("NSGA_II", 0, [[0.0, 1.0]])]
    rows = summary_rows(runs, [0.25])

    text = csv_text(run_rows(runs, [0.25]))
    latex = summary_latex(rows)

    assert text.splitlines()[0] == "Problem,Algorithm,Seed,HV,Points,Feasible points"
    assert r"NSGA\_II" in latex
    assert r"\textbf{0.2500}" in latex
    assert latex_escape("a&b%c") == r"a\&b\%c"
    assert csv_text([]) == ""


def _front(algorithm: str, points: list[list[float]], problem: str = "p") -> FrontRecord:
    return FrontRecord(problem, algorithm, np.asarray(points, dtype=float), None, extra={"seeds": [0, 1]})


def test_ranking_orders_each_front_of_the_selected_problem() -> None:
    fronts = [
        _front("A", [[0.9, 0.0], [0.1, 0.2], [0.0, 1.0]]),
        _front("B", [[0.5, 0.5]], problem="q"),
    ]

    ranking = rank_fronts(fronts, method="weighted_sum", top_k=2, problem="p")

    assert [row["Algorithm"] for row in ranking.rows] == ["A", "A"]
    assert ranking.rows[0]["f1"] == pytest.approx(0.1)
    assert [row["Rank"] for row in ranking.rows] == [1, 2]
    assert problems_in(fronts) == ["p", "q"]
    assert front_summary_rows(fronts)[0]["Seeds"] == 2


def test_a_method_that_cannot_rank_a_front_reports_it() -> None:
    fronts = [_front("A", [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])]

    ranking = rank_fronts(fronts, method="knee", top_k=5)

    assert ranking.rows == []
    assert "cannot rank" in ranking.warnings[0]
