"""Indicators derived from study summaries for ``vamos bench`` and ablation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vamos.experiment import study_analysis
from vamos.experiment.study_analysis import SummarySource, derive_summary_rows
from vamos.foundation.core.hv_stop import compute_hv_reference
from vamos.foundation.quality_indicators.hypervolume import hypervolume
from vamos.run_artifacts import load_result
from vamos.study_artifacts import StudySpec, create_study

pytest.importorskip("moocore")

_INDICATORS = ("hv", "igd_plus")
_CONSTRAINED_INDICATORS = ("hv", "igd_plus", "epsilon_additive")


def _run_study(
    root: Path,
    name: str,
    problem: str,
    *,
    n_var: int | None = None,
    n_obj: int | None = None,
    pop_size: int = 20,
    algorithm: str = "nsgaii",
    max_evaluations: int = 1200,
    algorithm_config: Mapping[str, object] | None = None,
) -> SummarySource:
    spec = StudySpec(
        problems=[problem],
        algorithms=[algorithm],
        seeds=[0, 1],
        max_evaluations=max_evaluations,
        pop_size=pop_size,
        engine="numpy",
        n_var=n_var,
        n_obj=n_obj,
        algorithm_configs=None if algorithm_config is None else {algorithm: dict(algorithm_config)},
    )
    study = create_study(spec, output=root / name).run()
    assert study.status == "completed"
    return SummarySource(study.root, study.summarize(), {})


@pytest.fixture(scope="module")
def studies(tmp_path_factory: pytest.TempPathFactory) -> dict[str, SummarySource]:
    root = tmp_path_factory.mktemp("study-analysis")
    return {
        "zdt1": _run_study(root, "zdt1", "zdt1", n_var=5),
        "zdt1-variant": _run_study(root, "zdt1-variant", "zdt1", n_var=5, pop_size=12),
        "zdt3": _run_study(root, "zdt3", "zdt3", n_var=5),
        "zdt4": _run_study(root, "zdt4", "zdt4", n_var=5),
    }


def _values(rows: Sequence[Mapping[str, Any]], problem: str) -> dict[tuple[str, int], tuple[object, ...]]:
    return {
        (row["study_id"], row["seed"]): (row["hv"], row["indicator_igd_plus"], row["hv_reference"], row["hv_source"])
        for row in rows
        if row["problem"] == problem
    }


def _front(source: SummarySource, run_manifest_path: str) -> np.ndarray:
    return np.asarray(load_result(source.root / Path(run_manifest_path).parent).F, dtype=float)


def _feasible_front(source: SummarySource, run_manifest_path: str) -> np.ndarray:
    result = load_result(source.root / Path(run_manifest_path).parent)
    return np.asarray(result.F, dtype=float)[np.all(np.asarray(result.data["G"]) <= 0.0, axis=1)]


@pytest.fixture(scope="module")
def mw2(tmp_path_factory: pytest.TempPathFactory) -> dict[str, SummarySource]:
    root = tmp_path_factory.mktemp("study-analysis-mw2")
    return {
        # 1200 evaluations reach the MW2 feasible region in both seeds.
        "feasible": _run_study(root, "feasible", "mw2"),
        # The random initial population (budget = population) has no feasible solution.
        "infeasible": _run_study(root, "infeasible", "mw2", pop_size=10, max_evaluations=10),
        # IBEA after 320 evaluations keeps 3 and 2 feasible solutions out of 20.
        "partial": _run_study(root, "partial", "mw2", algorithm="ibea", max_evaluations=320),
    }


def test_derive_summary_rows_uses_per_problem_references(studies: dict[str, SummarySource]) -> None:
    alone = _values(derive_summary_rows([studies["zdt1"]], indicators=_INDICATORS), "zdt1")
    in_suite = _values(
        derive_summary_rows([studies["zdt1"], studies["zdt3"], studies["zdt4"]], indicators=_INDICATORS),
        "zdt1",
    )

    assert len(alone) == 2
    # ZDT4 used to inflate the ZDT1 HV reference and ZDT3 used to pull the
    # IGD+ reference front below the ZDT1 front.
    assert in_suite == alone
    assert {values[3] for values in alone.values()} == {"problem"}


def test_derive_summary_rows_shares_references_across_sources_of_a_problem(studies: dict[str, SummarySource]) -> None:
    sources = [studies["zdt1"], studies["zdt1-variant"], studies["zdt4"]]

    rows = derive_summary_rows(sources, indicators=_INDICATORS)

    zdt1_fronts = [
        _front(source, row.run_manifest_path) for source in sources[:2] for row in source.summary.rows if row.run_manifest_path is not None
    ]
    expected = " ".join(f"{value:.6f}" for value in compute_hv_reference(zdt1_fronts))
    assert {row["hv_reference"] for row in rows if row["problem"] == "zdt1"} == {expected}
    assert len([row for row in rows if row["problem"] == "zdt1"]) == 4
    assert {row["hv_reference"] for row in rows if row["problem"] == "zdt4"} != {expected}


def test_derive_summary_rows_accepts_problems_with_different_objectives(
    studies: dict[str, SummarySource],
    tmp_path: Path,
) -> None:
    dtlz2_3 = _run_study(tmp_path, "dtlz2-3", "dtlz2", n_var=12, n_obj=3)
    dtlz2_2 = _run_study(tmp_path, "dtlz2-2", "dtlz2", n_var=11, n_obj=2)

    rows = derive_summary_rows([studies["zdt1"], dtlz2_3, dtlz2_2], indicators=_INDICATORS)

    by_study = {source.summary.study_id: source for source in (studies["zdt1"], dtlz2_3, dtlz2_2)}
    assert len(rows) == 6
    for row in rows:
        n_obj = _front(by_study[row["study_id"]], row["run_manifest_path"]).shape[1]
        assert len(row["hv_reference"].split()) == n_obj
        assert isinstance(row["hv"], float)
        assert isinstance(row["indicator_igd_plus"], float)
    # Same problem identity, different objective counts: separate references.
    assert dtlz2_2.summary.rows[0].problem_id == dtlz2_3.summary.rows[0].problem_id
    three = _values(derive_summary_rows([dtlz2_3], indicators=_INDICATORS), "dtlz2")
    two = _values(derive_summary_rows([dtlz2_2], indicators=_INDICATORS), "dtlz2")
    assert _values(rows, "dtlz2") == {**three, **two}


def test_derive_summary_rows_excludes_infeasible_solutions(mw2: dict[str, SummarySource]) -> None:
    rows = derive_summary_rows([mw2["feasible"], mw2["infeasible"]], indicators=_CONSTRAINED_INDICATORS)

    infeasible = [row for row in rows if row["study_id"] == mw2["infeasible"].summary.study_id]
    assert len(infeasible) == 2
    for row in infeasible:
        observed = {
            key: row.get(key) for key in ("hv", "indicator_igd_plus", "indicator_epsilon_additive", "n_feasible", "feasibility_known")
        }
        assert observed == {
            "hv": 0.0,
            "indicator_igd_plus": math.inf,
            "indicator_epsilon_additive": math.inf,
            "n_feasible": 0,
            "feasibility_known": True,
        }


def test_infeasible_run_does_not_change_feasible_indicators(mw2: dict[str, SummarySource]) -> None:
    feasible_study = mw2["feasible"].summary.study_id

    def feasible_values(rows: Sequence[Mapping[str, Any]]) -> dict[int, tuple[object, ...]]:
        return {
            row["seed"]: (row["hv"], row["indicator_igd_plus"], row["indicator_epsilon_additive"], row["hv_reference"])
            for row in rows
            if row["study_id"] == feasible_study
        }

    alone = feasible_values(derive_summary_rows([mw2["feasible"]], indicators=_CONSTRAINED_INDICATORS))
    mixed = feasible_values(derive_summary_rows([mw2["feasible"], mw2["infeasible"]], indicators=_CONSTRAINED_INDICATORS))

    assert len(alone) == 2
    assert mixed == alone


def test_derive_summary_rows_measures_only_the_feasible_part_of_a_run(mw2: dict[str, SummarySource]) -> None:
    source = mw2["partial"]
    fronts = [_feasible_front(source, row.run_manifest_path) for row in source.summary.rows if row.run_manifest_path is not None]
    totals = [len(_front(source, row.run_manifest_path)) for row in source.summary.rows if row.run_manifest_path is not None]
    assert all(0 < len(front) < total for front, total in zip(fronts, totals))

    rows = derive_summary_rows([source], indicators=("hv",))

    reference = compute_hv_reference(fronts)
    assert [row["hv"] for row in rows] == [float(hypervolume(front, reference)) for front in fronts]
    assert [row.get("n_feasible") for row in rows] == [len(front) for front in fronts]


def test_derive_summary_rows_marks_runs_without_constraint_values(studies: dict[str, SummarySource], tmp_path: Path) -> None:
    # SMPSO's non-dominated result comes from its archive, which stores no G.
    archive = _run_study(
        tmp_path, "archive", "mw2", algorithm="smpso", max_evaluations=400, algorithm_config={"result_mode": "non_dominated"}
    )

    rows = derive_summary_rows([studies["zdt1"], archive], indicators=_INDICATORS)

    by_study = {source.summary.study_id: source for source in (studies["zdt1"], archive)}
    assert len(rows) == 4
    for row in rows:
        assert row["feasibility_known"] is False
        assert row["n_feasible"] == len(_front(by_study[row["study_id"]], row["run_manifest_path"]))


def test_derive_summary_rows_scores_a_stored_empty_result_as_infeasible(
    mw2: dict[str, SummarySource],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The canonical bundle accepts F (0, n_obj) and G (0, n_constraints).
    source = mw2["feasible"]
    empty_row = source.summary.rows[0]
    assert empty_row.run_manifest_path is not None
    empty_run = source.root / Path(empty_row.run_manifest_path).parent

    def load_with_empty_run(path: Path) -> Any:
        result = load_result(path)
        if Path(path) != empty_run:
            return result
        n_obj, n_constraints = result.F.shape[1], np.asarray(result.data["G"]).shape[1]
        return type(result)({"F": np.empty((0, n_obj)), "X": np.empty((0, result.X.shape[1])), "G": np.empty((0, n_constraints))})

    monkeypatch.setattr(study_analysis, "load_result", load_with_empty_run)

    rows = derive_summary_rows([source, mw2["infeasible"]], indicators=_CONSTRAINED_INDICATORS)

    empty = [row for row in rows if row["study_id"] == source.summary.study_id and row["task_id"] == empty_row.task_id]
    assert [
        {key: row[key] for key in ("hv", "indicator_igd_plus", "indicator_epsilon_additive", "n_feasible", "feasibility_known")}
        for row in empty
    ] == [{"hv": 0.0, "indicator_igd_plus": math.inf, "indicator_epsilon_additive": math.inf, "n_feasible": 0, "feasibility_known": True}]
    other = [row for row in rows if row["study_id"] == source.summary.study_id and row["task_id"] != empty_row.task_id]
    assert len(other) == 1
    assert other[0]["hv"] > 0.0
    assert math.isfinite(other[0]["indicator_igd_plus"])


_ALIAS_FRONT = np.array([[1.0, 2.5], [2.0, 1.5], [3.0, 1.2]])
_ALIAS_REFERENCE_FRONT = np.array([[1.0, 2.0], [1.5, 1.5], [2.0, 1.0]])
_ALIAS_REFERENCE_POINT = np.array([4.0, 3.0])


@pytest.mark.parametrize(
    "name",
    ["eps", "epsilon", "EPSILON_ADDITIVE", "eps_mult", "epsilon_mult", "hausdorff", "IGD", "igd+", "hv_approx", "hypervolume"],
)
def test_indicator_aliases_receive_reference_front(name: str) -> None:
    value = study_analysis._row_indicator(name, _ALIAS_FRONT, _ALIAS_REFERENCE_FRONT, _ALIAS_REFERENCE_POINT)
    assert value is not None
    assert math.isfinite(float(np.asarray(value)))


def test_hv_contribution_columns_use_the_problem_reference_point() -> None:
    value = study_analysis._row_indicator("hv_contributions", _ALIAS_FRONT, _ALIAS_REFERENCE_FRONT, _ALIAS_REFERENCE_POINT)
    assert value is not None
    assert np.asarray(value) == pytest.approx([0.5, 1.0, 0.3])
