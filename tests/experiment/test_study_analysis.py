"""Indicators derived from study summaries for ``vamos bench`` and ablation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vamos.experiment.study_analysis import SummarySource, derive_summary_rows
from vamos.foundation.core.hv_stop import compute_hv_reference
from vamos.run_artifacts import load_result
from vamos.study_artifacts import StudySpec, create_study

pytest.importorskip("moocore")

_INDICATORS = ("hv", "igd_plus")


def _run_study(root: Path, name: str, problem: str, *, n_var: int, n_obj: int | None = None, pop_size: int = 20) -> SummarySource:
    spec = StudySpec(
        problems=[problem],
        algorithms=["nsgaii"],
        seeds=[0, 1],
        max_evaluations=1200,
        pop_size=pop_size,
        engine="numpy",
        n_var=n_var,
        n_obj=n_obj,
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
