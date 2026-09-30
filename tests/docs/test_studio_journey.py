from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from vamos import load_result, load_study

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "journeys" / "prepare_studio_demo.py"


def test_studio_demo_produces_real_single_problem_study(tmp_path: Path) -> None:
    example = runpy.run_path(str(EXAMPLE))
    output = tmp_path / "study"
    example["run"](output, max_evaluations=80, pop_size=20)

    study = load_study(output)
    report = study.inspect()
    rows = study.summarize().rows
    assert report.state == "completed"
    assert report.verified_run_count == 4
    assert {row.seed for row in rows} == {42, 43}
    assert len({row.algorithm_id for row in rows}) == 2
    assert len({row.problem_id for row in rows}) == 1
    for row in rows:
        assert row.evaluations == 80
        assert row.run_manifest_path is not None
        result = load_result(output / Path(row.run_manifest_path).parent)
        assert result.F.shape[1] == 2
        assert result.F.shape[0] > 0

    with pytest.raises(FileExistsError, match="Choose a new"):
        example["run"](output, max_evaluations=80, pop_size=20)


def test_studio_demo_plot_reads_canonical_algorithm_ids(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    example = runpy.run_path(str(EXAMPLE))
    plot = tmp_path / "study.svg"
    example["run"](tmp_path / "study", max_evaluations=80, pop_size=20, plot=plot)
    svg = plot.read_text(encoding="utf-8")
    assert "NSGA-II" in svg
    assert "MOEA/D" in svg
