from __future__ import annotations

from pathlib import Path

import pytest

from vamos.experiment.artifacts import save_result
from vamos.experiment.gui.catalog import discover_runs, load_front
from vamos.experiment.gui.tables import run_table, solution_table, solutions_csv
from vamos.experiment.unified import optimize

pytestmark = pytest.mark.gui


def _store(root: Path, relative: str, *, algorithm: str = "nsgaii", seed: int = 1) -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    result = optimize("zdt1", algorithm=algorithm, max_evaluations=300, pop_size=20, seed=seed)
    return save_result(result, target).root


def test_discovery_lists_runs_skips_hidden_dirs_and_reports_unreadable_ones(tmp_path: Path) -> None:
    _store(tmp_path, "a/run1")
    _store(tmp_path, "b/run2", algorithm="spea2", seed=2)
    hidden = tmp_path / ".staging" / "x"
    hidden.mkdir(parents=True)
    (hidden / "manifest.json").write_text("{}", encoding="utf-8")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")

    by_path = {summary.relative_path: summary for summary in discover_runs(tmp_path)}
    assert set(by_path) == {"a/run1", "b/run2", "broken"}
    first = by_path["a/run1"]
    assert first.loadable
    assert (first.algorithm, first.problem, first.seed, first.n_objectives, first.evaluations) == ("nsgaii", "zdt1", 1, 2, 300)
    assert by_path["b/run2"].algorithm == "spea2"
    assert by_path["broken"].status == "unreadable"
    assert by_path["broken"].error
    assert not by_path["broken"].loadable

    assert discover_runs(tmp_path / "missing") == []
    assert len(discover_runs(tmp_path, limit=1)) == 1
    assert discover_runs(tmp_path, max_depth=0) == []


def test_front_loading_tables_and_csv_export(tmp_path: Path) -> None:
    root = _store(tmp_path, "run")
    F, X = load_front(root)
    assert X is not None

    columns, rows = solution_table(F, X, [0, 1, 999], max_variables=3)
    assert [column["label"] for column in columns] == ["#", "f1", "f2", "x1", "x2", "x3"]
    assert [row["solution"] for row in rows] == [0, 1]

    lines = solutions_csv(F, X, [1]).strip().splitlines()
    header = lines[0].split(",")
    assert header[:3] == ["solution", "f1", "f2"]
    assert len(header) == 3 + X.shape[1]
    values = lines[1].split(",")
    assert values[0] == "1"
    assert float(values[1]) == F[1, 0]
    assert len(solutions_csv(F, X).strip().splitlines()) == F.shape[0] + 1

    run_columns, run_rows = run_table(discover_runs(tmp_path))
    assert run_columns[0]["name"] == "problem"
    assert (run_rows[0]["id"], run_rows[0]["algorithm"], run_rows[0]["relative_path"]) == (0, "nsgaii", "run")
