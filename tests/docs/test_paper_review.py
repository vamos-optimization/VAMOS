from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from paper import review

ROOT = Path(__file__).resolve().parents[2]


def _csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["algorithm", "framework", "problem", "n_evals", "seed", "runtime_seconds", "hypervolume"])
        writer.writerows(rows)


def test_summary_separates_budgets_and_counts_recorded_metrics(tmp_path: Path) -> None:
    source = tmp_path / "benchmark_paper.csv"
    _csv(
        source,
        [
            ["NSGA-II", "test", "zdt1", "50000", "0", "10", ""],
            ["NSGA-II", "test", "zdt1", "50000", "1", "20", ""],
            ["NSGA-II", "test", "zdt1", "200", "0", "0.1", "0.2"],
        ],
    )
    before = source.read_bytes()
    report, summaries = review.inspect_csv(source)
    full = next(row for row in summaries if row["n_evals"] == "50000")
    assert full["median_runtime_seconds"] == 15
    assert full["observations"] == 2
    assert full["median_hypervolume"] is None
    assert full["valid_hypervolume"] == 0
    assert len(summaries) == 2
    assert any("budgets" in warning for warning in report["warnings"])
    assert source.read_bytes() == before


@pytest.mark.parametrize("fault", ["duplicate", "nan", "negative", "malformed"])
def test_invalid_observations_fail_instead_of_producing_a_summary(tmp_path: Path, fault: str) -> None:
    source = tmp_path / "benchmark_paper.csv"
    row = ["NSGA-II", "test", "zdt1", "50000", "0", "10", "0.9"]
    rows = [row, row] if fault == "duplicate" else [row]
    if fault in {"nan", "negative"}:
        row[-2] = "nan" if fault == "nan" else "-1"
    elif fault == "malformed":
        row.append("unexpected")
    _csv(source, rows)
    with pytest.raises(ValueError):
        review.inspect_csv(source)


def test_missing_input_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        review.inspect_data(tmp_path)


def test_smoke_requires_source_revision_before_executing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(review.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Git clone"):
        review.run_smoke(ROOT, tmp_path)
    assert not (tmp_path / "run").exists()


def test_reviewer_outputs_refuse_collisions(tmp_path: Path) -> None:
    output = tmp_path / "review"
    output.mkdir()
    sentinel = output / "data-report.json"
    sentinel.write_text("keep this report", encoding="utf-8")
    assert review.main(["data", "--output-dir", str(output)]) == 1
    assert sentinel.read_text(encoding="utf-8") == "keep this report"
    assert list(output.iterdir()) == [sentinel]


def test_rebuild_stage_uses_sources_and_excludes_existing_generated_output(tmp_path: Path) -> None:
    output = tmp_path / "review"
    output.mkdir()
    workspace = review.stage_rebuild(ROOT, output)
    assert not (workspace / "paper" / "generated").exists()
    for name in review.DOCUMENTS:
        relative = Path("paper/manuscript") / f"{name}.tex"
        assert (workspace / relative).read_bytes() == (ROOT / relative).read_bytes()
    assert (workspace / "paper/manuscript/revision_marks.tex").is_file()
    assert (workspace / "paper/manuscript/figures/PROTOCOL.png").is_file()
    assert not (workspace / "paper/11_sync_overleaf_sources.py").exists()
    hashes = json.loads((output / "source-hashes.json").read_text(encoding="utf-8"))
    assert "experiments/benchmark_paper.csv" in hashes


@pytest.mark.smoke
def test_reviewer_smoke_saves_verifies_and_replays_from_another_cwd(tmp_path: Path) -> None:
    output = tmp_path / "review"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(
        [sys.executable, str(ROOT / "paper/review.py"), "smoke", "--output-dir", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((output / "smoke-report.json").read_text(encoding="utf-8"))
    assert report["verification"]["effective_replayability"] == "exact"
    assert report["replay"]["exact"] is True
    assert (output / "run/manifest.json").is_file()
    assert (output / "replay/result.npz").is_file()


def test_rebuild_child_failure_does_not_report_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "review"
    output.mkdir()

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(review.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        review.rebuild(ROOT, output, pdf=False)
    assert not (output / "rebuild-report.json").exists()
