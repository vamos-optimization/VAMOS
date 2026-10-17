from __future__ import annotations

import copy
import csv
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vamos import create_study, load_result, load_run, load_study, verify_run

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/assets/studies"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return runpy.run_path(str(ROOT / "examples/tutorials/study_workflow.py"))


@pytest.fixture(scope="module")
def measured(workflow: dict, tmp_path_factory: pytest.TempPathFactory) -> tuple:
    pytest.importorskip("moocore")
    root = tmp_path_factory.mktemp("study-lessons")
    paths = [root / f"study-{budget}" for budget in (80, 160)]
    for budget, path in zip((80, 160), paths):
        workflow["plan_or_execute_study"](path, budget=budget, execute=True)
    dataset = workflow["measure_studies"](paths, root / "metrics")
    return root, paths, dataset


def test_plan_is_read_only_and_execution_refuses_existing_evidence(workflow: dict, tmp_path: Path) -> None:
    output = tmp_path / "study"
    plan = workflow["plan_or_execute_study"](output, budget=80)
    assert plan.status == "ready"
    assert plan.task_count == 8
    assert plan.total_evaluation_budget == 640
    assert not output.exists()
    output.mkdir()
    evidence = output / "keep.txt"
    evidence.write_text("existing evidence", encoding="utf-8")
    with pytest.raises(FileExistsError):
        workflow["plan_or_execute_study"](output, budget=80, execute=True)
    assert evidence.read_text(encoding="utf-8") == "existing evidence"


@pytest.mark.parametrize("budget", [0, 19, 81])
def test_budget_must_be_a_whole_number_of_populations(workflow: dict, budget: int) -> None:
    with pytest.raises(ValueError, match="budget"):
        workflow["study_spec"](budget=budget)


def test_all_downloads_run_standalone(tmp_path: Path) -> None:
    pytest.importorskip("moocore")
    pytest.importorskip("matplotlib")
    scripts = ("t06-fair-study.py", "t07-quality-and-budget.py", "t08-study-report.py")
    for script in scripts:
        (tmp_path / script).write_bytes((ASSETS / script).read_bytes())

    def run(index: int, *args: str) -> str:
        result = subprocess.run(
            [sys.executable, scripts[index], *args], cwd=tmp_path, capture_output=True, text=True, timeout=60, check=True
        )
        return result.stdout

    assert "Plan only" in run(0, "--output", "study", "--budget", "80")
    assert not (tmp_path / "study").exists()
    assert "Succeeded tasks: 8/8" in run(0, "--output", "study", "--budget", "80", "--run")
    assert "Measured successful rows: 8" in run(1, "--studies", "study", "--output", "metrics")
    assert "Report observations: 8" in run(2, "--measurements", "metrics/measurements.json", "--output", "report")
    assert {p.name for p in (tmp_path / "report").iterdir()} == {
        "metrics.csv",
        "summary.csv",
        "report.md",
        "descriptive-ranks.json",
        "budget-comparison.svg",
        "final-distributions.svg",
    }


def test_measurements_trace_to_verified_runs_and_actual_objectives(measured: tuple) -> None:
    _, paths, dataset = measured
    studies = {load_study(path).study_id: path for path in paths}
    assert len(dataset["rows"]) == 16
    assert sum(row["evaluations"] for row in dataset["rows"]) == 8 * (80 + 160)
    for row in dataset["rows"]:
        path = studies[row["study_id"]] / Path(row["run_manifest_path"]).parent
        verified = verify_run(path)
        assert verified.run_id == row["selected_run_id"]
        assert row["run_manifest_sha256"] == load_run(path).manifest["integrity"]["manifest_sha256"]
        result = load_result(path)
        np.testing.assert_array_equal(row["front"], result.front())
        x = result.X
        g = 1 + 9 * x[:, 1:].mean(axis=1)
        second = g * (1 - np.sqrt(x[:, 0] / g)) if row["problem"] == "zdt1" else g * (1 - (x[:, 0] / g) ** 2)
        np.testing.assert_allclose(result.F, np.column_stack([x[:, 0], second]))


def test_metric_reference_and_direction_have_known_values(workflow: dict) -> None:
    pytest.importorskip("moocore")
    good = workflow["measure_front"]("zdt1", [[0.0, 0.0]])
    worse = workflow["measure_front"]("zdt1", [[0.5, 1.0]])
    assert good["hv"] == pytest.approx(1.1 * 11)
    assert worse["hv"] == pytest.approx((1.1 - 0.5) * (11 - 1))
    assert good["igd_plus"] == pytest.approx(0)
    assert worse["igd_plus"] > good["igd_plus"]
    with pytest.raises(ValueError, match="reference box"):
        workflow["measure_front"]("zdt1", [[0, 12]])
    with pytest.raises(ValueError, match="finite"):
        workflow["measure_front"]("zdt1", [[0, np.nan]])


def test_duplicate_or_incompatible_studies_are_not_pooled(workflow: dict, measured: tuple, tmp_path: Path) -> None:
    _, paths, _ = measured
    with pytest.raises(ValueError, match="incompatible or duplicate"):
        workflow["measure_studies"]([paths[0], paths[0]], tmp_path / "duplicate")
    other = tmp_path / "other"
    create_study(workflow["study_spec"]("recorded", 80), output=other)
    with pytest.raises(ValueError, match="incompatible or duplicate"):
        workflow["measure_studies"]([paths[0], other], tmp_path / "mixed")
    assert not (tmp_path / "duplicate").exists()
    assert not (tmp_path / "mixed").exists()


def test_unfinished_tasks_remain_visible_and_block_reporting(workflow: dict, tmp_path: Path) -> None:
    pytest.importorskip("moocore")
    study = tmp_path / "pending"
    create_study(workflow["study_spec"](budget=80), output=study)
    dataset = workflow["measure_studies"]([study], tmp_path / "metrics")
    assert len(dataset["rows"]) == 8
    assert all(row["hv"] is None and row["igd_plus"] is None and row["front"] is None for row in dataset["rows"])
    with pytest.raises(ValueError, match="incomplete studies"):
        workflow["render_report"](tmp_path / "metrics/measurements.json", tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("damage", ["missing", "duplicate", "metric", "front", "budget", "failed"])
def test_report_rejects_missing_or_changed_evidence(workflow: dict, measured: tuple, tmp_path: Path, damage: str) -> None:
    dataset = copy.deepcopy(measured[2])
    row = dataset["rows"][0]
    if damage == "missing":
        dataset["rows"].pop()
    elif damage == "duplicate":
        dataset["rows"].append(copy.deepcopy(row))
    elif damage == "metric":
        row["hv"] += 1
    elif damage == "front":
        row["front"][0][0] += 0.01
    elif damage == "budget":
        row["evaluations"] -= 1
    else:
        row["state"] = "failed"
    source = tmp_path / "changed.json"
    source.write_text(json.dumps(dataset), encoding="utf-8")
    with pytest.raises(ValueError):
        workflow["render_report"](source, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_report_uses_saved_observations_without_optimization(workflow: dict, measured: tuple, tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")
    import vamos

    def forbidden(*args, **kwargs):
        pytest.fail("A report must not rerun optimization")

    monkeypatch.setattr(vamos, "optimize", forbidden)
    monkeypatch.setattr(vamos, "create_study", forbidden)
    root, _, dataset = measured
    output = tmp_path / "report"
    summary = workflow["render_report"](root / "metrics/measurements.json", output)
    assert len(summary["groups"]) == 8
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 16
    assert {r["selected_run_id"] for r in table} == {r["selected_run_id"] for r in dataset["rows"]}
    text = (output / "report.md").read_text(encoding="utf-8")
    assert "What these observations show" in text
    assert "Source SHA-256" in text
    assert "No p-value" in text
    with pytest.raises(FileExistsError):
        workflow["render_report"](root / "metrics/measurements.json", output)


def test_retained_report_recomputes_from_all_120_observations(workflow: dict) -> None:
    pytest.importorskip("moocore")
    dataset = json.loads((ASSETS / "recorded-measurements.json").read_text(encoding="utf-8"))
    summary = workflow["summarize_measurements"](dataset)
    assert len(dataset["rows"]) == 120
    assert sum(r["evaluations"] for r in dataset["rows"]) == 120000
    assert {r["seed"] for r in dataset["rows"]} == set(range(10))
    assert {r["evaluation_budget"] for r in dataset["rows"]} == {200, 800, 2000}
    assert all(g["n"] == 10 for g in summary["groups"])
    with (ASSETS / "summary.csv").open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 12
    for row, group in zip(table, summary["groups"]):
        assert (row["problem"], row["algorithm"], int(row["budget"])) == (group["problem"], group["algorithm"], group["budget"])
        assert float(row["hv_median"]) == pytest.approx(group["hv_median"], abs=1e-12)
        assert float(row["igd_plus_median"]) == pytest.approx(group["igd_plus_median"], abs=1e-12)
    scores = np.array(summary["scores"])
    ranks = np.array(summary["ranks"])
    np.testing.assert_array_equal(np.argmin(ranks, axis=1), np.argmax(scores, axis=1))


def test_study_tutorial_generated_content_is_synchronized() -> None:
    subprocess.run([sys.executable, str(ROOT / "tools/build_study_tutorials.py"), "--check"], cwd=ROOT, check=True, timeout=30)
