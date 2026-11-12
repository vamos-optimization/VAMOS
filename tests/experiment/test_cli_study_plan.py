from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import vamos


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    return subprocess.run(
        [sys.executable, "-m", "vamos.experiment.cli.main", "study", *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def _write_spec(path: Path, **changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "problems": ["zdt1"],
        "algorithms": ["nsgaii"],
        "seeds": [0, 1],
        "max_evaluations": 24,
        "pop_size": 8,
    }
    value.update(changes)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def test_study_plan_help_is_side_effect_free(tmp_path: Path) -> None:
    before = set(tmp_path.iterdir())
    proc = _run("plan", "--help")

    assert proc.returncode == 0
    assert "vamos study plan" in proc.stdout
    assert "--output" in proc.stdout
    assert "--json" in proc.stdout
    assert set(tmp_path.iterdir()) == before


def test_top_level_help_lists_study_command() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "vamos.experiment.cli.main", "help"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0
    assert "vamos study" in proc.stdout


def test_pl_017_python_and_cli_json_are_semantically_equal(tmp_path: Path) -> None:
    config = tmp_path / "study.json"
    values = _write_spec(config)
    python_report = vamos.plan_study(vamos.StudySpec(**values))  # type: ignore[arg-type]

    proc = _run("plan", str(config), "--json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert proc.stderr.count("Warning:") == 1
    report = payload["payload"]
    assert payload["plan_id"] == python_report.plan_id
    assert report["task_ids"] == list(python_report.task_ids)
    assert report["task_count"] == python_report.task_count
    assert report["total_evaluation_budget"] == python_report.total_evaluation_budget
    assert report["components"]["backends"] == list(python_report.backend_ids)


def test_pl_018_json_success_is_one_stable_stdout_document(tmp_path: Path) -> None:
    config = tmp_path / "study.json"
    _write_spec(config, seeds=[7])

    first = _run("plan", str(config), "--json")
    second = _run("plan", str(config), "--json")

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr
    assert first.stderr.count("Warning:") == 1
    assert first.stdout == second.stdout
    assert first.stdout.count("\n") == 1
    payload = json.loads(first.stdout)
    assert payload["document_type"] == "vamos.study-command-result"
    assert payload["schema_version"] == "1.0.0"
    assert payload["status"] == "ready"
    assert payload["exit_code"] == 0
    assert payload["changed"] is False
    assert payload["payload"]["execution_occurred"] is False
    assert payload["payload"]["filesystem_write_occurred"] is False


def test_pl_018_json_collision_keeps_identities_and_exits_five(tmp_path: Path) -> None:
    config = tmp_path / "study.json"
    _write_spec(config, seeds=[7])
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    before = list(occupied.iterdir())

    proc = _run("plan", str(config), "--output", str(occupied), "--json")

    assert proc.returncode == 5
    assert proc.stderr.count("Warning:") == 2
    assert proc.stdout.count("\n") == 1
    payload = json.loads(proc.stdout)
    assert payload["status"] == "blocked"
    assert payload["payload"]["valid"] is True
    assert payload["plan_id"].startswith("sha256:")
    assert payload["payload"]["task_ids"]
    assert payload["payload"]["output"]["status"] == "empty_directory"
    assert payload["errors"][0]["reason"] == "OUTPUT_COLLISION"
    assert list(occupied.iterdir()) == before


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"problems": ["missing"], "algorithms": ["nsgaii"], "seeds": [0]}, "UNRESOLVED_TASK"),
        ({"problems": ["zdt1"], "algorithms": ["nsgaii"], "seeds": [0], "unknown": 1}, "UNKNOWN_FIELD"),
    ],
)
def test_pl_018_invalid_plan_is_one_structured_json_error(tmp_path: Path, payload: dict[str, object], reason: str) -> None:
    config = tmp_path / "study.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    proc = _run("plan", str(config), "--json")

    assert proc.returncode == 2
    assert proc.stderr == ""
    assert proc.stdout.count("\n") == 1
    envelope = json.loads(proc.stdout)
    assert envelope["document_type"] == "vamos.study-command-result"
    assert envelope["status"] == "error"
    assert envelope["exit_code"] == 2
    assert envelope["changed"] is False
    assert envelope["plan_id"] is None
    assert envelope["errors"][0]["reason"] == reason
    assert envelope["errors"][0]["execution_occurred"] is False


def test_invalid_human_plan_has_no_traceback(tmp_path: Path) -> None:
    config = tmp_path / "study.json"
    config.write_text("{not json", encoding="utf-8")

    proc = _run("plan", str(config))

    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "Error [study_spec]" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_human_plan_errors_show_field_and_value(tmp_path: Path) -> None:
    """The human output names the field and the value that the JSON error carries (VR-129)."""
    config = tmp_path / "study.json"
    _write_spec(config, algorithm_configs={"nsgaii": {"crossover": {"method": "nope"}}})

    proc = _run("plan", str(config))

    assert proc.returncode == 2
    assert "UNAVAILABLE_COMPONENT" in proc.stderr
    assert "$.matrix[0].operators.crossover" in proc.stderr
    assert "unavailable-crossover" in proc.stderr
