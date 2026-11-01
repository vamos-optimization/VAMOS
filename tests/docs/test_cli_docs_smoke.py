from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_vamos(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"})
    return subprocess.run(
        [sys.executable, "-m", "vamos.experiment.cli.main", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


@pytest.mark.smoke
def test_cli_guide_config_validation_command_smoke(tmp_path: Path) -> None:
    source_path = "docs/guide/cli.md"
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "1",
                "defaults": {
                    "problem": "zdt1",
                    "algorithm": "nsgaii",
                    "engine": "numpy",
                    "population_size": 8,
                    "max_evaluations": 16,
                    "output_root": str(tmp_path / "results"),
                },
            }
        ),
        encoding="utf-8",
    )

    proc = _run_vamos("--config", str(config_path), "--validate-config")
    assert proc.returncode == 0, f"{source_path}: {proc.stderr or proc.stdout}"
    assert "Config OK." in proc.stderr, source_path


@pytest.mark.smoke
def test_cli_guide_config_run_and_summarize_command_smoke(tmp_path: Path) -> None:
    source_path = "docs/guide/cli.md"
    output_root = tmp_path / "results"
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "1",
                "defaults": {
                    "problem": "zdt1",
                    "algorithm": "nsgaii",
                    "engine": "auto",
                    "population_size": 8,
                    "offspring_population_size": 8,
                    "max_evaluations": 16,
                    "seed": 11,
                    "output_root": str(output_root),
                },
            }
        ),
        encoding="utf-8",
    )

    run_proc = _run_vamos("--config", str(config_path), "--no-preflight")
    assert run_proc.returncode == 0, f"{source_path}: {run_proc.stderr or run_proc.stdout}"
    assert any(output_root.rglob("manifest.json")), source_path

    summarize_proc = _run_vamos("summarize", "--results", str(output_root), "--latest")
    assert summarize_proc.returncode == 0, f"{source_path}: {summarize_proc.stderr or summarize_proc.stdout}"
    assert "zdt1" in summarize_proc.stdout.lower(), source_path
    assert "nsgaii" in summarize_proc.stdout.lower(), source_path


@pytest.mark.smoke
def test_cli_guide_profile_command_smoke(tmp_path: Path) -> None:
    source_path = "docs/guide/cli.md"
    output_csv = tmp_path / "profile.csv"

    proc = _run_vamos(
        "profile",
        "--problem",
        "zdt1",
        "--engines",
        "numpy",
        "--budget",
        "200",
        "--repeats",
        "1",
        "--no-hv",
        "--output",
        str(output_csv),
    )
    assert proc.returncode == 0, f"{source_path}: {proc.stderr or proc.stdout}"
    assert output_csv.exists(), source_path


@pytest.mark.smoke
def test_cli_guide_zoo_commands_smoke(tmp_path: Path) -> None:
    source_path = "docs/guide/cli.md"
    list_proc = _run_vamos("zoo", "list")
    assert list_proc.returncode == 0, f"{source_path}: {list_proc.stderr or list_proc.stdout}"
    assert "zdt1" in (list_proc.stderr + list_proc.stdout).lower(), source_path

    info_proc = _run_vamos("zoo", "info", "zdt1")
    assert info_proc.returncode == 0, f"{source_path}: {info_proc.stderr or info_proc.stdout}"
    assert "name: zdt1" in (info_proc.stderr + info_proc.stdout).lower(), source_path

    output_root = tmp_path / "zoo_runs"
    run_proc = _run_vamos(
        "zoo",
        "run",
        "zdt1",
        "--algorithm",
        "nsgaii",
        "--budget",
        "16",
        "--pop-size",
        "8",
        "--seed",
        "5",
        "--output",
        str(output_root),
    )
    assert run_proc.returncode == 0, f"{source_path}: {run_proc.stderr or run_proc.stdout}"
    assert any(output_root.rglob("manifest.json")), source_path


@pytest.mark.smoke
def test_cli_guide_study_plan_is_read_only_json(tmp_path: Path) -> None:
    source_path = "docs/guide/cli.md"
    config_path = tmp_path / "study.json"
    output = tmp_path / "studies" / "comparison-01"
    config_path.write_text(
        json.dumps(
            {
                "problems": ["zdt1", "zdt2"],
                "algorithms": ["nsgaii"],
                "seeds": [0, 1],
                "max_evaluations": 24,
                "pop_size": 8,
            }
        ),
        encoding="utf-8",
    )

    proc = _run_vamos("study", "plan", str(config_path), "--output", str(output), "--json")

    assert proc.returncode == 0, f"{source_path}: {proc.stderr or proc.stdout}"
    payload = json.loads(proc.stdout)
    assert payload["document_type"] == "vamos.study-command-result", source_path
    assert payload["operation"] == "plan", source_path
    assert payload["payload"]["task_count"] == 4, source_path
    assert payload["payload"]["output"]["status"] == "available", source_path
    assert payload["changed"] is False, source_path
    assert payload["payload"]["execution_occurred"] is False, source_path
    assert payload["payload"]["filesystem_write_occurred"] is False, source_path
    assert not output.parent.exists(), source_path


@pytest.mark.smoke
def test_cli_guide_tune_smoke_command_smoke(tmp_path: Path) -> None:
    source_path = "docs/guide/cli.md"
    output_root = tmp_path / "tuning"
    proc = _run_vamos(
        "tune",
        "--instances",
        "zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1",
        "--algorithm",
        "nsgaii",
        "--backend",
        "random",
        "--smoke",
        "--output-dir",
        str(output_root),
        "--name",
        "docs_cli_tune_smoke",
        timeout=180,
    )
    assert proc.returncode == 0, f"{source_path}: {proc.stderr or proc.stdout}"
    assert (output_root / "docs_cli_tune_smoke" / "tuning_summary.json").exists(), source_path


@pytest.mark.smoke
def test_cli_guide_both_with_external_baselines_smoke(tmp_path: Path) -> None:
    from vamos.engine.algorithm.catalog import ENABLED_ALGORITHMS

    source_path = "docs/guide/cli.md"
    output_root = tmp_path / "results"
    proc = _run_vamos(
        "--problem",
        "zdt1",
        "--algorithm",
        "both",
        "--include-external",
        "--external-problem-source",
        "native",
        "--max-evaluations",
        "200",
        "--seed",
        "2",
        "--output-root",
        str(output_root),
        timeout=300,
    )
    assert proc.returncode == 0, f"{source_path}: {proc.stderr or proc.stdout}"
    assert len(list(output_root.rglob("manifest.json"))) == len(ENABLED_ALGORITHMS), source_path
