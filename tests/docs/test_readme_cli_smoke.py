from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_vamos(*args: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
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
def test_readme_quickstart_template_list_command_smoke() -> None:
    source_path = "README.md"
    proc = _run_vamos("quickstart", "--template", "list", "--yes")
    assert proc.returncode == 0, f"{source_path}: {proc.stderr or proc.stdout}"
    output = (proc.stderr + proc.stdout).lower()
    assert "demo" in output, source_path


@pytest.mark.smoke
def test_readme_create_problem_command_smoke(tmp_path: Path) -> None:
    source_path = "README.md"
    output_file = tmp_path / "readme_problem.py"
    proc = _run_vamos("create-problem", "--yes", "--output", str(output_file))
    assert proc.returncode == 0, f"{source_path}: {proc.stderr or proc.stdout}"
    assert output_file.exists(), source_path


@pytest.mark.smoke
def test_readme_profile_command_smoke(tmp_path: Path) -> None:
    source_path = "README.md"
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
def test_readme_bench_smoke_command_smoke(tmp_path: Path) -> None:
    source_path = "README.md"
    output_dir = tmp_path / "report"
    proc = _run_vamos(
        "bench",
        "ZDT_small",
        "--algorithms",
        "nsgaii",
        "--output",
        str(output_dir),
        "--smoke",
        timeout=240,
    )
    assert proc.returncode == 0, f"{source_path}: {proc.stderr or proc.stdout}"
    assert (output_dir / "summary" / "metrics.csv").exists(), source_path


@pytest.mark.smoke
def test_readme_tune_smoke_command_smoke(tmp_path: Path) -> None:
    source_path = "README.md"
    output_dir = tmp_path / "tuning_smoke"
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
        str(output_dir),
        "--name",
        "readme_tune_smoke",
        timeout=180,
    )
    assert proc.returncode == 0, f"{source_path}: {proc.stderr or proc.stdout}"
    assert (output_dir / "readme_tune_smoke" / "tuning_summary.json").exists(), source_path
