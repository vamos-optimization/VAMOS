"""Tests for the canonical CLI subcommand dispatch."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import pytest

from vamos.engine.tuning import available_model_based_backends
from vamos.experiment.runtime.catalog import resolve_engine


def _run_vamos(*args: str, timeout: int = 30, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run `python -m vamos.experiment.cli.main <args>` as a subprocess."""
    cmd = [sys.executable, "-m", "vamos.experiment.cli.main", *args]
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(cmd, capture_output=True, timeout=timeout, env=merged_env)


def _decode(stream: bytes) -> str:
    return stream.decode("utf-8", errors="replace")


# ---- help ----


def test_help_subcommand_lists_all():
    proc = _run_vamos("help")
    assert proc.returncode == 0
    stdout = proc.stdout.decode()
    # Each consolidated command should appear in help output
    for name in ("quickstart", "create-problem", "summarize", "check", "bench", "studio", "zoo", "tune", "profile"):
        assert name in stdout, f"'{name}' not found in help output"


# ---- consolidated commands: import smoke ----


def test_check_dispatches():
    """vamos check should call self_check and exit 0."""
    # The full Windows extras environment records five tiny run environments;
    # process/package discovery can exceed the generic subprocess timeout.
    proc = _run_vamos("check", timeout=120)
    assert proc.returncode == 0, proc.stderr.decode()


@pytest.mark.cli
def test_main_runner_subprocess_smoke(tmp_path):
    output_root = tmp_path / "results"
    proc = _run_vamos(
        "--problem",
        "zdt1",
        "--algorithm",
        "nsgaii",
        "--engine",
        "numpy",
        "--population-size",
        "6",
        "--offspring-population-size",
        "6",
        "--max-evaluations",
        "12",
        "--seed",
        "7",
        "--output-root",
        str(output_root),
        "--no-preflight",
        timeout=120,
        env={"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"},
    )
    assert proc.returncode == 0, proc.stderr.decode()

    run_dir = output_root / "ZDT1" / "nsgaii" / "numpy" / "seed_7"
    assert {path.name for path in run_dir.iterdir()} == {"manifest.json", "result.npz", "environment.json"}


@pytest.mark.cli
def test_main_runner_accepts_engine_auto(tmp_path):
    output_root = tmp_path / "results"
    proc = _run_vamos(
        "--problem",
        "zdt1",
        "--algorithm",
        "nsgaii",
        "--engine",
        "auto",
        "--population-size",
        "6",
        "--offspring-population-size",
        "6",
        "--max-evaluations",
        "12",
        "--seed",
        "9",
        "--output-root",
        str(output_root),
        "--no-preflight",
        timeout=120,
        env={"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"},
    )
    assert proc.returncode == 0, proc.stderr.decode()

    resolved_engine = resolve_engine("auto", algorithm="nsgaii")
    run_dir = output_root / "ZDT1" / "nsgaii" / resolved_engine / "seed_9"
    assert (run_dir / "manifest.json").exists()


@pytest.mark.cli
def test_config_only_subprocess_smoke(tmp_path):
    output_root = tmp_path / "results"
    config_path = tmp_path / "spec.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "1",
                "defaults": {
                    "problem": "zdt1",
                    "algorithm": "nsgaii",
                    "engine": "auto",
                    "population_size": 6,
                    "offspring_population_size": 6,
                    "max_evaluations": 12,
                    "seed": 13,
                    "output_root": str(output_root),
                },
            }
        ),
        encoding="utf-8",
    )

    proc = _run_vamos(
        "--config",
        str(config_path),
        "--no-preflight",
        timeout=120,
        env={"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"},
    )
    assert proc.returncode == 0, proc.stderr.decode()

    resolved_engine = resolve_engine("auto", algorithm="nsgaii")
    run_dir = output_root / "ZDT1" / "nsgaii" / resolved_engine / "seed_13"
    assert {path.name for path in run_dir.iterdir()} == {"manifest.json", "result.npz", "environment.json"}


@pytest.mark.cli
def test_validate_config_subprocess_smoke(tmp_path):
    config_path = tmp_path / "spec.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "1",
                "defaults": {
                    "problem": "zdt1",
                    "algorithm": "nsgaii",
                    "engine": "numpy",
                    "population_size": 6,
                    "max_evaluations": 12,
                },
            }
        ),
        encoding="utf-8",
    )

    proc = _run_vamos("--config", str(config_path), "--validate-config")
    assert proc.returncode == 0, proc.stderr.decode()
    assert "Config OK." in proc.stderr.decode()


def test_bench_help():
    """vamos bench --help should work."""
    proc = _run_vamos("bench", "--help")
    assert proc.returncode == 0


@pytest.mark.cli
def test_bench_smoke_executes_minimal_suite(tmp_path):
    output_dir = tmp_path / "bench_report"
    proc = _run_vamos(
        "bench",
        "ZDT_small",
        "--algorithms",
        "nsgaii",
        "--output",
        str(output_dir),
        "--smoke",
        timeout=240,
        env={"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"},
    )
    assert proc.returncode == 0, proc.stderr.decode()

    summary_dir = output_dir / "summary"
    metrics_csv = summary_dir / "metrics.csv"
    tidy_csv = summary_dir / "metrics_tidy.csv"
    suite_json = summary_dir / "suite.json"
    assert metrics_csv.exists()
    assert tidy_csv.exists()
    assert suite_json.exists()

    suite_payload = json.loads(suite_json.read_text(encoding="utf-8"))
    assert len(suite_payload.get("experiments", [])) == 1
    assert suite_payload["experiments"][0]["problem"] == "zdt1"
    assert suite_payload["experiments"][0]["seeds"] == [0]
    assert suite_payload["config_overrides"]["population_size"] == 8

    rows = metrics_csv.read_text(encoding="utf-8").splitlines()
    assert len(rows) >= 2


@pytest.mark.cli
def test_profile_help():
    proc = _run_vamos("profile", "--help")
    assert proc.returncode == 0, proc.stderr.decode()
    assert "vamos profile" in proc.stdout.decode()


@pytest.mark.cli
def test_profile_smoke_executes_and_writes_csv(tmp_path):
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
        "--seed",
        "3",
        "--no-hv",
        "--output",
        str(output_csv),
        timeout=180,
        env={"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"},
    )
    assert proc.returncode == 0, proc.stderr.decode()
    assert output_csv.exists()
    content = output_csv.read_text(encoding="utf-8")
    assert "engine,time_seconds,time_min,time_max,repeats,n_solutions,hypervolume,error" in content
    assert "numpy" in content


@pytest.mark.cli
def test_zoo_help():
    proc = _run_vamos("zoo", "--help")
    assert proc.returncode == 0


@pytest.mark.cli
def test_zoo_list_and_info_smoke():
    list_proc = _run_vamos("zoo", "list")
    assert list_proc.returncode == 0, _decode(list_proc.stderr)
    list_output = _decode(list_proc.stderr) or _decode(list_proc.stdout)
    assert "zdt1" in list_output.lower()

    info_proc = _run_vamos("zoo", "info", "zdt1")
    assert info_proc.returncode == 0, _decode(info_proc.stderr)
    info_output = _decode(info_proc.stderr) or _decode(info_proc.stdout)
    assert "Name: zdt1" in info_output


@pytest.mark.cli
def test_zoo_run_smoke(tmp_path):
    output_root = tmp_path / "zoo_runs"
    proc = _run_vamos(
        "zoo",
        "run",
        "zdt1",
        "--algorithm",
        "nsgaii",
        "--engine",
        "auto",
        "--budget",
        "16",
        "--pop-size",
        "8",
        "--seed",
        "4",
        "--output",
        str(output_root),
        timeout=180,
        env={"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"},
    )
    assert proc.returncode == 0, proc.stderr.decode()
    resolved_engine = resolve_engine("auto", algorithm="nsgaii")
    run_dir = output_root / "ZDT1" / "nsgaii" / resolved_engine / "seed_4"
    assert {path.name for path in run_dir.iterdir()} == {"manifest.json", "result.npz", "environment.json"}


@pytest.mark.cli
def test_create_problem_subprocess_smoke(tmp_path):
    output_file = tmp_path / "custom_problem.py"
    proc = _run_vamos(
        "create-problem",
        "--yes",
        "--name",
        "custom problem",
        "--n-var",
        "3",
        "--n-obj",
        "2",
        "--budget",
        "250",
        "--output",
        str(output_file),
    )
    assert proc.returncode == 0, proc.stderr.decode()
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "Generated by: vamos create-problem" in content
    assert "def custom_problem(x):" in content


# ---- _SUBCOMMANDS dict ----


def test_subcommands_dict_matches_dispatch():
    """The _SUBCOMMANDS constant must list every command that _dispatch_subcommand handles."""
    from vamos.experiment.cli.main import _SUBCOMMANDS

    assert isinstance(_SUBCOMMANDS, dict)
    # At minimum, original + consolidated = 12 entries
    assert len(_SUBCOMMANDS) >= 12
    for name in ("quickstart", "create-problem", "summarize", "check", "bench", "studio", "zoo", "tune", "profile"):
        assert name in _SUBCOMMANDS, f"'{name}' missing from _SUBCOMMANDS"


def test_tune_smoke_with_suite_split(tmp_path):
    out_root = tmp_path / "tune_results"
    proc = _run_vamos(
        "tune",
        "--instances",
        "zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1",
        "--algorithm",
        "nsgaii",
        "--backend",
        "random",
        "--split-strategy",
        "suite_stratified",
        "--budget",
        "30",
        "--tune-budget",
        "4",
        "--n-seeds",
        "2",
        "--n-jobs",
        "1",
        "--no-run-validation",
        "--no-run-test",
        "--no-run-statistical-finisher",
        "--output-dir",
        str(out_root),
        "--name",
        "cli_smoke_tune",
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode()

    run_dir = out_root / "cli_smoke_tune"
    split_path = run_dir / "split_instances.csv"
    summary_path = run_dir / "tuning_summary.json"
    assert split_path.exists()
    assert summary_path.exists()

    with split_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert rows
    assert "suite" in rows[0]
    assert any(row.get("split") == "train" for row in rows)
    assert any(row.get("split") == "validation" for row in rows)
    assert any(row.get("split") == "test" for row in rows)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary.get("backend_requested") == "random"
    assert summary.get("backend_effective") == "random"
    split_info = summary.get("split", {})
    assert split_info.get("split_strategy") == "suite_stratified"


@pytest.mark.cli
def test_tune_smoke_flag_subprocess(tmp_path):
    out_root = tmp_path / "tune_results"
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
        str(out_root),
        "--name",
        "cli_flag_smoke_tune",
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode()

    run_dir = out_root / "cli_flag_smoke_tune"
    summary_path = run_dir / "tuning_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary.get("smoke_mode") is True
    assert summary.get("backend_effective") == "random"
    smoke_profile = summary.get("smoke_profile", {})
    assert smoke_profile.get("budget") == 30
    assert smoke_profile.get("tune_budget") == 4
    assert smoke_profile.get("run_validation") is False
    assert smoke_profile.get("run_test") is False
    assert smoke_profile.get("run_statistical_finisher") is False
    split_info = summary.get("split", {})
    assert split_info.get("split_strategy") == "suite_stratified"


def test_tune_backend_fallback_subprocess(tmp_path):
    availability = available_model_based_backends()
    unavailable = [name for name, ok in availability.items() if not bool(ok)]
    if not unavailable:
        pytest.skip("All model-based backends are available in this environment.")

    requested = unavailable[0]
    out_root = tmp_path / "tune_results"
    proc = _run_vamos(
        "tune",
        "--instances",
        "zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1",
        "--algorithm",
        "nsgaii",
        "--backend",
        requested,
        "--backend-fallback",
        "random",
        "--split-strategy",
        "suite_stratified",
        "--budget",
        "30",
        "--tune-budget",
        "4",
        "--n-seeds",
        "2",
        "--n-jobs",
        "1",
        "--no-run-validation",
        "--no-run-test",
        "--no-run-statistical-finisher",
        "--output-dir",
        str(out_root),
        "--name",
        "cli_fallback_tune",
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode()

    run_dir = out_root / "cli_fallback_tune"
    summary_path = run_dir / "tuning_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary.get("backend_requested") == requested
    assert summary.get("backend_effective") == "random"
