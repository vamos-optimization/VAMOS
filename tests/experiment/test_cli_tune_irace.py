"""`vamos tune --backend irace`: parsing, availability, smoke profile and backend wiring.

The subprocess run needs R with irace >= 4.4.4: ``Rscript`` on PATH, or its path in
``VAMOS_IRACE_RSCRIPT``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vamos.engine.tuning._model_backend_irace import MIN_IRACE_VERSION
from vamos.experiment.cli import _tune_runtime
from vamos.experiment.cli import tune as tune_cli
from vamos.experiment.cli._tune_irace import SMOKE_TUNE_BUDGET, backend_availability, resolve_irace_work_dir


def test_parser_accepts_irace_and_types_options() -> None:
    args = tune_cli._parse_args(["--backend", "irace", "--irace-option", "nbIterations=3", "--irace-option", "elitist=false"])
    assert args.backend == "irace" and args.irace_rscript == "Rscript" and args.irace_instance_seeds == "task"
    assert args.irace_options == {"nbIterations": 3, "elitist": False}
    assert tune_cli._parse_args(["--irace-option", "testType=t-test"]).irace_options == {"testType": "t-test"}


@pytest.mark.parametrize(
    "argv",
    [
        ["--backend", "irace", "--timeout-seconds", "10"],
        ["--backend", "irace", "--fidelity-levels", "10,20"],
        ["--irace-option", "nbIterations"],
        ["--irace-option", "1x=2"],
        ["--irace-option", "seed=1", "--irace-option", "seed=2"],
    ],
)
def test_parser_rejects_settings_irace_cannot_honor(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        tune_cli._parse_args(argv)


def test_availability_follows_the_configured_rscript(tmp_path: Path) -> None:
    assert backend_availability(SimpleNamespace(irace_rscript=sys.executable))["irace"] is True
    assert backend_availability(SimpleNamespace(irace_rscript=str(tmp_path / "missing")))["irace"] is False


def test_list_backends_reports_irace(capsys: pytest.CaptureFixture[str]) -> None:
    assert tune_cli.main(["--list-backends", "--irace-rscript", sys.executable]) == 0
    assert "irace       : True" in capsys.readouterr().out


def test_smoke_profile_keeps_enough_irace_experiments() -> None:
    args = tune_cli._parse_args(["--backend", "irace", "--smoke"])
    profile = tune_cli._apply_smoke_mode(args)
    assert profile["tune_budget"] == SMOKE_TUNE_BUDGET and args.irace_options == {"nbIterations": 1}
    kept = tune_cli._parse_args(["--backend", "irace", "--irace-option", "nbIterations=2"])
    tune_cli._apply_smoke_mode(kept)
    assert kept.irace_options == {"nbIterations": 2}
    assert tune_cli._apply_smoke_mode(tune_cli._parse_args(["--backend", "random"]))["tune_budget"] == 4


def test_irace_work_dir_defaults_inside_the_output(tmp_path: Path) -> None:
    args = tune_cli._parse_args(["--backend", "irace"])
    resolve_irace_work_dir(args, tmp_path)
    assert args.irace_work_dir == str(tmp_path / "irace")
    explicit = tune_cli._parse_args(["--backend", "irace", "--irace-work-dir", "elsewhere"])
    resolve_irace_work_dir(explicit, tmp_path)
    assert explicit.irace_work_dir == "elsewhere"
    other = tune_cli._parse_args(["--backend", "random"])
    resolve_irace_work_dir(other, tmp_path)
    assert other.irace_work_dir == ""


def test_run_backend_passes_irace_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeTuner:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def run(self, eval_fn: object, verbose: bool = True) -> tuple[dict, list]:
            return {"x": 1}, []

    monkeypatch.setattr(_tune_runtime, "ModelBasedTuner", FakeTuner)
    args = tune_cli._parse_args(
        [
            "--backend",
            "irace",
            "--tune-budget",
            "300",
            "--irace-rscript",
            "R/bin/Rscript",
            "--irace-instance-seeds",
            "irace",
            "--irace-option",
            "firstTest=2",
        ]
    )
    args.irace_work_dir = "out/irace"
    task = SimpleNamespace()
    assert _tune_runtime.run_backend(args, task, lambda config, ctx: 0.0, 3) == ({"x": 1}, [])  # type: ignore[arg-type]
    assert captured["backend"] == "irace" and captured["n_jobs"] == 3 and captured["timeout_seconds"] is None
    assert (captured["irace_max_experiments"], captured["irace_rscript"], captured["irace_work_dir"]) == (300, "R/bin/Rscript", "out/irace")
    assert captured["irace_options"] == {"firstTest": 2} and captured["irace_instance_seeds"] == "irace"


def _rscript() -> str | None:
    rscript = os.environ.get("VAMOS_IRACE_RSCRIPT") or shutil.which("Rscript")
    if rscript is None:
        return None
    probe = subprocess.run(
        [rscript, "-e", 'cat(as.character(utils::packageVersion("irace")))'], capture_output=True, text=True, check=False
    )
    parts = probe.stdout.strip().split(".")[:3]
    if probe.returncode != 0 or not all(p.isdigit() for p in parts):
        return None
    return rscript if tuple(map(int, parts)) + (0,) * (3 - len(parts)) >= MIN_IRACE_VERSION else None


@pytest.mark.backends
@pytest.mark.slow
def test_vamos_tune_with_irace_smoke(tmp_path: Path) -> None:
    rscript = _rscript()
    if rscript is None:
        pytest.skip("R with irace >= 4.4.4 is not available.")
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0")
    argv = ["tune", "--problem", "zdt1", "--algorithm", "nsgaii", "--backend", "irace", "--backend-fallback", "error", "--smoke"]
    argv += ["--irace-rscript", rscript, "--output-dir", str(tmp_path), "--name", "irace_smoke"]
    proc = subprocess.run([sys.executable, "-m", "vamos.experiment.cli.main", *argv], capture_output=True, text=True, timeout=900, env=env)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    out = tmp_path / "irace_smoke"
    summary = json.loads((out / "tuning_summary.json").read_text(encoding="utf-8"))
    assert summary["backend_effective"] == "irace"
    assert (out / "irace" / "irace.Rdata").exists()
    assert "nbIterations = 1" in (out / "irace" / "scenario.txt").read_text(encoding="utf-8")
