from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "docs" / "topics" / "tuning.md"
EXAMPLE_PATH = ROOT / "examples" / "tuning" / "random_search_nsgaii.py"


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


def test_tuning_docs_match_current_contract() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    example = EXAMPLE_PATH.read_text(encoding="utf-8")

    assert "Experimental surface in VAMOS 1.0.0" in text
    assert "current default backend is `optuna`" in text
    assert "examples/tuning/random_search_nsgaii.py" in text
    assert "TuningTask.budget_per_run" in text
    assert "held-out problems and/or seeds" in text

    assert "return -hypervolume" not in text
    assert "racing` (default statistical racing flow)" not in text

    assert "from vamos import optimize" in example
    assert "from vamos.algorithms import NSGAIIConfig" in example
    assert "from vamos.engine.tuning import" in example
    assert "instances=[Instance(name=\"zdt1\", n_var=30), Instance(name=\"zdt2\", n_var=30)]" in example
    assert "seeds=[0, 1]" in example
    assert "budget_per_run=80" in example
    assert "maximize=False" in example
    assert "max_trials=2" in example


@pytest.mark.smoke
def test_tuning_docs_smoke_command(tmp_path: Path) -> None:
    source_path = "docs/topics/tuning.md"
    output_root = tmp_path / "tuning_docs"
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
        "docs_tuning_smoke",
    )
    assert proc.returncode == 0, f"{source_path}: {proc.stderr or proc.stdout}"
    summary_path = output_root / "docs_tuning_smoke" / "tuning_summary.json"
    assert summary_path.exists(), source_path
