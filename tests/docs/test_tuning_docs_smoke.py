from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "docs" / "topics" / "tuning.md"


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

    assert "Experimental surface in VAMOS 1.0.0" in text
    assert "does **not** expose a curated public programmatic facade" in text
    assert "Maintained user workflows should therefore use\n`vamos tune`" in text
    assert "advanced evaluation and contributors" in text
    assert "defaults to **`optuna`**" in text
    assert "held-out problems and/or seeds" in text
    assert "global tuning seed **and** the base used to derive" in text
    assert "does not\nlet you provide an independent training-seed list" in text
    assert "--split-seed" in text
    assert "--validation-seeds" in text
    assert "--test-seeds" in text
    assert "--backend random" in text
    assert "--budget" in text
    assert "--tune-budget" in text

    assert "Keep the tuner seed separate" not in text
    assert "return -hypervolume" not in text
    assert "RandomSearchTuner(" not in text
    assert "from vamos.engine.tuning import" not in text
    assert "racing` (default statistical racing flow)" not in text
    assert "examples/tuning/random_search_nsgaii.py" not in text


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
