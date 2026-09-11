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

    assert "no metric selector" in text
    assert "hypervolume (HV) and maximizes that score" in text
    assert "final population" in text
    assert "removes\ninfeasible rows" in text
    assert "Pareto-filters" in text
    assert "does not score top-level `result.F`" in text
    assert "use_external_archive" in text
    assert "archive_unbounded" in text
    assert "archive_prune_policy" in text
    assert "removes `use_external_archive`, `archive_unbounded`, and" in text
    assert "fixed final-population score source" in text
    assert "Archive studies are separate from ordinary CLI tuning" in text

    assert "Constrained AGE-MOEA and RVEA" in text
    assert "does not currently support" in text
    assert "fails explicitly" in text
    assert "population-aligned constraint values" in text
    assert "including problems later assigned to validation or\n    test splits" in text

    assert "Persistent Optuna studies are scoring-contract versioned" in text
    assert "__vamos_cli_final_population_hv_v1" in text
    assert "must\nnot compete with trials scored under the current final-population HV contract" in text
    assert "Reusing the same base study name resumes only studies from this contract" in text

    assert "[10.0, ..., 10.0]" in text
    assert "--runtime-penalty" in text
    assert "--failure-score" in text
    assert "not a\n  universal failure policy" in text
    assert "should not be relied on to rescue an invalid HV scoring" in text
    assert "current maintained CLI cannot select it" in text

    assert "global tuning seed **and** the base used to derive" in text
    assert "does not\nlet you provide an independent training-seed list" in text
    assert "--split-seed" in text
    assert "--validation-seeds" in text
    assert "--test-seeds" in text

    assert "Racing has its own fidelity-budget schedule" in text
    assert "enables multi-fidelity racing by default" in text
    assert "**`1000,3000,10000` evaluations**" in text
    assert "they are not capped by\n`--budget`" in text
    assert "--no-multi-fidelity" in text
    assert "--fidelity-levels 1000,3000,5000" in text
    assert "Treat `--fidelity-levels`, rather than `--budget`, as the authoritative" in text

    assert "--backend random" in text
    assert "--budget" in text
    assert "--tune-budget" in text
    assert "--optuna-storage" in text
    assert "--optuna-study-name" in text

    assert "Current result-source limitation" not in text
    assert "does not guarantee source-consistent HV comparisons" not in text
    assert "Keep the tuner seed separate" not in text
    assert "IGD+ (lower is better) or HV" not in text
    assert "--failure-score` is the score assigned when an evaluation fails" not in text
    assert "`--budget` is the MOEA objective-evaluation budget for each candidate run" not in text
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
