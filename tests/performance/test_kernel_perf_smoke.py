from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.smoke
def test_kernel_benchmark_harness_smoke(tmp_path: Path) -> None:
    output = tmp_path / "kernel_benchmarks.json"
    cmd = [
        sys.executable,
        "tools/benchmark_kernels.py",
        "--smoke",
        "--output",
        str(output),
    ]
    subprocess.run(cmd, check=True)

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["meta"]["smoke"] is True
    assert "polynomial_mutation.numpy" in data["benchmarks"]
    assert "tournament_selection.numpy" in data["benchmarks"]
    assert "sbx_crossover.numpy" in data["benchmarks"]
    assert "nsga2_ranking.numpy" in data["benchmarks"]
    assert "nsga2_survival.numpy" in data["benchmarks"]
    assert "archive_deduplication" in data["benchmarks"]
    assert "moead_neighborhood.python" in data["benchmarks"]

    for benchmark in data["benchmarks"].values():
        assert benchmark["timing"]["median_seconds"] >= 0.0
