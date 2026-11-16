from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_compare_pymoo_help() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/benchmark_compare_pymoo.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.backends
def test_benchmark_compare_pymoo_smoke(tmp_path: Path) -> None:
    if importlib.util.find_spec("pymoo") is None:
        pytest.skip("pymoo not installed")
    if importlib.util.find_spec("moocore") is None:
        pytest.skip("moocore not installed")

    json_path = tmp_path / "comparison.json"
    md_path = tmp_path / "comparison.md"
    proc = subprocess.run(
        [
            sys.executable,
            "tools/benchmark_compare_pymoo.py",
            "--smoke",
            "--cases",
            "zdt1",
            "--algorithms",
            "nsgaii",
            "--seeds",
            "42",
            "--engine",
            "numpy",
            "--output",
            str(json_path),
            "--markdown",
            str(md_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]
    assert {row["framework"] for row in payload["summary"]} == {"vamos", "pymoo"}
    assert md_path.is_file()


def _report(path: Path, vamos_hv: list[float], pymoo_hv: list[float]) -> Path:
    runs = [
        {"case": "zdt1", "algorithm": "nsgaii", "framework": framework, "seed": seed, "hv": hv, "runtime_ms": 1.0}
        for framework, values in (("vamos", vamos_hv), ("pymoo", pymoo_hv))
        for seed, hv in zip((42, 1337, 2024), values)
    ]
    path.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    return path


def _check(report: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "tools/benchmark_compare_pymoo.py", "--check", str(report), *extra],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_compare_fails_when_vamos_hv_drops(tmp_path: Path) -> None:
    dropped = _check(_report(tmp_path / "dropped.json", [0.52, 0.53, 0.54], [0.66, 0.66, 0.67]), "--fail-below", "5")
    close = _check(_report(tmp_path / "close.json", [0.64, 0.65, 0.70], [0.66, 0.66, 0.67]), "--fail-below", "5")

    assert dropped.returncode == 1, dropped.stderr
    assert "zdt1" in dropped.stderr and "nsgaii" in dropped.stderr
    assert close.returncode == 0, close.stderr
