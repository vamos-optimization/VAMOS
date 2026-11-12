from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUDGET_PATH = Path(__file__).with_name("ruff_format_budget.json")
REFORMAT_RE = re.compile(r"(\d+)\s+files?\s+would be reformatted")


def test_ruff_format_gate() -> None:
    cmd = [sys.executable, "-m", "ruff", "format", "--check", "--no-cache", "src/vamos", "tests"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode == 0:
        return

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if not BUDGET_PATH.exists():
        raise AssertionError(f"Missing ruff format budget file: {BUDGET_PATH}")
    budget = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    max_files = int(budget.get("max_files_to_reformat", 0))

    match = REFORMAT_RE.search(stdout)
    if match is not None:
        current = int(match.group(1))
        if current <= max_files:
            return
        raise AssertionError(
            "ruff format budget exceeded.\n"
            f"Current files to reformat: {current}\n"
            f"Budget: {max_files}\n"
            "Run `ruff format src/vamos tests` or reduce churn and update budget intentionally.\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}\n"
        )

    raise AssertionError(f"ruff format check failed.\nExit code: {proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}\n")
