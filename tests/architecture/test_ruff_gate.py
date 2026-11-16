from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ruff_targets_the_oldest_supported_python() -> None:
    # The gate then flags version checks and imports that only an unsupported Python needed (UP035, UP036).
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    oldest = project["project"]["requires-python"].removeprefix(">=")

    assert project["tool"]["ruff"]["target-version"] == "py" + oldest.replace(".", "")


def test_ruff_gate() -> None:
    cmd = [sys.executable, "-m", "ruff", "check", "--no-cache", "src/vamos", "tests"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode == 0:
        return

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    raise AssertionError(f"ruff check failed.\nExit code: {proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}\n")
