"""Build the distributions from a copy of the build inputs, so the checkout's build/, dist/ and egg-info stay untouched."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_inputs(root: Path) -> list[str]:
    # The project metadata, the files MANIFEST.in adds to the sdist, src/, and tests/: the release builds in a
    # full checkout, where setuptools adds tests/test*.py to the sdist unless MANIFEST.in prunes them.
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    included = [line.split(maxsplit=1)[1] for line in manifest if line.startswith("include ")]
    return ["pyproject.toml", "MANIFEST.in", *included, "src", "tests"]


def _copy_build_inputs(root: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for name in _build_inputs(root):
        source = root / name
        if source.is_dir():
            shutil.copytree(source, destination / name, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        else:
            shutil.copy2(source, destination / name)


def build_from_copy(workdir: Path, *args: str) -> Path:
    """Copy the checkout's build inputs into ``workdir``, run ``python -m build`` there and return the output directory.

    ``--no-isolation`` builds offline with the setuptools and wheel of the dev extras, as ``tools/release_check.py`` does.
    """
    tree = workdir / "tree"
    dist = workdir / "dist"
    _copy_build_inputs(REPO_ROOT, tree)
    cmd = [sys.executable, "-m", "build", "--no-isolation", *args, "--outdir", str(dist)]
    proc = subprocess.run(cmd, cwd=tree, capture_output=True, text=True)
    if proc.returncode != 0:
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        raise AssertionError(f"{' '.join(cmd[1:])} failed.\nExit code: {proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}\n")
    return dist


def checkout_build_outputs() -> dict[str, int]:
    """What a build in the checkout would write: ``build/``, ``dist/`` and ``src/*.egg-info``, with modification times."""
    roots = [REPO_ROOT / "build", REPO_ROOT / "dist", *(REPO_ROOT / "src").glob("*.egg-info")]
    return {
        path.relative_to(REPO_ROOT).as_posix(): path.stat().st_mtime_ns
        for root in roots
        if root.exists()
        for path in [root, *root.rglob("*")]
    }
