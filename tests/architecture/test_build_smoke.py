from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_inputs(root: Path) -> list[str]:
    # The project metadata, the files MANIFEST.in adds to the sdist, and src/.
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    included = [line.split(maxsplit=1)[1] for line in manifest if line.startswith("include ")]
    return ["pyproject.toml", "MANIFEST.in", *included, "src"]


def _copy_build_inputs(root: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for name in _build_inputs(root):
        source = root / name
        if source.is_dir():
            shutil.copytree(source, destination / name, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        else:
            shutil.copy2(source, destination / name)


def _checkout_build_outputs() -> dict[str, int]:
    # What a build in the checkout would write: build/, dist/ and src/*.egg-info.
    roots = [REPO_ROOT / "build", REPO_ROOT / "dist", *(REPO_ROOT / "src").glob("*.egg-info")]
    return {
        path.relative_to(REPO_ROOT).as_posix(): path.stat().st_mtime_ns
        for root in roots
        if root.exists()
        for path in [root, *root.rglob("*")]
    }


def _assert_wheel_matches_source(dist: Path) -> None:
    wheels = sorted(dist.glob("*.whl"))
    if not wheels:
        raise AssertionError(f"Build completed without producing a wheel under {dist}.")
    src_root = REPO_ROOT / "src"
    unexpected: list[str] = []
    with zipfile.ZipFile(wheels[-1]) as zf:
        for name in zf.namelist():
            if not name.startswith("vamos/") or not name.endswith(".py"):
                continue
            src_path = src_root / Path(name)
            if not src_path.is_file():
                unexpected.append(name)
    if unexpected:
        preview = "\n".join(f"- {name}" for name in unexpected[:20])
        if len(unexpected) > 20:
            preview += f"\n- ... and {len(unexpected) - 20} more"
        raise AssertionError(f"Wheel contains Python modules not present in src/vamos:\n{preview}")


def test_build_smoke(tmp_path: Path) -> None:
    # Build from a copy so the checkout's build/, dist/ and egg-info stay as the developer left them.
    # --no-isolation builds offline with the setuptools and wheel from the dev extras, as release_check.py does.
    before = _checkout_build_outputs()
    tree = tmp_path / "tree"
    dist = tmp_path / "dist"
    _copy_build_inputs(REPO_ROOT, tree)
    cmd = [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(dist)]
    proc = subprocess.run(cmd, cwd=tree, capture_output=True, text=True)
    if proc.returncode != 0:
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        raise AssertionError(f"python -m build failed.\nExit code: {proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}\n")
    _assert_wheel_matches_source(dist)
    assert _checkout_build_outputs() == before
