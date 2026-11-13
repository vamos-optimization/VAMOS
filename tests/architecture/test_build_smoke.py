from __future__ import annotations

import fnmatch
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib

from tests.architecture._build_tree import REPO_ROOT, build_from_copy, checkout_build_outputs


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


def _assert_sdist_has_no_tests(dist: Path) -> None:
    sdists = sorted(dist.glob("*.tar.gz"))
    if not sdists:
        raise AssertionError(f"Build completed without producing an sdist under {dist}.")
    with tarfile.open(sdists[-1]) as archive:
        # Member names start with the "<name>-<version>/" directory.
        names = [name.partition("/")[2] for name in archive.getnames()]
    tests = sorted(name for name in names if name == "tests" or name.startswith("tests/"))
    if tests:
        raise AssertionError(f"The sdist ships {len(tests)} files of the test suite without the rest of it: {tests[:5]}")


def test_build_smoke(tmp_path: Path) -> None:
    # The sdist and the wheel built from it; the checkout's build/, dist/ and egg-info stay as the developer left them.
    before = checkout_build_outputs()
    dist = build_from_copy(tmp_path)
    _assert_wheel_matches_source(dist)
    _assert_sdist_has_no_tests(dist)
    assert checkout_build_outputs() == before


def test_src_holds_only_code_agent_docs_and_shipped_data() -> None:
    # Any other file under src/vamos ships in no distribution, so it is left over, such as a script's output.
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data: dict[str, list[str]] = pyproject["tool"]["setuptools"]["package-data"]
    tracked = subprocess.run(
        ["git", "ls-files", "src/vamos"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()

    def shipped(path: PurePosixPath) -> bool:
        for package, patterns in package_data.items():
            package_dir = PurePosixPath("src", *package.split("."))
            if package_dir in path.parents:
                relative = path.relative_to(package_dir).as_posix()
                if any(fnmatch.fnmatch(relative, pattern) for pattern in patterns):
                    return True
        return False

    stray = [name for name in tracked if not (name.endswith(".py") or name.endswith("/AGENTS.md") or shipped(PurePosixPath(name)))]
    assert stray == [], f"Files under src/vamos that are neither code, agent documentation nor package data: {stray}"
