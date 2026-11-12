from __future__ import annotations

import zipfile
from pathlib import Path

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


def test_build_smoke(tmp_path: Path) -> None:
    # The sdist and the wheel built from it; the checkout's build/, dist/ and egg-info stay as the developer left them.
    before = checkout_build_outputs()
    _assert_wheel_matches_source(build_from_copy(tmp_path))
    assert checkout_build_outputs() == before
