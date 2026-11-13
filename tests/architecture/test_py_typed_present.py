from __future__ import annotations

import zipfile
from pathlib import Path

from tests.architecture._build_tree import REPO_ROOT, build_from_copy, checkout_build_outputs

SRC_PY_TYPED = REPO_ROOT / "src" / "vamos" / "py.typed"


def test_py_typed_present(tmp_path: Path) -> None:
    if not SRC_PY_TYPED.exists():
        raise AssertionError(f"Missing py.typed marker: {SRC_PY_TYPED}")

    before = checkout_build_outputs()
    wheels = sorted(build_from_copy(tmp_path, "--wheel").glob("*.whl"))
    if not wheels:
        raise AssertionError(f"No wheel produced in {tmp_path}")
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
    if "vamos/py.typed" not in names:
        raise AssertionError("Wheel missing vamos/py.typed marker.")
    assert checkout_build_outputs() == before
