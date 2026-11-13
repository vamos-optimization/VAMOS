"""Importing a public module loads only the base dependencies that pyproject.toml declares.

Each module is imported in a fresh interpreter. A finder at the front of ``sys.meta_path``
records which module first imports each top-level package, so a package that a base
dependency imports on its own (NumPy loads ``charset_normalizer`` when it is installed)
is not charged to VAMOS.
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.util import find_spec

import pytest

from tests.architecture._dependencies import base_distributions, import_names

# Known departures from "importing core VAMOS must not import optional packages"
# (src/vamos/foundation/kernel/AGENTS.md). They show only where moocore is installed.
_MOOCORE = find_spec("moocore") is not None
_EAGER_FACADE = pytest.mark.xfail(
    _MOOCORE,
    strict=True,
    reason="vamos/__init__.py imports the whole package, and with it moocore; resolving the top-level facade lazily removes this",
)
_GUARDED_MOOCORE = pytest.mark.xfail(
    _MOOCORE,
    strict=True,
    reason=(
        "engine/algorithm/components/archive_core.py, components/subset_selection.py and "
        "foundation/quality_indicators/moocore_indicators.py import moocore at module level inside try/except"
    ),
)

PUBLIC_MODULES = (
    pytest.param("vamos", marks=_EAGER_FACADE),
    pytest.param("vamos.api", marks=_EAGER_FACADE),
    pytest.param("vamos.algorithms", marks=_GUARDED_MOOCORE),
    pytest.param("vamos.problems", marks=_EAGER_FACADE),
    pytest.param("vamos.run_artifacts", marks=_GUARDED_MOOCORE),
    pytest.param("vamos.study_artifacts", marks=_GUARDED_MOOCORE),
    pytest.param("vamos.exceptions", marks=_EAGER_FACADE),
    pytest.param("vamos.selection", marks=_EAGER_FACADE),
    pytest.param("vamos.ux.api", marks=_GUARDED_MOOCORE),
)

_PROBE = """
import json
import sys


class _FirstImporter:
    def __init__(self):
        self.importers = {}

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition(".")[0]
        if root not in self.importers and root not in sys.stdlib_module_names:
            frame = sys._getframe(1)
            while frame is not None and frame.f_globals.get("__name__", "").startswith(("importlib", "_frozen_importlib")):
                frame = frame.f_back
            self.importers[root] = "" if frame is None else frame.f_globals.get("__name__", "")
        return None


recorder = _FirstImporter()
sys.meta_path.insert(0, recorder)
__import__(sys.argv[1])
print(json.dumps({root: importer for root, importer in recorder.importers.items() if root in sys.modules}))
"""


def _third_party_imported_by_vamos(module: str) -> dict[str, str]:
    result = subprocess.run([sys.executable, "-c", _PROBE, module], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"import {module} failed:\n{result.stderr}"
    loaded: dict[str, str] = json.loads(result.stdout.splitlines()[-1])
    return {root: importer for root, importer in loaded.items() if root != "vamos" and importer.partition(".")[0] == "vamos"}


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_import_vamos_loads_only_base_dependencies(module: str) -> None:
    allowed = {name.partition(".")[0] for distribution in base_distributions() for name in import_names(distribution)}
    extra = {root: importer for root, importer in _third_party_imported_by_vamos(module).items() if root not in allowed}
    assert not extra, f"import {module} loads packages outside [project].dependencies: " + ", ".join(
        f"{root} (imported by {importer})" for root, importer in sorted(extra.items())
    )
