from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib

from vamos.foundation.version import __version__

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_VERSION = "1.0.1"
RELEASE_DATE = "2026-10-01"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_runtime_is_the_single_package_version_source() -> None:
    pyproject = tomllib.loads(_read("pyproject.toml"))

    assert __version__ == EXPECTED_VERSION
    assert "version" not in pyproject["project"]
    assert pyproject["project"]["dynamic"] == ["version"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "vamos.foundation.version.__version__"}
    assert "version_toml" not in pyproject["tool"]["semantic_release"]


def test_public_release_documents_match_runtime_version() -> None:
    citation = _read("CITATION.cff")
    changelog = _read("CHANGELOG.md")
    readme = _read("README.md")
    release_notes = _read(f"docs/project/release-notes-{EXPECTED_VERSION}.md")

    assert re.search(rf"(?m)^version:\s*{re.escape(EXPECTED_VERSION)}\s*$", citation)
    assert re.search(rf"(?m)^date-released:\s*{RELEASE_DATE}\s*$", citation)
    assert f"## [{EXPECTED_VERSION}] - {RELEASE_DATE}" in changelog
    assert f"version = {{{EXPECTED_VERSION}}}" in readme
    assert f"VAMOS {EXPECTED_VERSION} release notes" in release_notes
    assert f"Released {RELEASE_DATE}." in release_notes
