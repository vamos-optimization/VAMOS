from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("filename", ["docs-preview.yml", "zensical-compat.yml"])
def test_release_tool_updates_trigger_documentation_validation(filename: str) -> None:
    workflow = yaml.load(
        (ROOT / ".github" / "workflows" / filename).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )

    # BaseLoader preserves GitHub's `on` key instead of YAML 1.1's boolean coercion.
    assert "release/requirements-tools.txt" in workflow["on"]["pull_request"]["paths"]
    assert "main" in workflow["on"]["pull_request"]["branches"]
    assert workflow["permissions"] == {"contents": "read"}
