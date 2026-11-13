"""Every test file a CI workflow names exists, so removing a test cannot break a CI step."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("workflow", sorted((ROOT / ".github" / "workflows").glob("*.yml")), ids=lambda path: path.name)
def test_workflow_test_paths_exist(workflow: Path) -> None:
    named = set(re.findall(r"\btests/[\w/.-]+\.py\b", workflow.read_text(encoding="utf-8")))
    assert sorted(path for path in named if not (ROOT / path).is_file()) == []
