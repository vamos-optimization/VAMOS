"""The replay and multiprocessing guidance matches what replay and the evaluation backends do."""

from __future__ import annotations

import re
from pathlib import Path

from vamos.experiment.artifacts.component_support import BUILT_IN_OPERATORS
from vamos.experiment.artifacts.specs import _OPERATOR_ALIASES
from vamos.experiment.unified import optimize

ROOT = Path(__file__).resolve().parents[2]
RUN_ARTIFACTS = ROOT / "docs" / "guide" / "run-artifacts.md"
COOKBOOK = ROOT / "docs" / "guide" / "cookbook.md"


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end < 0 else text[start:end]


def test_run_artifacts_guide_lists_replayable_operators() -> None:
    section = _section(RUN_ARTIFACTS.read_text(encoding="utf-8"), "### Replayable operators")
    table = [line for line in section.splitlines() if line.startswith("| `")]
    documented = {name: stable for line in table for name, stable in [re.findall(r"`([^`]+)`", line)[:2]]}
    # Each configuration name replay accepts, with the stable name the manifest records.
    assert documented == _OPERATOR_ALIASES
    assert set(documented.values()) == set(BUILT_IN_OPERATORS)


def test_multiprocessing_guidance_names_the_main_guard_and_best_effort() -> None:
    guide = RUN_ARTIFACTS.read_text(encoding="utf-8")
    cookbook = _section(COOKBOOK.read_text(encoding="utf-8"), "## 11. Multiprocessing evaluation")
    docstring = optimize.__doc__ or ""

    assert 'if __name__ == "__main__":' in docstring
    assert 'if __name__ == "__main__":' in cookbook
    assert 'eval_strategy="multiprocessing"' in cookbook
    for text in (guide, cookbook):
        words = " ".join(text.split())
        assert "best_effort" in words
        assert "serially with the same seed" in words
