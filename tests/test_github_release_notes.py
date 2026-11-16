"""The GitHub Release body: the release notes with every link absolute."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from github_release_notes import absolute_link, github_release_notes  # noqa: E402

SITE = "https://vamos-optimization.org/docs/1.3.0"


@pytest.mark.parametrize(
    "target, expected",
    [
        ("#which-results-change", f"{SITE}/project/release-notes-1.3.0/#which-results-change"),
        ("release-notes-1.2.1.md", f"{SITE}/project/release-notes-1.2.1/"),
        ("stability-and-versioning.md#scientific-corrections", f"{SITE}/project/stability-and-versioning/#scientific-corrections"),
        ("../topics/tuning.md#conditional-parameters", f"{SITE}/topics/tuning/#conditional-parameters"),
        ("../release_smoke.md", f"{SITE}/release_smoke/"),
        ("../index.md", f"{SITE}/"),
        ("../assets/vamos-mark.svg", f"{SITE}/assets/vamos-mark.svg"),
        ("../../CHANGELOG.md", "https://github.com/vamos-optimization/VAMOS/blob/v1.3.0/CHANGELOG.md"),
        ("https://pypi.org/project/vamos-optimization/", "https://pypi.org/project/vamos-optimization/"),
        ("mailto:someone@example.org", "mailto:someone@example.org"),
    ],
)
def test_links_point_to_the_versions_documentation_or_its_tag(target: str, expected: str) -> None:
    assert absolute_link(target, "1.3.0") == expected


def test_a_link_out_of_the_repository_is_an_error() -> None:
    with pytest.raises(ValueError, match="leaves the repository"):
        absolute_link("../../../elsewhere.md", "1.3.0")


def test_code_keeps_its_text_and_links_with_code_in_their_text_are_rewritten() -> None:
    notes = (
        "See [`vamos tune`](../topics/tuning.md) and [notes](#top).\n"
        "Write `[x](y.md)` to link.\n"
        "```markdown\n"
        "[kept](kept.md)\n"
        "```\n"
        "[ref]: ../reference/problems.md\n"
    )

    converted = github_release_notes(notes, "1.3.0")

    assert converted == (
        f"See [`vamos tune`]({SITE}/topics/tuning/) and [notes]({SITE}/project/release-notes-1.3.0/#top).\n"
        "Write `[x](y.md)` to link.\n"
        "```markdown\n"
        "[kept](kept.md)\n"
        "```\n"
        f"[ref]: {SITE}/reference/problems/\n"
    )


@pytest.mark.parametrize("notes", sorted((ROOT / "docs" / "project").glob("release-notes-*.md")), ids=lambda path: path.stem)
def test_every_release_body_has_only_absolute_links(notes: Path) -> None:
    version = notes.stem.removeprefix("release-notes-")
    converted = github_release_notes(notes.read_text(encoding="utf-8"), version)

    prose = re.sub(r"```.*?```", "", converted, flags=re.DOTALL)
    prose = re.sub(r"`[^`\n]+`", "", prose)
    targets = re.findall(r"\]\(([^)\s]+)", prose) + re.findall(r"^ {0,3}\[[^\]]+\]:\s*(\S+)", prose, flags=re.MULTILINE)
    assert [target for target in targets if not re.match(r"^(https?|mailto):", target)] == []


def test_the_release_workflow_publishes_the_converted_notes() -> None:
    workflow = (ROOT / ".github" / "workflows" / "upload_pypi.yml").read_text(encoding="utf-8")

    assert 'python3 tools/github_release_notes.py "$VAMOS_RELEASE_VERSION" --output frozen/github-release-notes.md' in workflow
    assert "--notes-file frozen/github-release-notes.md" in workflow
    assert '--notes-file "docs/project/release-notes-' not in workflow
