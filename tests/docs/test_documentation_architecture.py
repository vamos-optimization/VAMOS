from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict[str, object]:
    return yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))


def _top_level_nav_titles(nav: list[object]) -> list[str]:
    titles: list[str] = []
    for item in nav:
        if isinstance(item, str):
            titles.append(item)
        else:
            assert isinstance(item, dict) and len(item) == 1
            titles.append(next(iter(item)))
    return titles


def test_public_navigation_is_task_first() -> None:
    config = _config()
    nav = config["nav"]
    assert isinstance(nav, list)

    assert _top_level_nav_titles(nav) == [
        "Home",
        "Get Started",
        "Guides",
        "Examples",
        "Reference",
        "Project",
        "Developer",
    ]

    nav_text = yaml.safe_dump(nav, sort_keys=False)
    assert "guide/custom-problem.md" in nav_text
    assert "examples.md" in nav_text
    assert "dev/documentation-architecture.md" in nav_text
    assert "Engineering Audit" not in nav_text
    assert "website/" not in nav_text


def test_installation_tabs_have_required_markdown_extensions() -> None:
    config = _config()
    raw_extensions = config.get("markdown_extensions", [])
    names = {
        item if isinstance(item, str) else next(iter(item))
        for item in raw_extensions
    }

    assert "pymdownx.tabbed" in names
    assert "pymdownx.superfences" in names


def test_canonical_routing_pages_exist_and_cross_link() -> None:
    getting_started = (ROOT / "docs" / "guide" / "getting-started.md").read_text(encoding="utf-8")
    custom_problem = (ROOT / "docs" / "guide" / "custom-problem.md").read_text(encoding="utf-8")
    examples = (ROOT / "docs" / "examples.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs" / "dev" / "documentation-architecture.md").read_text(encoding="utf-8")

    assert "[Install VAMOS](installation.md)" in getting_started
    assert "[Solve your own problem](custom-problem.md)" in getting_started
    assert "[Durable studies](studies.md)" in getting_started
    assert "from vamos import make_problem, optimize" in custom_problem
    assert "examples/basics/quickstart.py" in examples
    assert "`website/docs/` is a legacy public-content tree" in architecture
    assert "does not deploy to Cloudflare" in architecture
