from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_homepage_hero_uses_container_width_instead_of_squeezing_two_columns() -> None:
    homepage = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    css = (ROOT / "docs" / "stylesheets" / "home-hero.css").read_text(encoding="utf-8")

    assert 'class="vamos-hero-shell"' in homepage
    assert 'class="vamos-hero__headline-main"' in homepage
    assert 'class="vamos-hero__headline-accent"' in homepage
    assert '<span class="vamos-hero__nowrap">Multi-objective</span> optimization' in homepage
    assert "optimization<br>" not in homepage

    assert "container-type: inline-size" in css
    assert "@container vamos-home-hero (max-width: 64rem)" in css
    assert "grid-template-columns: 1fr" in css
    assert "overflow-wrap: normal" in css
    assert "word-break: normal" in css
    assert "hyphens: none" in css
    assert "white-space: nowrap" in css


def test_homepage_hero_styles_are_loaded_after_the_base_identity_styles() -> None:
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))

    assert config["extra_css"] == [
        "stylesheets/vamos.css",
        "stylesheets/home-hero.css",
    ]
