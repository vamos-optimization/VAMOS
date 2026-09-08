from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_vamos_visual_identity_is_wired_into_canonical_docs() -> None:
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))

    assert config["site_name"] == "VAMOS"
    assert config["extra_css"] == ["stylesheets/vamos.css"]
    assert config["theme"]["font"] == {"text": "DM Sans", "code": "JetBrains Mono"}


def test_brand_tokens_and_accessible_link_color_are_explicit() -> None:
    css = (ROOT / "docs" / "stylesheets" / "vamos.css").read_text(encoding="utf-8")

    for token in ("#0B2545", "#14B8A6", "#374151", "#F8FAFC"):
        assert token in css
    assert "--vamos-teal-dark: #0F766E" in css
    assert "--md-typeset-a-color: var(--vamos-teal-dark)" in css
    assert "color: var(--vamos-navy-deep) !important" in css
    assert "@media (prefers-reduced-motion: reduce)" in css


def test_homepage_uses_three_scientific_user_journeys() -> None:
    homepage = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")

    assert 'class="vamos-hero"' in homepage
    assert "pip install vamos-optimization" in homepage
    assert homepage.count('class="vamos-journey"') == 3
    assert "Try VAMOS" in homepage
    assert "Solve my problem" in homepage
    assert "Run a reproducible study" in homepage
    assert "https://github.com/vamos-optimization/VAMOS/blob/main/CITATION.cff" in homepage
    assert "https://github.com/vamos-optimization/VAMOS/blob/main/SECURITY.md" in homepage
