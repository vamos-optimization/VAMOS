from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_zensical_probe_is_pinned_and_strict() -> None:
    workflow = (ROOT / ".github" / "workflows" / "zensical-compat.yml").read_text(encoding="utf-8")

    assert 'zensical==0.0.59' in workflow
    assert 'mkdocstrings==1.0.6' in workflow
    assert 'mkdocstrings-python==2.0.8' in workflow
    assert "zensical build --config-file mkdocs.yml --strict --clean" in workflow
    assert "website/mkdocs.yml" not in workflow


def test_zensical_probe_does_not_replace_release_deployment() -> None:
    deploy = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")

    assert "python tools/build_release_docs.py" in deploy
    assert "zensical build" not in deploy
