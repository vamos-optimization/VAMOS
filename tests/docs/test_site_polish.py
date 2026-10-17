from __future__ import annotations

import importlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    return importlib.import_module("check_docs_site")


def make_site(root: Path) -> None:
    head = '<html lang="en"><head><meta name="description" content="Learn"><meta property="og:title" content="Lesson"><meta property="og:description" content="Learn"></head>'
    (root / "index.html").write_text(
        head + '<body><main><h1 id="learn">Learn</h1><a href="example.py" download>Example</a></main></body></html>'
    )
    (root / "404.html").write_text(
        head
        + '<meta name="robots" content="noindex, follow"><main><h1>Missing</h1><input id="recovery-query"><a href="/">Home</a></main></html>'
    )
    (root / "example.py").write_text("print('example')\n")
    (root / "search").mkdir()
    (root / "search/search_index.json").write_text(json.dumps({"docs": [{"location": "#learn"}]}))


def test_site_gate_accepts_a_complete_reader_journey(checker, tmp_path):
    make_site(tmp_path)
    assert checker.check_site(tmp_path, base_url="https://docs.example/")["downloads"] == 1


@pytest.mark.parametrize("defect", ["download", "bookmark", "search", "language", "alt", "recovery"])
def test_site_gate_rejects_reader_failures(checker, tmp_path, defect):
    make_site(tmp_path)
    if defect == "download":
        (tmp_path / "example.py").unlink()
    elif defect == "search":
        (tmp_path / "search/search_index.json").write_text('{"docs": []}')
    elif defect == "recovery":
        path = tmp_path / "404.html"
        path.write_text(path.read_text().replace("noindex", "index"))
    else:
        path = tmp_path / "index.html"
        replacements = {
            "bookmark": ('id="learn"', 'id="moved"'),
            "language": ('lang="en"', 'lang=""'),
            "alt": ("</main>", '<img src="example.py"></main>'),
        }
        path.write_text(path.read_text().replace(*replacements[defect]))
    with pytest.raises(checker.SiteCheckError):
        checker.check_site(tmp_path, base_url="https://docs.example/")


def test_site_gate_rejects_removed_public_bookmarks(checker, tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    make_site(old)
    make_site(new)
    path = old / "index.html"
    path.write_text(path.read_text().replace("</main>", '<h2 id="older-heading">More</h2></main>'))
    with pytest.raises(checker.SiteCheckError, match="Removed public route or bookmark"):
        checker.check_site(new, base_url="https://docs.example/", baseline=old)


def test_learning_path_metadata_points_to_real_pages_and_downloads():
    for path in (ROOT / "docs/tutorials").glob("t0*.md"):
        metadata = yaml.safe_load(path.read_text().split("---", 2)[1])
        assert metadata["description"] and metadata["lesson"]["position"]
        for direction in ("previous", "next"):
            route = metadata["lesson"][direction]["url"].rstrip("/")
            assert (ROOT / "docs" / f"{route}.md").is_file()
        for download in metadata["downloads"]:
            assert (ROOT / "docs" / download["url"]).is_file()


def test_preview_and_production_enforce_site_and_artifact_checks():
    preview = (ROOT / ".github/workflows/docs-preview.yml").read_text()
    production = (ROOT / ".github/workflows/docs-cloudflare.yml").read_text()
    for text in (preview, production):
        assert "python tools/check_docs_site.py" in text
        assert "python tools/check_docs_artifact.py create" in text
        assert "python tools/check_docs_artifact.py verify" in text
    assert "--output preview-evidence/browser" in preview
    assert "if: always()" in preview
    assert "expected_source_sha:" in production
    assert "python tools/check_docs_artifact.py live" in production
    assert (
        production.index("check_docs_artifact.py create")
        < production.index("wrangler@${WRANGLER_VERSION}")
        < production.index("check_docs_artifact.py live")
    )


@pytest.mark.parametrize("prefix", ["/", "/docs/1.0.1/"])
def test_nested_404_keeps_theme_resources_on_the_preview_origin(monkeypatch, tmp_path, prefix):
    pytest.importorskip("mkdocs")
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    builder = importlib.import_module("build_release_docs")
    builder._build_site(ROOT, "mkdocs.yml", tmp_path / "site", "https://vamos-optimization.org" + prefix)
    resources = []

    class Resources(HTMLParser):
        def handle_starttag(self, tag, attrs):
            data = dict(attrs)
            if tag == "link" and data.get("rel") == "stylesheet":
                resources.append(data["href"])
            if tag in {"script", "img"} and data.get("src"):
                resources.append(data["src"])

    Resources().feed((tmp_path / "site/404.html").read_text(encoding="utf-8"))
    checked = 0
    for resource in resources:
        if urlsplit(resource).scheme:
            assert urlsplit(resource).netloc == "fonts.googleapis.com"
            continue
        resolved = urlsplit(urljoin("https://preview.example" + prefix + "missing/nested/page/", resource))
        assert resolved.netloc == "preview.example"
        assert resolved.path.startswith(prefix)
        assert (tmp_path / "site" / resolved.path.removeprefix(prefix)).is_file(), resource
        checked += 1
    assert checked >= 7
