from __future__ import annotations

import json
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


class ApiHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.bookmarks: dict[str, str] = {}
        self.h1_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = dict(attrs)
        if identifier := data.get("id"):
            self.ids.add(identifier)
        if tag == "a" and "data-api-legacy" in data:
            identifier, href = data.get("id"), data.get("href")
            assert identifier and href
            assert identifier not in self.bookmarks
            self.bookmarks[identifier] = href
        if tag == "h1":
            self.h1_count += 1


def parse_api_html(text: str) -> ApiHTML:
    parser = ApiHTML()
    parser.feed(text)
    return parser


def test_api_is_a_topic_index_with_public_facades_and_explicit_source_policy() -> None:
    docs = ROOT / "docs"
    index = (docs / "reference/api_reference.md").read_text(encoding="utf-8")
    assert ":::" not in index
    assert "<details" in index
    assert len(index.split("<details", 1)[0].split()) < 500
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    nav = yaml.safe_dump(config["nav"], sort_keys=False)
    for page in (docs / "reference/api").rglob("*.md"):
        assert page.relative_to(docs).as_posix() in nav
        text = page.read_text(encoding="utf-8")
        assert "show_source: true" not in text
        assert text.count("show_source: false") == text.count("::: ")
    for page, symbol in (
        ("optimization", "vamos.optimize"),
        ("problem-definition", "vamos.make_problem"),
        ("results", "vamos.OptimizationResult"),
    ):
        assert f"::: {symbol}" in (docs / f"reference/api/{page}.md").read_text(encoding="utf-8")
    assert "javascripts/api-reference.js" in config["extra_javascript"]


def test_legacy_bookmark_inventory_is_not_silently_dropped() -> None:
    docs = ROOT / "docs"
    frozen = json.loads((docs / "reference/api/legacy-bookmarks.json").read_text(encoding="utf-8"))
    assert frozen["source_commit"] == "ed100ea43a3f5dec0204c8297e7defdd5e020bcc"
    assert len(frozen["bookmarks"]) == 152
    index = parse_api_html((docs / "reference/api_reference.md").read_text(encoding="utf-8"))
    assert index.bookmarks == frozen["bookmarks"]
    for target in index.bookmarks.values():
        assert target.startswith("../api/")
        assert not urlsplit(target).netloc


def test_legacy_redirect_script_preserves_origin_version_and_query() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the JavaScript behavior test")
    completed = subprocess.run(
        [node, "--test", "tests/docs/api_reference_redirects.test.cjs"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_every_legacy_bookmark_resolves_in_current_and_immutable_portal(tmp_path: Path) -> None:
    pytest.importorskip("mkdocs")
    output = tmp_path / "portal"
    built = subprocess.run(
        [sys.executable, "tools/build_release_docs.py", "--version", "1.0.0", "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr

    channels = (
        (output, "https://example.invalid/"),
        (output / "docs" / "1.0.0", "https://example.invalid/docs/1.0.0/"),
    )
    for tree, base_root in channels:
        source = tree / "reference/api_reference/index.html"
        parsed = parse_api_html(source.read_text(encoding="utf-8"))
        assert len(parsed.bookmarks) == 152
        assert parsed.h1_count == 1
        base = f"{base_root}reference/api_reference/"
        cache: dict[Path, ApiHTML] = {}
        for old, target in parsed.bookmarks.items():
            resolved = urlsplit(urljoin(base, target))
            expected_prefix = urlsplit(base_root).path + "reference/api/"
            assert resolved.path.startswith(expected_prefix)
            relative = unquote(resolved.path.removeprefix(urlsplit(base_root).path).lstrip("/"))
            path = tree / relative / "index.html"
            assert path.is_file(), (old, target)
            if path not in cache:
                cache[path] = parse_api_html(path.read_text(encoding="utf-8"))
            if resolved.fragment:
                assert unquote(resolved.fragment) in cache[path].ids, (old, target)
        assert len(source.read_bytes()) < 100_000
        optimization = tree / "reference/api/optimization/index.html"
        assert len(optimization.read_bytes()) < 150_000
