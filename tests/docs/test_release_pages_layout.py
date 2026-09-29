from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://vamos-optimization.org/"


def test_versioned_pages_canonical_urls_resolve_to_deployed_files(tmp_path: Path) -> None:
    pytest.importorskip("mkdocs")
    archive = tmp_path / "previous"
    frozen = archive / "docs" / "1.0.0" / "guide" / "frozen"
    frozen.mkdir(parents=True)
    (frozen / "index.html").write_text("frozen release", encoding="utf-8")
    (archive / "docs" / "1.0.0" / "asset.txt").write_text("frozen asset", encoding="utf-8")

    output = tmp_path / "public"
    result = subprocess.run(
        [
            sys.executable,
            "tools/build_release_docs.py",
            "--version",
            "1.1.0",
            "--output",
            str(output),
            "--archive-from",
            str(archive),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    count = 0
    for page in output.rglob("*.html"):
        for canonical in re.findall(r'<link\s+rel="canonical"\s+href="([^"]+)"', page.read_text(encoding="utf-8")):
            url = urlsplit(canonical)
            assert url.scheme == "https" and url.netloc == "vamos-optimization.org"
            assert not url.path.startswith("/docs/stable/")
            assert not url.path.startswith("/latest/")
            target = output / unquote(url.path.lstrip("/"))
            if url.path.endswith("/"):
                target /= "index.html"
            assert target.is_file(), canonical
            count += 1
    assert count > 80

    manifest = json.loads((output / "docs" / "versions.json").read_text(encoding="utf-8"))
    assert manifest == {"stable": "1.1.0", "versions": ["1.0.0", "1.1.0"]}

    assert (output / "docs" / "1.0.0" / "guide" / "frozen" / "index.html").read_text(encoding="utf-8") == "frozen release"
    assert (output / "docs" / "1.0.0" / "asset.txt").read_text(encoding="utf-8") == "frozen asset"
    assert "docs/1.0.0/guide/frozen/" in (output / "1.0.0" / "guide" / "frozen" / "index.html").read_text(encoding="utf-8")

    current_home = (output / "index.html").read_text(encoding="utf-8")
    assert f'rel="canonical" href="{BASE_URL}"' in current_home
    assert "http-equiv=\"refresh\"" not in current_home

    assert f"url={BASE_URL}" in (output / "docs" / "index.html").read_text(encoding="utf-8")
    assert f"url={BASE_URL}" in (output / "latest" / "index.html").read_text(encoding="utf-8")
    assert f"url={BASE_URL}" in (output / "docs" / "stable" / "index.html").read_text(encoding="utf-8")
    assert "docs/1.1.0/" in (output / "1.1.0" / "index.html").read_text(encoding="utf-8")

    legacy_api = (output / "docs" / "stable" / "reference" / "api_reference" / "index.html").read_text(encoding="utf-8")
    assert 'const redirectBase = "https://vamos-optimization.org/reference/api_reference/";' in legacy_api
    assert "window.location.search" in legacy_api
    assert "window.location.hash" in legacy_api
    assert "window.location.replace" in legacy_api

    immutable_home = (output / "docs" / "1.1.0" / "index.html").read_text(encoding="utf-8")
    assert 'rel="canonical" href="https://vamos-optimization.org/docs/1.1.0/"' in immutable_home
    assert (output / "docs" / "stable" / "index.html").read_bytes() != (output / "docs" / "1.1.0" / "index.html").read_bytes()

    assert "https://github.com/vamos-optimization/VAMOS/blob/main/CITATION.cff" in immutable_home
    assert "https://github.com/vamos-optimization/VAMOS/blob/main/SECURITY.md" in immutable_home


def test_same_version_republish_reuses_immutable_archive(tmp_path: Path) -> None:
    pytest.importorskip("mkdocs")
    archive = tmp_path / "previous"
    immutable = archive / "docs" / "1.0.0"
    immutable.mkdir(parents=True)
    frozen_home = (
        '<link rel="canonical" href="https://vamos-optimization.org/docs/1.0.0/">\n'
        "<!-- frozen production archive -->\n"
    )
    (immutable / "index.html").write_text(frozen_home, encoding="utf-8")
    (immutable / "frozen-marker.txt").write_text("do not rebuild", encoding="utf-8")

    output = tmp_path / "republished"
    result = subprocess.run(
        [
            sys.executable,
            "tools/build_release_docs.py",
            "--version",
            "1.0.0",
            "--output",
            str(output),
            "--archive-from",
            str(archive),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    assert (output / "docs" / "1.0.0" / "index.html").read_text(encoding="utf-8") == frozen_home
    assert (output / "docs" / "1.0.0" / "frozen-marker.txt").read_text(encoding="utf-8") == "do not rebuild"
    assert "frozen production archive" not in (output / "index.html").read_text(encoding="utf-8")
    manifest = json.loads((output / "docs" / "versions.json").read_text(encoding="utf-8"))
    assert manifest == {"stable": "1.0.0", "versions": ["1.0.0"]}


def test_archive_carry_forward_removes_withdrawn_routes(tmp_path: Path) -> None:
    pytest.importorskip("mkdocs")
    archive = tmp_path / "previous"
    immutable = archive / "docs" / "1.0.0"
    prefix = f"{BASE_URL}docs/1.0.0/"
    kept_entry = f"    <url>\n         <loc>{prefix}guide/</loc>\n    </url>\n"
    withdrawn_entry = f"    <url>\n         <loc>{prefix}audit/commands_used/</loc>\n    </url>\n"
    files = {
        "index.html": f'<link rel="canonical" href="{prefix}">\n',
        "guide/index.html": f'<link rel="canonical" href="{prefix}guide/">\n',
        "audit/commands_used/index.html": f'<link rel="canonical" href="{prefix}audit/commands_used/">\n',
        "audit/findings.csv": "id,severity\n",
        "topics/engineering_audit/index.html": f'<link rel="canonical" href="{prefix}topics/engineering_audit/">\n',
        "topics/tuning/index.html": f'<link rel="canonical" href="{prefix}topics/tuning/">\n',
        "sitemap.xml": f"<urlset>\n{kept_entry}{withdrawn_entry}</urlset>\n",
        "search/search_index.json": json.dumps(
            {
                "config": {"lang": ["en"]},
                "docs": [
                    {"location": "guide/", "title": "Guide", "text": ""},
                    {"location": "audit/software_engineering_audit/#summary", "title": "Audit", "text": ""},
                    {"location": "topics/engineering_audit/", "title": "Audit", "text": ""},
                ],
            }
        ),
    }
    for relative, content in files.items():
        path = immutable / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (immutable / "sitemap.xml.gz").write_bytes(gzip.compress(files["sitemap.xml"].encode("utf-8")))
    (immutable / "dev" / "adr").mkdir(parents=True)
    (immutable / "dev" / "adr" / "index.html").write_text(f'<link rel="canonical" href="{prefix}dev/adr/">\n', encoding="utf-8")
    studies = immutable / "dev" / "studies" / "index.html"
    studies.parent.mkdir(parents=True)
    studies.write_bytes(
        (
            f'<head><link rel="canonical" href="{prefix}dev/studies/">\n'
            '<link rel="prev" href="../testing/">\n'
            '<link rel="next" href="../adr/">\n'
            "</head><body>\n"
            '<ul class="md-nav__list">\n'
            '<li class="md-nav__item"><a href="../testing/" class="md-nav__link">Testing</a></li>\n'
            '<li class="md-nav__item md-nav__item--nested"><label>Architecture Decisions</label><ul>'
            '<li class="md-nav__item"><a href="../adr/" class="md-nav__link">ADR Index</a></li>'
            '<li class="md-nav__item"><a href="../adr/0008-durable-study-manifest-contract/" class="md-nav__link">0008</a></li>'
            "</ul></li>\n"
            "</ul>\n"
            '<p>Decision record: <a href="../adr/0008-durable-study-manifest-contract/#decision">ADR 0008</a>.</p>\n'
            '<ul><li>Read <a href="../adr/">the records</a> first.</li></ul>\n'
            "</body>\n"
        ).encode()
    )

    output = tmp_path / "public"
    result = subprocess.run(
        [
            sys.executable,
            "tools/build_release_docs.py",
            "--version",
            "1.0.0",
            "--output",
            str(output),
            "--archive-from",
            str(archive),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    carried = output / "docs" / "1.0.0"
    assert not (carried / "audit").exists()
    assert not (carried / "topics" / "engineering_audit").exists()
    assert not (carried / "dev" / "adr").exists()
    assert not (output / "1.0.0" / "audit").exists()
    assert not (output / "1.0.0" / "dev" / "adr").exists()
    assert (carried / "dev" / "studies" / "index.html").read_bytes().decode("utf-8") == (
        f'<head><link rel="canonical" href="{prefix}dev/studies/">\n'
        '<link rel="prev" href="../testing/">\n'
        "\n"
        "</head><body>\n"
        '<ul class="md-nav__list">\n'
        '<li class="md-nav__item"><a href="../testing/" class="md-nav__link">Testing</a></li>\n'
        "\n"
        "</ul>\n"
        "<p>Decision record: ADR 0008.</p>\n"
        "<ul><li>Read the records first.</li></ul>\n"
        "</body>\n"
    )
    assert not (output / "website").exists()
    for relative in ("index.html", "guide/index.html", "topics/tuning/index.html"):
        assert (carried / relative).read_text(encoding="utf-8") == files[relative]

    sitemap = (carried / "sitemap.xml").read_text(encoding="utf-8")
    assert sitemap == f"<urlset>\n{kept_entry}</urlset>\n"
    assert gzip.decompress((carried / "sitemap.xml.gz").read_bytes()) == (carried / "sitemap.xml").read_bytes()
    index = json.loads((carried / "search" / "search_index.json").read_text(encoding="utf-8"))
    assert index["config"] == {"lang": ["en"]}
    assert [entry["location"] for entry in index["docs"]] == ["guide/"]
