from __future__ import annotations

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

    immutable_home = (output / "docs" / "1.1.0" / "index.html").read_text(encoding="utf-8")
    assert 'rel="canonical" href="https://vamos-optimization.org/docs/1.1.0/"' in immutable_home
    assert (output / "docs" / "stable" / "index.html").read_bytes() != (output / "docs" / "1.1.0" / "index.html").read_bytes()

    assert "https://github.com/vamos-optimization/VAMOS/blob/main/CITATION.cff" in immutable_home
    assert "https://github.com/vamos-optimization/VAMOS/blob/main/SECURITY.md" in immutable_home
