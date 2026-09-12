from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASE_URL = "https://vamos-optimization.org/"
VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[2]


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _canonical(target: str, *, refresh: str | None = None) -> str:
    meta = "" if refresh is None else f'<meta http-equiv="refresh" content="0; url={refresh}">\n'
    return (
        "<!doctype html><html><head>\n"
        f'<link rel="canonical" href="{target}">\n'
        f"{meta}"
        "</head><body></body></html>\n"
    )


def _redirect(target: str) -> str:
    return _canonical(target, refresh=target)


def _build_clean(root: Path, *, deep_refresh: str | None = None) -> None:
    immutable = f"{BASE_URL}docs/{VERSION}/"
    deep = f"{BASE_URL}algorithms/nsgaii/"
    _write(root, "docs/versions.json", json.dumps({"stable": VERSION, "versions": [VERSION]}))
    _write(root, "index.html", _canonical(BASE_URL))
    _write(root, "algorithms/nsgaii/index.html", _canonical(deep, refresh=deep_refresh))
    _write(root, f"docs/{VERSION}/index.html", _canonical(immutable))
    _write(root, "docs/index.html", _redirect(BASE_URL))
    _write(root, "docs/stable/index.html", _redirect(BASE_URL))
    _write(root, "docs/stable/algorithms/nsgaii/index.html", _redirect(deep))
    _write(root, "latest/index.html", _redirect(BASE_URL))
    _write(root, "latest/algorithms/nsgaii/index.html", _redirect(deep))
    _write(root, f"{VERSION}/index.html", _redirect(immutable))
    _write(root, "website/index.html", _canonical(f"{BASE_URL}website/"))


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "tools/check_docs_portal_transition_content.py",
            "--root",
            str(root),
            "--version",
            VERSION,
            "--base-url",
            BASE_URL,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_clean_content_pages_may_not_meta_refresh(tmp_path: Path) -> None:
    root = tmp_path / "clean"
    _build_clean(root, deep_refresh="https://evil.invalid/")

    completed = _run(root)
    assert completed.returncode != 0
    assert "Clean-current content page must not meta-refresh" in completed.stderr


def test_clean_content_pages_without_refresh_pass(tmp_path: Path) -> None:
    root = tmp_path / "clean"
    _build_clean(root)

    completed = _run(root)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_trusted_preview_runs_supplemental_content_gate() -> None:
    workflow = (ROOT / ".github/workflows/docs-cloudflare-preview.yml").read_text(encoding="utf-8")
    assert "python tools/check_docs_portal_transition_content.py" in workflow
