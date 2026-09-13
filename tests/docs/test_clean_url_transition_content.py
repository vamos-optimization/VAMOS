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


def _canonical(
    target: str,
    *,
    head_refresh: str | None = None,
    body_refresh: str | None = None,
    noscript_refresh: str | None = None,
    include_canonical: bool = True,
) -> str:
    canonical = "" if not include_canonical else f'<link rel="canonical" href="{target}">\n'
    head_meta = "" if head_refresh is None else f'<meta http-equiv="refresh" content="0; url={head_refresh}">\n'
    body_meta = "" if body_refresh is None else f'<meta http-equiv="refresh" content="0; url={body_refresh}">\n'
    noscript = (
        ""
        if noscript_refresh is None
        else f'<noscript><meta http-equiv="refresh" content="0; url={noscript_refresh}"></noscript>\n'
    )
    return (
        "<!doctype html><html><head>\n"
        f"{canonical}{head_meta}"
        "</head><body>\n"
        f"{body_meta}{noscript}"
        "</body></html>\n"
    )


def _redirect(
    target: str,
    *,
    body_refresh: str | None = None,
    noscript_refresh: str | None = None,
) -> str:
    return _canonical(
        target,
        head_refresh=target,
        body_refresh=body_refresh,
        noscript_refresh=noscript_refresh,
    )


def _build_clean(
    root: Path,
    *,
    deep_head_refresh: str | None = None,
    deep_body_refresh: str | None = None,
    deep_noscript_refresh: str | None = None,
    deep_canonical: bool = True,
    alias_body_refresh: str | None = None,
) -> None:
    immutable = f"{BASE_URL}docs/{VERSION}/"
    deep = f"{BASE_URL}algorithms/nsgaii/"
    missing = f"{BASE_URL}404.html"
    _write(root, "docs/versions.json", json.dumps({"stable": VERSION, "versions": [VERSION]}))
    _write(root, "index.html", _canonical(BASE_URL))
    _write(
        root,
        "algorithms/nsgaii/index.html",
        _canonical(
            deep,
            head_refresh=deep_head_refresh,
            body_refresh=deep_body_refresh,
            noscript_refresh=deep_noscript_refresh,
            include_canonical=deep_canonical,
        ),
    )
    # MkDocs emits a 404 document without canonical metadata; it is still content,
    # not a redirect, and must remain refresh-free.
    _write(root, "404.html", _canonical(missing, include_canonical=False))
    _write(root, f"docs/{VERSION}/index.html", _canonical(immutable))
    _write(root, "docs/index.html", _redirect(BASE_URL))
    _write(root, "docs/stable/index.html", _redirect(BASE_URL))
    _write(
        root,
        "docs/stable/algorithms/nsgaii/index.html",
        _redirect(deep, body_refresh=alias_body_refresh),
    )
    _write(root, "docs/stable/404.html", _redirect(missing))
    _write(root, "latest/index.html", _redirect(BASE_URL))
    _write(root, "latest/algorithms/nsgaii/index.html", _redirect(deep))
    _write(root, "latest/404.html", _redirect(missing))
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


def test_clean_content_pages_may_not_meta_refresh_in_head(tmp_path: Path) -> None:
    root = tmp_path / "clean-head-refresh"
    _build_clean(root, deep_head_refresh="https://evil.invalid/")

    completed = _run(root)
    assert completed.returncode != 0
    assert "Clean-current content page must not meta-refresh" in completed.stderr


def test_clean_content_pages_may_not_meta_refresh_in_body(tmp_path: Path) -> None:
    root = tmp_path / "clean-body-refresh"
    _build_clean(root, deep_body_refresh="https://evil.invalid/")

    completed = _run(root)
    assert completed.returncode != 0
    assert "Clean-current content page must not meta-refresh" in completed.stderr


def test_clean_content_pages_may_not_meta_refresh_in_noscript(tmp_path: Path) -> None:
    root = tmp_path / "clean-noscript-refresh"
    _build_clean(root, deep_noscript_refresh="https://evil.invalid/")

    completed = _run(root)
    assert completed.returncode != 0
    assert "Clean-current content page must not meta-refresh" in completed.stderr


def test_clean_content_pages_may_not_hide_refresh_in_svg_foreign_object(tmp_path: Path) -> None:
    root = tmp_path / "clean-svg-foreign-object-refresh"
    _build_clean(root)
    deep = f"{BASE_URL}algorithms/nsgaii/"
    malicious = (
        "<!doctype html><html><head>\n"
        f'<link rel="canonical" href="{deep}">\n'
        "</head><body><svg><foreignObject>\n"
        '<meta http-equiv="refresh" content="0; url=https://evil.invalid/">\n'
        "</foreignObject></svg></body></html>\n"
    )
    _write(root, "algorithms/nsgaii/index.html", malicious)

    completed = _run(root)
    assert completed.returncode != 0
    assert "Clean-current content page must not meta-refresh" in completed.stderr


def test_clean_content_pages_require_exact_canonical(tmp_path: Path) -> None:
    root = tmp_path / "clean-missing-canonical"
    _build_clean(root, deep_canonical=False)

    completed = _run(root)
    assert completed.returncode != 0
    assert "Clean-current canonical mismatch" in completed.stderr


def test_clean_generated_404_without_canonical_is_allowed(tmp_path: Path) -> None:
    root = tmp_path / "clean-404"
    _build_clean(root)

    completed = _run(root)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_clean_alias_may_not_add_second_active_refresh(tmp_path: Path) -> None:
    root = tmp_path / "clean-alias-extra-refresh"
    _build_clean(root, alias_body_refresh="https://evil.invalid/")

    completed = _run(root)
    assert completed.returncode != 0
    assert "Compatibility redirect refresh mismatch" in completed.stderr


def test_clean_content_pages_without_refresh_pass(tmp_path: Path) -> None:
    root = tmp_path / "clean"
    _build_clean(root)

    completed = _run(root)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_trusted_preview_runs_supplemental_content_gate_safely() -> None:
    workflow = (ROOT / ".github/workflows/docs-cloudflare-preview.yml").read_text(encoding="utf-8")
    assert "python tools/check_docs_portal_transition_content.py" in workflow
    assert "PORTAL_VERSION: ${{ steps.metadata.outputs.version }}" in workflow
    assert "PORTAL_DIR: ${{ steps.metadata.outputs.portal_dir }}" in workflow
    assert '--version "$PORTAL_VERSION"' in workflow
    assert '--root "$PORTAL_DIR"' in workflow
    assert "--version '${{ steps.metadata.outputs.version }}'" not in workflow
    assert "re.fullmatch(r\"\\d+\\.\\d+\\.\\d+\", version)" in workflow
