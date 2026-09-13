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


def _canonical(target: str, *, body: str = "") -> str:
    return (
        "<!doctype html><html><head>\n"
        f'<link rel="canonical" href="{target}">\n'
        "</head><body>\n"
        f"{body}"
        "</body></html>\n"
    )


def _redirect(target: str, *, body: str = "") -> str:
    return (
        "<!doctype html><html><head>\n"
        f'<link rel="canonical" href="{target}">\n'
        f'<meta http-equiv="refresh" content="0; url={target}">\n'
        "</head><body>\n"
        f"{body}"
        "</body></html>\n"
    )


def _build_clean(root: Path) -> None:
    immutable = f"{BASE_URL}docs/{VERSION}/"
    deep = f"{BASE_URL}algorithms/nsgaii/"
    _write(root, "docs/versions.json", json.dumps({"stable": VERSION, "versions": [VERSION]}))
    _write(root, "index.html", _canonical(BASE_URL))
    _write(root, "algorithms/nsgaii/index.html", _canonical(deep))
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
            "tools/check_docs_portal_active_content.py",
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


def test_post_cutover_active_content_contract_passes(tmp_path: Path) -> None:
    root = tmp_path / "clean"
    _build_clean(root)

    completed = _run(root)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_svg_foreign_object_cannot_hide_meta_refresh(tmp_path: Path) -> None:
    root = tmp_path / "svg-refresh"
    _build_clean(root)
    deep = f"{BASE_URL}algorithms/nsgaii/"
    body = (
        "<svg><foreignObject>"
        '<meta http-equiv="refresh" content="0; url=https://evil.invalid/">'
        "</foreignObject></svg>"
    )
    _write(root, "algorithms/nsgaii/index.html", _canonical(deep, body=body))

    completed = _run(root)
    assert completed.returncode != 0
    assert "Canonical content page must not meta-refresh" in completed.stderr


def test_template_refresh_remains_inert(tmp_path: Path) -> None:
    root = tmp_path / "template-refresh"
    _build_clean(root)
    deep = f"{BASE_URL}algorithms/nsgaii/"
    body = (
        "<template>"
        '<meta http-equiv="refresh" content="0; url=https://evil.invalid/">'
        "</template>"
    )
    _write(root, "algorithms/nsgaii/index.html", _canonical(deep, body=body))

    completed = _run(root)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_alias_cannot_add_second_active_refresh(tmp_path: Path) -> None:
    root = tmp_path / "alias-extra-refresh"
    _build_clean(root)
    deep = f"{BASE_URL}algorithms/nsgaii/"
    extra = '<meta http-equiv="refresh" content="0; url=https://evil.invalid/">'
    _write(root, "docs/stable/algorithms/nsgaii/index.html", _redirect(deep, body=extra))

    completed = _run(root)
    assert completed.returncode != 0
    assert "Compatibility redirect refresh mismatch" in completed.stderr


def test_cloudflare_workflows_use_post_cutover_validators_and_trusted_pr_identity() -> None:
    preview = (ROOT / ".github/workflows/docs-cloudflare-preview.yml").read_text(encoding="utf-8")
    production = (ROOT / ".github/workflows/docs-cloudflare.yml").read_text(encoding="utf-8")

    for workflow in (preview, production):
        assert "python tools/check_docs_portal.py" in workflow
        assert "python tools/check_docs_portal_active_content.py" in workflow
        assert "check_docs_portal_transition.py" not in workflow
        assert "check_docs_portal_transition_content.py" not in workflow

    assert "TRUSTED_PR_NUMBER: ${{ github.event.workflow_run.pull_requests[0].number }}" in preview
    assert 'test "$pr_number" = "$TRUSTED_PR_NUMBER"' in preview
    assert 'portal_dir="$ARTIFACT_ROOT/docs-preview-$TRUSTED_PR_NUMBER"' in preview
    assert 'echo "pr_number=$TRUSTED_PR_NUMBER" >> "$GITHUB_OUTPUT"' in preview
    assert "PORTAL_VERSION: ${{ steps.metadata.outputs.version }}" in preview
    assert "PORTAL_DIR: ${{ steps.metadata.outputs.portal_dir }}" in preview
