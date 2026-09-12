from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
UPLOAD_ARTIFACT = "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
DOWNLOAD_ARTIFACT = "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093"
BASE_URL = "https://vamos-optimization.org/"
VERSION = "1.0.0"


def test_preview_workflow_is_read_only_and_artifact_only() -> None:
    workflow = (ROOT / ".github" / "workflows" / "docs-preview.yml").read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "python tools/build_release_docs.py" in workflow
    assert "python tools/check_docs_portal.py" in workflow
    assert UPLOAD_ARTIFACT in workflow
    assert "retention-days: 14" in workflow
    assert "pages: write" not in workflow
    assert "id-token: write" not in workflow
    assert "actions/deploy-pages@" not in workflow
    assert f"DOC_BASE_URL: {BASE_URL}" in workflow


def test_release_workflow_checks_portal_and_requires_archive_for_later_versions() -> None:
    workflow = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")

    assert "archive_run_id:" in workflow
    assert "archive_artifact_name:" in workflow
    assert "if: env.DOC_VERSION != '1.0.0'" in workflow
    assert DOWNLOAD_ARTIFACT in workflow
    assert "--archive-from previous-public" in workflow
    assert "python tools/check_docs_portal.py" in workflow
    assert UPLOAD_ARTIFACT in workflow
    assert "vamos-docs-portal-${{ env.DOC_VERSION }}" in workflow
    assert "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9" in workflow
    assert "actions/deploy-pages@368f82528645a54fb793d4d04e342629a3f51346" in workflow
    assert f"DOC_BASE_URL: {BASE_URL}" in workflow


def test_documentation_delivery_page_is_discoverable() -> None:
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    nav_text = yaml.safe_dump(config["nav"], sort_keys=False)
    delivery = (ROOT / "docs" / "dev" / "documentation-delivery.md").read_text(encoding="utf-8")

    assert "dev/documentation-delivery.md" in nav_text
    assert "does not deploy" in delivery
    assert "archive_run_id" in delivery
    assert "GitHub Pages fallback mirror" in delivery
    assert "canonical production path" in delivery
    assert "clean current" in delivery


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _redirect(target: str) -> str:
    return (
        f'<link rel="canonical" href="{target}">\n'
        f'<meta http-equiv="refresh" content="0; url={target}">\n'
    )


def _build_minimal_clean_portal(root: Path, *, include_deep_current: bool = False) -> None:
    immutable_url = f"{BASE_URL}docs/{VERSION}/"
    _write(root, "index.html", f'<link rel="canonical" href="{BASE_URL}">\n')
    _write(root, f"docs/{VERSION}/index.html", f'<link rel="canonical" href="{immutable_url}">\n')
    _write(root, "docs/versions.json", json.dumps({"stable": VERSION, "versions": [VERSION]}))
    _write(root, "docs/index.html", _redirect(BASE_URL))
    _write(root, "docs/stable/index.html", _redirect(BASE_URL))
    _write(root, "latest/index.html", _redirect(BASE_URL))
    _write(root, f"{VERSION}/index.html", _redirect(immutable_url))
    _write(root, "website/index.html", f'<link rel="canonical" href="{BASE_URL}website/">')
    if include_deep_current:
        target = f"{BASE_URL}algorithms/nsgaii/"
        _write(root, "algorithms/nsgaii/index.html", f'<link rel="canonical" href="{target}">\n')
        _write(root, "docs/stable/algorithms/nsgaii/index.html", _redirect(target))
        _write(root, "latest/algorithms/nsgaii/index.html", _redirect(target))


def _run_portal_check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "tools/check_docs_portal.py",
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


def test_portal_checker_accepts_minimal_clean_current_contract(tmp_path: Path) -> None:
    root = tmp_path / "portal"
    _build_minimal_clean_portal(root, include_deep_current=True)

    completed = _run_portal_check(root)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["stable"] == VERSION
    assert payload["versions"] == [VERSION]
    assert payload["current_url"] == BASE_URL
    assert payload["canonical_links_checked"] >= 7


def test_portal_checker_rejects_missing_deep_compatibility_route(tmp_path: Path) -> None:
    root = tmp_path / "missing-alias"
    _build_minimal_clean_portal(root, include_deep_current=True)
    (root / "latest" / "algorithms" / "nsgaii" / "index.html").unlink()

    completed = _run_portal_check(root)
    assert completed.returncode != 0
    assert "route inventory does not match its source tree" in completed.stderr


def test_portal_checker_rejects_commented_redirect_spoof(tmp_path: Path) -> None:
    root = tmp_path / "spoof"
    _build_minimal_clean_portal(root)
    spoofed = (
        f'<!-- <link rel="canonical" href="{BASE_URL}">'
        f'<meta http-equiv="refresh" content="0; url={BASE_URL}"> -->\n'
        '<link rel="canonical" href="https://evil.invalid/">\n'
        '<meta http-equiv="refresh" content="0; url=https://evil.invalid/">\n'
    )
    _write(root, "docs/stable/index.html", spoofed)

    completed = _run_portal_check(root)
    assert completed.returncode != 0
    assert "Redirect canonical mismatch" in completed.stderr
