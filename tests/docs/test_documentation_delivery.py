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


def test_portal_checker_accepts_minimal_clean_current_contract(tmp_path: Path) -> None:
    root = tmp_path / "portal"
    version = "1.0.0"

    def write(relative: str, content: str) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    immutable_url = f"{BASE_URL}docs/{version}/"
    write("index.html", f'<link rel="canonical" href="{BASE_URL}">\n')
    write(f"docs/{version}/index.html", f'<link rel="canonical" href="{immutable_url}">\n')
    write("docs/versions.json", json.dumps({"stable": version, "versions": [version]}))
    write("docs/index.html", f'<link rel="canonical" href="{BASE_URL}"><meta http-equiv="refresh" content="0; url={BASE_URL}">')
    write("docs/stable/index.html", f'<link rel="canonical" href="{BASE_URL}"><meta http-equiv="refresh" content="0; url={BASE_URL}">')
    write("latest/index.html", f'<link rel="canonical" href="{BASE_URL}"><meta http-equiv="refresh" content="0; url={BASE_URL}">')
    write(f"{version}/index.html", f'<link rel="canonical" href="{immutable_url}"><meta http-equiv="refresh" content="0; url={immutable_url}">')
    write("website/index.html", f'<link rel="canonical" href="{BASE_URL}website/">')

    completed = subprocess.run(
        [
            sys.executable,
            "tools/check_docs_portal.py",
            "--root",
            str(root),
            "--version",
            version,
            "--base-url",
            BASE_URL,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["stable"] == version
    assert payload["versions"] == [version]
    assert payload["current_url"] == BASE_URL
    assert payload["canonical_links_checked"] >= 5
