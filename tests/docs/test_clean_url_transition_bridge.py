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


def _canonical(target: str) -> str:
    return (
        "<!doctype html><html><head>\n"
        f'<link rel="canonical" href="{target}">\n'
        "</head><body></body></html>\n"
    )


def _redirect(target: str) -> str:
    return (
        "<!doctype html><html><head>\n"
        f'<link rel="canonical" href="{target}">\n'
        f'<meta http-equiv="refresh" content="0; url={target}">\n'
        "</head><body></body></html>\n"
    )


def _manifest(root: Path) -> None:
    _write(
        root,
        "docs/versions.json",
        json.dumps({"stable": VERSION, "versions": [VERSION]}),
    )


def _check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "tools/check_docs_portal_transition.py",
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


def test_transition_bridge_accepts_legacy_portal_contract(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    immutable_url = f"{BASE_URL}docs/{VERSION}/"
    stable_url = f"{BASE_URL}docs/stable/"
    homepage = _canonical(immutable_url)

    _manifest(root)
    _write(root, f"docs/{VERSION}/index.html", homepage)
    _write(root, "docs/stable/index.html", homepage)
    _write(root, "index.html", _redirect(stable_url))
    _write(root, "docs/index.html", _redirect(stable_url))
    _write(root, "latest/index.html", _redirect(stable_url))
    _write(root, f"{VERSION}/index.html", _redirect(immutable_url))
    _write(root, "website/index.html", _canonical(f"{BASE_URL}website/"))

    completed = _check(root)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["contract"] == "legacy-stable-tree"


def _build_clean_fixture(root: Path) -> None:
    immutable_url = f"{BASE_URL}docs/{VERSION}/"
    _manifest(root)
    _write(root, "index.html", _canonical(BASE_URL))
    _write(root, "algorithms/nsgaii/index.html", _canonical(f"{BASE_URL}algorithms/nsgaii/"))
    _write(root, f"docs/{VERSION}/index.html", _canonical(immutable_url))
    _write(root, "docs/index.html", _redirect(BASE_URL))
    _write(root, "docs/stable/index.html", _redirect(BASE_URL))
    _write(root, "docs/stable/algorithms/nsgaii/index.html", _redirect(f"{BASE_URL}algorithms/nsgaii/"))
    _write(root, "latest/index.html", _redirect(BASE_URL))
    _write(root, "latest/algorithms/nsgaii/index.html", _redirect(f"{BASE_URL}algorithms/nsgaii/"))
    _write(root, f"{VERSION}/index.html", _redirect(immutable_url))
    _write(root, "website/index.html", _canonical(f"{BASE_URL}website/"))


def test_transition_bridge_accepts_only_explicit_clean_current_contract(tmp_path: Path) -> None:
    root = tmp_path / "clean"
    _build_clean_fixture(root)

    completed = _check(root)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["contract"] == "clean-current-transition"


def test_transition_bridge_rejects_missing_clean_homepage(tmp_path: Path) -> None:
    root = tmp_path / "missing-home"
    _build_clean_fixture(root)
    (root / "index.html").unlink()
    (root / "docs" / "stable" / "index.html").unlink()
    (root / "latest" / "index.html").unlink()

    rejected = _check(root)
    assert rejected.returncode != 0
    assert "Missing required portal file" in rejected.stderr


def test_transition_bridge_rejects_missing_deep_alias_route(tmp_path: Path) -> None:
    root = tmp_path / "missing-alias"
    _build_clean_fixture(root)
    (root / "latest" / "algorithms" / "nsgaii" / "index.html").unlink()

    rejected = _check(root)
    assert rejected.returncode != 0
    assert "route inventory does not match its source tree" in rejected.stderr


def test_transition_bridge_rejects_canonical_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "path-escape"
    _build_clean_fixture(root)
    outside = tmp_path / "outside.txt"
    outside.write_text("not part of the portal", encoding="utf-8")
    malicious = (
        f"{BASE_URL}docs/{VERSION}/"
        "%2e%2e/%2e%2e/%2e%2e/outside.txt"
    )
    _write(root, f"docs/{VERSION}/index.html", _canonical(malicious))

    rejected = _check(root)
    assert rejected.returncode != 0
    assert "Canonical URL escapes portal root" in rejected.stderr


def test_transition_bridge_rejects_commented_spoof_with_active_foreign_redirect(tmp_path: Path) -> None:
    root = tmp_path / "spoofed-redirect"
    _build_clean_fixture(root)
    expected = BASE_URL
    spoofed = (
        "<!doctype html><html><head>\n"
        f'<!-- <link rel="canonical" href="{expected}">'
        f'<meta http-equiv="refresh" content="0; url={expected}"> -->\n'
        '<link rel="canonical" href="https://evil.invalid/">\n'
        '<meta http-equiv="refresh" content="0; url=https://evil.invalid/">\n'
        "</head><body></body></html>\n"
    )
    _write(root, "docs/stable/index.html", spoofed)

    rejected = _check(root)
    assert rejected.returncode != 0
    assert "Redirect canonical mismatch" in rejected.stderr


def test_transition_bridge_rejects_duplicate_redirect_attributes(tmp_path: Path) -> None:
    root = tmp_path / "duplicate-attributes"
    _build_clean_fixture(root)
    duplicated = (
        "<!doctype html><html><head>\n"
        f'<link rel="canonical" href="https://evil.invalid/" href="{BASE_URL}">\n'
        f'<meta http-equiv="refresh" content="0; url=https://evil.invalid/" content="0; url={BASE_URL}">\n'
        "</head><body></body></html>\n"
    )
    _write(root, "docs/stable/index.html", duplicated)

    rejected = _check(root)
    assert rejected.returncode != 0
    assert "duplicate 'href' attribute" in rejected.stderr
    assert "duplicate 'content' attribute" in rejected.stderr


def test_transition_bridge_ignores_redirect_directives_inside_template(tmp_path: Path) -> None:
    root = tmp_path / "template-spoof"
    _build_clean_fixture(root)
    inert = (
        "<!doctype html><html><head><template>\n"
        f'<link rel="canonical" href="{BASE_URL}">\n'
        f'<meta http-equiv="refresh" content="0; url={BASE_URL}">\n'
        "</template></head><body></body></html>\n"
    )
    _write(root, "docs/stable/index.html", inert)

    rejected = _check(root)
    assert rejected.returncode != 0
    assert "Redirect canonical mismatch" in rejected.stderr


def test_trusted_preview_workflow_uses_transition_validator() -> None:
    workflow = Path(".github/workflows/docs-cloudflare-preview.yml").read_text(encoding="utf-8")
    assert "python tools/check_docs_portal_transition.py" in workflow
    assert "ref: main" in workflow
    assert "persist-credentials: false" in workflow
