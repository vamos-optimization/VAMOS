from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def artifact(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    return importlib.import_module("check_docs_artifact")


def make_portal(root: Path) -> None:
    (root / "index.html").write_bytes(b"<h1>Approved home</h1>")
    (root / "example.py").write_bytes(b"print('approved')\n")


def test_artifact_binds_bytes_and_source_commit(artifact, tmp_path):
    make_portal(tmp_path)
    payload = artifact.create_manifest(tmp_path, "a" * 40)
    assert artifact.verify_manifest(tmp_path, payload, "a" * 40) == 2
    with pytest.raises(ValueError, match="source commit"):
        artifact.verify_manifest(tmp_path, payload, "b" * 40)
    (tmp_path / "example.py").write_text("print('changed')\n")
    with pytest.raises(ValueError, match="bytes or file inventory"):
        artifact.verify_manifest(tmp_path, payload)


@pytest.mark.parametrize("mutation", ["missing", "extra", "symlink"])
def test_artifact_rejects_changed_inventory(artifact, tmp_path, mutation):
    make_portal(tmp_path)
    payload = artifact.create_manifest(tmp_path, "a" * 40)
    if mutation == "missing":
        (tmp_path / "example.py").unlink()
    elif mutation == "extra":
        (tmp_path / "unexpected.html").write_text("extra")
    else:
        try:
            (tmp_path / "alias").symlink_to(tmp_path / "index.html")
        except OSError:
            pytest.skip("Creating symlinks is unavailable on this platform")
    with pytest.raises(ValueError):
        artifact.verify_manifest(tmp_path, payload)


@pytest.mark.parametrize("name", ["../secret", "/absolute", "x/../y", "a\\b", "a//b"])
def test_artifact_rejects_escaping_or_ambiguous_paths(artifact, tmp_path, name):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": artifact.SCHEMA, "source_sha": "a" * 40, "files": {name: "b" * 64}, "current": [name]}))
    with pytest.raises(ValueError, match="Invalid artifact entry"):
        artifact.read_manifest(manifest)


def test_live_check_detects_stale_content_and_checks_every_current_file(artifact, tmp_path):
    make_portal(tmp_path)
    payload = artifact.create_manifest(tmp_path, "a" * 40)
    requested = []

    def fetch(url, timeout):
        requested.append(url)
        return (tmp_path / ("index.html" if url.endswith("/") else "example.py")).read_bytes()

    assert artifact.verify_live(payload, "https://docs.example/", fetch=fetch, attempts=1) == 2
    assert set(requested) == {"https://docs.example/", "https://docs.example/example.py"}
    with pytest.raises(ValueError, match="Deployed bytes differ"):
        artifact.verify_live(payload, "https://docs.example/", fetch=lambda *_: b"stale", attempts=1)


def test_manifest_current_inventory_cannot_omit_a_file(artifact, tmp_path):
    make_portal(tmp_path)
    payload = artifact.create_manifest(tmp_path, "a" * 40)
    payload["current"] = ["index.html"]
    with pytest.raises(ValueError, match="Current-content inventory"):
        artifact.verify_manifest(tmp_path, payload)
