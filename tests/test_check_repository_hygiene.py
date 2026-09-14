from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path

from tools.check_repository_hygiene import collect_violations, distribution_violations, load_policy

ROOT = Path(__file__).resolve().parents[1]


def _policy() -> dict[str, object]:
    return copy.deepcopy(load_policy(ROOT / "release" / "repository-hygiene-policy.json"))


def _codes(tmp_path: Path, paths: list[str], *, policy: dict[str, object] | None = None) -> set[str]:
    violations = collect_violations(tmp_path, tracked_paths=paths, policy=policy or _policy(), exceptions=[])
    return {item.code for item in violations}


def test_detects_root_fun_csv(tmp_path: Path) -> None:
    (tmp_path / "FUN.csv").write_text("1,2\n", encoding="utf-8")
    assert "forbidden_root_name" in _codes(tmp_path, ["FUN.csv"])


def test_detects_root_png(tmp_path: Path) -> None:
    (tmp_path / "plot.png").write_bytes(b"png")
    assert "root_output_file" in _codes(tmp_path, ["plot.png"])


def test_rejects_tracked_manuscript_but_allows_local_workspace(tmp_path: Path) -> None:
    path = tmp_path / "paper" / "manuscript" / "main.tex"
    path.parent.mkdir(parents=True)
    path.write_text("local manuscript", encoding="utf-8")
    assert _codes(tmp_path, []) == set()
    assert "forbidden_top_level" in _codes(tmp_path, ["paper/manuscript/main.tex"])


def test_rejects_manuscript_sources_in_distribution(tmp_path: Path) -> None:
    wheel = tmp_path / "example.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("project/paper/manuscript/main.tex", "local manuscript")
    violations = distribution_violations([wheel], _policy())
    assert {item.code for item in violations} == {"forbidden_distribution_content"}


def test_detects_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "tools" / ".tmp_probe.txt"
    path.parent.mkdir()
    path.write_text("scratch", encoding="utf-8")
    assert "temporary_name" in _codes(tmp_path, ["tools/.tmp_probe.txt"])


def test_detects_duplicate_audit_file(tmp_path: Path) -> None:
    first = tmp_path / "final_audit_latest.md"
    second = tmp_path / "docs" / "final_audit_001.md"
    second.parent.mkdir()
    first.write_text("audit", encoding="utf-8")
    second.write_text("audit", encoding="utf-8")
    assert "audit_evidence" in _codes(tmp_path, ["final_audit_latest.md", "docs/final_audit_001.md"])


def test_detects_zip_with_expanded_duplicate(tmp_path: Path) -> None:
    expanded = tmp_path / "bundle" / "source.txt"
    expanded.parent.mkdir()
    expanded.write_text("same bytes", encoding="utf-8")
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("bundle/source.txt", b"same bytes")
    codes = _codes(tmp_path, ["bundle.zip", "bundle/source.txt"])
    assert "archive_extracted_duplicate" in codes


def test_detects_oversized_unexplained_binary(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "assets" / "huge.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * 64)
    policy = _policy()
    policy["large_file_bytes"] = 32
    assert "unapproved_large_file" in _codes(tmp_path, ["docs/assets/huge.bin"], policy=policy)


def test_detects_notebook_with_outputs(tmp_path: Path) -> None:
    path = tmp_path / "notebooks" / "dirty.ipynb"
    path.parent.mkdir()
    payload = {
        "cells": [{"cell_type": "code", "execution_count": 1, "outputs": [{"output_type": "stream", "text": "x"}]}],
        "metadata": {"kernelspec": {"display_name": "Python 3 (VAMOS)", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert "notebook_output" in _codes(tmp_path, ["notebooks/dirty.ipynb"])


def test_detects_audit_handoff_in_root(tmp_path: Path) -> None:
    path = tmp_path / "goal_handoff.md"
    path.write_text("handoff", encoding="utf-8")
    assert "audit_evidence" in _codes(tmp_path, ["goal_handoff.md"])


def test_detects_personal_windows_path(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "bad.md"
    path.parent.mkdir()
    personal = "C:" + "\\Users\\alice\\project"
    path.write_text(personal, encoding="utf-8")
    assert "personal_absolute_path" in _codes(tmp_path, ["docs/bad.md"])


def test_detects_personal_posix_home_path(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "bad.md"
    path.parent.mkdir()
    personal = "/" + "home/alice/project"
    path.write_text(personal, encoding="utf-8")
    assert "personal_absolute_path" in _codes(tmp_path, ["docs/bad.md"])


def test_detects_forbidden_distribution_content(tmp_path: Path) -> None:
    wheel = tmp_path / "example.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("project/reports/local.json", "{}")
    violations = distribution_violations([wheel], _policy())
    assert {item.code for item in violations} == {"forbidden_distribution_content"}
