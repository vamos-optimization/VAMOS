"""Every script in tools/ has an owner that runs or documents it, and points at code that exists (VR-314, VR-322)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_TEXT_SUFFIXES = {".cfg", ".ini", ".json", ".md", ".ps1", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}


def _tracked_texts() -> dict[str, str]:
    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True).stdout
    texts = {}
    for item in listed.split(b"\0"):
        relative = item.decode("utf-8")
        path = ROOT / relative
        if relative and path.suffix in _TEXT_SUFFIXES and path.is_file():
            texts[relative] = path.read_text(encoding="utf-8", errors="replace")
    return texts


def test_every_tool_script_is_referenced() -> None:
    texts = _tracked_texts()
    tools = sorted(
        relative for relative in texts if relative.startswith("tools/") and relative.count("/") == 1 and relative.endswith(".py")
    )

    # A script nothing runs, imports or documents can break, or do damage, without anyone noticing.
    def referenced(tool: str) -> bool:
        stem = Path(tool).stem
        needles = (f"{stem}.py", f"tools.{stem}", f"import {stem}", f"from {stem} import")
        return any(needle in text for relative, text in texts.items() if relative != tool for needle in needles)

    assert [tool for tool in tools if not referenced(tool)] == []


def test_audit_facade_paths_exist() -> None:
    from tools.audit_full_repo import FACADE_PATHS

    assert [path for path in FACADE_PATHS if not path.is_file()] == []


@pytest.mark.parametrize("checker", ["check_learning_content", "check_notebook_links", "check_notebook_quality"])
def test_notebook_checkers_run_in_health(checker: str) -> None:
    assert f'[python, "tools/{checker}.py"]' in (ROOT / "tools" / "health.py").read_text(encoding="utf-8")


def _repository_with_a_checkpoint(root: Path) -> Path:
    # A tracked notebook, and the ignored checkpoint copy Jupyter writes next to it.
    notebook = root / "notebooks" / "lesson.ipynb"
    notebook.parent.mkdir(parents=True)
    notebook.write_text('{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}', encoding="utf-8")
    (root / ".gitignore").write_text(".ipynb_checkpoints/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    checkpoint = root / "notebooks" / ".ipynb_checkpoints" / "lesson-checkpoint.ipynb"
    checkpoint.parent.mkdir()
    checkpoint.write_text(notebook.read_text(encoding="utf-8"), encoding="utf-8")
    return notebook


@pytest.mark.parametrize("checker", ["check_notebook_links", "check_notebook_quality"])
def test_notebook_checkers_read_tracked_notebooks_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checker: str) -> None:
    import importlib

    module = importlib.import_module(f"tools.{checker}")
    notebook = _repository_with_a_checkpoint(tmp_path)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "NOTEBOOK_ROOT", tmp_path / "notebooks")
    assert module.iter_notebooks() == [notebook]


def test_learning_content_check_reads_tracked_files_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools import check_learning_content as module

    notebook = _repository_with_a_checkpoint(tmp_path)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    assert module.iter_files(tmp_path / "notebooks") == [notebook]
