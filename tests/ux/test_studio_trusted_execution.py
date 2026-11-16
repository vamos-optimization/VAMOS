from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vamos.ux.panel import launcher

ROOT = Path(__file__).resolve().parents[2]


def test_studio_compiles_and_runs_no_python_text() -> None:
    # Studio runs registered problems only; the Problem Builder, which ran Python typed or drafted by an LLM, was removed.
    calls = []
    for path in sorted([*(ROOT / "src/vamos/ux/panel").rglob("*.py"), *(ROOT / "src/vamos/ux/studio").rglob("*.py")]):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval", "compile"}:
                calls.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}: {node.func.id}()")

    assert calls == []


def test_launcher_command_binds_to_loopback_by_default() -> None:
    command = launcher._build_panel_command("results", (), show=False)

    assert "--address=127.0.0.1" in command
    assert "--port=5006" in command
    assert "--show" not in command


def test_remote_binding_is_rejected_without_dangerous_use_acknowledgement() -> None:
    with pytest.raises(SystemExit) as raised:
        launcher.main(["--address", "0.0.0.0", "--no-browser"])

    assert raised.value.code == 2


def test_remote_binding_requires_and_accepts_separate_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setitem(sys.modules, "panel", ModuleType("panel"))

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert launcher.main(["--address", "0.0.0.0", "--allow-remote-binding", "--no-browser"]) == 0
    assert "--address=0.0.0.0" in commands[0]
    assert "WARNING" in capsys.readouterr().err


def test_studio_inspection_module_has_no_execution_primitives() -> None:
    source = (ROOT / "src/vamos/ux/studio/data.py").read_text(encoding="utf-8")

    for forbidden in ("optimize(", "reproduce(", "subprocess", "exec(", "eval("):
        assert forbidden not in source
