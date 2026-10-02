from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from vamos.experiment.gui import cli

pytestmark = [pytest.mark.gui, pytest.mark.cli]


def _run_vamos(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "vamos.experiment.cli.main", *args], cwd=cwd, capture_output=True, text=True, timeout=120)


def test_gui_help_is_side_effect_free(tmp_path: Path) -> None:
    proc = _run_vamos("gui", "--help", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "usage: vamos gui" in proc.stdout
    assert "--allow-remote-binding" in proc.stdout
    assert list(tmp_path.iterdir()) == []


def test_vamos_help_lists_the_gui_command(tmp_path: Path) -> None:
    proc = _run_vamos("help", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "vamos gui" in proc.stdout


def test_parsing_does_not_import_nicegui() -> None:
    code = "import sys\nfrom vamos.experiment.gui.cli import build_parser\nbuild_parser().format_help()\nprint('nicegui' in sys.modules)"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_remote_binding_requires_acknowledgement(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--address", "0.0.0.0"])
    assert exc.value.code == 2
    assert "--allow-remote-binding" in capsys.readouterr().err


def test_invalid_port_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--port", "0"])
    assert exc.value.code == 2
    assert "--port" in capsys.readouterr().err


def test_missing_optional_dependencies_exit_with_an_install_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "missing_dependencies", lambda: ["nicegui"])
    assert cli.main([str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "nicegui" in err
    assert 'pip install "vamos-optimization[gui]"' in err


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("[::1]", True),
        ("0.0.0.0", False),
        ("192.168.1.10", False),
        ("example.org", False),
    ],
)
def test_loopback_detection(address: str, expected: bool) -> None:
    assert cli.is_loopback_address(address) is expected


def test_launch_passes_resolved_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("nicegui")
    pytest.importorskip("plotly")
    import vamos.experiment.gui.app as app

    captured: dict[str, object] = {}

    def fake_run_gui(settings: app.GuiSettings, **kwargs: object) -> None:
        captured.update(settings=settings, **kwargs)

    monkeypatch.setattr(app, "run_gui", fake_run_gui)
    assert cli.main([str(tmp_path), "--no-browser", "--port", "9123"]) == 0
    settings = captured["settings"]
    assert isinstance(settings, app.GuiSettings)
    assert settings.results_root == tmp_path.resolve()
    assert settings.jobs_root == tmp_path.resolve() / cli.JOBS_DIRNAME
    assert (captured["host"], captured["port"], captured["show"], captured["native"]) == ("127.0.0.1", 9123, False, False)
