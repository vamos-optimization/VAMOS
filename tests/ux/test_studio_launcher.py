"""`vamos studio` without its optional extra (review finding VR-128)."""

from __future__ import annotations

import sys

import pytest

from vamos.ux.panel import launcher


def test_studio_without_the_extra_exits_with_an_install_hint(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setitem(sys.modules, "panel", None)  # `import panel` raises ImportError

    with pytest.raises(SystemExit) as exited:
        launcher.main(["--no-browser"])

    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert 'pip install "vamos-optimization[studio]"' in err and "Traceback" not in err
