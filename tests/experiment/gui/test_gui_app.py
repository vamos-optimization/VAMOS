"""Simulated-user tests of the NiceGUI pages (no browser and no pytest-asyncio required).

Each scenario in ``gui_scenarios.py`` runs in its own interpreter because the NiceGUI
test harness removes page-defining modules from ``sys.modules`` when it exits.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("nicegui")
pytest.importorskip("plotly")

pytestmark = pytest.mark.gui

SCENARIOS = Path(__file__).with_name("gui_scenarios.py")


@pytest.mark.parametrize("scenario", ["run", "explorer", "validate-and-cancel"])
def test_gui_scenario(scenario: str, tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(SCENARIOS), scenario, str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"scenario {scenario!r} failed:\n{proc.stdout}\n{proc.stderr}"
