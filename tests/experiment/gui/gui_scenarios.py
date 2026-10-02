"""Simulated-user scenarios for the NiceGUI pages, run in a separate interpreter.

``test_gui_app.py`` executes ``python gui_scenarios.py <scenario> <root>`` so that the
NiceGUI test harness never shares a process with the rest of the suite: on exit,
``nicegui.testing.user_simulation`` removes the modules that define page functions
(here ``vamos`` and its subpackages) from ``sys.modules``. A failing assertion exits
with status 1 and a traceback on stderr.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from nicegui.testing.user_simulation import user_simulation

from vamos.experiment.artifacts import save_result
from vamos.experiment.gui.app import GuiSettings, build_app
from vamos.experiment.unified import optimize

WAIT = 600  # should_see retries are 0.1 s apart: wait up to a minute for worker processes

Scenario = Callable[[Any, Path], Awaitable[None]]


def _element(user: Any, marker: str) -> Any:
    return user.find(marker=marker).elements.pop()


async def run_flow(user: Any, root: Path) -> None:
    await user.open("/run")
    await user.should_see("Run configuration")
    _element(user, "max-evaluations").set_value(2000)
    user.find(marker="launch").click()
    await user.should_see("Succeeded", retries=WAIT)
    await user.should_see("Stored canonical run")
    live = _element(user, "live-front")
    assert live.figure.layout.uirevision == "vamos-live-front"
    assert live.figure.data[0].type == "scatter"

    user.find(marker="front-plot").trigger("plotly_selected", [0, 2, 10_000])
    await user.should_see("2 of ")
    assert [row["solution"] for row in _element(user, "solutions-table").rows] == [0, 2]

    user.find("Download CSV").click()
    response = await user.download.next(timeout=5.0)
    lines = response.text.strip().splitlines()
    assert lines[0].startswith("solution,f1,f2")
    assert [line.split(",")[0] for line in lines[1:]] == ["0", "2"]
    assert any(path.name == "run" for path in (root / "gui-runs").glob("*/run"))


async def explorer_flow(user: Any, root: Path) -> None:
    (root / "study").mkdir()
    save_result(optimize("zdt1", algorithm="nsgaii", max_evaluations=300, pop_size=20, seed=4), root / "study" / "run")
    broken = root / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{}", encoding="utf-8")

    await user.open("/")
    await user.should_see("Results root:")
    rows = {row["relative_path"]: row for row in _element(user, "runs-table").rows}
    assert set(rows) == {"study/run", "broken"}, rows
    assert rows["study/run"]["algorithm"] == "nsgaii"
    assert rows["broken"]["status"] == "unreadable"

    table = user.find(marker="runs-table")
    table.trigger("selection", {"added": True, "rows": [rows["broken"]], "keys": [rows["broken"]["id"]]})
    await user.should_see("broken")
    await user.should_not_see("Verify integrity")

    table.trigger("selection", {"added": True, "rows": [rows["study/run"]], "keys": [rows["study/run"]["id"]]})
    await user.should_see("Verify integrity")
    user.find(marker="front-plot").trigger("plotly_click", [3])
    await user.should_see("1 of ")
    user.find("Verify integrity").click()
    for _ in range(WAIT):
        if any("Integrity valid" in str(message) for message in user.notify.messages):
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"verification notification not shown: {user.notify.messages}")


async def validate_and_cancel_flow(user: Any, root: Path) -> None:
    await user.open("/run")
    _element(user, "pop-size").set_value(50)
    _element(user, "max-evaluations").set_value(10)
    user.find(marker="launch").click()
    await asyncio.sleep(0.1)
    assert any(">= pop_size" in str(message) for message in user.notify.messages), user.notify.messages
    assert not (root / "gui-runs").exists()

    _element(user, "max-evaluations").set_value(5_000_000)
    user.find(marker="launch").click()
    await user.should_see("Running", retries=WAIT)
    user.find(marker="cancel").click()
    await user.should_see("Cancelled", retries=WAIT)
    await user.should_see("nothing was stored")


SCENARIOS: dict[str, Scenario] = {
    "run": run_flow,
    "explorer": explorer_flow,
    "validate-and-cancel": validate_and_cancel_flow,
}


async def _simulate(root: Path, scenario: Scenario) -> None:
    async with user_simulation() as user:
        build_app(GuiSettings(results_root=root, jobs_root=root / "gui-runs"))
        await scenario(user, root)


def main(argv: list[str]) -> int:
    name, root = argv[1], Path(argv[2])
    asyncio.run(_simulate(root, SCENARIOS[name]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
