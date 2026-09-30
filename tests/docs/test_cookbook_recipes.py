"""Execute the actual published cookbook snippets and beginner CLI workflow."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COOKBOOK = ROOT / "docs/guide/cookbook.md"
MINIMAL = ROOT / "docs/guide/minimal-python.md"


def _sections() -> dict[str, str]:
    pieces = re.split(r"^## ", COOKBOOK.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return {part.splitlines()[0]: part for part in pieces[1:]}


def _blocks(text: str, language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)\n```", text, re.DOTALL)


def _run_python(code: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "recipe.py"
    script.write_text(code, encoding="utf-8")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


PYTHON_RECIPES = [title for title, body in _sections().items() if _blocks(body, "python")]
OPTIONAL_MODULES = {"5.": "numba", "6.": "matplotlib", "17.": "pandas", "19.": "moocore", "20.": "moocore"}


@pytest.mark.smoke
@pytest.mark.parametrize("title", PYTHON_RECIPES)
def test_cookbook_python_recipe(title: str, tmp_path: Path) -> None:
    for prefix, module in OPTIONAL_MODULES.items():
        if title.startswith(prefix) and importlib.util.find_spec(module) is None:
            pytest.skip(f"Recipe needs optional {module} dependency")
    sections = _sections()
    code = "\n\n".join(_blocks(sections[title], "python"))
    if title.startswith("15."):
        setup = next(body for heading, body in sections.items() if heading.startswith("13."))
        code = "\n\n".join([*_blocks(setup, "python"), code])
    completed = _run_python(code, tmp_path)
    assert completed.returncode == 0, f"{title}\n{completed.stdout}\n{completed.stderr}"
    if title.startswith("6."):
        assert (tmp_path / "algorithm-comparison.png").stat().st_size > 0
    if title.startswith("17."):
        assert (tmp_path / "exports/zdt1-results.csv").stat().st_size > 0
    if title.startswith("15."):
        assert (tmp_path / "results/cookbook-replay/manifest.json").is_file()


def _run_cli(command: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    args = shlex.split(command)
    assert args.pop(0) == "vamos"
    return subprocess.run(
        [sys.executable, "-m", "vamos.experiment.cli.main", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.smoke
@pytest.mark.parametrize("number", ["11.", "12.", "16."])
def test_cookbook_cli_recipe(number: str, tmp_path: Path) -> None:
    body = next(body for title, body in _sections().items() if title.startswith(number))
    for config in _blocks(body, "json"):
        (tmp_path / "experiment.json").write_text(config, encoding="utf-8")
    for block in _blocks(body, "bash"):
        for command in block.splitlines():
            completed = _run_cli(command, tmp_path)
            assert completed.returncode == 0, f"{command}\n{completed.stdout}\n{completed.stderr}"
    manifests = list((tmp_path / "results").rglob("manifest.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["status"] == "succeeded"


@pytest.mark.smoke
def test_minimal_python_cli_path(tmp_path: Path) -> None:
    source = MINIMAL.read_text(encoding="utf-8")
    commands = [line for block in _blocks(source, "bash") for line in block.splitlines() if line.startswith("vamos ")]
    # The interactive wizard is an optional alternative, not an unattended recipe.
    commands = [command for command in commands if command != "vamos quickstart"]
    assert len(commands) == 8
    for command in commands:
        completed = _run_cli(command, tmp_path)
        assert completed.returncode == 0, f"{command}\n{completed.stdout}\n{completed.stderr}"
    assert (tmp_path / "quickstart.json").is_file()
    assert len(list((tmp_path / "results").rglob("manifest.json"))) == 5
