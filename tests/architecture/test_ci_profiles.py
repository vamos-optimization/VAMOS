"""CI installs what users and the release get: declared pins, declared minimums, and the release build (VR-308, VR-309, VR-315)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

ROOT = Path(__file__).resolve().parents[2]


def _name(requirement: str) -> str:
    return re.sub(r"[-_.]+", "-", re.split(r"[\s<>=!~;\[]", requirement.strip(), maxsplit=1)[0]).lower()


def _pins(path: Path) -> dict[str, list[str]]:
    pins: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, _, rest = line.partition("==")
            pins.setdefault(_name(name), []).append(rest.split(";", 1)[0].strip())
    return pins


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _ci_jobs() -> dict:
    return yaml.load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)["jobs"]


_PIP_VALUE_OPTIONS = {"-c", "-r", "-e", "--constraint", "--requirement", "--editable"}


def _workflows() -> list[tuple[str, str]]:
    return [(path.name, path.read_text(encoding="utf-8")) for path in sorted((ROOT / ".github" / "workflows").glob("*.yml"))]


def _pip_installs() -> list[tuple[str, list[str]]]:
    """The arguments of each ``pip install`` command of the workflows, with the workflow's name."""
    installs = []
    for name, text in _workflows():
        for command in text.replace("\\\n", " ").splitlines():
            if "pip install" in command:
                installs.append((name, command.split("pip install", 1)[1].split()))
    return installs


def _installed_by_name(arguments: list[str]) -> list[str]:
    """The distributions that ``pip install`` arguments name directly, not through an extra or a file."""
    names = []
    skip = False
    for argument in arguments:
        if skip or argument in _PIP_VALUE_OPTIONS:
            skip = not skip
            continue
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*", argument):
            names.append(_name(argument))
    return names


def _installed_extras() -> list[tuple[str, str]]:
    """The extras that the workflows install: ``.[a,b]`` in a command, or the lists of the ``EXTRAS`` variable it expands."""
    found = []
    for name, text in _workflows():
        groups = [
            group
            for workflow, arguments in _pip_installs()
            if workflow == name
            for group in re.findall(r"\.\[([A-Za-z0-9_,-]+)\]", " ".join(arguments))
        ]
        for line in text.splitlines():
            key, _, value = line.strip().partition(":")
            if key == "EXTRAS":
                # The literal lists inside the ${{ }} expressions ('a,b' && 'c' || ',d'), and the text outside them.
                groups += re.findall(r"(?:&&|\|\|)\s*'([^']*)'", value) + [re.sub(r"\$\{\{.*?\}\}", ",", value)]
        found += [(name, extra.strip()) for group in groups for extra in group.split(",") if extra.strip()]
    return found


def test_ci_constraints_pin_only_declared_distributions() -> None:
    data = _project()
    requirements = [
        *data["project"]["dependencies"],
        *(requirement for group in data["project"]["optional-dependencies"].values() for requirement in group),
        *data["build-system"]["requires"],
    ]
    declared = {_name(requirement) for requirement in requirements}
    # A requirement such as dask[distributed] also installs the distribution its extra names.
    declared |= {_name(extra) for requirement in requirements for extra in re.findall(r"\[([^\]]+)\]", requirement)}
    # A maintainer tool's dependency that a workflow installs by name, such as pymoo for tools/benchmark_compare_pymoo.py.
    declared |= {name for _workflow, arguments in _pip_installs() for name in _installed_by_name(arguments)}

    undeclared = sorted(set(_pins(ROOT / "constraints" / "ci.txt")) - declared)
    assert undeclared == [], f"constraints/ci.txt pins distributions that nothing installs: {undeclared}"


def test_workflows_install_declared_extras_and_pinned_tools() -> None:
    # pip only warns about an extra that the package does not declare, and installs nothing for it.
    extras = set(_project()["project"]["optional-dependencies"])
    undeclared = sorted({f"{workflow}: {extra}" for workflow, extra in _installed_extras() if extra not in extras})
    assert undeclared == [], f"workflows install extras that pyproject.toml does not declare: {undeclared}"

    pins = set(_pins(ROOT / "constraints" / "ci.txt"))
    unpinned = [
        f"{workflow}: {name}"
        for workflow, arguments in _pip_installs()
        if "constraints/ci.txt" in arguments
        for name in _installed_by_name(arguments)
        if name not in pins
    ]
    assert unpinned == [], f"workflows install by name, with constraints/ci.txt, distributions that it does not pin: {unpinned}"


def test_minimum_profile_uses_declared_minimums() -> None:
    import subprocess
    import sys

    declared = {}
    for requirement in _project()["project"]["dependencies"]:
        match = re.fullmatch(r"([A-Za-z0-9._-]+)>=([0-9.]+)", requirement.strip())
        assert match is not None, f"core dependency without a plain minimum: {requirement}"
        declared[_name(match[1])] = match[2]
    printed = subprocess.run(
        [sys.executable, "tools/minimum_constraints.py"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert {_name(pin.split("==")[0]): pin.split("==")[1] for pin in printed} == declared

    job = _ci_jobs()["tests-minimum"]
    assert job["strategy"]["matrix"]["python-version"] == ["3.10"]
    assert _project()["project"]["requires-python"] == ">=3.10"
    scripts = "\n".join(step.get("run", "") for step in job["steps"])
    assert "python tools/minimum_constraints.py" in scripts
    assert "constraints/ci.txt ." not in scripts
    # pytest and VAMOS resolve together: a separate pytest install could keep a dependency that needs a newer core pin.
    installs = [line for line in scripts.splitlines() if "pip install" in line and "minimum.txt" in line]
    assert installs and all(" . pytest" in line for line in installs)
    assert "tests-minimum" in _ci_jobs()["build-wheel"]["needs"]


def test_build_wheel_job_builds_like_the_release() -> None:
    steps = _ci_jobs()["build-wheel"]["steps"]
    scripts = "\n".join(step.get("run", "") for step in steps)

    assert "-r release/requirements-build.txt" in scripts
    assert "python -m build --no-isolation" in scripts
    assert "constraints/ci.txt" not in scripts
