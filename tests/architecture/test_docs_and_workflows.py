from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_vscode_python_modules_exist() -> None:
    launch = json.loads((ROOT / ".vscode" / "launch.json").read_text(encoding="utf-8"))
    modules = [cfg["module"] for cfg in launch["configurations"] if "module" in cfg and cfg["module"] != "pytest"]
    modules.extend(
        [
            "vamos.experiment.cli.main",
            "vamos.experiment.diagnostics.self_check",
        ]
    )

    missing = [module for module in sorted(set(modules)) if importlib.util.find_spec(module) is None]

    assert missing == []


def test_vscode_tasks_use_current_entrypoints() -> None:
    text = (ROOT / ".vscode" / "tasks.json").read_text(encoding="utf-8")

    assert "python -m vamos.experiment.cli.main" in text
    assert "python -m vamos.experiment.diagnostics.self_check" in text
    assert "vamos.cli.main" not in text
    assert "vamos.diagnostics.self_check" not in text


def test_readme_notebook_links_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    notebook_paths = re.findall(r"`(notebooks/[^`]+\.ipynb)`", readme)

    missing = [path for path in notebook_paths if not (ROOT / path).exists()]

    assert missing == []


def test_no_supported_jax_engine_claims_remain() -> None:
    paths = [
        ROOT / "README.md",
        ROOT / "CHANGELOG.md",
        ROOT / "notebooks" / "0_basic" / "05_interactive_tutorial.ipynb",
        ROOT / "notebooks" / "2_advanced" / "23_backends_and_performance.ipynb",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert 'engine="jax"' not in text
    assert '\\"jax\\"' not in text
    assert "JAX Support" not in text


def _health_gate_commands() -> list[str]:
    tree = ast.parse((ROOT / "tools" / "health.py").read_text(encoding="utf-8"))
    gates = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "commands"
    )
    assert isinstance(gates, ast.List)
    commands = []
    for gate in gates.elts:
        assert isinstance(gate, ast.Tuple) and isinstance(gate.elts[1], ast.List)
        words = ["python" if isinstance(word, ast.Name) else ast.literal_eval(word) for word in gate.elts[1].elts]
        commands.append(" ".join(words))
    return commands


def test_architecture_health_lists_the_health_gates() -> None:
    text = (ROOT / "docs" / "dev" / "architecture_health.md").read_text(encoding="utf-8")
    heading = re.search(r"^## Health gates.*$", text, flags=re.MULTILINE | re.IGNORECASE)
    assert heading is not None
    section = text[heading.end() :].split("\n## ", 1)[0]
    listed = re.findall(r"^(?:-|\d+\.) `([^`]+)`", section, flags=re.MULTILINE)

    assert listed == _health_gate_commands()


def test_ci_architecture_step_runs_the_health_test_files() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    step = workflow.split("- name: Architecture health checks", 1)[1].split("- name:", 1)[0]
    ci_files = set(re.findall(r"tests/\S+\.py", step))
    health_files = {command.split()[-1] for command in _health_gate_commands() if command.startswith("python -m pytest")}

    assert sorted(health_files - ci_files) == []
