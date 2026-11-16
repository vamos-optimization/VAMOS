from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src" / "vamos"
LAYER_DIRS = ("foundation", "engine", "ux", "experiment", "assist")
# Layers import the modules these facades re-export. The artifact facades
# `vamos.run_artifacts` and `vamos.study_artifacts` are the exception: ux reads
# stored runs and studies through them (tests/architecture/test_layer_boundaries.py).
FACADES = {"vamos.api", "vamos.algorithms", "vamos.ux.api", "vamos.problems", "vamos.exceptions", "vamos.selection"}


def _iter_layer_files() -> list[Path]:
    paths: list[Path] = []
    for layer in LAYER_DIRS:
        root = SRC_ROOT / layer
        if root.exists():
            paths.extend(root.rglob("*.py"))
    return sorted(paths)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _detect_violations() -> list[tuple[str, int, str]]:
    violations: list[tuple[str, int, str]] = []

    for path in _iter_layer_files():
        text = _read_text(path)
        tree = ast.parse(text)
        rel_path = path.relative_to(ROOT).as_posix()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "vamos" or alias.name in FACADES:
                        violations.append((rel_path, int(node.lineno), alias.name))
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                if module == "vamos" or module in FACADES:
                    imported = ", ".join(alias.name for alias in node.names) or "*"
                    violations.append((rel_path, int(node.lineno), f"{module}:{imported}"))
                elif module == "vamos.ux" and any(alias.name == "api" for alias in node.names):
                    violations.append((rel_path, int(node.lineno), "vamos.ux:api"))

    return violations


def test_no_facade_imports_in_layers() -> None:
    violations = _detect_violations()
    if not violations:
        return

    lines = ["Facade imports detected inside layer packages:"]
    for rel_path, lineno, detail in sorted(violations):
        lines.append(f"- {rel_path}:{lineno} -> {detail}")
    raise AssertionError("\n".join(lines))
