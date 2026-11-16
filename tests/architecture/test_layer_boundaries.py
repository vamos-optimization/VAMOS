from __future__ import annotations

import ast
from pathlib import Path

BASELINE_VIOLATIONS: set[tuple[str, str, str]] = set()

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src" / "vamos"
LAYERS = {"foundation", "engine", "ux", "experiment", "resources"}

# Allowed direct layer imports (source -> set(targets)).
ALLOWED_LAYER_IMPORTS: dict[str, set[str]] = {
    "foundation": {"foundation", "resources"},
    "engine": {"engine", "foundation", "resources"},
    "experiment": {"experiment", "foundation", "engine", "ux", "resources"},
    "ux": {"ux", "foundation", "engine", "resources"},
    "resources": {"resources"},
}


def _layer_from_path(path: Path) -> str | None:
    try:
        rel = path.relative_to(SRC_ROOT)
    except ValueError:
        return None
    if not rel.parts:
        return None
    layer = rel.parts[0]
    return layer if layer in LAYERS else None


def _module_from_path(path: Path) -> str:
    rel = path.relative_to(ROOT / "src")
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def _layer_from_module(module: str) -> str | None:
    if not module.startswith("vamos."):
        return None
    parts = module.split(".")
    if len(parts) < 2:
        return None
    return parts[1] if parts[1] in LAYERS else None


def _resolve_relative(current_module: str, module: str | None, level: int) -> str:
    cur_parts = current_module.split(".") if current_module else []
    if level > len(cur_parts):
        base_parts: list[str] = []
    else:
        base_parts = cur_parts[:-level]
    if module:
        base_parts += module.split(".")
    return ".".join(base_parts)


def _iter_python_files() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _detect_violations():
    violations: set[tuple[str, str, str]] = set()
    line_map: dict[tuple[str, str, str], list[int]] = {}

    for path in _iter_python_files():
        source_layer = _layer_from_path(path)
        if source_layer not in LAYERS:
            continue
        text = _read_text(path)
        tree = ast.parse(text)
        current_module = _module_from_path(path)
        rel_path = path.relative_to(ROOT).as_posix()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    if not module.startswith("vamos."):
                        continue
                    target_layer = _layer_from_module(module)
                    record = _violation_record(source_layer, target_layer, rel_path, module)
                    if record:
                        _add_violation(record, node.lineno, violations, line_map)
            elif isinstance(node, ast.ImportFrom):
                level = node.level or 0
                module = node.module

                if level == 0 and module:
                    if module == "vamos":
                        for alias in node.names:
                            if alias.name == "*":
                                continue
                            resolved = f"vamos.{alias.name}"
                            target_layer = _layer_from_module(resolved)
                            record = _violation_record(source_layer, target_layer, rel_path, resolved)
                            if record:
                                _add_violation(record, node.lineno, violations, line_map)
                    elif module.startswith("vamos."):
                        target_layer = _layer_from_module(module)
                        record = _violation_record(source_layer, target_layer, rel_path, module)
                        if record:
                            _add_violation(record, node.lineno, violations, line_map)
                    continue

                if level > 0:
                    if module is not None:
                        resolved = _resolve_relative(current_module, module, level)
                        if resolved.startswith("vamos."):
                            target_layer = _layer_from_module(resolved)
                            record = _violation_record(source_layer, target_layer, rel_path, resolved)
                            if record:
                                _add_violation(record, node.lineno, violations, line_map)
                    else:
                        base = _resolve_relative(current_module, None, level)
                        for alias in node.names:
                            if alias.name == "*":
                                continue
                            resolved = f"{base}.{alias.name}" if base else alias.name
                            if not resolved.startswith("vamos."):
                                continue
                            target_layer = _layer_from_module(resolved)
                            record = _violation_record(source_layer, target_layer, rel_path, resolved)
                            if record:
                                _add_violation(record, node.lineno, violations, line_map)

    for record, lines in line_map.items():
        line_map[record] = sorted(set(lines))
    return violations, line_map


def _violation_record(source_layer: str, target_layer: str | None, rel_path: str, module: str):
    if target_layer is None:
        return None
    if source_layer == target_layer:
        return None
    allowed = ALLOWED_LAYER_IMPORTS.get(source_layer, {source_layer})
    if target_layer in allowed:
        return None
    return (f"{source_layer}->{target_layer}", rel_path, module)


def _add_violation(record, lineno: int, violations, line_map):
    violations.add(record)
    line_map.setdefault(record, []).append(int(lineno))


def _format_record(record: tuple[str, str, str]) -> str:
    vtype, rel_path, module = record
    return f"({vtype}) {rel_path} -> {module}"


def _format_diff(new, missing, line_map) -> str:
    lines = []
    if new:
        lines.append("NEW VIOLATIONS")
        for record in sorted(new):
            line_list = line_map.get(record, [])
            preview = ", ".join(str(n) for n in line_list[:3]) or "n/a"
            if len(line_list) > 3:
                preview = f"{preview}, ..."
            lines.append(f"- {_format_record(record)} (lines: {preview})")
    if missing:
        lines.append("MISSING (baseline no longer detected)")
        for record in sorted(missing):
            lines.append(f"- {_format_record(record)}")
    return "\n".join(lines)


def test_every_vamos_package_is_a_layer():
    # A new top-level package needs a place in the layering policy of AGENTS.md first.
    packages = {path.parent.name for path in SRC_ROOT.glob("*/__init__.py")}

    assert packages == LAYERS
    # vamos.assist, the LLM-assisted planning workflow, was removed; stale bytecode may remain in a checkout.
    assert not list((SRC_ROOT / "assist").rglob("*.py"))


def test_layer_boundaries_do_not_regress():
    detected, line_map = _detect_violations()
    new = detected - BASELINE_VIOLATIONS
    missing = BASELINE_VIOLATIONS - detected

    if new or missing:
        raise AssertionError(_format_diff(new, missing, line_map))
