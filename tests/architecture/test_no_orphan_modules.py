"""Every module under ``src/vamos`` is reachable through imports from a facade or an entry point."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PACKAGE = SRC / "vamos"

# Curated facades that users import directly, besides the top-level ``vamos.*`` modules.
FACADES = ("vamos.ux.api",)

# Modules that nothing imports on purpose, and why. Remove an entry when its
# module gains an importer or is deleted; the test fails on stale entries.
ALLOWED_UNREACHABLE: dict[str, str] = {
    "vamos.engine.tuning._irace_target_runner": "irace runs it as a script (the targetRunner of the irace backend).",
    "vamos.ux.studio.app": "Studio entry module, run with `python -m vamos.ux.studio.app`.",
    "vamos.resources.reference_fronts": "Data package read through importlib.resources.",
    "vamos.foundation.constraints.dsl": "Unused constraint DSL; its removal is review finding VR-164.",
    "vamos.engine.tuning.api": "Experimental tuning facade for direct imports.",
    "vamos.ux.analysis.tuning_viz": "Experimental tuning plots for direct imports.",
    "vamos.ux.analytics": "Experimental genealogy analytics for direct imports.",
    "vamos.ux.analytics.genealogy": "Experimental genealogy analytics for direct imports.",
}

_MODULE_STRING = re.compile(r"vamos(?:\.\w+)+")
_CONSOLE_SCRIPT = re.compile(r'^\s*[\w-]+\s*=\s*"(vamos(?:\.\w+)*):\w+"', re.MULTILINE)


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _modules() -> dict[str, Path]:
    return {_module_name(path): path for path in sorted(PACKAGE.rglob("*.py"))}


def _referenced_names(name: str, path: Path) -> set[str]:
    """Absolute names that ``path`` imports, anywhere in the file, or names as a module string."""
    package = name.split(".") if path.name == "__init__.py" else name.split(".")[:-1]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                module = ".".join([*base, node.module] if node.module else base)
            else:
                module = node.module or ""
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Lazy loaders name modules as strings, such as "vamos.a.b" or "vamos.a.b:attr".
            match = _MODULE_STRING.fullmatch(node.value.split(":", 1)[0])
            if match:
                found.add(match.group(0))
    return found


def _import_graph(modules: dict[str, Path]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for name, path in modules.items():
        edges: set[str] = set()
        for target in _referenced_names(name, path):
            parts = target.split(".")
            # Importing a.b.c also imports the packages a and a.b.
            edges.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))
        graph[name] = {edge for edge in edges if edge in modules and edge != name}
    return graph


def _roots(modules: dict[str, Path]) -> set[str]:
    top_level = {name for name, path in modules.items() if path.parent == PACKAGE}
    scripts = set(_CONSOLE_SCRIPT.findall((ROOT / "pyproject.toml").read_text(encoding="utf-8")))
    main_modules = {name for name in modules if name.endswith(".__main__")}
    return {"vamos", *top_level, *FACADES, *scripts, *main_modules}


def _unreachable_modules() -> set[str]:
    modules = _modules()
    graph = _import_graph(modules)
    seen: set[str] = set()
    stack = list(_roots(modules))
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(graph.get(name, ()))
    return set(modules) - seen


def test_every_module_has_an_importer() -> None:
    unexpected = sorted(_unreachable_modules() - set(ALLOWED_UNREACHABLE))
    assert not unexpected, (
        "These modules are not imported, directly or through other modules, from a facade, "
        "the console script or a `__main__` module. Delete them, import them, or add them to "
        f"ALLOWED_UNREACHABLE with the reason: {unexpected}"
    )


def test_allowed_unreachable_entries_are_current() -> None:
    stale = sorted(set(ALLOWED_UNREACHABLE) - _unreachable_modules())
    assert not stale, f"Remove these entries from ALLOWED_UNREACHABLE; they are reachable or gone: {stale}"
