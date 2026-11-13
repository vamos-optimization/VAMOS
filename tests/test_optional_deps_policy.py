"""Optional dependencies are imported only where docs/dev/architecture_health.md allows.

The watched packages are the extras of ``[project.optional-dependencies]`` in pyproject.toml,
by import name. Only module-level statements count; an import inside a function is lazy.
"""

from __future__ import annotations

import ast
import functools
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

from tests.architecture._dependencies import (
    OPTIMIZATION_FRAMEWORKS,
    base_distributions,
    distribution_name,
    import_names,
    optional_distributions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "vamos"

EXTRAS = optional_distributions()
STUDIO = EXTRAS["studio"]
OPTIONAL = sorted(set().union(*EXTRAS.values()))

_GUARD_EXCEPTIONS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
_STUDIO_UI = ("src/vamos/ux/panel/", "src/vamos/ux/studio/")


@dataclass(frozen=True)
class _ModuleLevelImport:
    path: str
    line: int
    names: tuple[str, ...]
    guarded: bool
    dynamic: bool


def _guards_import_error(node: ast.Try) -> bool:
    for handler in node.handlers:
        if handler.type is None:
            return True
        types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
        if any(isinstance(item, ast.Name) and item.id in _GUARD_EXCEPTIONS for item in types):
            return True
    return False


def _module_level_statements(tree: ast.Module) -> Iterator[tuple[ast.stmt, bool]]:
    for node in tree.body:
        if isinstance(node, ast.Try):
            guarded = _guards_import_error(node)
            for statement in node.body:
                yield statement, guarded
        else:
            yield node, False


def _calls_run_at_import(node: ast.AST) -> Iterator[ast.Call]:
    # Function bodies run when called, not at import; class bodies run at import.
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return
    if isinstance(node, ast.Call):
        yield node
    for child in ast.iter_child_nodes(node):
        yield from _calls_run_at_import(child)


def _static_names(node: ast.Import | ast.ImportFrom) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if node.level or not node.module:
        return ()
    return (node.module, *(f"{node.module}.{alias.name}" for alias in node.names))


def _dynamic_name(call: ast.Call) -> str | None:
    func = call.func
    called = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
    if called not in {"import_module", "__import__"} or not call.args:
        return None
    argument = call.args[0]
    return argument.value if isinstance(argument, ast.Constant) and isinstance(argument.value, str) else None


@functools.cache
def _imported_names() -> frozenset[str]:
    """Every absolute module name that src/vamos imports, at module level or inside a function."""
    found: set[str] = set()
    for path in SRC_ROOT.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                found.update(_static_names(node))
            elif isinstance(node, ast.Call) and (name := _dynamic_name(node)) is not None:
                found.add(name)
    return frozenset(found)


def _imports(modules: tuple[str, ...]) -> bool:
    return any(name == module or name.startswith(f"{module}.") for name in _imported_names() for module in modules)


@functools.cache
def _module_level_imports() -> tuple[_ModuleLevelImport, ...]:
    found: list[_ModuleLevelImport] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel_path)
        for statement, guarded in _module_level_statements(tree):
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                found.append(_ModuleLevelImport(rel_path, statement.lineno, _static_names(statement), guarded, dynamic=False))
            for call in _calls_run_at_import(statement):
                name = _dynamic_name(call)
                if name is not None:
                    found.append(_ModuleLevelImport(rel_path, call.lineno, (name,), guarded, dynamic=True))
    return tuple(found)


def _is_kernel_backend_of(path: str, modules: tuple[str, ...]) -> bool:
    # foundation/kernel/<backend>_*.py imports its own package; registry.py loads it on demand.
    stem = Path(path).stem.lstrip("_")
    return path.startswith("src/vamos/foundation/kernel/") and any(stem.startswith(f"{module.partition('.')[0]}_") for module in modules)


def _violation(record: _ModuleLevelImport, distribution: str, modules: tuple[str, ...]) -> str | None:
    if record.dynamic:
        return "dynamic import at module level"
    if record.path.startswith(("src/vamos/foundation/", "src/vamos/engine/")):
        return None if _is_kernel_backend_of(record.path, modules) else "foundation and engine import it only inside functions"
    if distribution in STUDIO and record.path.startswith(_STUDIO_UI):
        return None
    return None if record.guarded else "module-level import outside try/except ImportError"


_KNOWN_DEPARTURES = {
    "moocore": pytest.mark.xfail(
        strict=True,
        reason=(
            "engine/algorithm/components/archive_core.py, components/subset_selection.py and "
            "foundation/quality_indicators/moocore_indicators.py import moocore at module level inside try/except"
        ),
    ),
}


@pytest.mark.parametrize(
    "distribution", [pytest.param(name, marks=_KNOWN_DEPARTURES[name]) if name in _KNOWN_DEPARTURES else name for name in OPTIONAL]
)
def test_optional_deps_policy(distribution: str) -> None:
    modules = import_names(distribution)
    violations = []
    for record in _module_level_imports():
        if not any(name == module or name.startswith(f"{module}.") for name in record.names for module in modules):
            continue
        reason = _violation(record, distribution, modules)
        if reason is not None:
            violations.append(f"- {record.path}:{record.line}: {reason}")
    assert not violations, f"{distribution} ({', '.join(modules)}) is imported where the policy forbids it:\n" + "\n".join(violations)


def test_declared_dependencies_map_to_their_import_names() -> None:
    # Checks the installed distributions only; CI installs every extra on Linux.
    provided: dict[str, set[str]] = {}
    for top_level, distributions in packages_distributions().items():
        for distribution in distributions:
            provided.setdefault(distribution_name(distribution), set()).add(top_level)
    wrong = {
        distribution: sorted(provided[distribution])
        for distribution in sorted(base_distributions().union(*EXTRAS.values()))
        if distribution in provided and not {name.partition(".")[0] for name in import_names(distribution)} <= provided[distribution]
    }
    assert not wrong, f"add these distributions to IMPORT_NAMES in tests/architecture/_dependencies.py (they provide): {wrong}"


def test_analysis_extra_holds_only_what_vamos_imports() -> None:
    # The examples and notebooks import more; those packages belong to the examples extra.
    unused = sorted(distribution for distribution in EXTRAS["analysis"] if not _imports(import_names(distribution)))
    assert unused == [], f"src/vamos imports none of these analysis packages: {unused}"


def test_vamos_imports_no_other_optimization_framework() -> None:
    imported = sorted(distribution for distribution, modules in OPTIMIZATION_FRAMEWORKS.items() if _imports(modules))
    assert imported == [], f"src/vamos imports these optimization frameworks: {imported}"
