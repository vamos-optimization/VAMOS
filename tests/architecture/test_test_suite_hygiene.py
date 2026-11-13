"""The test suite contains only tests that can fail and modules pytest collects as intended (VR-131, VR-133, VR-319)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parents[1]
_MODULES = sorted(TESTS.rglob("*.py"))


def _test_functions(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")]


def _body_without_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def _is_length_tautology(node: ast.AST) -> bool:
    # assert len(x) >= 0 holds for every sized object.
    return (
        isinstance(node, ast.Assert)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Call)
        and isinstance(node.test.left.func, ast.Name)
        and node.test.left.func.id == "len"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.GtE)
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == 0
    )


@pytest.mark.parametrize("path", _MODULES, ids=lambda path: path.relative_to(TESTS).as_posix())
def test_module_tests_can_fail(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports_source_tree = [
        node.lineno
        for node in ast.walk(tree)
        if (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src."))
        or (isinstance(node, ast.Import) and any(alias.name.startswith("src.") for alias in node.names))
    ]
    # Importing src.vamos loads a second copy of the package next to the installed vamos.
    assert not imports_source_tree, f"imports src.vamos at lines {imports_source_tree}; import vamos instead"
    empty = [node.name for node in _test_functions(tree) if all(isinstance(stmt, ast.Pass) for stmt in _body_without_docstring(node))]
    assert not empty, f"tests without a body: {empty}"
    tautologies = [node.lineno for node in ast.walk(tree) if _is_length_tautology(node)]
    assert not tautologies, f"assert len(...) >= 0 cannot fail (lines {tautologies})"


def test_no_uncollected_test_scripts_or_empty_test_packages() -> None:
    # pytest collects test_*.py only, so verify_*.py scripts never run.
    scripts = sorted(path.relative_to(TESTS).as_posix() for path in TESTS.rglob("verify_*.py"))
    assert not scripts, f"modules pytest never collects: {scripts}"
    empty_packages = sorted(
        directory.relative_to(TESTS).as_posix()
        for directory in TESTS.rglob("*")
        if directory.is_dir()
        and directory.name != "__pycache__"
        and {child.name for child in directory.iterdir() if child.name != "__pycache__"} == {"__init__.py"}
    )
    assert not empty_packages, f"empty test packages: {empty_packages}"
