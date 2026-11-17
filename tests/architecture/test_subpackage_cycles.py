"""Import cycles between the subpackages of ``src/vamos``, counting the imports inside functions.

A lazy import hides a cycle from module import order, but the two subpackages
still depend on each other. Imports under ``if TYPE_CHECKING:`` never run, so
they do not count. Each cycle that exists today is listed with the reason it
stays; a new cycle, or a known one that grows, fails. When a cycle shrinks or
disappears, update its entry, so the list only shrinks.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PACKAGE = SRC / "vamos"

# Each known cycle, as the set of its subpackages, with the reason it stays.
KNOWN_CYCLES: dict[frozenset[str], str] = {
    frozenset({"vamos.foundation.kernel", "vamos.foundation.quality_indicators"}): (
        "Kernels compute hypervolume and dominance, and the indicators use the NumPy kernel's sorting."
    ),
    frozenset(
        {
            "vamos.engine.algorithm",
            "vamos.engine.archive",
            "vamos.engine.config",
            "vamos.engine.hooks",
            "vamos.engine.operators",
            "vamos.engine.variation",
        }
    ): (
        "Algorithm-specific variation lives in engine.operators.policies and uses algorithm internals; "
        "see the known debt in docs/dev/architecture_health.md."
    ),
    frozenset({"vamos.experiment.cli", "vamos.experiment.presentation"}): (
        "The CLI's entry point and quickstart run experiments through presentation, which runs them with the CLI's "
        "orchestration. It was part of the assist cycle until vamos.assist was removed."
    ),
    frozenset({"vamos.ux.panel", "vamos.ux.studio"}): (
        "The Panel pages render Studio's state, and Studio's entry module starts the Panel launcher."
    ),
}


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_type_checking(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _runtime_imports(tree: ast.AST) -> list[ast.Import | ast.ImportFrom]:
    """Every import that can run, at module level or inside a function; not ``if TYPE_CHECKING:``."""
    found: list[ast.Import | ast.ImportFrom] = []
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.If) and _is_type_checking(node):
            stack.extend(node.orelse)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return found


def _subpackage_graph() -> dict[str, set[str]]:
    modules = {_module_name(path): path for path in sorted(PACKAGE.rglob("*.py"))}
    packages = {name for name, path in modules.items() if path.name == "__init__.py"}

    def node_of(module: str) -> str:
        # A subpackage of a layer (vamos.engine.algorithm) is one node; any other module is its own node.
        subpackage = ".".join(module.split(".")[:3])
        return subpackage if subpackage in packages and subpackage.count(".") == 2 else module

    def resolve(name: str) -> str | None:
        while name and name not in modules:
            name = name.rpartition(".")[0]
        return name or None

    graph: dict[str, set[str]] = defaultdict(set)
    for name, path in modules.items():
        package = name.split(".") if path.name == "__init__.py" else name.split(".")[:-1]
        for node in _runtime_imports(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            else:
                base = node.module or ""
                if node.level:
                    base = ".".join([*package[: len(package) - node.level + 1], *([node.module] if node.module else [])])
                targets = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            for target in targets:
                resolved = resolve(target) if target.startswith("vamos") else None
                if resolved is not None and node_of(resolved) != node_of(name):
                    graph[node_of(name)].add(node_of(resolved))
    return graph


def _cycles(graph: dict[str, set[str]]) -> set[frozenset[str]]:
    """The strongly connected components with more than one node (Tarjan's algorithm, iterative)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: set[frozenset[str]] = set()
    for root in sorted(graph):
        if root in index:
            continue
        work: list[tuple[str, list[str]]] = [(root, sorted(graph[root]))]
        index[root] = low[root] = len(index)
        stack.append(root)
        on_stack.add(root)
        while work:
            node, pending = work[-1]
            if pending:
                child = pending.pop()
                if child not in index:
                    index[child] = low[child] = len(index)
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, sorted(graph.get(child, ()))))
                elif child in on_stack:
                    low[node] = min(low[node], index[child])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = set()
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.add(member)
                    if member == node:
                        break
                if len(component) > 1:
                    components.add(frozenset(component))
    return components


def test_cycle_finder_sees_a_cycle_through_a_lazy_import() -> None:
    graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": {"a"}}

    assert _cycles(graph) == {frozenset({"a", "b", "c"})}
    tree = ast.parse("import x\nif TYPE_CHECKING:\n    import y\ndef f():\n    import z\n")
    assert sorted(alias.name for node in _runtime_imports(tree) for alias in node.names) == ["x", "z"]


def test_subpackage_cycles_are_known() -> None:
    detected = _cycles(_subpackage_graph())

    new = sorted(sorted(cycle) for cycle in detected - set(KNOWN_CYCLES))
    stale = sorted(sorted(cycle) for cycle in set(KNOWN_CYCLES) - detected)
    assert not new and not stale, (
        f"Import cycles between subpackages changed.\nNew or grown: {new}\nGone or shrunk (update KNOWN_CYCLES): {stale}"
    )
