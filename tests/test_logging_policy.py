from __future__ import annotations

import ast
import logging
from pathlib import Path

ALLOWED_BASICCONFIG_PATHS: tuple[str, ...] = ()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_basic_config_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name) and func.value.id == "logging":
            return func.attr == "basicConfig"
        return func.attr == "basicConfig"
    if isinstance(func, ast.Name):
        return func.id == "basicConfig"
    return False


def test_logging_policy() -> None:
    repo_root = _repo_root()
    src_root = repo_root / "src" / "vamos"
    violations: list[str] = []

    for path in src_root.rglob("*.py"):
        rel_path = path.relative_to(repo_root).as_posix()
        if rel_path in ALLOWED_BASICCONFIG_PATHS:
            continue

        text = path.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:  # pragma: no cover - should not happen
            raise AssertionError(f"Failed to parse {rel_path}: {exc}") from exc

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _is_basic_config_call(node):
                lineno = getattr(node, "lineno", "?")
                violations.append(f"{rel_path}:{lineno}: logging.basicConfig")

    if violations:
        msg = ["logging.basicConfig is forbidden in library modules:"]
        msg.extend(f"- {item}" for item in sorted(violations))
        raise AssertionError("\n".join(msg))


def test_auto_selected_log_lists_only_auto_values(caplog) -> None:
    """The "Auto-selected defaults" log names only what optimize chose (VR-127)."""
    import vamos
    from vamos.algorithms import NSGAIIConfig

    with caplog.at_level(logging.INFO, logger="vamos"):
        vamos.optimize(
            "zdt1",
            algorithm="nsgaii",
            algorithm_config=NSGAIIConfig.default(pop_size=10, n_var=30),
            termination=("max_evaluations", 40),
            seed=1,
        )
    assert not [record.getMessage() for record in caplog.records if "Auto-selected" in record.getMessage()]

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="vamos"):
        result = vamos.optimize("dtlz2", algorithm="nsgaiii", max_evaluations=200, seed=1)
    (message,) = [record.getMessage() for record in caplog.records if "Auto-selected" in record.getMessage()]
    # NSGA-III's population is its reference lattice, not the size derived from the problem.
    assert f"'pop_size': {len(result.data['population']['F'])}" in message, message


def test_verbose_does_not_configure_logger_permanently() -> None:
    """``verbose=True`` logs the run and leaves the "vamos" logger as it was (VR-127)."""
    import json
    import subprocess
    import sys

    code = (
        "import json, logging, vamos\n"
        "vamos.optimize('zdt1', max_evaluations=100, pop_size=10, seed=1, verbose=True)\n"
        "logger = logging.getLogger('vamos')\n"
        "print(json.dumps([len(logger.handlers), logger.propagate, logger.level]))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=300)
    assert "[vamos] Problem: n_var=30" in proc.stderr
    assert json.loads(proc.stdout.splitlines()[-1]) == [0, True, logging.NOTSET]
