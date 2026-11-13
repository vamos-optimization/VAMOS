from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "vamos"
SNAPSHOT_PATH = Path(__file__).resolve().parent / "public_api_snapshot.json"

FACADE_PATHS = {
    "vamos": SRC_ROOT / "__init__.py",
    "vamos.api": SRC_ROOT / "api.py",
    "vamos.algorithms": SRC_ROOT / "algorithms.py",
    "vamos.problems": SRC_ROOT / "problems.py",
    "vamos.run_artifacts": SRC_ROOT / "run_artifacts.py",
    "vamos.study_artifacts": SRC_ROOT / "study_artifacts.py",
    "vamos.exceptions": SRC_ROOT / "exceptions.py",
    "vamos.selection": SRC_ROOT / "selection.py",
    "vamos.ux.api": SRC_ROOT / "ux" / "api.py",
}


def _extract_all(path: Path) -> list[str] | None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        items: list[str] = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                items.append(elt.value)
                        return sorted(items)
                    return None
    return None


def test_public_api_snapshot() -> None:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    current: dict[str, list[str]] = {}
    missing: list[str] = []

    for module, path in FACADE_PATHS.items():
        if not path.exists():
            missing.append(f"{module}: missing file {path}")
            continue
        exports = _extract_all(path)
        if exports is None:
            missing.append(f"{module}: missing or non-literal __all__")
            continue
        current[module] = exports

    if missing:
        raise AssertionError("Public API snapshot failed:\n" + "\n".join(f"- {item}" for item in missing))

    if current != snapshot:
        lines = [
            "Public API snapshot mismatch.",
            "If this change is intentional, run: python tools/update_public_api_snapshot.py",
        ]
        for module in sorted(set(snapshot) | set(current)):
            expected = snapshot.get(module, [])
            got = current.get(module, [])
            if expected == got:
                continue
            removed = sorted(set(expected) - set(got))
            added = sorted(set(got) - set(expected))
            lines.append(f"- {module}")
            if added:
                lines.append(f"  + added: {', '.join(added)}")
            if removed:
                lines.append(f"  - removed: {', '.join(removed)}")
        raise AssertionError("\n".join(lines))


def test_public_api_snapshot_covers_every_public_module() -> None:
    # AGENTS.md: "Public exports are guarded by tests/architecture/test_public_api_snapshot.py".
    public = {"vamos"} | {f"vamos.{path.stem}" for path in SRC_ROOT.glob("*.py") if not path.name.startswith("_")}
    missing = sorted(public - set(FACADE_PATHS))
    assert not missing, f"public modules outside the snapshot: {missing}"


def test_public_api_snapshot_writer_uses_lf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # On Windows, a text-mode write would turn every line of the snapshot into CRLF, a full diff with no change.
    from tools import update_public_api_snapshot as writer

    target = tmp_path / "public_api_snapshot.json"
    monkeypatch.setattr(writer, "SNAPSHOT_PATH", target)
    writer.main()
    assert b"\r\n" not in target.read_bytes()
    assert target.read_bytes() == (Path(__file__).with_name("public_api_snapshot.json")).read_bytes().replace(b"\r\n", b"\n")
