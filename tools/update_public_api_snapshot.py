from __future__ import annotations

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "vamos"
SNAPSHOT_PATH = REPO_ROOT / "tests" / "architecture" / "public_api_snapshot.json"

FACADE_PATHS = {
    "vamos": SRC_ROOT / "__init__.py",
    "vamos.api": SRC_ROOT / "api.py",
    "vamos.algorithms": SRC_ROOT / "algorithms.py",
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


def main() -> None:
    snapshot: dict[str, list[str]] = {}
    for module, path in sorted(FACADE_PATHS.items()):
        if not path.exists():
            raise SystemExit(f"Missing facade file: {path}")
        exports = _extract_all(path)
        if exports is None:
            raise SystemExit(f"Missing or non-literal __all__ in {path}")
        snapshot[module] = exports

    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote public API snapshot to {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
