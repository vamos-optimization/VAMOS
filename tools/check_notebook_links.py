from __future__ import annotations

import json
import re
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.check_pre_release_remnants import tracked_paths

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = REPO_ROOT / "notebooks"
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]+)?\)")


def iter_notebooks() -> list[Path]:
    # Tracked notebooks only: Jupyter's ignored .ipynb_checkpoints/ copies are not part of the repository.
    return [path for path in tracked_paths(REPO_ROOT) if path.suffix == ".ipynb" and NOTEBOOK_ROOT in path.parents]


def main() -> int:
    errors: list[str] = []

    for notebook in iter_notebooks():
        data = json.loads(notebook.read_text(encoding="utf-8"))
        for cell_idx, cell in enumerate(data.get("cells", []), start=1):
            if cell.get("cell_type") != "markdown":
                continue
            text = "".join(cell.get("source", []))
            for match in LINK_RE.finditer(text):
                target = match.group(1).strip()
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                target_path = (notebook.parent / target).resolve()
                if not target_path.exists():
                    errors.append(f"{notebook.relative_to(REPO_ROOT)} cell {cell_idx}: missing link target {target!r}")

    if errors:
        for error in errors:
            print(error)
        return 1

    print(f"Notebook link check passed for {len(iter_notebooks())} notebooks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
