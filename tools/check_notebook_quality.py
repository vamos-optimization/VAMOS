from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.check_pre_release_remnants import tracked_paths

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = REPO_ROOT / "notebooks"
EXPECTED_KERNEL = "Python 3 (VAMOS)"


def iter_notebooks() -> list[Path]:
    # Tracked notebooks only: Jupyter's ignored .ipynb_checkpoints/ copies are not part of the repository.
    return [path for path in tracked_paths(REPO_ROOT) if path.suffix == ".ipynb" and NOTEBOOK_ROOT in path.parents]


def main() -> int:
    failures: list[str] = []

    for path in iter_notebooks():
        data = json.loads(path.read_text(encoding="utf-8"))
        rel_path = path.relative_to(REPO_ROOT)
        metadata = data.get("metadata", {})
        kernelspec = metadata.get("kernelspec", {})
        language_info = metadata.get("language_info", {})

        if kernelspec.get("display_name") != EXPECTED_KERNEL:
            failures.append(f"{rel_path}: unexpected kernelspec display name")
        if language_info.get("name") != "python":
            failures.append(f"{rel_path}: missing python language_info")

        first_heading = None
        for cell in data.get("cells", []):
            if cell.get("cell_type") != "markdown":
                continue
            for raw_line in "".join(cell.get("source", [])).splitlines():
                line = raw_line.strip()
                if line:
                    first_heading = line
                    break
            if first_heading is not None:
                break

        if not isinstance(first_heading, str) or not first_heading.startswith("# "):
            failures.append(f"{rel_path}: missing top-level title")

        for idx, cell in enumerate(data.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("execution_count") is not None:
                failures.append(f"{rel_path} cell {idx}: execution_count should be cleared")
            outputs = cell.get("outputs", [])
            if outputs:
                failures.append(f"{rel_path} cell {idx}: outputs should be cleared")

    if failures:
        for failure in failures:
            print(failure)
        return 1

    print(f"Notebook quality check passed for {len(iter_notebooks())} notebooks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
