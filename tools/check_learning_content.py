from __future__ import annotations

import json
import re
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.check_pre_release_remnants import tracked_paths

REPO_ROOT = Path(__file__).resolve().parents[1]

PATHS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs",
    REPO_ROOT / "examples",
    REPO_ROOT / "notebooks",
]

STALE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "removed archive builder",
        re.compile(r"\.archive\s*\(\s*size\s*=|\.\s*archive_type\s*\("),
    ),
    (
        "stale optimize population_size kwarg",
        re.compile(r"optimize\s*\([^)]*population_size\s*=", re.DOTALL),
    ),
    (
        "old CLI module entrypoint",
        re.compile(r"python\s+-m\s+vamos\.experiment\.cli\.main"),
    ),
]

NOTEBOOK_TUNING_STALE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "tuning notebooks should use Optuna, not racing/random tuners",
        re.compile(r"\b(RacingTuner|RandomSearchTuner)\b"),
    ),
]


def iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    # Tracked files only: ignored copies such as Jupyter's .ipynb_checkpoints/ are not part of the repository.
    suffixes = {".md", ".py", ".yml", ".yaml", ".ipynb"}
    return [path for path in tracked_paths(REPO_ROOT) if path.suffix in suffixes and root in path.parents]


def load_text(path: Path) -> str:
    if path.suffix == ".ipynb":
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks: list[str] = []
        for cell in data.get("cells", []):
            chunks.append("".join(cell.get("source", [])))
        return "\n".join(chunks)
    return path.read_text(encoding="utf-8")


def main() -> int:
    failures: list[str] = []

    for root in PATHS:
        for path in iter_files(root):
            text = load_text(path)
            rel_path = path.relative_to(REPO_ROOT)
            for label, pattern in STALE_PATTERNS:
                if pattern.search(text):
                    failures.append(f"{rel_path}: {label}")
            if (
                path.suffix == ".ipynb"
                and path.parent.name in {"0_basic", "1_intermediate", "2_advanced"}
                and "tuning" in path.stem.lower()
            ):
                for label, pattern in NOTEBOOK_TUNING_STALE_PATTERNS:
                    if pattern.search(text):
                        failures.append(f"{rel_path}: {label}")

    if failures:
        for failure in failures:
            print(failure)
        return 1

    print("Learning content check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
