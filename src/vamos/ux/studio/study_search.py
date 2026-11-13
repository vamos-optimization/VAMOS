"""Find the studies under a folder without reading them, for the study pickers of VAMOS Studio."""

from __future__ import annotations

import os
from pathlib import Path

STUDY_MANIFEST = "study-manifest.json"
STUDY_SEARCH_DEPTH = 4  # How many folders below the chosen one Studio looks for studies.


def study_roots(base: Path, *, max_depth: int = STUDY_SEARCH_DEPTH) -> list[Path]:
    """The folders at ``base`` or at most ``max_depth`` levels below it that hold a study manifest.

    The most recently changed studies come first, judged by their event journals, so that a caller
    that reads only the first few studies reads the latest. Links to other folders are not followed.
    """
    roots: list[Path] = []
    level = [Path(base)]
    for depth in range(max_depth + 1):
        below: list[Path] = []
        for folder in level:
            try:
                with os.scandir(folder) as entries:
                    for entry in entries:
                        if entry.name == STUDY_MANIFEST and entry.is_file():
                            roots.append(folder)
                        elif depth < max_depth and entry.is_dir(follow_symlinks=False):
                            below.append(Path(entry.path))
            except OSError:
                continue
        level = below
    return sorted(roots, key=_last_change, reverse=True)


def _last_change(root: Path) -> float:
    for path in (root / "events", root):
        try:
            return path.stat().st_mtime
        except OSError:
            continue
    return 0.0


__all__ = ["STUDY_MANIFEST", "STUDY_SEARCH_DEPTH", "study_roots"]
