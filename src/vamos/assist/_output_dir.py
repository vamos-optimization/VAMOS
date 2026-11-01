"""Output folders of ``vamos assist apply`` and ``vamos assist run``."""

from __future__ import annotations

import shutil
from pathlib import Path

# Marks a folder assist created, the only kind ``--overwrite`` may clear.
MARKER = ".vamos-assist"


def prepare_output_dir(path: Path, *, plan_dir: Path, overwrite: bool, owned: tuple[str, ...], kind: str) -> None:
    """Create ``path`` for assist output, or clear the entries assist wrote there when ``overwrite``.

    ``path`` cannot be the plan folder or contain it. ``overwrite`` applies only to a folder with
    the marker, and removes only the ``owned`` entries; anything else in the folder is kept.
    """
    target = path.resolve()
    plan = plan_dir.resolve()
    if target == plan or target in plan.parents:
        raise ValueError(f"{kind} directory {path} is the plan directory or contains it; choose another --out.")
    if path.exists():
        if not overwrite:
            raise ValueError(f"{kind} directory already exists: {path}")
        if not (path / MARKER).is_file():
            raise ValueError(
                f"{kind} directory {path} was not created by vamos assist (it has no {MARKER} file); "
                "remove it yourself or choose another --out."
            )
        for name in owned:
            entry = path / name
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            elif entry.exists() or entry.is_symlink():
                entry.unlink()
    path.mkdir(parents=True, exist_ok=True)
    (path / MARKER).write_text("Created by vamos assist; --overwrite replaces only the files assist wrote here.\n", encoding="utf-8")


__all__ = ["MARKER", "prepare_output_dir"]
