"""
Packaged data assets for VAMOS (reference fronts, weight vectors, etc.).
"""

from __future__ import annotations

import os
import sys
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

ResourceLocation = Path | Traversable


def user_reference_fronts_dir() -> Path:
    """The per-user directory of the reference fronts that VAMOS does not package.

    VAMOS 1.5.1: ``vamos zoo fetch-fronts`` writes the fronts it downloads there, and
    :func:`reference_front_path` searches it after the packaged fronts, so such a front never replaces a packaged
    one. The directory is ``reference_fronts`` under ``VAMOS_DATA_DIR`` when that variable is set, and otherwise under
    the platform's per-user data directory: ``%LOCALAPPDATA%\\vamos`` on Windows, ``~/Library/Application
    Support/vamos`` on macOS, and ``$XDG_DATA_HOME/vamos`` (by default ``~/.local/share/vamos``) elsewhere.
    """
    root = os.environ.get("VAMOS_DATA_DIR")
    if root:
        return Path(root) / "reference_fronts"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "vamos" / "reference_fronts"


def _reference_front_locations() -> list[ResourceLocation]:
    """Return candidate locations containing reference front CSV files."""
    locations: list[ResourceLocation] = []

    # Explicit override for external runs (e.g., remote servers).
    override = os.environ.get("VAMOS_REFERENCE_FRONTS_DIR")
    if override:
        override_path = Path(override)
        if override_path.exists():
            locations.append(override_path)

    # Packaged fronts are the canonical built-in source.
    packaged = resources.files(__name__) / "reference_fronts"
    if packaged.is_dir():
        locations.append(packaged)

    # Downloaded fronts come last: they add fronts and never replace a packaged one.
    try:
        downloaded = user_reference_fronts_dir()
    except RuntimeError:  # Path.home() without a home directory
        return locations
    if downloaded.is_dir():
        locations.append(downloaded)

    return locations


def reference_front_path(name: str) -> Path:
    """
    Return a filesystem path to a reference front CSV by key (e.g., "zdt1").

    The locations are searched in order: ``VAMOS_REFERENCE_FRONTS_DIR`` when set, the packaged fronts, and
    (VAMOS 1.5.1) the downloaded fronts of :func:`user_reference_fronts_dir`.
    """
    target = f"{name.lower()}.csv"
    for base in _reference_front_locations():
        if not base.is_dir():
            continue
        candidates = (
            f"{name}.csv",
            f"{name.lower()}.csv",
            f"{name.upper()}.csv",
        )
        for cand in candidates:
            p = base / cand
            if p.is_file():
                return Path(str(p))

        # Case-insensitive lookup for robustness across filesystems and naming.
        try:
            entries = tuple(base.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name.lower() == target and entry.is_file():
                return Path(str(entry))

    raise ValueError(f"Unknown reference front '{name}'.")


def weight_path(filename: str) -> Path:
    """
    Return a filesystem path to a packaged weight vector file.
    """
    path = resources.files(__name__) / "weights" / filename
    if not path.is_file():
        raise ValueError(f"Unknown weights file '{filename}'.")
    return Path(str(path))


__all__ = ["reference_front_path", "user_reference_fronts_dir", "weight_path"]
