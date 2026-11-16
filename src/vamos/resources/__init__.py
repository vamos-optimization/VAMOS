"""
Packaged data assets for VAMOS (reference fronts, weight vectors, etc.).
"""

from __future__ import annotations

import os
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

ResourceLocation = Path | Traversable


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

    return locations


def reference_front_path(name: str) -> Path:
    """
    Return a filesystem path to a packaged reference front CSV by key (e.g., "zdt1").
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


__all__ = ["reference_front_path", "weight_path"]
