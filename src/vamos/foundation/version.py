"""
Version helpers for VAMOS.
"""

from __future__ import annotations

# Bumped by hand in the release pull request; see docs/release_smoke.md.
__version__ = "1.2.0"


def get_version() -> str:
    return __version__


__all__ = ["__version__", "get_version"]
