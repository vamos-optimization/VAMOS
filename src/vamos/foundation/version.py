"""
Version helpers for VAMOS.
"""

from __future__ import annotations

# Managed by python-semantic-release
__version__ = "1.1.0"


def get_version() -> str:
    return __version__


__all__ = ["__version__", "get_version"]
