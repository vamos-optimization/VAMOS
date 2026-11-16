"""Public typing helpers for the experiment-facing optimization API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

from vamos.engine.hooks.live_viz import LiveVisualization

CheckpointPayload: TypeAlias = Mapping[str, Any]
TerminationValue: TypeAlias = int | float | Mapping[str, Any]
TerminationSpec: TypeAlias = tuple[str, TerminationValue]

__all__ = ["CheckpointPayload", "LiveVisualization", "TerminationSpec", "TerminationValue"]
