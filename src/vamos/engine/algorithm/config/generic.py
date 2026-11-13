from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .types import AlgorithmConfigMapping


@dataclass(frozen=True, slots=True)
class GenericAlgorithmConfig:
    """
    Minimal config wrapper for plugin/custom algorithms.

    This exists to keep the public-facing APIs strongly typed while still
    allowing users (or tests) to pass a free-form mapping when integrating a
    third-party algorithm via the algorithm registry.

    Experimental: exported by ``vamos.algorithms`` for custom and plugin
    algorithms, but not covered by the 1.x stability policy. Built-in
    algorithms take their own configuration classes.
    """

    data: Mapping[str, object]

    def to_dict(self) -> AlgorithmConfigMapping:
        return dict(self.data)


__all__ = ["GenericAlgorithmConfig"]
