from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass as _dataclass
from typing import Any

import numpy as np

from vamos.engine.config.variation import VariationConfig
from vamos.foundation.observer import Observer, RunContext

_VARIATION_FIELDS = (
    "nsgaii",
    "moead",
    "smsemoa",
    "nsgaiii",
    "spea2",
    "ibea",
    "smpso",
    "agemoea",
    "rvea",
)


@_dataclass
class VariationConfigs:
    """Bundles per-algorithm variation configurations into a single object."""

    nsgaii: VariationConfig | None = None
    moead: VariationConfig | None = None
    smsemoa: VariationConfig | None = None
    nsgaiii: VariationConfig | None = None
    spea2: VariationConfig | None = None
    ibea: VariationConfig | None = None
    smpso: VariationConfig | None = None
    agemoea: VariationConfig | None = None
    rvea: VariationConfig | None = None

    @classmethod
    def from_namespace(cls, args: Namespace) -> VariationConfigs:
        return cls(
            nsgaii=getattr(args, "nsgaii_variation", None),
            moead=getattr(args, "moead_variation", None),
            smsemoa=getattr(args, "smsemoa_variation", None),
            nsgaiii=getattr(args, "nsgaiii_variation", None),
            spea2=getattr(args, "spea2_variation", None),
            ibea=getattr(args, "ibea_variation", None),
            smpso=getattr(args, "smpso_variation", None),
            agemoea=getattr(args, "agemoea_variation", None),
            rvea=getattr(args, "rvea_variation", None),
        )

    def as_storage_dict(self) -> dict[str, VariationConfig | None]:
        return {
            "nsgaii_variation": self.nsgaii,
            "moead_variation": self.moead,
            "smsemoa_variation": self.smsemoa,
            "nsgaiii_variation": self.nsgaiii,
            "spea2_variation": self.spea2,
            "ibea_variation": self.ibea,
            "smpso_variation": self.smpso,
            "agemoea_variation": self.agemoea,
            "rvea_variation": self.rvea,
        }

    def has_any(self) -> bool:
        return any(getattr(self, field_name) is not None for field_name in _VARIATION_FIELDS)

    def for_algorithm(self, algorithm: str) -> VariationConfigs:
        key = algorithm.strip().lower()
        if key not in _VARIATION_FIELDS:
            return VariationConfigs()
        return VariationConfigs(**{key: self.copy_field(key)})

    def copy(self) -> VariationConfigs:
        return VariationConfigs(**{field_name: self.copy_field(field_name) for field_name in _VARIATION_FIELDS})

    def copy_field(self, field_name: str) -> VariationConfig | None:
        value = getattr(self, field_name)
        return None if value is None else dict(value)


class CompositeObserver(Observer):
    """Fans out observer events to all registered observers."""

    def __init__(self, observers: list[Observer]):
        self.observers = [o for o in observers if o is not None]

    def on_start(self, ctx: RunContext) -> None:
        for obs in self.observers:
            obs.on_start(ctx)

    def on_generation(
        self,
        generation: int,
        F: np.ndarray[Any, Any] | None = None,
        X: np.ndarray[Any, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        for obs in self.observers:
            obs.on_generation(generation, F, X, stats)

    def on_end(
        self,
        final_F: np.ndarray[Any, Any] | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None:
        for obs in self.observers:
            obs.on_end(final_F, final_stats)

    def should_stop(self) -> bool:
        for obs in self.observers:
            if hasattr(obs, "should_stop") and obs.should_stop():
                return True
        return False


class LiveVizAdapter:
    """Bridge algorithm live-viz callbacks to observer on_generation events."""

    def __init__(self, observer: CompositeObserver):
        self._observer = observer

    def on_start(self, ctx: RunContext) -> None:
        return None

    def on_generation(
        self,
        generation: int,
        F: np.ndarray[Any, Any] | None = None,
        X: np.ndarray[Any, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        self._observer.on_generation(generation, F, X, stats)

    def on_end(
        self,
        final_F: np.ndarray[Any, Any] | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None:
        return None

    def should_stop(self) -> bool:
        return self._observer.should_stop()
