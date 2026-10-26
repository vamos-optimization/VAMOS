"""IBEA configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, overload

from vamos.engine.archive import ExternalArchiveConfig

from .base import (
    ConstraintModeStr,
    IndicatorType,
    ResultMode,
    _ConfigBuilderState,
    _ConstraintModeBuilder,
    _CrossoverBuilder,
    _InitializerBuilder,
    _MutationBuilder,
    _MutationProbFactorBuilder,
    _PopSizeBuilder,
    _RepairBuilder,
    _require_fields,
    _ResultArchiveBuilder,
    _SelectionBuilder,
    _SerializableConfig,
    _TrackGenealogyBuilder,
    _validate_operators,
)
from .types import RepairConfigValue


@dataclass(frozen=True)
class IBEAConfig(_SerializableConfig):
    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    selection: tuple[str, dict[str, Any]]
    indicator: IndicatorType
    kappa: float
    repair: RepairConfigValue = "auto"
    initializer: dict[str, Any] | None = None
    mutation_prob_factor: float | None = None
    constraint_mode: ConstraintModeStr = "feasibility"
    track_genealogy: bool = False
    result_mode: ResultMode | None = None
    external_archive: ExternalArchiveConfig | None = None

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
    ) -> IBEAConfig:
        """Create a default IBEA configuration."""
        mut_prob = 1.0 / n_var if n_var else 0.1
        return (
            cls.builder()
            .pop_size(pop_size)
            .crossover("sbx", prob=1.0, eta=20.0)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .selection("tournament")
            .indicator("eps")
            .kappa(0.05)
            .build()
        )

    @classmethod
    def builder(cls) -> _IBEAConfigBuilder:
        return _IBEAConfigBuilder()


class _IBEAConfigBuilder(
    _ConfigBuilderState,
    _PopSizeBuilder,
    _CrossoverBuilder,
    _MutationBuilder,
    _SelectionBuilder,
    _RepairBuilder,
    _InitializerBuilder,
    _MutationProbFactorBuilder,
    _ConstraintModeBuilder,
    _TrackGenealogyBuilder,
    _ResultArchiveBuilder,
):
    """Declarative configuration holder for IBEA settings."""

    @overload
    def indicator(self, name: IndicatorType) -> _IBEAConfigBuilder: ...

    @overload
    def indicator(self, name: str) -> _IBEAConfigBuilder: ...

    def indicator(self, name: str) -> _IBEAConfigBuilder:
        self._cfg["indicator"] = name
        return self

    def kappa(self, value: float) -> _IBEAConfigBuilder:
        self._cfg["kappa"] = value
        return self

    def build(self) -> IBEAConfig:
        _require_fields(
            self._cfg,
            ("pop_size", "crossover", "mutation", "selection", "indicator", "kappa"),
            "IBEA",
        )
        _validate_operators(self._cfg)
        return IBEAConfig(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            selection=self._cfg["selection"],
            indicator=cast(IndicatorType, str(self._cfg["indicator"])),
            kappa=float(self._cfg["kappa"]),
            repair=self._cfg.get("repair", "auto"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            external_archive=self._cfg.get("external_archive"),
        )
