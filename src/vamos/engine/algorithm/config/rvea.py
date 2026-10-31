"""RVEA configuration."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

from vamos.engine.archive import ExternalArchiveConfig

from .base import (
    ConstraintModeStr,
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
    _SerializableConfig,
    _TrackGenealogyBuilder,
    _validate_operators,
)
from .types import RepairConfigValue


@dataclass(frozen=True)
class RVEAConfig(_SerializableConfig):
    pop_size: int
    n_partitions: int
    alpha: float
    adapt_freq: float | None
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
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
        pop_size: int | None = None,
        n_var: int | None = None,
        n_obj: int = 3,
    ) -> RVEAConfig:
        """Create a default RVEA configuration.

        ``n_partitions`` is 12 for three objectives, 6 for four or more, and with
        two objectives ``pop_size - 1`` (one reference vector per individual), or
        99 without ``pop_size``.
        """
        mut_prob = 1.0 / n_var if n_var else 0.1
        if n_obj == 2:
            n_partitions = 99 if pop_size is None else max(1, int(pop_size) - 1)
        else:
            n_partitions = 12 if n_obj == 3 else 6
        if pop_size is None:
            pop_size = comb(n_partitions + n_obj - 1, n_obj - 1)
        return (
            cls.builder()
            .pop_size(pop_size)
            .n_partitions(n_partitions)
            .alpha(2.0)
            .adapt_freq(0.1)
            .crossover("sbx", prob=1.0, eta=30.0)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .build()
        )

    @classmethod
    def builder(cls) -> _RVEAConfigBuilder:
        return _RVEAConfigBuilder()


class _RVEAConfigBuilder(
    _ConfigBuilderState,
    _PopSizeBuilder,
    _CrossoverBuilder,
    _MutationBuilder,
    _RepairBuilder,
    _InitializerBuilder,
    _MutationProbFactorBuilder,
    _ConstraintModeBuilder,
    _TrackGenealogyBuilder,
    _ResultArchiveBuilder,
):
    """
    Declarative configuration holder for RVEA settings.

    Examples:
        cfg = RVEAConfig.default()
        cfg = RVEAConfig.builder().pop_size(100).n_partitions(12).build()
    """

    def n_partitions(self, value: int) -> _RVEAConfigBuilder:
        self._cfg["n_partitions"] = value
        return self

    def alpha(self, value: float) -> _RVEAConfigBuilder:
        self._cfg["alpha"] = float(value)
        return self

    def adapt_freq(self, value: float | None) -> _RVEAConfigBuilder:
        self._cfg["adapt_freq"] = None if value is None else float(value)
        return self

    def build(self) -> RVEAConfig:
        _require_fields(
            self._cfg,
            ("pop_size", "n_partitions", "alpha", "crossover", "mutation"),
            "RVEA",
        )
        _validate_operators(self._cfg)
        return RVEAConfig(
            pop_size=self._cfg["pop_size"],
            n_partitions=self._cfg.get("n_partitions", 12),
            alpha=float(self._cfg.get("alpha", 2.0)),
            adapt_freq=self._cfg.get("adapt_freq", 0.1),
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            repair=self._cfg.get("repair", "auto"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            external_archive=self._cfg.get("external_archive"),
        )
