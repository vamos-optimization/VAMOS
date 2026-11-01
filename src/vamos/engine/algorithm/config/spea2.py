"""SPEA2 configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vamos.engine.archive import ExternalArchiveConfig
from vamos.foundation.encoding import EncodingLike

from .base import (
    DEFAULT_POP_SIZE,
    ConstraintModeStr,
    ResultMode,
    _build_external_archive_config,
    _ConfigBuilderState,
    _ConstraintModeBuilder,
    _CrossoverBuilder,
    _encoding_operators,
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
class SPEA2Config(_SerializableConfig):
    pop_size: int
    archive_size: int  # Internal archive (part of SPEA2 algorithm)
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    selection: tuple[str, dict[str, Any]]
    k_neighbors: int | None = None
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
        encoding: EncodingLike | None = None,
    ) -> SPEA2Config:
        """Create a default SPEA2 configuration.

        ``pop_size=None`` means 100. Discrete and mixed ``encoding`` values take
        their crossover and mutation from the shared encoding table.
        """
        mut_prob = 1.0 / n_var if n_var else 0.1
        size = DEFAULT_POP_SIZE if pop_size is None else pop_size
        crossover, mutation = _encoding_operators(
            encoding, mut_prob, real=(("sbx", {"prob": 0.9, "eta": 20.0}), ("pm", {"prob": mut_prob, "eta": 20.0}))
        )
        return (
            cls.builder()
            .pop_size(size)
            .archive_size(size)
            .crossover(crossover[0], **crossover[1])
            .mutation(mutation[0], **mutation[1])
            .selection("tournament")
            .build()
        )

    @classmethod
    def builder(cls) -> _SPEA2ConfigBuilder:
        return _SPEA2ConfigBuilder()


class _SPEA2ConfigBuilder(
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
    """
    Declarative configuration holder for SPEA2 settings.

    Examples:
        cfg = SPEA2Config.default()
        cfg = SPEA2Config.builder().pop_size(100).archive_size(100).build()
    """

    def archive_size(self, value: int) -> _SPEA2ConfigBuilder:
        self._cfg["archive_size"] = value
        return self

    def k_neighbors(self, value: int) -> _SPEA2ConfigBuilder:
        self._cfg["k_neighbors"] = value
        return self

    def external_archive(self, capacity: int | None = None, **kwargs: Any) -> _SPEA2ConfigBuilder:
        """Configure an external archive for result storage.

        Note
        ----
        This is separate from ``archive_size``, which controls the internal SPEA2 archive.

        Parameters
        ----------
        capacity
            Maximum number of solutions. ``None`` means unbounded.
        **kwargs
            Forwarded to :class:`ExternalArchiveConfig`.
        """
        self._cfg["external_archive"] = _build_external_archive_config(capacity, kwargs)
        self._cfg.setdefault("result_mode", "non_dominated")
        return self

    def build(self) -> SPEA2Config:
        _require_fields(
            self._cfg,
            ("pop_size", "archive_size", "crossover", "mutation", "selection"),
            "SPEA2",
        )
        _validate_operators(self._cfg)
        return SPEA2Config(
            pop_size=self._cfg["pop_size"],
            archive_size=self._cfg["archive_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            selection=self._cfg["selection"],
            k_neighbors=self._cfg.get("k_neighbors"),
            repair=self._cfg.get("repair", "auto"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            external_archive=self._cfg.get("external_archive"),
        )
