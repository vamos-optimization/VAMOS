"""SMPSO configuration."""

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
    _encoding_operators,
    _InitializerBuilder,
    _MutationBuilder,
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
class SMPSOConfig(_SerializableConfig):
    pop_size: int
    archive_size: int  # Internal archive (part of SMPSO algorithm)
    mutation: tuple[str, dict[str, Any]]
    inertia: float = 0.1
    c1: float = 1.5
    c2: float = 1.5
    vmax_fraction: float = 0.5
    repair: RepairConfigValue = "auto"
    initializer: dict[str, Any] | None = None
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
    ) -> SMPSOConfig:
        """Create a default SMPSO configuration.

        ``pop_size=None`` means 100. A mixed ``encoding`` takes its mutation from
        the shared encoding table; SMPSO supports real and mixed encodings.
        """
        mut_prob = 1.0 / n_var if n_var else 0.1
        size = DEFAULT_POP_SIZE if pop_size is None else pop_size
        _, mutation = _encoding_operators(encoding, mut_prob, real=(("sbx", {}), ("polynomial", {"prob": mut_prob, "eta": 20.0})))
        return cls.builder().pop_size(size).archive_size(size).mutation(mutation[0], **mutation[1]).build()

    @classmethod
    def builder(cls) -> _SMPSOConfigBuilder:
        return _SMPSOConfigBuilder()


class _SMPSOConfigBuilder(
    _ConfigBuilderState,
    _PopSizeBuilder,
    _MutationBuilder,
    _RepairBuilder,
    _InitializerBuilder,
    _ConstraintModeBuilder,
    _TrackGenealogyBuilder,
    _ResultArchiveBuilder,
):
    """Declarative configuration holder for SMPSO settings."""

    def archive_size(self, value: int) -> _SMPSOConfigBuilder:
        self._cfg["archive_size"] = value
        return self

    def inertia(self, value: float) -> _SMPSOConfigBuilder:
        self._cfg["inertia"] = value
        return self

    def c1(self, value: float) -> _SMPSOConfigBuilder:
        self._cfg["c1"] = value
        return self

    def c2(self, value: float) -> _SMPSOConfigBuilder:
        self._cfg["c2"] = value
        return self

    def vmax_fraction(self, value: float) -> _SMPSOConfigBuilder:
        self._cfg["vmax_fraction"] = value
        return self

    def external_archive(self, capacity: int | None = None, **kwargs: Any) -> _SMPSOConfigBuilder:
        """Configure an external archive for result storage.

        Note
        ----
        This is separate from ``archive_size``, which controls the internal SMPSO archive.

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

    def build(self) -> SMPSOConfig:
        _require_fields(
            self._cfg,
            ("pop_size", "archive_size", "mutation"),
            "SMPSO",
        )
        _validate_operators(self._cfg)
        return SMPSOConfig(
            pop_size=self._cfg["pop_size"],
            archive_size=self._cfg["archive_size"],
            mutation=self._cfg["mutation"],
            inertia=float(self._cfg.get("inertia", 0.1)),
            c1=float(self._cfg.get("c1", 1.5)),
            c2=float(self._cfg.get("c2", 1.5)),
            vmax_fraction=float(self._cfg.get("vmax_fraction", 0.5)),
            repair=self._cfg.get("repair", "auto"),
            initializer=self._cfg.get("initializer"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            external_archive=self._cfg.get("external_archive"),
        )
