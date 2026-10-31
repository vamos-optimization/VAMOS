"""MOEA/D configuration."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, overload

from vamos.engine.archive import ExternalArchiveConfig
from vamos.foundation.encoding import EncodingLike

from .base import (
    ConstraintModeStr,
    ResultMode,
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
    _SerializableConfig,
    _TrackGenealogyBuilder,
    _validate_operators,
)
from .types import AggregationName, RepairConfigValue


@dataclass(frozen=True)
class MOEADConfig(_SerializableConfig):
    pop_size: int
    batch_size: int
    neighbor_size: int
    delta: float
    replace_limit: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    aggregation: tuple[str, dict[str, Any]]
    weight_vectors: dict[str, int | str | None] | None
    constraint_mode: ConstraintModeStr = "feasibility"
    repair: RepairConfigValue = "auto"
    initializer: dict[str, Any] | None = None
    mutation_prob_factor: float | None = None
    use_numba_variation: bool | None = None
    track_genealogy: bool = False
    result_mode: ResultMode | None = None
    external_archive: ExternalArchiveConfig | None = None

    @classmethod
    def default(
        cls,
        pop_size: int | None = None,
        n_var: int | None = None,
        n_obj: int = 3,
        encoding: EncodingLike | None = None,
    ) -> MOEADConfig:
        """Create a default MOEA/D configuration with sensible defaults.

        Without ``pop_size``, the population is the weight-vector lattice: 99
        divisions with two objectives, 12 with three and 6 with more. Discrete
        and mixed ``encoding`` values take their crossover and mutation from the
        shared encoding table.
        """
        divisions = 99 if n_obj == 2 else (12 if n_obj == 3 else 6)
        if pop_size is not None and n_obj == 2:
            divisions = max(1, int(pop_size) - 1)
        if pop_size is None:
            pop_size = divisions + 1 if n_obj == 2 else comb(divisions + n_obj - 1, n_obj - 1)
        mut_prob = 1.0 / n_var if n_var else 0.1
        crossover, mutation = _encoding_operators(
            encoding, mut_prob, real=(("de", {"cr": 1.0, "f": 0.5}), ("pm", {"prob": mut_prob, "eta": 20.0}))
        )
        return (
            cls.builder()
            .pop_size(pop_size)
            .batch_size(1)
            .neighbor_size(20)
            .delta(0.9)
            .replace_limit(2)
            .crossover(crossover[0], **crossover[1])
            .mutation(mutation[0], **mutation[1])
            .aggregation("pbi", theta=5.0)
            .weight_vectors(divisions=divisions)
            .build()
        )

    @classmethod
    def builder(cls) -> _MOEADConfigBuilder:
        return _MOEADConfigBuilder()


class _MOEADConfigBuilder(
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
    Declarative configuration holder for MOEA/D settings.

    Examples:
        # Fluent builder
        cfg = MOEADConfig.builder().pop_size(100).neighbor_size(20).build()

        # Quick default configuration
        cfg = MOEADConfig.default()
    """

    def batch_size(self, value: int) -> _MOEADConfigBuilder:
        self._cfg["batch_size"] = value
        return self

    def neighbor_size(self, value: int) -> _MOEADConfigBuilder:
        self._cfg["neighbor_size"] = value
        return self

    def delta(self, value: float) -> _MOEADConfigBuilder:
        self._cfg["delta"] = value
        return self

    def replace_limit(self, value: int) -> _MOEADConfigBuilder:
        self._cfg["replace_limit"] = value
        return self

    @overload
    def aggregation(self, method: AggregationName, **kwargs: Any) -> _MOEADConfigBuilder: ...

    @overload
    def aggregation(self, method: str, **kwargs: Any) -> _MOEADConfigBuilder: ...

    def aggregation(self, method: str, **kwargs: Any) -> _MOEADConfigBuilder:
        self._cfg["aggregation"] = (method, kwargs)
        return self

    def weight_vectors(self, *, path: str | None = None, divisions: int | None = None) -> _MOEADConfigBuilder:
        self._cfg["weight_vectors"] = {"path": path, "divisions": divisions}
        return self

    def use_numba_variation(self, enabled: bool = True) -> _MOEADConfigBuilder:
        self._cfg["use_numba_variation"] = bool(enabled)
        return self

    def build(self) -> MOEADConfig:
        _require_fields(
            self._cfg,
            (
                "pop_size",
                "neighbor_size",
                "delta",
                "replace_limit",
                "crossover",
                "mutation",
                "aggregation",
            ),
            "MOEA/D",
        )
        _validate_operators(self._cfg)
        return MOEADConfig(
            pop_size=self._cfg["pop_size"],
            batch_size=int(self._cfg.get("batch_size", 1)),
            neighbor_size=self._cfg["neighbor_size"],
            delta=self._cfg["delta"],
            replace_limit=self._cfg["replace_limit"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            aggregation=self._cfg["aggregation"],
            weight_vectors=self._cfg.get("weight_vectors"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            repair=self._cfg.get("repair", "auto"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            use_numba_variation=self._cfg.get("use_numba_variation"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            external_archive=self._cfg.get("external_archive"),
        )
