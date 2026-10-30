"""NSGA-III configuration."""

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
    _SelectionBuilder,
    _SerializableConfig,
    _TrackGenealogyBuilder,
    _validate_operators,
)
from .types import RepairConfigValue


@dataclass(frozen=True)
class NSGAIIIConfig(_SerializableConfig):
    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    selection: tuple[str, dict[str, Any]]
    reference_directions: dict[str, int | str | None]
    enforce_ref_dirs: bool = True
    pop_size_auto: bool = False
    constraint_mode: ConstraintModeStr = "feasibility"
    repair: RepairConfigValue = "auto"
    initializer: dict[str, Any] | None = None
    mutation_prob_factor: float | None = None
    track_genealogy: bool = False
    result_mode: ResultMode | None = None
    external_archive: ExternalArchiveConfig | None = None

    @classmethod
    def default(
        cls,
        pop_size: int | None = None,
        n_var: int | None = None,
        n_obj: int = 3,
    ) -> NSGAIIIConfig:
        """
        Create a default NSGA-III configuration.

        Parameters
        ----------
        pop_size
            Population size. Defaults to the generated reference-direction count.
        n_var
            Number of variables used for the default mutation probability.
        n_obj
            Number of objectives used to choose the default reference directions:
            12 divisions for three objectives, 6 for four or more, and with two
            objectives ``pop_size - 1`` (one direction per individual), or 99
            without ``pop_size``.
        """
        mut_prob = 1.0 / n_var if n_var else 0.1
        if n_obj == 2:
            divisions = 99 if pop_size is None else max(1, int(pop_size) - 1)
        else:
            divisions = 12 if n_obj == 3 else 6
        if pop_size is None:
            pop_size = comb(divisions + n_obj - 1, n_obj - 1)
        return (
            cls.builder()
            .pop_size(pop_size)
            .crossover("sbx", prob=1.0, eta=30.0)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .selection("tournament")
            .reference_directions(divisions=divisions)
            .pop_size_auto(True)
            .build()
        )

    @classmethod
    def builder(cls) -> _NSGAIIIConfigBuilder:
        return _NSGAIIIConfigBuilder()


class _NSGAIIIConfigBuilder(
    _ConfigBuilderState,
    _PopSizeBuilder,
    _CrossoverBuilder,
    _MutationBuilder,
    _SelectionBuilder,
    _RepairBuilder,
    _InitializerBuilder,
    _MutationProbFactorBuilder,
    _ResultArchiveBuilder,
    _ConstraintModeBuilder,
    _TrackGenealogyBuilder,
):
    """
    Declarative configuration holder for NSGA-III settings.

    Examples:
        cfg = NSGAIIIConfig.default(n_obj=3)
        cfg = NSGAIIIConfig.builder().pop_size(92).crossover("sbx", prob=1.0).build()
    """

    def reference_directions(
        self,
        *,
        path: str | None = None,
        divisions: int | None = None,
    ) -> _NSGAIIIConfigBuilder:
        self._cfg["reference_directions"] = {"path": path, "divisions": divisions}
        return self

    def enforce_ref_dirs(self, enabled: bool = True) -> _NSGAIIIConfigBuilder:
        self._cfg["enforce_ref_dirs"] = bool(enabled)
        return self

    def pop_size_auto(self, enabled: bool = True) -> _NSGAIIIConfigBuilder:
        self._cfg["pop_size_auto"] = bool(enabled)
        return self

    def build(self) -> NSGAIIIConfig:
        _require_fields(
            self._cfg,
            ("pop_size", "crossover", "mutation", "selection"),
            "NSGA-III",
        )
        _validate_operators(self._cfg)
        ref_dirs = self._cfg.get("reference_directions", {})
        return NSGAIIIConfig(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            selection=self._cfg["selection"],
            reference_directions=ref_dirs,
            enforce_ref_dirs=bool(self._cfg.get("enforce_ref_dirs", True)),
            pop_size_auto=bool(self._cfg.get("pop_size_auto", False)),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            repair=self._cfg.get("repair", "auto"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "population"),
            external_archive=self._cfg.get("external_archive"),
        )
