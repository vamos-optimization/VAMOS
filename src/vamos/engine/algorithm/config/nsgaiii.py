"""NSGA-III configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.reference_directions import lattice_size
from vamos.foundation.encoding import EncodingLike

from .base import (
    ConstraintModeStr,
    ResultMode,
    _ConfigBuilderState,
    _ConstraintModeBuilder,
    _CrossoverBuilder,
    _encoding_operators,
    _InitializerBuilder,
    _lattice_divisions_within,
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
        encoding: EncodingLike | None = None,
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
            Number of objectives used to choose the reference directions. With
            ``pop_size``, the complete Das-Dennis lattice with the most directions
            that fit in it (``pop_size - 1`` divisions with two objectives).
            Without it, 99 divisions for two objectives, 12 for three and 6 for
            four or more, and ``pop_size`` is the direction count.
        encoding
            Problem encoding. Discrete and mixed encodings take their crossover and
            mutation from the shared encoding table.
        """
        mut_prob = 1.0 / n_var if n_var else 0.1
        if pop_size is None:
            divisions = 99 if n_obj == 2 else (12 if n_obj == 3 else 6)
            pop_size = lattice_size(n_obj, divisions)
        else:
            divisions = _lattice_divisions_within(pop_size, n_obj, algorithm="NSGA-III")
        crossover, mutation = _encoding_operators(
            encoding, mut_prob, real=(("sbx", {"prob": 1.0, "eta": 30.0}), ("pm", {"prob": mut_prob, "eta": 20.0}))
        )
        return (
            cls.builder()
            .pop_size(pop_size)
            .crossover(crossover[0], **crossover[1])
            .mutation(mutation[0], **mutation[1])
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
