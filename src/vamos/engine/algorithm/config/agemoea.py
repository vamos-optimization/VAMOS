"""AGE-MOEA configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Self

from vamos.engine.algorithm.components.constraint_handling import FRONT_TECHNIQUES, TechniqueParameter
from vamos.engine.archive import ExternalArchiveConfig
from vamos.foundation.encoding import EncodingLike

from .base import (
    DEFAULT_POP_SIZE,
    ConstraintModeStr,
    LiveCallbackMode,
    ResultMode,
    _ConfigBuilderState,
    _ConstraintHandlingBuilder,
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
    _warn_without_effect,
)
from .types import AGEMOEAVariant, RepairConfigValue, resolve_agemoea_variant


class _AGEMOEAConfigBuilder(
    _ConfigBuilderState,
    _PopSizeBuilder,
    _CrossoverBuilder,
    _MutationBuilder,
    _RepairBuilder,
    _InitializerBuilder,
    _MutationProbFactorBuilder,
    _ConstraintHandlingBuilder,
    _ConstraintModeBuilder,
    _TrackGenealogyBuilder,
    _ResultArchiveBuilder,
):
    """
    Fluent builder for AGE-MOEA configs.
    """

    def variant(self, value: AGEMOEAVariant) -> Self:
        """The survival: ``"agemoea"`` (the default) or ``"agemoea_ii"``, AGE-MOEA-II's (Panichella, GECCO 2022).

        AGE-MOEA-II estimates the geometry of the first front by Newton-Raphson from the point
        nearest to the ideal point and measures diversity with geodesic distances on the estimated
        manifold; the rest of the algorithm is the same.
        """
        self._cfg["variant"] = resolve_agemoea_variant(value)
        return self

    def build(self) -> AGEMOEAConfig:
        _require_fields(
            self._cfg,
            ("pop_size", "crossover", "mutation"),
            "AGE-MOEA",
        )
        _validate_operators(self._cfg)
        return AGEMOEAConfig(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            repair=self._cfg.get("repair", "auto"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            constraint_handling=self._cfg.get("constraint_handling"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            external_archive=self._cfg.get("external_archive"),
            live_callback_mode=self._cfg.get("live_callback_mode", "population"),
            variant=self._cfg.get("variant", "agemoea"),
        )


@dataclass(frozen=True)
class AGEMOEAConfig(_SerializableConfig):
    #: The techniques that rank a population into fronts; ``two_rankings`` is defined for NSGA-II only.
    constraint_handling_techniques: ClassVar[tuple[str, ...]] = FRONT_TECHNIQUES
    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    repair: RepairConfigValue = "auto"
    initializer: dict[str, Any] | None = None
    mutation_prob_factor: float | None = None
    constraint_mode: ConstraintModeStr = "feasibility"
    track_genealogy: bool = False
    result_mode: ResultMode | None = None
    external_archive: ExternalArchiveConfig | None = None
    # Fields added after 1.0 go last, so that the constructor keeps the positions of its parameters.
    constraint_handling: tuple[str, dict[str, TechniqueParameter]] | None = None
    live_callback_mode: LiveCallbackMode = "population"
    variant: AGEMOEAVariant = "agemoea"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "variant", resolve_agemoea_variant(self.variant))
        if self.track_genealogy:
            _warn_without_effect("AGE-MOEA", "track_genealogy", "AGE-MOEA records no genealogy", stacklevel=3)

    @classmethod
    def default(
        cls,
        pop_size: int | None = None,
        n_var: int | None = None,
        encoding: EncodingLike | None = None,
    ) -> AGEMOEAConfig:
        """Create a default AGE-MOEA configuration.

        ``pop_size=None`` means 100. Discrete and mixed ``encoding`` values take
        their crossover and mutation from the shared encoding table.
        """
        mut_prob = 1.0 / n_var if n_var else 0.1
        crossover, mutation = _encoding_operators(
            encoding, mut_prob, real=(("sbx", {"prob": 0.9, "eta": 15.0}), ("pm", {"prob": mut_prob, "eta": 20.0}))
        )
        return (
            cls.builder()
            .pop_size(DEFAULT_POP_SIZE if pop_size is None else pop_size)
            .crossover(crossover[0], **crossover[1])
            .mutation(mutation[0], **mutation[1])
            .build()
        )

    @classmethod
    def builder(cls) -> _AGEMOEAConfigBuilder:
        return _AGEMOEAConfigBuilder()
