"""SMS-EMOA configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, TypeAlias

from vamos.engine.algorithm.components.constraint_handling import FRONT_TECHNIQUES
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
    _SelectionBuilder,
    _SerializableConfig,
    _TrackGenealogyBuilder,
    _validate_operators,
)
from .types import RepairConfigValue

ReferencePointRule: TypeAlias = Literal["relative", "jmetal"]

# The default rule goes unrecorded, so that configurations without a rule keep their 1.0 form.
_REFERENCE_POINT_RULES = ("relative", "jmetal")


@dataclass(frozen=True)
class SMSEMOAConfig(_SerializableConfig):
    #: The techniques that rank a population into fronts; ``two_rankings`` is defined for NSGA-II only.
    constraint_handling_techniques: ClassVar[tuple[str, ...]] = FRONT_TECHNIQUES
    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    selection: tuple[str, dict[str, Any]]
    reference_point: dict[str, Any]
    eliminate_duplicates: bool = False
    constraint_mode: ConstraintModeStr = "feasibility"
    repair: RepairConfigValue = "auto"
    initializer: dict[str, Any] | None = None
    mutation_prob_factor: float | None = None
    track_genealogy: bool = False
    result_mode: ResultMode | None = None
    external_archive: ExternalArchiveConfig | None = None
    # Fields added after 1.0 go last, so that the constructor keeps the positions of its parameters.
    constraint_handling: tuple[str, dict[str, float]] | None = None
    live_callback_mode: LiveCallbackMode = "population"

    @classmethod
    def default(
        cls,
        pop_size: int | None = None,
        n_var: int | None = None,
        encoding: EncodingLike | None = None,
    ) -> SMSEMOAConfig:
        """Create a default SMS-EMOA configuration.

        ``pop_size=None`` means 100. Discrete and mixed ``encoding`` values take
        their crossover and mutation from the shared encoding table.
        """
        mut_prob = 1.0 / n_var if n_var else 0.1
        crossover, mutation = _encoding_operators(
            encoding, mut_prob, real=(("sbx", {"prob": 0.9, "eta": 20.0}), ("pm", {"prob": mut_prob, "eta": 20.0}))
        )
        return (
            cls.builder()
            .pop_size(DEFAULT_POP_SIZE if pop_size is None else pop_size)
            .crossover(crossover[0], **crossover[1])
            .mutation(mutation[0], **mutation[1])
            .selection("random")
            .reference_point(adaptive=True)
            .build()
        )

    @classmethod
    def builder(cls) -> _SMSEMOAConfigBuilder:
        return _SMSEMOAConfigBuilder()


class _SMSEMOAConfigBuilder(
    _ConfigBuilderState,
    _PopSizeBuilder,
    _CrossoverBuilder,
    _MutationBuilder,
    _SelectionBuilder,
    _RepairBuilder,
    _InitializerBuilder,
    _MutationProbFactorBuilder,
    _ConstraintHandlingBuilder,
    _ConstraintModeBuilder,
    _TrackGenealogyBuilder,
    _ResultArchiveBuilder,
):
    """
    Declarative configuration holder for SMS-EMOA settings.

    Examples:
        cfg = SMSEMOAConfig.default()
        cfg = SMSEMOAConfig.builder().pop_size(100).crossover("sbx", prob=1.0).build()
    """

    def eliminate_duplicates(self, enabled: bool = True) -> _SMSEMOAConfigBuilder:
        self._cfg["eliminate_duplicates"] = bool(enabled)
        return self

    def reference_point(
        self,
        *,
        vector: Any = None,
        offset: float | None = None,
        adaptive: bool | None = None,
        rule: ReferencePointRule = "relative",
    ) -> _SMSEMOAConfigBuilder:
        """Reference point of the hypervolume contributions.

        With the ``"relative"`` rule (the default) and without ``vector``, the reference
        is ``nadir + offset * (nadir - ideal)`` of the population and the new child, so
        ``offset`` (default 1.0) is a fraction of each objective's range and the
        selection does not depend on objective units; it is recomputed at every step
        unless ``adaptive=False``. A ``vector`` is used as given and stays fixed, so it
        cannot be combined with ``adaptive=True``; points outside its box contribute 0.

        ``rule="jmetal"`` is the reference of jMetal's and Evolver's SMS-EMOA
        (``SMSEMOAReplacement``): each coordinate is the worst value of the population
        and the new child times 1.1, or divided by 1.1 when it is negative, recomputed
        at every step. It depends on where each objective's zero lies, and takes no
        ``vector``, ``offset`` or ``adaptive=False``.
        """
        if rule not in _REFERENCE_POINT_RULES:
            raise ValueError(f"reference_point rule must be one of {', '.join(map(repr, _REFERENCE_POINT_RULES))}; got {rule!r}.")
        if rule == "jmetal":
            if vector is not None or offset is not None or adaptive is False:
                raise ValueError("The 'jmetal' reference_point rule takes no vector, offset or adaptive=False.")
            self._cfg["reference_point"] = {"vector": None, "adaptive": True, "rule": "jmetal"}
            return self
        offset_value = 1.0 if offset is None else float(offset)
        if not math.isfinite(offset_value) or offset_value < 0.0:
            raise ValueError(f"reference_point offset must be a finite number >= 0, got {offset!r}.")
        if vector is not None and adaptive:
            raise ValueError("A reference_point vector is fixed; do not combine vector= with adaptive=True.")
        self._cfg["reference_point"] = {
            "vector": vector,
            "offset": offset_value,
            "adaptive": vector is None if adaptive is None else bool(adaptive),
        }
        return self

    def build(self) -> SMSEMOAConfig:
        _require_fields(
            self._cfg,
            ("pop_size", "crossover", "mutation", "selection"),
            "SMS-EMOA",
        )
        _validate_operators(self._cfg)
        reference_point = self._cfg.get("reference_point", {"offset": 1.0, "adaptive": True})
        return SMSEMOAConfig(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            selection=self._cfg["selection"],
            reference_point=reference_point,
            eliminate_duplicates=bool(self._cfg.get("eliminate_duplicates", False)),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            constraint_handling=self._cfg.get("constraint_handling"),
            repair=self._cfg.get("repair", "auto"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            external_archive=self._cfg.get("external_archive"),
            live_callback_mode=self._cfg.get("live_callback_mode", "population"),
        )
