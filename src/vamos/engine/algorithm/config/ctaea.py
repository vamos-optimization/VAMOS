"""C-TAEA configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vamos.engine.archive import ExternalArchiveConfig
from vamos.foundation.encoding import EncodingLike

from .base import (
    DEFAULT_POP_SIZE,
    ConstraintModeStr,
    ResultMode,
    _ConfigBuilderState,
    _ConstraintModeBuilder,
    _CrossoverBuilder,
    _encoding_operators,
    _InitializerBuilder,
    _MutationBuilder,
    _PopSizeBuilder,
    _RepairBuilder,
    _require_fields,
    _ResultArchiveBuilder,
    _SerializableConfig,
    _validate_operators,
)
from .types import RepairConfigValue


@dataclass(frozen=True)
class CTAEAConfig(_SerializableConfig):
    """Configuration of C-TAEA (Li et al., 2019).

    ``pop_size`` is the requested number of weight vectors, at least 1. The number of vectors
    generated from it, N, is the size of each of the two archives (91 for 100 and three
    objectives), so a run's initial population and generations have N individuals. The crossover
    must mate parents two by two; only its first child is kept.
    """

    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    repair: RepairConfigValue = "auto"
    initializer: dict[str, Any] | None = None
    constraint_mode: ConstraintModeStr = "feasibility"
    result_mode: ResultMode | None = None
    external_archive: ExternalArchiveConfig | None = None

    @classmethod
    def default(
        cls,
        pop_size: int | None = None,
        n_var: int | None = None,
        encoding: EncodingLike | None = None,
    ) -> CTAEAConfig:
        """Create the default C-TAEA configuration: that of the paper and of jMetal's port.

        ``pop_size=None`` means 100 requested weight vectors. Real variables use SBX (probability
        1.0, distribution index 20) and polynomial mutation (probability ``1/n``, distribution
        index 20). Discrete and mixed ``encoding`` values take their crossover and mutation from
        the shared encoding table.
        """
        mut_prob = 1.0 / n_var if n_var else 0.1
        crossover, mutation = _encoding_operators(
            encoding, mut_prob, real=(("sbx", {"prob": 1.0, "eta": 20.0}), ("pm", {"prob": mut_prob, "eta": 20.0}))
        )
        return (
            cls.builder()
            .pop_size(DEFAULT_POP_SIZE if pop_size is None else pop_size)
            .crossover(crossover[0], **crossover[1])
            .mutation(mutation[0], **mutation[1])
            .build()
        )

    @classmethod
    def builder(cls) -> _CTAEAConfigBuilder:
        return _CTAEAConfigBuilder()


class _CTAEAConfigBuilder(
    _ConfigBuilderState,
    _PopSizeBuilder,
    _CrossoverBuilder,
    _MutationBuilder,
    _RepairBuilder,
    _InitializerBuilder,
    _ConstraintModeBuilder,
    _ResultArchiveBuilder,
):
    """
    Fluent builder for C-TAEA configs.

    Examples:
        cfg = CTAEAConfig.default(n_var=12)
        cfg = CTAEAConfig.builder().pop_size(100).crossover("sbx", prob=1.0, eta=20.0).mutation("pm", prob="1/n", eta=20.0).build()
    """

    def build(self) -> CTAEAConfig:
        _require_fields(self._cfg, ("pop_size", "crossover", "mutation"), "C-TAEA")
        _validate_operators(self._cfg)
        return CTAEAConfig(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            repair=self._cfg.get("repair", "auto"),
            initializer=self._cfg.get("initializer"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            external_archive=self._cfg.get("external_archive"),
        )


__all__ = ["CTAEAConfig"]
