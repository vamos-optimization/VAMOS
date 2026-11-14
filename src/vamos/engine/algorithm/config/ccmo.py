"""CCMO configuration."""

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
class CCMOConfig(_SerializableConfig):
    """Configuration of CCMO (Tian et al., 2021).

    ``pop_size`` is the size of each of the two populations, at least 2. The crossover selects
    the variation: ``("de", {"cr": ..., "f": ...})`` the DE variation (real variables only), any
    two-parent crossover the GA variation, which keeps one child per pair of parents.
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
    ) -> CCMOConfig:
        """Create the default CCMO configuration: that of the paper and of jMetal's port.

        ``pop_size=None`` means 100. Real variables use the GA variation with SBX (probability
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
    def builder(cls) -> _CCMOConfigBuilder:
        return _CCMOConfigBuilder()


class _CCMOConfigBuilder(
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
    Fluent builder for CCMO configs.

    Examples:
        cfg = CCMOConfig.default(n_var=15)
        cfg = CCMOConfig.builder().pop_size(100).crossover("de", cr=1.0, f=0.5).mutation("pm", prob="1/n", eta=20.0).build()
    """

    def build(self) -> CCMOConfig:
        _require_fields(self._cfg, ("pop_size", "crossover", "mutation"), "CCMO")
        _validate_operators(self._cfg)
        return CCMOConfig(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            repair=self._cfg.get("repair", "auto"),
            initializer=self._cfg.get("initializer"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            external_archive=self._cfg.get("external_archive"),
        )


__all__ = ["CCMOConfig"]
