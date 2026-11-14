"""ToP configuration."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any

from vamos.engine.archive import ExternalArchiveConfig
from vamos.foundation.exceptions import ConfigurationError

from .base import (
    DEFAULT_POP_SIZE,
    ConstraintModeStr,
    ResultMode,
    _ConfigBuilderState,
    _ConstraintModeBuilder,
    _CrossoverBuilder,
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

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


@dataclass(frozen=True)
class ToPConfig(_SerializableConfig):
    """Configuration of ToP (Liu and Wang, 2019), with NSGA-II in its second phase.

    ``pop_size`` is the size of the population, at least 1. The variation is ToP's differential
    evolution for real variables: the crossover must be ``("de", {"cr": ..., "f": ...})``, whose
    ``cr`` and ``f`` are the crossover rate and the scale factor of both phases, and the mutation
    is applied to every trial vector. Phase 1 lasts while ``(delta >= delta_threshold or Pf <=
    feasible_ratio_threshold)`` and the evaluations done are not more than
    ``phase_one_evaluation_fraction`` of the budget.
    """

    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    delta_threshold: float = 0.2
    feasible_ratio_threshold: float = 1.0 / 3.0
    phase_one_evaluation_fraction: float = 0.9
    repair: RepairConfigValue = "auto"
    initializer: dict[str, Any] | None = None
    constraint_mode: ConstraintModeStr = "feasibility"
    result_mode: ResultMode | None = None
    external_archive: ExternalArchiveConfig | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        name, params = self.crossover if isinstance(self.crossover, tuple) else (None, {})
        if not isinstance(name, str) or name.strip().lower() != "de":
            raise ConfigurationError(f"ToP's variation is differential evolution: its crossover must be 'de', got {name!r}.")
        unknown = sorted(set(params) - {"cr", "CR", "f", "F"})
        if unknown:
            raise ConfigurationError(f"ToP's crossover 'de' takes the parameters cr and f; got {', '.join(unknown)}.")
        f = params.get("f", params.get("F", 0.5))
        if isinstance(f, bool) or not isinstance(f, (int, float)) or not (math.isfinite(f) and f > 0.0):
            raise ConfigurationError(f"ToP's crossover 'de' parameter f must be positive and finite; got {f!r}.")

    @classmethod
    def default(cls, pop_size: int | None = None, n_var: int | None = None) -> ToPConfig:
        """Create the default ToP configuration: PlatEMO's, which jMetal's port also uses.

        ``pop_size=None`` means 100. The differential evolution has CR = 1.0 and F = 0.5, and the
        polynomial mutation probability ``1/n`` and distribution index 20. The switch thresholds
        are 0.2 for ``delta``, 1/3 for ``Pf`` and 90% of the budget.
        """
        mut_prob = 1.0 / n_var if n_var else 0.1
        return (
            cls.builder()
            .pop_size(DEFAULT_POP_SIZE if pop_size is None else pop_size)
            .crossover("de", cr=1.0, f=0.5)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .build()
        )

    @classmethod
    def builder(cls) -> _ToPConfigBuilder:
        return _ToPConfigBuilder()


class _ToPConfigBuilder(
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
    Fluent builder for ToP configs.

    Examples:
        cfg = ToPConfig.default(n_var=10)
        cfg = ToPConfig.builder().pop_size(100).crossover("de", cr=1.0, f=0.5).mutation("pm", prob="1/n", eta=20.0).build()
    """

    def delta_threshold(self, value: float) -> Self:
        """Phase 1 goes on while ``delta`` is at least this value (or ``Pf`` is not above its threshold); 0.2 by default."""
        self._cfg["delta_threshold"] = float(value)
        return self

    def feasible_ratio_threshold(self, value: float) -> Self:
        """Phase 1 goes on while ``Pf`` is not above this value (or ``delta`` is at least its threshold); 1/3 by default."""
        self._cfg["feasible_ratio_threshold"] = float(value)
        return self

    def phase_one_evaluation_fraction(self, value: float) -> Self:
        """Phase 1 ends when the evaluations done exceed this fraction of the budget; 0.9 by default."""
        self._cfg["phase_one_evaluation_fraction"] = float(value)
        return self

    def build(self) -> ToPConfig:
        _require_fields(self._cfg, ("pop_size", "crossover", "mutation"), "ToP")
        _validate_operators(self._cfg)
        return ToPConfig(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            delta_threshold=self._cfg.get("delta_threshold", 0.2),
            feasible_ratio_threshold=self._cfg.get("feasible_ratio_threshold", 1.0 / 3.0),
            phase_one_evaluation_fraction=self._cfg.get("phase_one_evaluation_fraction", 0.9),
            repair=self._cfg.get("repair", "auto"),
            initializer=self._cfg.get("initializer"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            external_archive=self._cfg.get("external_archive"),
        )


__all__ = ["ToPConfig"]
