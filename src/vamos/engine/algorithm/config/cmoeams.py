"""CMOEA-MS configuration."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from vamos.engine.archive import ExternalArchiveConfig
from vamos.foundation.encoding import EncodingLike
from vamos.foundation.exceptions import ConfigurationError

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

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

_DE_VARIANTS = ("rand_1_bin", "rand_1_exp")


@dataclass(frozen=True)
class CMOEAMSConfig(_SerializableConfig):
    """Configuration of CMOEA-MS (Tian et al., 2022).

    ``pop_size`` is the size of the population, at least 2. The crossover selects the variation:
    ``("de", {"cr": ..., "f": ...})`` the DE variation (real variables only), whose crossover is
    ``de_variant``, ``"rand_1_bin"`` (binomial, PlatEMO's) or ``"rand_1_exp"`` (exponential); any
    crossover with two parents and two children the GA variation, which keeps both children.
    ``feasible_ratio_threshold`` is the paper's ``lambda``: the fitness is that of stage B when the
    ratio of feasible solutions of the union of the population and the offspring is greater than
    it, and at least 10% of the evaluations have been done.
    """

    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    feasible_ratio_threshold: float = 0.5
    de_variant: str = "rand_1_bin"
    repair: RepairConfigValue = "auto"
    initializer: dict[str, Any] | None = None
    constraint_mode: ConstraintModeStr = "feasibility"
    result_mode: ResultMode | None = None
    external_archive: ExternalArchiveConfig | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.de_variant, str) or self.de_variant.strip().lower() not in _DE_VARIANTS:
            raise ConfigurationError(f"CMOEA-MS's de_variant must be one of {', '.join(_DE_VARIANTS)}; got {self.de_variant!r}.")
        object.__setattr__(self, "de_variant", self.de_variant.strip().lower())
        name = self.crossover[0] if isinstance(self.crossover, tuple) else None
        is_de = isinstance(name, str) and name.strip().lower() == "de"
        if self.de_variant != "rand_1_bin" and not is_de:
            raise ConfigurationError(f"CMOEA-MS's de_variant applies to the DE variation (crossover 'de'), not to crossover {name!r}.")
        if is_de:
            params = self.crossover[1]
            unknown = sorted(set(params) - {"cr", "CR", "f", "F"})
            if unknown:
                raise ConfigurationError(f"CMOEA-MS's crossover 'de' takes the parameters cr and f; got {', '.join(unknown)}.")

    @classmethod
    def default(
        cls,
        pop_size: int | None = None,
        n_var: int | None = None,
        encoding: EncodingLike | None = None,
    ) -> CMOEAMSConfig:
        """Create the default CMOEA-MS configuration: that of the paper, of PlatEMO's implementation and of jMetal's port.

        ``pop_size=None`` means 100. Real variables use the GA variation with SBX (probability
        1.0, distribution index 20) and polynomial mutation (probability ``1/n``, distribution
        index 20), and ``feasible_ratio_threshold`` (``lambda``) is 0.5. Discrete and mixed
        ``encoding`` values take their crossover and mutation from the shared encoding table.
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
    def builder(cls) -> _CMOEAMSConfigBuilder:
        return _CMOEAMSConfigBuilder()


class _CMOEAMSConfigBuilder(
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
    Fluent builder for CMOEA-MS configs.

    Examples:
        cfg = CMOEAMSConfig.default(n_var=10)
        cfg = CMOEAMSConfig.builder().pop_size(100).crossover("sbx", prob=1.0, eta=20.0).mutation("pm", prob="1/n", eta=20.0).build()
        cfg = CMOEAMSConfig.builder().pop_size(100).crossover("de", cr=1.0, f=0.5).de_variant("rand_1_exp").mutation("pm").build()
    """

    def feasible_ratio_threshold(self, value: float) -> Self:
        """The paper's ``lambda``: stage B needs a ratio of feasible solutions greater than this value; 0.5 by default."""
        self._cfg["feasible_ratio_threshold"] = float(value)
        return self

    def de_variant(self, value: str) -> Self:
        """The crossover of the DE variation: ``"rand_1_bin"`` (binomial, the default) or ``"rand_1_exp"`` (exponential)."""
        self._cfg["de_variant"] = value
        return self

    def build(self) -> CMOEAMSConfig:
        _require_fields(self._cfg, ("pop_size", "crossover", "mutation"), "CMOEA-MS")
        _validate_operators(self._cfg)
        return CMOEAMSConfig(
            pop_size=self._cfg["pop_size"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            feasible_ratio_threshold=self._cfg.get("feasible_ratio_threshold", 0.5),
            de_variant=self._cfg.get("de_variant", "rand_1_bin"),
            repair=self._cfg.get("repair", "auto"),
            initializer=self._cfg.get("initializer"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            external_archive=self._cfg.get("external_archive"),
        )


__all__ = ["CMOEAMSConfig"]
