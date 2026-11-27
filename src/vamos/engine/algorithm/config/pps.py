"""PPS configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Self

from vamos.engine.algorithm.components.weight_vectors import largest_weighted_population, vector_source
from vamos.engine.archive import ExternalArchiveConfig
from vamos.foundation.exceptions import ConfigurationError

from .base import (
    ConstraintModeStr,
    LiveCallbackMode,
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

#: The paper's population size and neighbourhood size (Fan et al., 2019, section 4.1).
_PAPER_POP_SIZE = 300
_PAPER_NEIGHBOR_SIZE = 30
# The admissible values of each real parameter: (low, high, low closed, high closed).
_RANGES: dict[str, tuple[float, float, bool, bool]] = {
    "delta": (0.0, 1.0, True, True),
    "switch_threshold": (0.0, math.inf, True, False),
    # A bound on the ratio of feasible to infeasible solutions, which has no upper bound.
    "feasibility_ratio_threshold": (0.0, math.inf, True, False),
    "tao": (0.0, 1.0, False, False),
    "cp": (0.0, math.inf, False, False),
    "decay_fraction": (0.0, 1.0, False, True),
}


def _check_number(name: str, value: object) -> float:
    number = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.nan
    low, high, low_closed, high_closed = _RANGES[name]
    above = number >= low if low_closed else number > low
    below = number <= high if high_closed else number < high
    if not (math.isfinite(number) and above and below):
        interval = f"{'[' if low_closed else '('}{low:g}, {high:g}{']' if high_closed else ')'}"
        raise ConfigurationError(f"PPS's {name} must be in {interval}; got {value!r}.")
    return number


def _check_integer(name: str, value: object, low: int, high: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
        bound = f"in [{low}, {high}]" if high is not None else f">= {low}"
        raise ConfigurationError(f"PPS's {name} must be an integer {bound}; got {value!r}.")
    return value


@dataclass(frozen=True)
class PPSConfig(_SerializableConfig):
    """Configuration of PPS-MOEA/D, the push and pull search in MOEA/D (Fan et al., 2019).

    ``pop_size`` is the number of subproblems, one per weight vector, which MOEA/D's
    ``weight_vectors`` give (see :class:`MOEADConfig`); ``neighbor_size`` (``T``) the size of the
    neighbourhoods, in ``[2, pop_size]``; ``delta`` the probability of drawing the parents from
    the neighbourhood; ``replace_limit`` (``n_r``) the most solutions a new one replaces. The
    variation is differential evolution for real variables: the crossover must be ``("de",
    {"cr": ..., "f": ...})``, and the mutation is applied to every new solution.

    The push stage switches to the pull stage when the largest relative change of the ideal and
    nadir points over the last ``switch_window`` generations (``l``) is at most
    ``switch_threshold`` (the ``epsilon`` of Eq. 4). In the pull stage, the epsilon level shrinks
    by ``tao`` (``tau``) while the ratio of feasible to infeasible solutions of the population is below
    ``feasibility_ratio_threshold`` (``alpha``), and otherwise follows ``epsilon(0) (1 - k /
    Tc) ** cp``; it is 0 from the generation ``Tc``, ``decay_fraction`` of the run's generations.
    The defaults are the paper's: ``N = 300``, ``T = 30``, ``delta = 0.9``, ``n_r = 2``,
    ``epsilon = 1e-3``, ``l = 20``, ``alpha = 0.95``, ``tau = 0.1``, ``cp = 2`` and ``Tc = 800`` of
    1000 generations, DE with CR = 1.0 and F = 0.5 and polynomial mutation ``1/n`` and 20.
    """

    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    neighbor_size: int = _PAPER_NEIGHBOR_SIZE
    delta: float = 0.9
    replace_limit: int = 2
    switch_threshold: float = 1e-3
    switch_window: int = 20
    feasibility_ratio_threshold: float = 0.95
    tao: float = 0.1
    cp: float = 2.0
    decay_fraction: float = 0.8
    weight_vectors: dict[str, int | str | None] | None = None
    repair: RepairConfigValue = "auto"
    initializer: dict[str, Any] | None = None
    constraint_mode: ConstraintModeStr = "feasibility"
    result_mode: ResultMode | None = None
    external_archive: ExternalArchiveConfig | None = None
    live_callback_mode: LiveCallbackMode = "population"

    def __post_init__(self) -> None:
        super().__post_init__()
        name, params = self.crossover if isinstance(self.crossover, tuple) else (None, {})
        if not isinstance(name, str) or name.strip().lower() != "de":
            raise ConfigurationError(f"PPS's variation is differential evolution: its crossover must be 'de', got {name!r}.")
        unknown = sorted(set(params) - {"cr", "CR", "f", "F"})
        if unknown:
            raise ConfigurationError(f"PPS's crossover 'de' takes the parameters cr and f; got {', '.join(unknown)}.")
        f = params.get("f", params.get("F", 0.5))
        if isinstance(f, bool) or not isinstance(f, (int, float)) or not (math.isfinite(f) and f > 0.0):
            raise ConfigurationError(f"PPS's crossover 'de' parameter f must be positive and finite; got {f!r}.")
        pop_size = _check_integer("pop_size", self.pop_size, 2)
        _check_integer("neighbor_size", self.neighbor_size, 2, pop_size)
        _check_integer("replace_limit", self.replace_limit, 1)
        _check_integer("switch_window", self.switch_window, 1)
        for field_name in _RANGES:
            object.__setattr__(self, field_name, _check_number(field_name, getattr(self, field_name)))
        object.__setattr__(self, "weight_vectors", vector_source(self.weight_vectors, field="weight_vectors", allow_method=True))

    @classmethod
    def default(cls, pop_size: int | None = None, n_var: int | None = None, n_obj: int | None = None) -> PPSConfig:
        """Create the default PPS configuration: the paper's.

        ``pop_size=None`` means the paper's 300, which has weight vectors with two and three
        objectives (the lattice of 23 divisions); with ``n_obj`` of four or more, it means the
        largest population up to 300 that has them, as MOEA/D's weight vectors need. ``neighbor_size``
        is 30, or ``pop_size`` when it is smaller. The differential evolution has CR = 1.0 and
        F = 0.5, the polynomial mutation probability ``1/n`` and distribution index 20; the stages
        have the paper's parameters.
        """
        size = pop_size if pop_size is not None else largest_weighted_population(_PAPER_POP_SIZE, int(n_obj or 2))
        mut_prob = 1.0 / n_var if n_var else 0.1
        return (
            cls.builder()
            .pop_size(size)
            .neighbor_size(min(_PAPER_NEIGHBOR_SIZE, size))
            .crossover("de", cr=1.0, f=0.5)
            .mutation("pm", prob=mut_prob, eta=20.0)
            .build()
        )

    @classmethod
    def builder(cls) -> _PPSConfigBuilder:
        return _PPSConfigBuilder()


class _PPSConfigBuilder(
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
    Fluent builder for PPS configs.

    Examples:
        cfg = PPSConfig.default(n_var=30)
        cfg = PPSConfig.builder().pop_size(300).crossover("de", cr=1.0, f=0.5).mutation("pm", prob="1/n", eta=20.0).build()
    """

    def neighbor_size(self, value: int) -> Self:
        """``T``, the size of each subproblem's neighbourhood; 30 by default."""
        self._cfg["neighbor_size"] = value
        return self

    def delta(self, value: float) -> Self:
        """The probability of drawing the parents, and the solutions to replace, from the neighbourhood; 0.9 by default."""
        self._cfg["delta"] = value
        return self

    def replace_limit(self, value: int) -> Self:
        """``n_r``, the most solutions a new solution replaces; 2 by default."""
        self._cfg["replace_limit"] = value
        return self

    def switch_threshold(self, value: float) -> Self:
        """The ``epsilon`` of Eq. 4: the push stage ends when the change rate ``r_k`` is at most this value; 1e-3 by default."""
        self._cfg["switch_threshold"] = value
        return self

    def switch_window(self, value: int) -> Self:
        """``l``, the generations over which the change rate of the ideal and nadir points is measured; 20 by default."""
        self._cfg["switch_window"] = value
        return self

    def feasibility_ratio_threshold(self, value: float) -> Self:
        """``alpha``: the epsilon level shrinks while the ratio of feasible to infeasible solutions is below it; 0.95 by default."""
        self._cfg["feasibility_ratio_threshold"] = value
        return self

    def tao(self, value: float) -> Self:
        """``tau``, the rate at which the epsilon level shrinks; 0.1 by default."""
        self._cfg["tao"] = value
        return self

    def cp(self, value: float) -> Self:
        """``cp``, the exponent of ``epsilon(0) (1 - k / Tc) ** cp``; 2 by default."""
        self._cfg["cp"] = value
        return self

    def decay_fraction(self, value: float) -> Self:
        """``Tc`` as a fraction of the run's generations, from which the epsilon level is 0; 0.8 by default (800 of 1000)."""
        self._cfg["decay_fraction"] = value
        return self

    def weight_vectors(
        self,
        *,
        path: str | None = None,
        divisions: int | None = None,
        inner_divisions: int | None = None,
        method: str | None = None,
        seed: int | None = None,
    ) -> Self:
        """The weight vectors, as MOEA/D's (see :class:`MOEADConfig`); by default uniform with two objectives, else a lattice of ``pop_size``.

        **Development after 1.5.2:** ``method="riesz_energy"`` and ``seed``, as MOEA/D's ``weight_vectors``.
        """
        source = {"path": path, "divisions": divisions, "inner_divisions": inner_divisions, "method": method, "seed": seed}
        self._cfg["weight_vectors"] = vector_source(source, allow_method=True)
        return self

    def build(self) -> PPSConfig:
        _require_fields(self._cfg, ("pop_size", "crossover", "mutation"), "PPS")
        _validate_operators(self._cfg)
        pop_size = self._cfg["pop_size"]
        return PPSConfig(
            pop_size=pop_size,
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            neighbor_size=self._cfg.get(
                "neighbor_size", min(_PAPER_NEIGHBOR_SIZE, pop_size) if isinstance(pop_size, int) else _PAPER_NEIGHBOR_SIZE
            ),
            delta=self._cfg.get("delta", 0.9),
            replace_limit=self._cfg.get("replace_limit", 2),
            switch_threshold=self._cfg.get("switch_threshold", 1e-3),
            switch_window=self._cfg.get("switch_window", 20),
            feasibility_ratio_threshold=self._cfg.get("feasibility_ratio_threshold", 0.95),
            tao=self._cfg.get("tao", 0.1),
            cp=self._cfg.get("cp", 2.0),
            decay_fraction=self._cfg.get("decay_fraction", 0.8),
            weight_vectors=self._cfg.get("weight_vectors"),
            repair=self._cfg.get("repair", "auto"),
            initializer=self._cfg.get("initializer"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            external_archive=self._cfg.get("external_archive"),
            live_callback_mode=self._cfg.get("live_callback_mode", "population"),
        )


__all__ = ["PPSConfig"]
