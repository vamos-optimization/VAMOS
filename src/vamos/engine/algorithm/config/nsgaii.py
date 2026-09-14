"""NSGA-II configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vamos.engine.archive import ExternalArchiveConfig
from vamos.foundation.encoding import normalize_encoding

from .base import (
    ConstraintModeStr,
    LiveCallbackMode,
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


def normalize_nsgaii_steady_state(
    *,
    steady_state: bool,
    offspring_size: int | None,
    replacement_size: int | None,
) -> tuple[int | None, int | None]:
    """Validate NSGA-II incremental settings and resolve steady-state sizes.

    ``steady_state=True`` is the public shorthand for classic one-offspring
    NSGA-II: one offspring is evaluated and environmental selection is applied
    immediately, so both the offspring and replacement sizes resolve to one.

    Explicit contradictory sizes are rejected rather than silently ignored.
    When steady-state mode is disabled, the supplied values are preserved so
    the existing ``offspring_size`` incremental-batch behavior is unchanged.
    """
    resolved_offspring = None if offspring_size is None else int(offspring_size)
    resolved_replacement = None if replacement_size is None else int(replacement_size)

    if resolved_offspring is not None and resolved_offspring <= 0:
        raise ValueError("offspring size must be positive.")
    if resolved_replacement is not None and resolved_replacement <= 0:
        raise ValueError("replacement size must be positive.")

    if not steady_state:
        return resolved_offspring, resolved_replacement

    if resolved_offspring not in {None, 1}:
        raise ValueError("steady-state NSGA-II requires offspring_size=1 when offspring_size is specified.")
    if resolved_replacement not in {None, 1}:
        raise ValueError("steady-state NSGA-II requires replacement_size=1 when replacement_size is specified.")
    return 1, 1


class _NSGAIIConfigBuilder(
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
    Fluent builder for NSGA-II configs.
    """

    def offspring_size(self, value: int) -> _NSGAIIConfigBuilder:
        if value <= 0:
            raise ValueError("offspring size must be positive.")
        self._cfg["offspring_size"] = value
        return self

    def steady_state(self, enabled: bool = True) -> _NSGAIIConfigBuilder:
        """Enable classic one-offspring steady-state NSGA-II.

        When enabled, the built configuration resolves ``offspring_size`` and
        ``replacement_size`` to one. Explicit conflicting values fail fast.
        """
        self._cfg["steady_state"] = bool(enabled)
        return self

    def replacement_size(self, value: int) -> _NSGAIIConfigBuilder:
        if value <= 0:
            raise ValueError("replacement size must be positive.")
        self._cfg["replacement_size"] = value
        return self

    def immigration(self, config: dict[str, Any] | None) -> _NSGAIIConfigBuilder:
        if config is None:
            self._cfg["immigration"] = None
        else:
            self._cfg["immigration"] = dict(config)
        return self

    def parent_selection_filter(self, fn: Any | None) -> _NSGAIIConfigBuilder:
        self._cfg["parent_selection_filter"] = fn
        return self

    def live_callback_mode(self, mode: LiveCallbackMode) -> _NSGAIIConfigBuilder:
        self._cfg["live_callback_mode"] = str(mode)
        return self

    def generation_callback(
        self,
        fn: Any | None,
        *,
        copy_arrays: bool = True,
    ) -> _NSGAIIConfigBuilder:
        self._cfg["generation_callback"] = fn
        self._cfg["generation_callback_copy"] = bool(copy_arrays)
        return self

    def build(self) -> NSGAIIConfig:
        _require_fields(
            self._cfg,
            ("crossover", "mutation"),
            "NSGA-II",
        )
        _validate_operators(self._cfg)
        pop_size = int(self._cfg.get("pop_size", 100))
        selection = self._cfg.get("selection", ("tournament", {}))
        return NSGAIIConfig(
            pop_size=pop_size,
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            selection=selection,
            offspring_size=self._cfg.get("offspring_size"),
            steady_state=bool(self._cfg.get("steady_state", False)),
            replacement_size=self._cfg.get("replacement_size"),
            repair=self._cfg.get("repair", "auto"),
            external_archive=self._cfg.get("external_archive"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            immigration=self._cfg.get("immigration"),
            parent_selection_filter=self._cfg.get("parent_selection_filter"),
            live_callback_mode=self._cfg.get("live_callback_mode", "nd_only"),
            generation_callback=self._cfg.get("generation_callback"),
            generation_callback_copy=bool(self._cfg.get("generation_callback_copy", True)),
        )


@dataclass(frozen=True)
class NSGAIIConfig(_SerializableConfig):
    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    selection: tuple[str, dict[str, Any]]
    offspring_size: int | None = None
    steady_state: bool = False
    replacement_size: int | None = None
    repair: RepairConfigValue = "auto"
    external_archive: ExternalArchiveConfig | None = None
    initializer: dict[str, Any] | None = None
    mutation_prob_factor: float | None = None
    result_mode: ResultMode | None = None
    constraint_mode: ConstraintModeStr = "feasibility"
    track_genealogy: bool = False
    immigration: dict[str, Any] | None = None
    parent_selection_filter: Any | None = None
    live_callback_mode: LiveCallbackMode = "nd_only"
    generation_callback: Any | None = None
    generation_callback_copy: bool = True

    def __post_init__(self) -> None:
        offspring_size, replacement_size = normalize_nsgaii_steady_state(
            steady_state=bool(self.steady_state),
            offspring_size=self.offspring_size,
            replacement_size=self.replacement_size,
        )
        object.__setattr__(self, "offspring_size", offspring_size)
        object.__setattr__(self, "replacement_size", replacement_size)

    @classmethod
    def default(
        cls,
        pop_size: int = 100,
        n_var: int | None = None,
        encoding: str | None = None,
    ) -> NSGAIIConfig:
        """
        Create a default NSGA-II configuration with sensible defaults.

        Parameters
        ----------
        pop_size
            Population size.
        n_var
            Number of variables used for the default mutation probability.
        encoding
            Problem encoding. If omitted, defaults to ``"real"``.

        Returns
        -------
        NSGAIIConfig
            Frozen configuration ready to use.
        """
        normalized = normalize_encoding(encoding, default="real")
        mut_prob = 1.0 / n_var if n_var else 0.1
        builder = cls.builder().pop_size(pop_size).selection("tournament")

        if normalized == "permutation":
            return builder.crossover("ox").mutation("swap").build()
        if normalized == "binary":
            return builder.crossover("uniform", prob=0.9).mutation("bitflip", prob=mut_prob).build()
        if normalized == "integer":
            return builder.crossover("sbx", prob=0.9, eta=20.0).mutation("pm", prob=mut_prob, eta=20.0).build()
        if normalized == "mixed":
            return builder.crossover("mixed", prob=0.9).mutation("mixed", prob=mut_prob).build()

        return builder.crossover("sbx", prob=1.0, eta=20.0).mutation("pm", prob=mut_prob, eta=20.0).build()

    @classmethod
    def builder(cls) -> _NSGAIIConfigBuilder:
        return _NSGAIIConfigBuilder()
