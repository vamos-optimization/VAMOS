"""NSGA-III configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.reference_directions import lattice_size
from vamos.foundation.encoding import EncodingLike
from vamos.foundation.exceptions import ConfigurationError

from .base import (
    ConstraintModeStr,
    LiveCallbackMode,
    ResultMode,
    _ConfigBuilderState,
    _ConstraintHandlingBuilder,
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
    _warn_without_effect,
)
from .types import RepairConfigValue

_REFERENCE_LATTICE_RULE = "since VAMOS 1.2.0, NSGA-III uses the largest reference-direction lattice that fits in pop_size and runs N >= H"

# Development after 1.4.0: NSGA-III's parent selections. "random" is Deb and Jain's (2014), the
# default; "tournament" is jMetal's NSGA-III tournament on rank, with ties decided at random.
NSGAIII_SELECTIONS = ("random", "tournament")


def nsgaiii_selection(selection: object) -> tuple[str, int]:
    """The method of NSGA-III's parent selection and its tournament size (2 for ``"random"``).

    Development after 1.4.0: NSGA-III runs ``"random"`` and ``"tournament"`` (with an optional
    ``size``), and rejects any other method or option instead of running a tournament.

    Raises
    ------
    ConfigurationError
        For a method other than ``"random"`` and ``"tournament"``, an option of ``"random"``, or an
        option of ``"tournament"`` other than ``size``.
    """
    if not (isinstance(selection, (tuple, list)) and len(selection) == 2 and isinstance(selection[0], str)):
        raise ConfigurationError(f"NSGA-III selection must be a (method, options) pair; got {selection!r}.")
    method, options = str(selection[0]).strip().lower(), selection[1]
    if not isinstance(options, dict):
        raise ConfigurationError(f"NSGA-III selection options must be a mapping; got {options!r}.")
    if method not in NSGAIII_SELECTIONS:
        raise ConfigurationError(
            f"NSGA-III does not support selection {selection[0]!r}: its parent selection is 'random' "
            "(Deb and Jain, 2014; the default) or 'tournament' (on rank, as jMetal's NSGA-III)."
        )
    allowed = {"size"} if method == "tournament" else set()
    unexpected = sorted(set(options) - allowed)
    if unexpected:
        detail = "only 'size'" if allowed else "no options"
        raise ConfigurationError(f"NSGA-III's {method} selection takes {detail}; got {', '.join(map(repr, unexpected))}.")
    size = options.get("size", 2)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ConfigurationError(f"NSGA-III's tournament size must be a positive integer; got {size!r}.")
    return method, size


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
    # Fields added after 1.0 go last, so that the constructor keeps the positions of its parameters.
    constraint_handling: tuple[str, dict[str, float]] | None = None
    live_callback_mode: LiveCallbackMode = "population"

    def __post_init__(self) -> None:
        super().__post_init__()
        nsgaiii_selection(self.selection)
        # ``enforce_ref_dirs`` is true in every default, so ``False`` warns wherever it comes from.
        # ``pop_size_auto`` is true in ``default()`` and false otherwise: only the builder method can
        # tell that a user set it.
        if not self.enforce_ref_dirs:
            _warn_without_effect("NSGA-III", "enforce_ref_dirs", _REFERENCE_LATTICE_RULE, stacklevel=3)

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

        Development after 1.4.0: parents are selected at random, as Deb and Jain (2014); the
        default used to be a binary tournament on rank and crowding distance.

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
            encoding, mut_prob, real=(("sbx", {"prob": 0.9, "eta": 30.0}), ("pm", {"prob": mut_prob, "eta": 20.0}))
        )
        builder = (
            cls.builder()
            .pop_size(pop_size)
            .crossover(crossover[0], **crossover[1])
            .mutation(mutation[0], **mutation[1])
            .selection("random")
            .reference_directions(divisions=divisions)
        )
        # The stable default keeps ``pop_size_auto=True`` without the warning of the builder method.
        builder._cfg["pop_size_auto"] = True
        return builder.build()

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
    _ConstraintHandlingBuilder,
    _ConstraintModeBuilder,
    _TrackGenealogyBuilder,
):
    """
    Declarative configuration holder for NSGA-III settings.

    Examples:
        cfg = NSGAIIIConfig.default(n_obj=3)
        cfg = NSGAIIIConfig.builder().pop_size(92).crossover("sbx", prob=1.0).build()
    """

    def selection(self, method: str, **kwargs: Any) -> Self:
        """Parent selection: ``"random"`` or ``"tournament"`` (with an optional ``size``, 2 by default).

        ``"random"`` draws the parents at random, as Deb and Jain (2014). With constraint values and
        feasibility first (``constraint_mode="feasibility"`` without ``constraint_handling``), each
        parent wins a binary tournament on the constraint violation that is decided at random between
        two feasible solutions, as the constrained NSGA-III of Jain and Deb (2014) and pymoo's NSGA-III.

        ``"tournament"`` takes ``size`` solutions at random and keeps the one of the lowest
        non-dominated rank (feasible solutions first, then by violation, or the fronts of
        ``constraint_handling``), with ties decided at random: jMetal's NSGA-III tournament. It does not
        use crowding distance.

        Development after 1.4.0: other methods and options raise ``ConfigurationError``; before,
        every method ran a binary tournament on rank and crowding distance.
        """
        nsgaiii_selection((method, kwargs))
        return super().selection(method, **kwargs)

    def reference_directions(
        self,
        *,
        path: str | None = None,
        divisions: int | None = None,
    ) -> _NSGAIIIConfigBuilder:
        """The reference directions: a file, or the complete Das-Dennis lattice of ``divisions``.

        ``path`` names a file with one direction per line, its components separated by whitespace or
        by commas, such as jMetal's weight files (``W3D_100.dat``, which VAMOS ships). Without either,
        NSGA-III runs the complete lattice with the most directions that fit in ``pop_size``. The
        population needs at least one individual per direction.

        **Development after 1.4.0:** the file may use whitespace as well as commas, and a ``path``
        that is not a file raises :class:`~vamos.exceptions.ConfigurationError` when the run starts;
        VAMOS 1.4.0 read commas only, and generated the lattice and saved it at a missing path.
        """
        self._cfg["reference_directions"] = {"path": path, "divisions": divisions}
        return self

    def enforce_ref_dirs(self, enabled: bool = True) -> _NSGAIIIConfigBuilder:
        """Kept for compatibility: it has no effect, warns, and goes in VAMOS 2.0."""
        if enabled:
            # ``False`` warns once, when the configuration is built.
            _warn_without_effect("NSGA-III", "enforce_ref_dirs", _REFERENCE_LATTICE_RULE, stacklevel=2)
        self._cfg["enforce_ref_dirs"] = bool(enabled)
        return self

    def pop_size_auto(self, enabled: bool = True) -> _NSGAIIIConfigBuilder:
        """Kept for compatibility: it has no effect, warns, and goes in VAMOS 2.0."""
        _warn_without_effect("NSGA-III", "pop_size_auto", _REFERENCE_LATTICE_RULE, stacklevel=2)
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
            constraint_handling=self._cfg.get("constraint_handling"),
            repair=self._cfg.get("repair", "auto"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "population"),
            external_archive=self._cfg.get("external_archive"),
            live_callback_mode=self._cfg.get("live_callback_mode", "population"),
        )
