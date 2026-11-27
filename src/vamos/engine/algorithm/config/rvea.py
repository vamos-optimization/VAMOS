"""RVEA configuration."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, ClassVar, Literal, TypeAlias, cast

from vamos.engine.algorithm.components.constraint_handling import PAIRWISE_TECHNIQUES, TechniqueParameter
from vamos.engine.algorithm.components.weight_vectors import vector_source
from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.reference_directions import lattice_size
from vamos.foundation.encoding import EncodingLike
from vamos.foundation.exceptions import ConfigurationError

from .base import (
    ConstraintModeStr,
    LiveCallbackMode,
    ReferenceDirectionPopulationError,
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
    _SerializableConfig,
    _TrackGenealogyBuilder,
    _validate_operators,
    _warn_without_effect,
)
from .types import RepairConfigValue

RVEAVariant: TypeAlias = Literal["pymoo", "platemo"]
_RVEA_VARIANTS = ("pymoo", "platemo")


def normalize_rvea_variant(value: object) -> RVEAVariant:
    """RVEA's survival variant in lower case, or ``ConfigurationError``. VAMOS 1.5.0: new."""
    name = value.strip().lower() if isinstance(value, str) else None
    if name not in _RVEA_VARIANTS:
        raise ConfigurationError(f"RVEA variant must be one of {', '.join(map(repr, _RVEA_VARIANTS))}; got {value!r}.")
    return cast(RVEAVariant, name)


def rvea_partitions_for_population(pop_size: int, n_obj: int) -> int:
    """Return the Das-Dennis partitions whose lattice has exactly ``pop_size`` vectors.

    RVEA keeps one reference vector per individual, so ``pop_size`` must equal
    ``comb(n_partitions + n_obj - 1, n_obj - 1)``. With two objectives the
    answer is ``pop_size - 1``.

    Raises
    ------
    ConfigurationError
        If no lattice for ``n_obj`` objectives has ``pop_size`` vectors. The
        message names the nearest valid population sizes.
    """
    if n_obj < 2:
        raise ConfigurationError(f"RVEA needs at least two objectives, got n_obj={n_obj}.")
    if pop_size < 1:
        raise ConfigurationError(f"RVEA needs a positive pop_size, got {pop_size}.")
    partitions = 1
    size = comb(partitions + n_obj - 1, n_obj - 1)
    while size < pop_size:
        partitions += 1
        size = comb(partitions + n_obj - 1, n_obj - 1)
    if size == pop_size:
        return partitions
    nearest = [f"{size} (n_partitions={partitions})"]
    if partitions > 1:
        nearest.insert(0, f"{comb(partitions + n_obj - 2, n_obj - 1)} (n_partitions={partitions - 1})")
    raise ConfigurationError(
        f"RVEA keeps one reference vector per individual, and no Das-Dennis lattice for n_obj={n_obj} "
        f"has exactly pop_size={pop_size} vectors.",
        suggestion=f"Use a population size of {' or '.join(nearest)}, or set n_partitions explicitly.",
        details={"pop_size": pop_size, "n_obj": n_obj},
    )


@dataclass(frozen=True)
class RVEAConfig(_SerializableConfig):
    """Configuration of RVEA.

    RVEA keeps one individual per reference vector, so ``pop_size`` must equal their number. By
    default the vectors are the complete Das-Dennis lattice of ``n_partitions``.

    **VAMOS 1.5.0:** ``reference_vectors`` takes the vectors from elsewhere, with the
    keys of MOEA/D's ``weight_vectors`` and NSGA-III's ``reference_directions``:

    - ``path``: a file with one vector per line, its components separated by whitespace or commas,
      or a directory with jMetal's ``W{n_obj}D_{pop_size}.dat`` files, such as the jMetal files
      VAMOS ships (``W3D_100.dat`` runs 100 individuals with three objectives). The file must exist
      and hold ``pop_size`` vectors;
    - ``divisions``: the lattice of that many divisions, as ``n_partitions``;
    - ``divisions`` and ``inner_divisions``: Deb and Jain's two-layer lattice, the boundary layer of
      ``divisions`` and an inside layer of ``inner_divisions`` (see
      :func:`~vamos.engine.reference_directions.two_layer_lattice`).

    With ``reference_vectors``, ``n_partitions`` is not used. ``to_dict()`` leaves the field out
    while it is ``None``, so a configuration without it keeps the resolved specification, and the
    task id, of VAMOS 1.4.0.

    **VAMOS 1.5.0:** ``variant`` chooses between the two readings of the survival.
    ``"pymoo"``, the default and VAMOS's RVEA so far, follows pymoo: with two objectives the angle
    penalty uses M = 1, the objectives are translated by the ideal point accumulated across
    generations, and the vectors adapt to the nadir point minus that ideal point. ``"platemo"``
    follows the authors' code in PlatEMO (``RVEA/EnvironmentalSelection.m`` and
    ``ReferenceVectorAdaptation.m``) and jMetal's ``RVEAEnvironmentalSelection``: M is the number of
    objectives, two included, so the penalty is ``M (t / t_max)^alpha theta / gamma`` for every
    number of objectives, the objectives are translated by the minimum of the candidates of each
    generation, and the vectors adapt to the range (maximum minus minimum) of the population. Both
    keep VAMOS's progress ``t / t_max`` in generations, its feasibility-first constrained survival
    and its parent selection. ``to_dict()`` leaves out the default.
    """

    pop_size: int
    n_partitions: int
    alpha: float
    adapt_freq: float | None
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
    reference_vectors: dict[str, int | str | None] | None = None
    variant: RVEAVariant = "pymoo"
    #: The techniques that judge the violation of a single solution, the only ones this algorithm takes.
    constraint_handling_techniques: ClassVar[tuple[str, ...]] = PAIRWISE_TECHNIQUES

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "reference_vectors", vector_source(self.reference_vectors, field="reference_vectors", allow_method=True))
        object.__setattr__(self, "variant", normalize_rvea_variant(self.variant))
        if self.track_genealogy:
            _warn_without_effect("RVEA", "track_genealogy", "RVEA records no genealogy", stacklevel=3)

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        # VAMOS 1.5.0: a configuration without reference_vectors is serialized as in VAMOS 1.4.0.
        if payload.get("reference_vectors") is None:
            payload.pop("reference_vectors", None)
        # VAMOS 1.5.0: so is a configuration with the default variant.
        if payload.get("variant") == "pymoo":
            payload.pop("variant", None)
        return payload

    @classmethod
    def default(
        cls,
        pop_size: int | None = None,
        n_var: int | None = None,
        n_obj: int = 3,
        encoding: EncodingLike | None = None,
    ) -> RVEAConfig:
        """Create a default RVEA configuration.

        RVEA needs one individual per reference vector. With ``pop_size``,
        ``n_partitions`` is the Das-Dennis lattice of exactly that size, and a size
        that matches no lattice raises :class:`~vamos.exceptions.ConfigurationError`
        naming the nearest valid ones. Without it, ``n_partitions`` is 99 for two
        objectives, 12 for three and 6 for four or more. Discrete and mixed
        ``encoding`` values take their crossover and mutation from the shared
        encoding table.
        """
        mut_prob = 1.0 / n_var if n_var else 0.1
        if pop_size is None:
            n_partitions = 99 if n_obj == 2 else (12 if n_obj == 3 else 6)
            pop_size = lattice_size(n_obj, n_partitions)
        else:
            n_partitions = _lattice_divisions_within(pop_size, n_obj, algorithm="RVEA")
            if lattice_size(n_obj, n_partitions) != pop_size:
                raise ReferenceDirectionPopulationError(
                    f"RVEA needs one individual per reference vector, and no Das-Dennis lattice for n_obj={n_obj} "
                    f"has {pop_size} vectors. Use pop_size={lattice_size(n_obj, n_partitions)} ({n_partitions} partitions) "
                    f"or pop_size={lattice_size(n_obj, n_partitions + 1)} ({n_partitions + 1} partitions)."
                )
        crossover, mutation = _encoding_operators(
            encoding, mut_prob, real=(("sbx", {"prob": 0.9, "eta": 30.0}), ("pm", {"prob": mut_prob, "eta": 20.0}))
        )
        return (
            cls.builder()
            .pop_size(pop_size)
            .n_partitions(n_partitions)
            .alpha(2.0)
            .adapt_freq(0.1)
            .crossover(crossover[0], **crossover[1])
            .mutation(mutation[0], **mutation[1])
            .build()
        )

    @classmethod
    def builder(cls) -> _RVEAConfigBuilder:
        return _RVEAConfigBuilder()


class _RVEAConfigBuilder(
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
    Declarative configuration holder for RVEA settings.

    Examples:
        cfg = RVEAConfig.default()
        cfg = RVEAConfig.builder().pop_size(100).n_partitions(12).build()
    """

    def n_partitions(self, value: int) -> _RVEAConfigBuilder:
        self._cfg["n_partitions"] = value
        return self

    def reference_vectors(
        self,
        *,
        path: str | None = None,
        divisions: int | None = None,
        inner_divisions: int | None = None,
        method: str | None = None,
        seed: int | None = None,
    ) -> _RVEAConfigBuilder:
        """**VAMOS 1.5.0:** the reference vectors: a file or directory, or a lattice.

        ``path`` names a file with one vector per line, its components separated by whitespace or
        commas, or a directory with jMetal's ``W{n_obj}D_{pop_size}.dat`` files; the file must exist
        and hold ``pop_size`` vectors. Without it, ``divisions`` gives the lattice, and with
        ``inner_divisions`` Deb and Jain's two-layer lattice. ``n_partitions`` is then not needed:
        without it, the configuration records ``divisions``, or 12 for a file.

        **Development after 1.5.2:** ``method="riesz_energy"`` gives the Riesz s-energy set of Blank et al.
        (2021) with one vector per individual, for any population, seeded by ``seed`` (0 by default; see
        :func:`~vamos.engine.reference_directions.riesz_energy`). It excludes the other options, and the
        configuration records 12 for ``n_partitions``, as for a file.
        """
        source = {"path": path, "divisions": divisions, "inner_divisions": inner_divisions, "method": method, "seed": seed}
        self._cfg["reference_vectors"] = vector_source(source, allow_method=True)
        return self

    def variant(self, value: RVEAVariant) -> _RVEAConfigBuilder:
        """**VAMOS 1.5.0:** the reading of RVEA's survival, ``"pymoo"`` (default) or ``"platemo"``.

        ``"platemo"`` follows the authors' PlatEMO code and jMetal: M is the number of objectives in the
        angle penalty also with two objectives, the objectives are translated by the minimum of each
        generation's candidates, and the vectors adapt to the population's range. See :class:`RVEAConfig`.
        """
        self._cfg["variant"] = normalize_rvea_variant(value)
        return self

    def alpha(self, value: float) -> _RVEAConfigBuilder:
        self._cfg["alpha"] = float(value)
        return self

    def adapt_freq(self, value: float | None) -> _RVEAConfigBuilder:
        self._cfg["adapt_freq"] = None if value is None else float(value)
        return self

    def build(self) -> RVEAConfig:
        vectors = self._cfg.get("reference_vectors")
        # With reference vectors, n_partitions is not used: it defaults to their divisions, or to 12 for a file.
        required = (
            ("pop_size", "alpha", "crossover", "mutation") if vectors else ("pop_size", "n_partitions", "alpha", "crossover", "mutation")
        )
        _require_fields(self._cfg, required, "RVEA")
        _validate_operators(self._cfg)
        default_partitions = vectors.get("divisions") if vectors and vectors.get("divisions") is not None else 12
        return RVEAConfig(
            pop_size=self._cfg["pop_size"],
            n_partitions=self._cfg.get("n_partitions", default_partitions),
            alpha=float(self._cfg.get("alpha", 2.0)),
            adapt_freq=self._cfg.get("adapt_freq", 0.1),
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
            reference_vectors=vectors,
            variant=self._cfg.get("variant", "pymoo"),
        )
