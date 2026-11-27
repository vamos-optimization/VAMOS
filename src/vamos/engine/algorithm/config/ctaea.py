"""C-TAEA configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

from vamos.engine.algorithm.components.weight_vectors import vector_source
from vamos.engine.archive import ExternalArchiveConfig
from vamos.foundation.encoding import EncodingLike

from .base import (
    DEFAULT_POP_SIZE,
    ConstraintModeStr,
    LiveCallbackMode,
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


def ctaea_weight_source(value: object) -> dict[str, int | str | None] | None:
    """C-TAEA's ``weight_vectors`` as :class:`CTAEAConfig` holds them: a checked source, or ``None`` for its default vectors.

    :func:`~vamos.engine.algorithm.components.weight_vectors.vector_source` checks the value and makes a
    path ``{"path": path, "divisions": None}``. A source with neither ``path`` nor ``divisions``, such as
    ``{}``, names no vectors, so it is ``None``: C-TAEA generates its default vectors and ``to_dict()``
    leaves the field out. Study planning normalizes a ``weight_vectors`` override with it, so that a task
    holds the configuration its execution rebuilds. Development after 1.5.2: new.
    """
    source = vector_source(value, field="weight_vectors")
    if source is not None and source.get("path") is None and source.get("divisions") is None:
        return None
    return source


@dataclass(frozen=True)
class CTAEAConfig(_SerializableConfig):
    """Configuration of C-TAEA (Li et al., 2019).

    ``pop_size`` is the requested number of weight vectors, at least 1. The number of vectors
    generated from it, N, is the size of each of the two archives (91 for 100 and three
    objectives), so a run's initial population and generations have N individuals. The crossover
    must mate parents two by two; only its first child is kept.

    **Development after 1.5.2:** ``weight_vectors`` gives the weight vectors instead, with the
    keys of MOEA/D's (see :class:`MOEADConfig`): ``path``, a file with one vector per line or a
    directory with jMetal's ``W{n_obj}D_{pop_size}.dat`` files, such as those VAMOS ships;
    ``divisions``, the complete Das-Dennis lattice; ``divisions`` and ``inner_divisions``, the
    two-layer lattice; and ``sha256``, the digest a file must hold. There must be exactly
    ``pop_size`` vectors, one per individual, so ``pop_size=100`` with jMetal's ``W3D_100.dat``
    runs 100 individuals on three objectives; another number raises ``ConfigurationError`` when
    the run starts, before any evaluation. As with the default vectors, every component is raised
    to at least ``1e-6`` (see :func:`~vamos.engine.algorithm.ctaea.weights.configured_weight_vectors`).
    Without the option, the vectors, the results and ``to_dict()`` are those of earlier versions.
    """

    pop_size: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    repair: RepairConfigValue = "auto"
    initializer: dict[str, Any] | None = None
    constraint_mode: ConstraintModeStr = "feasibility"
    result_mode: ResultMode | None = None
    external_archive: ExternalArchiveConfig | None = None
    live_callback_mode: LiveCallbackMode = "population"
    weight_vectors: dict[str, int | str | None] | None = None

    # C-TAEA reads a weight file with any number of objectives, where MOEA/D's and PPS's weights are uniform with two:
    # a run binds the file it reads to its digest with two objectives too (see bind_config_vector_files).
    _uniform_two_objective_weights = False

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "weight_vectors", ctaea_weight_source(self.weight_vectors))

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        # Without weight_vectors, a configuration serializes as before the field, so it keeps its resolved
        # specification and its study task id.
        if payload.get("weight_vectors") is None:
            payload.pop("weight_vectors", None)
        return payload

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
        # 100 individuals with three objectives, one per vector of jMetal's W3D_100.dat (vamos.resources.weight_path):
        cfg = (
            CTAEAConfig.builder()
            .pop_size(100)
            .crossover("sbx", prob=1.0, eta=20.0)
            .mutation("pm", prob="1/n", eta=20.0)
            .weight_vectors(path=str(weight_path("W3D_100.dat")))
            .build()
        )
    """

    def weight_vectors(self, *, path: str | None = None, divisions: int | None = None, inner_divisions: int | None = None) -> Self:
        """The weight vectors, one per individual: a file or directory, or a lattice (see :class:`CTAEAConfig`).

        By default C-TAEA generates its own vectors from ``pop_size`` (91 for 100 and three objectives). With this
        option it runs exactly ``pop_size`` individuals, one per vector: ``path`` names a file with one vector per line,
        its components separated by whitespace or commas, or a directory with jMetal's ``W{n_obj}D_{pop_size}.dat``;
        ``divisions`` gives the complete Das-Dennis lattice, and with ``inner_divisions`` the two-layer lattice. The
        ``sha256`` of a file's vectors, which a run records, is a key of the field's mapping. Development after 1.5.2: new.
        """
        self._cfg["weight_vectors"] = vector_source({"path": path, "divisions": divisions, "inner_divisions": inner_divisions})
        return self

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
            live_callback_mode=self._cfg.get("live_callback_mode", "population"),
            weight_vectors=self._cfg.get("weight_vectors"),
        )


__all__ = ["CTAEAConfig", "ctaea_weight_source"]
