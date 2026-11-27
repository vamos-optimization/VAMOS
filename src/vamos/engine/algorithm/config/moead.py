"""MOEA/D configuration."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, Literal, TypeAlias, overload

from vamos.engine.algorithm.components.weight_vectors import vector_source
from vamos.engine.algorithm.moead.subproblems import normalize_subproblem_order
from vamos.engine.algorithm.moead.update_criteria import normalize_subproblem_update_criterion
from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.operators.policies.moead import check_de_neighbourhood, normalize_de_base, normalize_de_scheme
from vamos.foundation.encoding import EncodingLike
from vamos.foundation.exceptions import ConfigurationError

from .base import (
    ConstraintModeStr,
    LiveCallbackMode,
    ResultMode,
    _ConfigBuilderState,
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
    _SerializableConfig,
    _TrackGenealogyBuilder,
    _validate_operators,
)
from .types import AggregationName, RepairConfigValue

MOEADVariant: TypeAlias = Literal["de", "sbx"]
SubproblemOrder: TypeAlias = Literal["random_permutation", "cyclic"]
_DE_NAMES = ("de", "differential", "differential_evolution")

# The real-coded crossover and the aggregation of each default: VAMOS's own (no variant), and jMetal's
# MOEADDEBuilder (MOEA/D-DE) and MOEADBuilder, which are Evolver's MOEADDEDefault and MOEADDoubleDefault.
_VARIANTS: dict[str | None, tuple[tuple[str, dict[str, Any]], tuple[str, dict[str, Any]]]] = {
    None: (("de", {"cr": 1.0, "f": 0.5}), ("pbi", {"theta": 5.0})),
    "de": (("de", {"cr": 1.0, "f": 0.5}), ("tchebycheff", {})),
    "sbx": (("sbx", {"prob": 0.9, "eta": 20.0}), ("pbi", {"theta": 5.0})),
}


@dataclass(frozen=True)
class MOEADConfig(_SerializableConfig):
    """Configuration of MOEA/D.

    MOEA/D gives each individual one weight vector, so ``pop_size`` must be a size that has
    weights. With one or two objectives every size has them: uniform weights, as in jMetal. With
    three or more, the weights are the file of ``weight_vectors(path=...)`` when there is one (a
    directory holds ``W{n_obj}D_{pop_size}.dat``); otherwise the complete Das-Dennis lattice with
    exactly ``pop_size`` vectors; otherwise the jMetal file ``W{n_obj}D_{pop_size}.dat`` that VAMOS
    ships (100 individuals with three objectives). Any other size raises ``ValueError`` when the run
    starts: a lattice is never truncated, and the configuration never changes ``pop_size``. The
    tuning bridge does round a sampled population down to the largest size that has weights, so
    150 becomes 136 with three objectives. The resolved spec of a run records the source in
    ``algorithm.resolution.weight_vectors``.

    **VAMOS 1.4.0:** the options of Evolver's MOEA/D, all off by default:

    - ``subproblem_order``: ``"random_permutation"`` (the default) or ``"cyclic"``, the order in
      which the subproblems are visited (Evolver's ``subProblemIdGenerator``,
      ``randomPermutationCycle`` or ``cyclicIntegerSequence``);
    - the ``scheme`` of the ``de`` crossover: ``"rand_1_bin"`` (the default), ``"rand_1_exp"``,
      ``"rand_2_bin"`` or ``"rand_2_exp"`` (jMetal's ``DifferentialEvolutionCrossover`` variants),
      and its ``base``: ``"current"`` (the default), the subproblem's solution, or ``"random"``, one
      more random parent, as Evolver's MOEA/D; the random parents (2 for rand/1, 4 for rand/2, one
      more with a random base) must fit in ``neighbor_size``;
    - the aggregations ``augmented_tchebycheff`` (``rho``, default ``1e-4``), ``inverted_pbi``
      (``theta``, default ``0.1``) and ``inverse_weight_tchebycheff`` (jMetal's and Evolver's
      modified Tchebycheff), and ``normalize=True`` with ``epsilon`` (default ``1e-6``) for any
      aggregation, which normalizes the objectives with the ideal and nadir points as jMetal does.

    See :mod:`vamos.engine.algorithm.moead.aggregation` and :mod:`vamos.engine.operators.policies.moead`
    for the definitions.

    **VAMOS 1.5.0:** ``weight_vectors`` takes ``inner_divisions`` with ``divisions``, Deb and
    Jain's two-layer lattice, which gives usual populations with many objectives (156 vectors with eight
    objectives, 3 and 2 divisions), and NSGA-III's ``reference_directions`` and RVEA's ``reference_vectors``
    take the same keys. :func:`vamos.algorithms.moead_populations` lists the populations that have weights
    without options.

    **Development after 1.5.2:** ``weight_vectors(method="riesz_energy", seed=...)`` gives any population its
    own weights, the Riesz s-energy set of ``pop_size`` vectors (see
    :func:`~vamos.engine.reference_directions.riesz_energy`).
    """

    pop_size: int
    batch_size: int
    neighbor_size: int
    delta: float
    replace_limit: int
    crossover: tuple[str, dict[str, Any]]
    mutation: tuple[str, dict[str, Any]]
    aggregation: tuple[str, dict[str, Any]]
    weight_vectors: dict[str, int | str | None] | None
    constraint_mode: ConstraintModeStr = "feasibility"
    repair: RepairConfigValue = "auto"
    initializer: dict[str, Any] | None = None
    mutation_prob_factor: float | None = None
    use_numba_variation: bool | None = None
    track_genealogy: bool = False
    result_mode: ResultMode | None = None
    external_archive: ExternalArchiveConfig | None = None
    # Fields added after 1.0 go last, so that the constructor keeps the positions of its parameters.
    subproblem_update_criterion: tuple[str, dict[str, float]] | None = None
    live_callback_mode: LiveCallbackMode = "population"
    subproblem_order: SubproblemOrder = "random_permutation"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "weight_vectors", vector_source(self.weight_vectors, field="weight_vectors", allow_method=True))
        criterion = normalize_subproblem_update_criterion(self.subproblem_update_criterion)
        if criterion is not None and criterion[0] != "feasibility_rules" and self.constraint_mode == "none":
            raise ValueError(f"subproblem_update_criterion {criterion[0]!r} needs constraint_mode 'feasibility'.")
        object.__setattr__(self, "subproblem_update_criterion", criterion)
        object.__setattr__(self, "subproblem_order", normalize_subproblem_order(self.subproblem_order))
        object.__setattr__(self, "crossover", _check_de_scheme(self.crossover, self.neighbor_size))

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        # The default order is serialized as in VAMOS 1.3.0, without the field, so a configuration that does not
        # set it keeps the resolved specification, and the task id, that VAMOS 1.3.0 gives it.
        if payload.get("subproblem_order") == "random_permutation":
            del payload["subproblem_order"]
        return payload

    @classmethod
    def default(
        cls,
        pop_size: int | None = None,
        n_var: int | None = None,
        n_obj: int = 3,
        encoding: EncodingLike | None = None,
        *,
        variant: MOEADVariant | None = None,
    ) -> MOEADConfig:
        """Create a default MOEA/D configuration: VAMOS's, or one of jMetal's two defaults.

        Without ``variant``, VAMOS's default: DE rand/1/bin with ``cr=1.0`` and
        ``f=0.5``, and PBI aggregation with ``theta=5``, which is neither of
        jMetal's. ``"de"`` is MOEA/D-DE (Li and Zhang, 2009): the same DE with
        Tchebycheff aggregation; ``"sbx"`` is SBX with ``prob=0.9`` and ``eta=20``,
        and PBI with ``theta=5``. They are jMetal's ``MOEADDEBuilder`` and
        ``MOEADBuilder``, and Evolver's ``MOEADDEDefault`` and
        ``MOEADDoubleDefault``. All three use polynomial mutation with probability
        ``1/n_var`` and ``eta=20``, a neighbourhood of 20, a probability of 0.9
        of mating in it and at most 2 replaced solutions. VAMOS 2.0 will make
        ``"de"`` the default.

        Without ``pop_size``, the population is the weight-vector lattice: 99
        divisions with two objectives, 12 with three and 6 with more. With
        ``pop_size`` and three or more objectives, the weights are the lattice
        with exactly ``pop_size`` vectors or, without one, the packaged
        ``W{n_obj}D_{pop_size}.dat`` file (as jMetal uses). Discrete and mixed
        ``encoding`` values take their crossover and mutation from the shared
        encoding table, and the variant's aggregation.
        """
        if variant not in _VARIANTS:
            names = ", ".join(repr(name) for name in _VARIANTS if name is not None)
            raise ConfigurationError(f"MOEA/D variant must be one of {names}, or None for VAMOS's default; got {variant!r}.")
        real_crossover, aggregation = _VARIANTS[variant]
        divisions: int | None
        if pop_size is None:
            divisions = 99 if n_obj == 2 else (12 if n_obj == 3 else 6)
            pop_size = divisions + 1 if n_obj == 2 else comb(divisions + n_obj - 1, n_obj - 1)
        elif n_obj == 2:
            divisions = max(1, int(pop_size) - 1)
        elif n_obj > 2:
            divisions = _exact_lattice_divisions(n_obj, int(pop_size))
        else:
            # One objective: every weight vector is (1,), whatever the divisions.
            divisions = 6
        mut_prob = 1.0 / n_var if n_var else 0.1
        crossover, mutation = _encoding_operators(encoding, mut_prob, real=(real_crossover, ("pm", {"prob": mut_prob, "eta": 20.0})))
        return (
            cls.builder()
            .pop_size(pop_size)
            .batch_size(1)
            .neighbor_size(min(20, pop_size))
            .delta(0.9)
            .replace_limit(2)
            .crossover(crossover[0], **crossover[1])
            .mutation(mutation[0], **mutation[1])
            .aggregation(aggregation[0], **aggregation[1])
            .weight_vectors(divisions=divisions)
            .build()
        )

    @classmethod
    def builder(cls) -> _MOEADConfigBuilder:
        return _MOEADConfigBuilder()


class _MOEADConfigBuilder(
    _ConfigBuilderState,
    _PopSizeBuilder,
    _CrossoverBuilder,
    _MutationBuilder,
    _RepairBuilder,
    _InitializerBuilder,
    _MutationProbFactorBuilder,
    _ConstraintModeBuilder,
    _TrackGenealogyBuilder,
    _ResultArchiveBuilder,
):
    """
    Declarative configuration holder for MOEA/D settings.

    Examples:
        # Fluent builder
        cfg = MOEADConfig.builder().pop_size(100).neighbor_size(20).build()

        # Quick default configuration
        cfg = MOEADConfig.default()
    """

    def subproblem_update_criterion(self, name: str, **params: float) -> _MOEADConfigBuilder:
        """How a new solution replaces the solution of a subproblem under constraints.

        ``name`` is ``feasibility_rules`` (the default), ``improved_epsilon`` (``tao``,
        ``feasibility_ratio_threshold``, ``decay_fraction``) or ``violation_threshold``, the criteria of
        jMetal's package ``subproblemupdate``. See :mod:`vamos.engine.algorithm.moead.update_criteria`.
        """
        self._cfg["subproblem_update_criterion"] = normalize_subproblem_update_criterion((name, params))
        return self

    def subproblem_order(self, value: SubproblemOrder) -> _MOEADConfigBuilder:
        """**VAMOS 1.4.0:** the order of the subproblems, ``"random_permutation"`` (default) or ``"cyclic"``.

        ``"random_permutation"`` visits a new random permutation of the subproblems in each cycle;
        ``"cyclic"`` visits them in index order and draws no random number for it (jMetal's
        ``RandomPermutationCycle`` and ``CyclicIntegerSequence``).
        """
        self._cfg["subproblem_order"] = normalize_subproblem_order(value)
        return self

    def batch_size(self, value: int) -> _MOEADConfigBuilder:
        self._cfg["batch_size"] = value
        return self

    def neighbor_size(self, value: int) -> _MOEADConfigBuilder:
        self._cfg["neighbor_size"] = value
        return self

    def delta(self, value: float) -> _MOEADConfigBuilder:
        self._cfg["delta"] = value
        return self

    def replace_limit(self, value: int) -> _MOEADConfigBuilder:
        self._cfg["replace_limit"] = value
        return self

    @overload
    def aggregation(self, method: AggregationName, **kwargs: Any) -> _MOEADConfigBuilder: ...

    @overload
    def aggregation(self, method: str, **kwargs: Any) -> _MOEADConfigBuilder: ...

    def aggregation(self, method: str, **kwargs: Any) -> _MOEADConfigBuilder:
        """The aggregation function and its parameters.

        ``tchebycheff``, ``weighted_sum``, ``pbi`` (``theta``, default 5), ``modified_tchebycheff``
        (``rho``, default ``1e-3``) and, **VAMOS 1.4.0**, ``augmented_tchebycheff``
        (``rho``, default ``1e-4``), ``inverted_pbi`` (``theta``, default 0.1) and
        ``inverse_weight_tchebycheff`` (jMetal's ``ModifiedTschebyscheff``, ``max |f - z*| / w``). Any of them takes
        ``normalize=True`` and ``epsilon`` (default ``1e-6``): each objective becomes
        ``(f - z*) / (z^nad - z* + epsilon)`` with the ideal point ``z*`` and the nadir point
        ``z^nad`` of the non-dominated solutions evaluated so far, as in jMetal.
        """
        self._cfg["aggregation"] = (method, kwargs)
        return self

    def weight_vectors(
        self,
        *,
        path: str | None = None,
        divisions: int | None = None,
        inner_divisions: int | None = None,
        method: str | None = None,
        seed: int | None = None,
    ) -> _MOEADConfigBuilder:
        """The weight vectors: a file or directory, a lattice, or by default the lattice or packaged file of ``pop_size``.

        ``path`` names a file with one vector per line, its components separated by whitespace or commas,
        or a directory with jMetal's ``W{n_obj}D_{pop_size}.dat`` files; a directory without the file gets
        the lattice of exactly ``pop_size`` vectors, saved there with a ``RuntimeWarning``. ``divisions``
        gives the complete Das-Dennis lattice. MOEA/D needs exactly one vector per individual.

        **VAMOS 1.5.0:** ``inner_divisions`` with ``divisions`` gives Deb and Jain's two-layer
        lattice, the boundary lattice of ``divisions`` and the lattice of ``inner_divisions`` shrunk by half
        towards the centre of the simplex, for any number of objectives; a ``path`` that does not exist
        raises :class:`~vamos.exceptions.ConfigurationError` when the run starts, where VAMOS 1.4.0 created
        the directory.

        **Development after 1.5.2:** ``method="riesz_energy"`` gives the Riesz s-energy set of Blank et al.
        (2021) with exactly ``pop_size`` vectors, for any population and any number of objectives, such as
        100 with three objectives; ``seed`` (0 by default) seeds its random start, so the same seed gives the
        same vectors (:func:`~vamos.engine.reference_directions.riesz_energy`). It excludes the other options.
        """
        source = {"path": path, "divisions": divisions, "inner_divisions": inner_divisions, "method": method, "seed": seed}
        self._cfg["weight_vectors"] = vector_source(source, allow_method=True)
        return self

    def use_numba_variation(self, enabled: bool = True) -> _MOEADConfigBuilder:
        self._cfg["use_numba_variation"] = bool(enabled)
        return self

    def build(self) -> MOEADConfig:
        _require_fields(
            self._cfg,
            (
                "pop_size",
                "neighbor_size",
                "delta",
                "replace_limit",
                "crossover",
                "mutation",
                "aggregation",
            ),
            "MOEA/D",
        )
        _validate_operators(self._cfg)
        _check_neighbourhood(self._cfg["neighbor_size"], self._cfg["pop_size"])
        return MOEADConfig(
            pop_size=self._cfg["pop_size"],
            batch_size=int(self._cfg.get("batch_size", 1)),
            neighbor_size=self._cfg["neighbor_size"],
            delta=self._cfg["delta"],
            replace_limit=self._cfg["replace_limit"],
            crossover=self._cfg["crossover"],
            mutation=self._cfg["mutation"],
            aggregation=self._cfg["aggregation"],
            weight_vectors=self._cfg.get("weight_vectors"),
            constraint_mode=self._cfg.get("constraint_mode", "feasibility"),
            subproblem_update_criterion=self._cfg.get("subproblem_update_criterion"),
            repair=self._cfg.get("repair", "auto"),
            initializer=self._cfg.get("initializer"),
            mutation_prob_factor=self._cfg.get("mutation_prob_factor"),
            use_numba_variation=self._cfg.get("use_numba_variation"),
            track_genealogy=bool(self._cfg.get("track_genealogy", False)),
            result_mode=self._cfg.get("result_mode", "non_dominated"),
            external_archive=self._cfg.get("external_archive"),
            live_callback_mode=self._cfg.get("live_callback_mode", "population"),
            subproblem_order=self._cfg.get("subproblem_order", "random_permutation"),
        )


def _exact_lattice_divisions(n_obj: int, pop_size: int) -> int | None:
    """Divisions of the complete Das-Dennis lattice with exactly ``pop_size`` vectors, or ``None``."""
    if n_obj < 2:
        return None
    divisions = 1
    while (size := comb(divisions + n_obj - 1, n_obj - 1)) < pop_size:
        divisions += 1
    return divisions if size == pop_size else None


def _check_de_scheme(crossover: object, neighbor_size: object) -> object:
    """The crossover with its DE ``scheme`` and ``base`` in lower case.

    Both belong to the DE crossover, and its random parents must fit in the neighbourhood: 2 for
    rand/1, 4 for rand/2, one more with ``base="random"``.
    """
    if not (isinstance(crossover, tuple) and len(crossover) == 2 and isinstance(crossover[1], dict)):
        return crossover
    name, params = crossover
    options = [key for key in ("scheme", "base") if key in params]
    if not options:
        return crossover
    if str(name).strip().lower() not in _DE_NAMES:
        raise ConfigurationError(f"The {options[0]} applies to the DE crossover ('de'), not to crossover {name!r}.")
    scheme = normalize_de_scheme(params.get("scheme", "rand_1_bin"))
    base = normalize_de_base(params.get("base", "current"))
    if isinstance(neighbor_size, int):
        check_de_neighbourhood(scheme, base, neighbor_size)
    normalized = {**params}
    if "scheme" in params:
        normalized["scheme"] = scheme
    if "base" in params:
        normalized["base"] = base
    return (name, normalized)


def _check_neighbourhood(neighbor_size: object, pop_size: int) -> None:
    """Reject a neighbourhood larger than the population; MOEA/D would shrink it to the population silently."""
    if isinstance(neighbor_size, int) and not isinstance(neighbor_size, bool) and neighbor_size > pop_size:
        raise ConfigurationError(f"neighbor_size must not exceed pop_size ({pop_size}); got {neighbor_size}.")
