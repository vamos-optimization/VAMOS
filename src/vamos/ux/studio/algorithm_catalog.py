"""Algorithms, configuration spaces and population rules behind the Studio algorithm forms."""

from __future__ import annotations

from vamos.engine.algorithm.catalog import ENABLED_ALGORITHMS
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.engine.tuning.racing import bridge_spaces
from vamos.engine.tuning.racing.config_space import AlgorithmConfigSpace
from vamos.ux.studio.services import _build_algorithm_config

ALGORITHM_LABELS: dict[str, str] = {
    "nsgaii": "NSGA-II",
    "moead": "MOEA/D",
    "smsemoa": "SMS-EMOA",
    "nsgaiii": "NSGA-III",
    "spea2": "SPEA2",
    "ibea": "IBEA",
    "smpso": "SMPSO",
    "agemoea": "AGE-MOEA",
    "rvea": "RVEA",
}

_SPACE_SUFFIX = {
    "real": "",
    "continuous": "",
    "binary": "_binary",
    "integer": "_integer",
    "permutation": "_permutation",
    "mixed": "_mixed",
}
_ARCHIVE_PARAMETERS = {"use_external_archive", "archive_unbounded", "archive_prune_policy"}
_CROSSOVER_PREFIXES = ("blx", "wa_", "laplace_", "fuzzy_", "pcx_", "undx_", "simplex_", "de_")
_MUTATION_PREFIXES = ("nonuniform_", "gaussian_", "cauchy_", "uniform_", "levy_", "power_", "creep_")
# Algorithms whose population size must equal their number of reference directions.
_REFERENCE_DIRECTION_ALGORITHMS = {"nsgaiii", "rvea"}


def studio_algorithms() -> tuple[str, ...]:
    return tuple(ENABLED_ALGORITHMS)


def algorithm_label(algorithm: str) -> str:
    return ALGORITHM_LABELS.get(algorithm, algorithm.upper())


def config_space_for(algorithm: str, encoding: str | None) -> AlgorithmConfigSpace | None:
    """Return the tuning space of ``algorithm`` for ``encoding``, if VAMOS defines one."""
    suffix = _SPACE_SUFFIX.get(str(encoding or "real").lower())
    if suffix is None:
        return None
    builder = getattr(bridge_spaces, f"build_{algorithm.lower()}{suffix}_config_space", None)
    if builder is None:
        return None
    space = builder()
    return space if isinstance(space, AlgorithmConfigSpace) else None


def algorithms_for_encoding(encoding: str | None) -> tuple[str, ...]:
    return tuple(name for name in studio_algorithms() if config_space_for(name, encoding) is not None)


def population_is_derived(algorithm: str, n_obj: int | None) -> bool:
    """Whether the population size follows from reference directions or weight vectors.

    These algorithms preserve their population/reference-direction cardinality,
    so Studio lets VAMOS choose the population instead of passing its own size.
    """
    if algorithm in _REFERENCE_DIRECTION_ALGORITHMS:
        return True
    return algorithm == "moead" and (n_obj or 2) >= 3


def default_algorithm_config(
    algorithm: str,
    *,
    encoding: str | None,
    n_var: int | None,
    n_obj: int | None,
    pop_size: int | None,
) -> AlgorithmConfigProtocol:
    """Build the VAMOS default configuration that Studio runs for ``algorithm``."""
    effective_pop = None if population_is_derived(algorithm, n_obj) else pop_size
    return _build_algorithm_config(algorithm, pop_size=effective_pop, n_var=n_var, n_obj=n_obj, encoding=encoding)


def config_fields_for(parameter: str) -> tuple[str, ...]:
    """Return the configuration fields that a space parameter determines."""
    if parameter in {"repair", "blx_repair"}:
        return ("repair",)
    if parameter in _ARCHIVE_PARAMETERS:
        return ("external_archive",)
    if parameter == "mutation_prob_factor":
        return ("mutation_prob_factor",)
    if parameter == "offspring_ratio":
        return ("offspring_size",)
    if parameter == "n_partitions":
        return ("n_partitions", "pop_size")
    if parameter == "initializer" or parameter.startswith("scatter_"):
        return ("initializer",)
    if parameter in {"aggregation", "pbi_theta"}:
        return ("aggregation",)
    if parameter.startswith("selection"):
        return ("selection",)
    if "crossover" in parameter or parameter.startswith(_CROSSOVER_PREFIXES):
        return ("crossover",)
    if "mutation" in parameter or parameter.startswith(_MUTATION_PREFIXES):
        return ("mutation",)
    return (parameter,)


__all__ = [
    "ALGORITHM_LABELS",
    "algorithm_label",
    "algorithms_for_encoding",
    "config_fields_for",
    "config_space_for",
    "default_algorithm_config",
    "population_is_derived",
    "studio_algorithms",
]
