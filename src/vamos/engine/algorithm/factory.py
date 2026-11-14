from __future__ import annotations

from collections.abc import Callable, Mapping

from vamos.engine.algorithm.builders import (
    build_agemoea_algorithm,
    build_ccmo_algorithm,
    build_ctaea_algorithm,
    build_ibea_algorithm,
    build_moead_algorithm,
    build_nsgaii_algorithm,
    build_nsgaiii_algorithm,
    build_rvea_algorithm,
    build_smpso_algorithm,
    build_smsemoa_algorithm,
    build_spea2_algorithm,
)
from vamos.engine.algorithm.config import GenericAlgorithmConfig
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.engine.algorithm.registry import resolve_algorithm
from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.config.variation import VariationConfig
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.types import ProblemProtocol

# Built-in algorithms that record offspring genealogy through survival.
GENEALOGY_ALGORITHMS = ("nsgaii", "moead", "smsemoa", "nsgaiii", "spea2", "ibea", "smpso")
_ALGORITHMS_WITHOUT_GENEALOGY = ("agemoea", "rvea", "ccmo", "ctaea")


def _plugin_default_config(
    problem: ProblemProtocol,
    config: ExperimentConfig,
) -> GenericAlgorithmConfig:
    return GenericAlgorithmConfig(
        {
            "pop_size": config.population_size,
            "offspring_size": config.offspring_size(),
            "n_var": problem.n_var,
            "n_obj": problem.n_obj,
            "encoding": normalize_encoding(getattr(problem, "encoding", "real")),
        }
    )


def build_algorithm(
    algorithm_name: str,
    engine_name: str,
    problem: ProblemProtocol,
    config: ExperimentConfig,
    *,
    algorithm_config: AlgorithmConfigProtocol | None = None,
    external_archive: ExternalArchiveConfig | None = None,
    selection_pressure: int = 2,
    variations: Mapping[str, VariationConfig | None] | None = None,
    track_genealogy: bool = False,
) -> tuple[object, AlgorithmConfigProtocol]:
    """
    Factory to build the algorithm instance.
    """
    kernel = resolve_kernel(engine_name)
    algorithm_key = algorithm_name.lower()
    if track_genealogy and algorithm_key in _ALGORITHMS_WITHOUT_GENEALOGY:
        raise ConfigurationError(
            f"track_genealogy is not supported by {algorithm_name}: it does not record genealogy.",
            suggestion=f"Use one of {', '.join(GENEALOGY_ALGORITHMS)}, or drop track_genealogy.",
        )
    variations = variations or {}
    pop_size = config.population_size
    offspring_size = config.offspring_size()

    if algorithm_config is not None:
        algo_ctor = resolve_algorithm(algorithm_key)
        return algo_ctor(dict(algorithm_config.to_dict()), kernel), algorithm_config

    builders: dict[str, Callable[[], tuple[object, AlgorithmConfigProtocol]]] = {
        "nsgaii": lambda: build_nsgaii_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            offspring_size=offspring_size,
            selection_pressure=selection_pressure,
            external_archive=external_archive,
            nsgaii_variation=variations.get("nsgaii"),
            track_genealogy=track_genealogy,
        ),
        "moead": lambda: build_moead_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            moead_variation=variations.get("moead"),
            track_genealogy=track_genealogy,
        ),
        "smsemoa": lambda: build_smsemoa_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            smsemoa_variation=variations.get("smsemoa"),
            track_genealogy=track_genealogy,
        ),
        "nsgaiii": lambda: build_nsgaiii_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            nsgaiii_variation=variations.get("nsgaiii"),
            track_genealogy=track_genealogy,
            selection_pressure=selection_pressure,
        ),
        "spea2": lambda: build_spea2_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            selection_pressure=selection_pressure,
            external_archive=external_archive,
            spea2_variation=variations.get("spea2"),
            track_genealogy=track_genealogy,
        ),
        "ibea": lambda: build_ibea_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            selection_pressure=selection_pressure,
            ibea_variation=variations.get("ibea"),
            track_genealogy=track_genealogy,
        ),
        "smpso": lambda: build_smpso_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            external_archive=external_archive,
            smpso_variation=variations.get("smpso"),
            track_genealogy=track_genealogy,
        ),
        "agemoea": lambda: build_agemoea_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            agemoea_variation=variations.get("agemoea"),
        ),
        "rvea": lambda: build_rvea_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            rvea_variation=variations.get("rvea"),
        ),
        "ccmo": lambda: build_ccmo_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            ccmo_variation=variations.get("ccmo"),
        ),
        "ctaea": lambda: build_ctaea_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            ctaea_variation=variations.get("ctaea"),
        ),
    }

    builder = builders.get(algorithm_key)
    if builder is not None:
        return builder()

    plugin_config = _plugin_default_config(problem, config)
    algo_ctor = resolve_algorithm(algorithm_key)
    return algo_ctor(dict(plugin_config.to_dict()), kernel), plugin_config
