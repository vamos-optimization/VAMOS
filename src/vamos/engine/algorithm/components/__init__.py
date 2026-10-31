# algorithm/components/__init__.py
"""
Shared algorithm components.

This package contains building blocks used across multiple algorithms:
- archive: External archive implementations (crowding, hypervolume)
- base: Base algorithm infrastructure (state, setup, result building)
- population: Population initialization and evaluation
- selection: Parent selection strategies
- termination: Termination criteria and trackers
- weight_vectors: Weight vector generation for decomposition
- variation: Variation operators and pipelines (subpackage)
- protocol: Algorithm interface definitions and enums
- utils: Shared utility functions
"""

from vamos.engine.algorithm.components.archive import (
    CrowdingDistanceArchive,
    HypervolumeArchive,
    MaxMinArchive,
    ReferenceDirectionsArchive,
    SPEA2Archive,
)
from vamos.engine.algorithm.components.hooks import (
    get_live_viz,
    notify_generation,
)
from vamos.engine.algorithm.components.lifecycle import (
    get_eval_strategy,
    setup_initial_population,
)
from vamos.engine.algorithm.components.metrics import setup_hv_tracker
from vamos.engine.algorithm.components.population import (
    initialize_population,
    resolve_bounds,
)
from vamos.engine.algorithm.components.protocol import (
    AlgorithmProtocol,
    ConstraintMode,
    InteractiveAlgorithmProtocol,
    SelectionMethod,
    SurvivalMethod,
)
from vamos.engine.algorithm.components.results import build_result
from vamos.engine.algorithm.components.selection import RandomSelection, TournamentSelection
from vamos.engine.algorithm.components.state import AlgorithmState
from vamos.engine.algorithm.components.subset_selection import (
    _hv_contributions,
    _single_front_crowding,
    select_top_k_angle,
    select_top_k_crowding,
    select_top_k_farthest,
    select_top_k_hv_greedy,
    select_top_k_kmeans,
    select_top_k_knn,
    select_top_k_reference_directions,
)
from vamos.engine.algorithm.components.termination import HVTracker, parse_termination
from vamos.engine.algorithm.components.utils import (
    compute_ideal_nadir,
    normalize_objectives,
    parse_operator_config,
    resolve_bounds_array,
    validate_termination,
)

__all__ = [
    # protocol
    "AlgorithmProtocol",
    "InteractiveAlgorithmProtocol",
    "SelectionMethod",
    "SurvivalMethod",
    "ConstraintMode",
    # base
    "AlgorithmState",
    "parse_termination",
    "setup_initial_population",
    "setup_hv_tracker",
    "get_live_viz",
    "notify_generation",
    "get_eval_strategy",
    "build_result",
    # archive
    "CrowdingDistanceArchive",
    "HypervolumeArchive",
    "MaxMinArchive",
    "ReferenceDirectionsArchive",
    "SPEA2Archive",
    # subset selection
    "_hv_contributions",
    "_single_front_crowding",
    "select_top_k_angle",
    "select_top_k_crowding",
    "select_top_k_farthest",
    "select_top_k_hv_greedy",
    "select_top_k_kmeans",
    "select_top_k_knn",
    "select_top_k_reference_directions",
    # population
    "initialize_population",
    "resolve_bounds",
    # selection
    "TournamentSelection",
    "RandomSelection",
    # termination
    "HVTracker",
    # utils
    "resolve_bounds_array",
    "validate_termination",
    "parse_operator_config",
    "compute_ideal_nadir",
    "normalize_objectives",
]
