from __future__ import annotations

import argparse

from vamos.engine.algorithm.catalog import (
    DEFAULT_ALGORITHM,
    ENABLED_ALGORITHMS,
    OPTIONAL_ALGORITHMS,
)
from vamos.engine.algorithm.registry import get_algorithms_registry
from vamos.experiment.runtime.catalog import DEFAULT_PROBLEM, EXPERIMENT_BACKENDS, EXTERNAL_ALGORITHM_NAMES
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import available_problem_names

from .defaults import spec_default
from .spec_args import add_spec_argument
from .types import SpecDefaults

_CLI_ENGINE_CHOICES = (*EXPERIMENT_BACKENDS, "auto")


def add_core_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_config: ExperimentConfig,
    spec_defaults: SpecDefaults,
) -> None:
    """Register core experiment arguments on the parser."""
    experiment_defaults = spec_defaults.experiment_defaults
    algorithm_choices = tuple(
        dict.fromkeys(
            (
                *sorted(get_algorithms_registry().keys()),
                *ENABLED_ALGORITHMS,
                *OPTIONAL_ALGORITHMS,
                *EXTERNAL_ALGORITHM_NAMES,
                "both",
            )
        )
    )

    add_spec_argument(
        parser,
        "--algorithm",
        choices=algorithm_choices,
        default=spec_default(experiment_defaults, "algorithm", DEFAULT_ALGORITHM),
        help=(
            "Algorithm to run (use 'both' to execute the default internal algorithms sequentially; "
            "combine with --include-external to add third-party baselines)."
        ),
    )
    add_spec_argument(
        parser,
        "--engine",
        choices=_CLI_ENGINE_CHOICES,
        default=spec_default(experiment_defaults, "engine", None),
        help=(
            "Kernel backend to use. Default is deterministic numpy. Use 'auto' to "
            "prefer available Numba for built-in algorithms, with a NumPy fallback."
        ),
    )
    add_spec_argument(
        parser,
        "--output-root",
        default=spec_default(experiment_defaults, "output_root", default_config.output_root),
        help="Directory where run artifacts are stored (default: VAMOS_OUTPUT_ROOT or 'results').",
    )
    add_spec_argument(
        parser,
        "--eval-strategy",
        choices=("serial", "multiprocessing", "dask"),
        default=spec_default(experiment_defaults, "eval_strategy", "serial"),
        help="Evaluation strategy to use (default: serial). Multiprocessing uses --n-workers; dask uses --dask-address.",
    )
    add_spec_argument(
        parser,
        "--n-workers",
        type=int,
        default=experiment_defaults.get("n_workers"),
        help="Number of workers for multiprocessing evaluation backend (ignored for serial).",
    )
    add_spec_argument(
        parser,
        "--dask-address",
        default=experiment_defaults.get("dask_address"),
        help="Address of Dask scheduler (e.g. 'tcp://localhost:8786'). Only used with --eval-strategy dask.",
    )
    add_spec_argument(
        parser,
        "--problem",
        choices=available_problem_names(),
        default=spec_default(experiment_defaults, "problem", DEFAULT_PROBLEM),
        help="Benchmark problem to solve (see available_problem_names() or vamos zoo list).",
    )
    add_spec_argument(
        parser,
        "--n-var",
        type=int,
        default=experiment_defaults.get("n_var"),
        help="Override the number of decision variables for the selected problem.",
    )
    add_spec_argument(
        parser,
        "--n-obj",
        type=int,
        default=experiment_defaults.get("n_obj"),
        help="Override the number of objectives (if the problem supports it).",
    )
    add_spec_argument(
        parser,
        "--include-external",
        action="store_true",
        default=bool(spec_default(experiment_defaults, "include_external", False)),
        help="Include PyMOO/jMetalPy/PyGMO baselines when running algorithms (requires research extra).",
    )
    add_spec_argument(
        parser,
        "--external-problem-source",
        choices=("native", "vamos"),
        default=spec_default(experiment_defaults, "external_problem_source", "native"),
        help=(
            "For external baselines, choose whether to use each library's native benchmark "
            "implementation ('native') or wrap the VAMOS problem definition ('vamos')."
        ),
    )
    add_spec_argument(
        parser,
        "--max-evaluations",
        type=int,
        default=spec_default(experiment_defaults, "max_evaluations", default_config.max_evaluations),
        help="Maximum number of evaluations per run (default: %(default)s).",
    )
    add_spec_argument(
        parser,
        "--population-size",
        type=int,
        default=spec_default(experiment_defaults, "population_size", default_config.population_size),
        help="Population size for all internal algorithms (default: %(default)s).",
    )
    add_spec_argument(
        parser,
        "--offspring-population-size",
        type=int,
        default=experiment_defaults.get("offspring_population_size"),
        help=("Number of offspring generated per NSGA-II generation. Defaults to --population-size."),
    )
    add_spec_argument(
        parser,
        "--selection-pressure",
        type=int,
        default=spec_default(experiment_defaults, "selection_pressure", 2),
        help=("Tournament size for selection operators (default: %(default)s). Applies to NSGA-II/III and SMS-EMOA."),
    )
    add_spec_argument(
        parser,
        "--seed",
        type=int,
        default=spec_default(experiment_defaults, "seed", default_config.seed),
        help="Random seed used by internal and external runs (default: %(default)s).",
    )
    add_spec_argument(
        parser,
        "--verbose",
        action="store_true",
        default=bool(experiment_defaults.get("verbose", False)),
        help="Enable verbose logging (DEBUG).",
    )
    add_spec_argument(
        parser,
        "--quiet",
        action="store_true",
        default=bool(experiment_defaults.get("quiet", False)),
        help="Suppress informational logging (only WARNING+).",
    )
    add_spec_argument(
        parser,
        "--no-preflight",
        action="store_true",
        default=bool(spec_default(experiment_defaults, "no_preflight", False)),
        help="Skip optional dependency warnings (useful for scripted runs).",
    )
