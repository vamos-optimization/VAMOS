from __future__ import annotations

import argparse

from vamos.foundation.core.experiment_config import ExperimentConfig

from .args_algorithms import add_algorithm_arguments
from .args_benchmarks import add_benchmark_arguments
from .args_core import add_core_arguments
from .args_outputs import add_output_arguments
from .args_tuning import add_tuning_arguments
from .spec_args import SpecArgumentParser
from .types import SpecDefaults


def build_pre_parser() -> argparse.ArgumentParser:
    """Create a pre-parser for config-file discovery."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        help="Path to a YAML/JSON experiment specification. CLI arguments override file values; use --validate-config to check and exit.",
    )
    pre_parser.add_argument(
        "--validate-config",
        action="store_true",
        help=(
            "Validate --config with the checks that precede a run and exit without running (status 0 when valid, 2 with the first error)."
        ),
    )
    return pre_parser


def build_parser(
    *,
    default_config: ExperimentConfig,
    pre_parser: argparse.ArgumentParser,
    spec_defaults: SpecDefaults,
) -> SpecArgumentParser:
    """Construct the full CLI parser with defaults applied."""
    parser = SpecArgumentParser(
        description="Vectorized multi-objective optimization demo across benchmark problems.",
        parents=[pre_parser],
    )
    add_core_arguments(
        parser,
        default_config=default_config,
        spec_defaults=spec_defaults,
    )
    add_benchmark_arguments(
        parser,
        spec_defaults=spec_defaults,
    )
    add_output_arguments(
        parser,
        spec_defaults=spec_defaults,
    )
    add_algorithm_arguments(
        parser,
        spec_defaults=spec_defaults,
    )
    add_tuning_arguments(
        parser,
        spec_defaults=spec_defaults,
    )
    return parser
