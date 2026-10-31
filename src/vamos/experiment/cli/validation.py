from __future__ import annotations

import argparse
import logging

from vamos.engine.archive import ExternalArchiveConfig
from vamos.experiment.runtime.catalog import EXPERIMENT_TYPES
from vamos.foundation.problem.resolver import PROBLEM_SET_PRESETS, resolve_problem_selections, resolve_reference_front_path

from .common import _collect_generic_variation, _collect_moead_variation, _normalize_operator_args, collect_nsgaii_variation_args
from .types import SpecDefaults

# Algorithm-specific options: the destination prefix of each algorithm's variation flags, and the
# options that several algorithms read. Any other selected algorithm ignores them.
_VARIATION_OPTION_PREFIXES: dict[str, str] = {
    "nsgaii": "nsgaii_",
    "moead": "moead_",
    "smsemoa": "smsemoa_",
    "nsgaiii": "nsgaiii_",
    "agemoea": "agemoea_",
    "rvea": "rvea_",
}
_SHARED_ALGORITHM_OPTIONS: dict[str, tuple[str, ...]] = {
    "external_archive_size": ("nsgaii", "spea2", "smpso"),
}
_VARIATION_FLAG_PREFIXES: dict[str, str] = {"nsgaiii": "nsga3"}
_EXAMPLE_METHODS: dict[str, str] = {"crossover": "sbx", "mutation": "polynomial"}


def _algorithm_options(parser: argparse.ArgumentParser) -> dict[str, tuple[str, frozenset[str]]]:
    """Map each algorithm-specific option destination to its flag and the algorithms that read it."""
    options: dict[str, tuple[str, frozenset[str]]] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        shared = _SHARED_ALGORITHM_OPTIONS.get(action.dest)
        if shared is not None:
            algorithms = frozenset(shared)
        else:
            algorithms = frozenset(name for name, prefix in _VARIATION_OPTION_PREFIXES.items() if action.dest.startswith(prefix))
        if algorithms:
            options[action.dest] = (action.option_strings[0], algorithms)
    return options


def _warn_ignored_algorithm_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Warn about algorithm-specific flags given on the command line that the selected algorithm ignores.

    A value equal to the parser default (which includes config-file defaults) is not reported: a config
    may carry blocks for several algorithms.
    """
    algorithm = getattr(args, "algorithm", None)
    if not algorithm or algorithm == "both":
        return
    for dest, (flag, algorithms) in _algorithm_options(parser).items():
        value = getattr(args, dest, None)
        if algorithm in algorithms or value is None or value == parser.get_default(dest):
            continue
        logging.getLogger(__name__).warning(
            "Ignoring %s: only %s use%s it (selected algorithm is '%s').",
            flag,
            ", ".join(sorted(algorithms)),
            "s" if len(algorithms) == 1 else "",
            algorithm,
        )


def _reject_unnamed_operators(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject operator parameters (``--nsgaii-crossover-prob`` ...) given without the operator they configure."""
    algorithm = getattr(args, "algorithm", None)
    for name in _VARIATION_OPTION_PREFIXES:
        if algorithm not in (name, "both"):
            continue
        variation = getattr(args, f"{name}_variation", None) or {}
        flag = _VARIATION_FLAG_PREFIXES.get(name, name)
        for operator in ("crossover", "mutation"):
            block = variation.get(operator)
            if not isinstance(block, dict) or block.get("method"):
                continue
            given = ", ".join(f"--{flag}-{operator}-{key.replace('_', '-')}" for key in block)
            parser.error(
                f"{given} configure{'s' if len(block) == 1 else ''} an operator that is not named; "
                f"pass --{flag}-{operator} too (for example --{flag}-{operator} {_EXAMPLE_METHODS[operator]}), "
                f"or 'method' under {name}.{operator} in the config file."
            )


def _validate_problem_dimensions(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.n_var is not None and args.n_var <= 0:
        parser.error("--n-var must be a positive integer.")
    if args.n_obj is not None and args.n_obj <= 0:
        parser.error("--n-obj must be a positive integer.")
    try:
        resolve_problem_selections(args)
    except ValueError as exc:
        parser.error(str(exc))


def finalize_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    spec_defaults: SpecDefaults,
    config_path: str | None,
) -> argparse.Namespace:
    """Normalize and validate parsed CLI arguments."""
    args.config_path = config_path
    args.config_spec = spec_defaults.spec
    args.problem_overrides = spec_defaults.problem_overrides
    args.experiment_defaults = spec_defaults.experiment_defaults

    _warn_ignored_algorithm_options(parser, args)
    _normalize_operator_args(parser, args)

    if getattr(args, "quiet", False) and getattr(args, "verbose", False):
        parser.error("--quiet and --verbose cannot be used together.")

    if getattr(args, "problem_set", None) is not None:
        if not PROBLEM_SET_PRESETS:
            parser.error("--problem-set is not available because no presets are registered.")
        if args.problem_set not in PROBLEM_SET_PRESETS:
            parser.error(f"--problem-set must be one of: {', '.join(sorted(PROBLEM_SET_PRESETS))}.")
    if getattr(args, "experiment", None) is not None and args.experiment not in EXPERIMENT_TYPES:
        parser.error(f"--experiment must be one of: {', '.join(EXPERIMENT_TYPES)}.")

    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be a non-negative integer.")
    _validate_problem_dimensions(parser, args)
    if args.population_size <= 0:
        parser.error("--population-size must be a positive integer.")
    if args.offspring_population_size is None:
        args.offspring_population_size = args.population_size
    if args.offspring_population_size <= 0:
        parser.error("--offspring-population-size must be positive.")
    if args.selection_pressure <= 0:
        parser.error("--selection-pressure must be a positive integer.")
    if getattr(args, "track_genealogy", False) and getattr(args, "algorithm", None) != "nsgaii":
        parser.error("--track-genealogy is supported only with --algorithm nsgaii.")
    if args.external_archive_size is not None and args.external_archive_size <= 0:
        parser.error("--external-archive-size must be a positive integer.")
    if args.external_archive_size is not None:
        args.external_archive = ExternalArchiveConfig(
            capacity=args.external_archive_size,
            pruning=args.external_archive_pruning,
        )
    else:
        args.external_archive = None
    if args.max_evaluations <= 0:
        parser.error("--max-evaluations must be a positive integer.")
    if args.max_evaluations < args.population_size:
        parser.error("--max-evaluations must be greater than or equal to --population-size.")
    if args.n_workers is not None and args.n_workers <= 0:
        parser.error("--n-workers must be positive if provided.")
    if args.live_viz_interval is not None and args.live_viz_interval <= 0:
        parser.error("--live-viz-interval must be positive.")
    if args.hv_threshold is not None:
        if not (0.0 < args.hv_threshold < 1.0):
            parser.error("--hv-threshold must be in the (0, 1) range.")
        if not args.hv_reference_front:
            problem_key = str(getattr(args, "problem", "")).strip().lower()
            if problem_key.startswith("zdt") or problem_key.startswith("zcat"):
                default_front = resolve_reference_front_path(problem_key, None, n_obj=getattr(args, "n_obj", None))
                if default_front is None:
                    parser.error(
                        "--hv-reference-front is required for the selected problem because no default reference front is available."
                    )
                args.hv_reference_front = str(default_front)
            else:
                parser.error("--hv-reference-front is required for non-ZDT/ZCAT problems when --hv-threshold is set.")
    elif args.hv_reference_front:
        parser.error("--hv-reference-front requires --hv-threshold to be set.")

    args.nsgaii_variation = collect_nsgaii_variation_args(args)
    args.moead_variation = _collect_moead_variation(args)
    args.smsemoa_variation = _collect_generic_variation(args, "smsemoa")
    args.nsgaiii_variation = _collect_generic_variation(args, "nsgaiii")
    args.spea2_variation = spec_defaults.experiment_defaults.get("spea2", {})
    args.ibea_variation = spec_defaults.experiment_defaults.get("ibea", {})
    args.smpso_variation = spec_defaults.experiment_defaults.get("smpso", {})
    args.agemoea_variation = _collect_generic_variation(args, "agemoea")
    args.rvea_variation = _collect_generic_variation(args, "rvea")
    _reject_unnamed_operators(parser, args)

    return args
