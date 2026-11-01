from __future__ import annotations

import logging
import secrets
from argparse import Namespace
from collections.abc import Callable, Mapping
from copy import deepcopy
from types import SimpleNamespace
from typing import TypeVar, cast

from vamos.engine.algorithm.catalog import ENABLED_ALGORITHMS
from vamos.engine.algorithm.config.base import normalize_constraint_mode
from vamos.engine.algorithm.config.validation import validate_config_fields
from vamos.engine.archive import ExternalArchiveConfig, normalize_prune_policy
from vamos.engine.config.spec import ExperimentSpec, ProblemOverrides, SpecBlock
from vamos.engine.config.variation import VariationOverrides, merge_variation_overrides
from vamos.engine.hooks import LiveVisualization
from vamos.experiment._execution_support import VariationConfigs
from vamos.experiment.execution import build_hook_manager, execute_problem_suite, raise_run_failures
from vamos.experiment.services.orchestrator import run_single
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.core.hv_stop import build_hv_stop_config
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.problem.registry import ProblemSelection, make_problem_selection
from vamos.foundation.problem.resolver import resolve_problem_selections

# Algorithms whose tournament reads the selection pressure ("both" runs them too).
_TOURNAMENT_ALGORITHMS = ("nsgaii", "nsgaiii", "spea2", "ibea", "both")
# Algorithms with CLI variation options, and their flag spelling when it differs from the name.
_VARIATION_ALGORITHMS = ("nsgaii", "moead", "smsemoa", "nsgaiii", "agemoea", "rvea")
_VARIATION_FLAGS = {"nsgaiii": "nsga3"}
_EXAMPLE_METHODS = {"crossover": "sbx", "mutation": "polynomial"}


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


T = TypeVar("T")


def _override_value(override: SpecBlock, key: str, default: T) -> T:
    return cast(T, override.get(key, default))


def _override_mapping(override: SpecBlock, key: str) -> VariationOverrides | None:
    value = override.get(key)
    return cast(VariationOverrides, value) if isinstance(value, dict) else None


def _effective_variation(args: Namespace, override: SpecBlock, name: str) -> Mapping[str, object]:
    """The variation options ``name`` runs with for one problem: CLI and config defaults, then its override."""
    return merge_variation_overrides(getattr(args, f"{name}_variation", None), _override_mapping(override, name))


def _check_unnamed_operators(args: Namespace, override: SpecBlock, algorithm: object, problem_key: str) -> None:
    """Reject operator parameters (``--nsgaii-crossover-prob`` ...) that still lack their operator for this problem."""
    for name in _VARIATION_ALGORITHMS:
        if algorithm not in (name, "both"):
            continue
        variation = _effective_variation(args, override, name)
        flag = _VARIATION_FLAGS.get(name, name)
        for operator in ("crossover", "mutation"):
            block = variation.get(operator)
            # Operator specs name the operator under "method" or "name".
            if not isinstance(block, dict) or block.get("method") or block.get("name"):
                continue
            given = ", ".join(f"--{flag}-{operator}-{key.replace('_', '-')}" for key in block)
            raise ConfigurationError(
                f"{given} configure{'s' if len(block) == 1 else ''} an operator that is not named (problem '{problem_key}').",
                suggestion=(
                    f"Pass --{flag}-{operator} too (for example --{flag}-{operator} {_EXAMPLE_METHODS[operator]}), "
                    f"or 'method' under {name}.{operator} in the config file."
                ),
            )


# Operator fields of an algorithm block, which the CLI builders check when they apply them.
_OPERATOR_FIELDS = ("crossover", "mutation", "selection", "repair", "aggregation", "initializer")


def _check_algorithm_fields(
    args: Namespace, override: SpecBlock, algorithm: object, problem_key: str, population: int, offspring: object
) -> None:
    """Reject the out-of-domain fields of the algorithm blocks a problem runs with.

    Building the configuration checks them again, with the fields that depend on each other.
    """
    for name in ENABLED_ALGORITHMS if algorithm == "both" else (str(algorithm),):
        block = _effective_variation(args, override, name)
        values = {key: value for key, value in block.items() if key not in _OPERATOR_FIELDS and value is not None}
        try:
            validate_config_fields(SimpleNamespace(**values), values)
            if "constraint_mode" in values:
                normalize_constraint_mode(values["constraint_mode"])
        except ValueError as exc:
            raise ConfigurationError(f"{name} block of problem '{problem_key}': {exc}") from exc
        if name == "nsgaii" and values.get("steady_state") and offspring != 1:
            raise ConfigurationError(
                f"nsgaii.steady_state needs offspring_population_size: 1 for problem '{problem_key}'; it is {offspring or population}.",
                suggestion="Set offspring_population_size: 1 in defaults or the problem block, or drop steady_state.",
            )
        neighbourhood = values.get("neighbor_size")
        if name == "moead" and isinstance(neighbourhood, int) and neighbourhood > population:
            raise ConfigurationError(
                f"moead.neighbor_size ({neighbourhood}) cannot exceed population_size ({population}) for problem '{problem_key}'.",
                suggestion="Lower neighbor_size or raise the population size.",
            )


def _check_effective_settings(
    args: Namespace, config: ExperimentConfig, selections: list[ProblemSelection], overrides: ProblemOverrides
) -> None:
    """Reject, before any run starts, settings a selected problem cannot run with.

    Each problem is checked with its effective values, after its ``problems.<name>`` overrides,
    and with the stopping and archive hooks its runs build.
    """
    config_spec: ExperimentSpec | None = getattr(args, "config_spec", None)
    for selection in selections:
        key = selection.spec.key
        raw = overrides.get(key)
        override: SpecBlock = raw if isinstance(raw, dict) else {}
        algorithm = override.get("algorithm", args.algorithm)
        pressure = int(cast(int, override.get("selection_pressure", args.selection_pressure)))
        population = int(cast(int, override.get("population_size", config.population_size)))
        # An explicit selection operator sets its own tournament size; the fallback pressure is then unused.
        selector = "nsgaii" if algorithm == "both" else str(algorithm)
        explicit_selection = isinstance(_effective_variation(args, override, selector).get("selection"), dict)
        if algorithm in _TOURNAMENT_ALGORITHMS and not explicit_selection and pressure > population:
            raise ConfigurationError(
                f"selection_pressure ({pressure}) cannot exceed population_size ({population}) for problem '{key}'.",
                suggestion="Lower --selection-pressure (selection_pressure in a config) or raise the population size.",
            )
        if override.get("track_genealogy", getattr(args, "track_genealogy", False)) and algorithm != "nsgaii":
            raise ConfigurationError(
                f"track_genealogy is supported only with the nsgaii algorithm; problem '{key}' runs '{algorithm}'.",
                suggestion="Drop --track-genealogy (track_genealogy in a config) or select nsgaii for that problem.",
            )
        _check_unnamed_operators(args, override, algorithm, key)
        offspring = override.get("offspring_population_size", config.offspring_population_size)
        _check_algorithm_fields(args, override, algorithm, key, population, offspring)
        build_hook_manager(config_spec, key)


def check_run_settings(args: Namespace, config: ExperimentConfig) -> None:
    """Run the checks that precede the first run of ``vamos`` without running anything.

    ``--validate-config`` calls it after parsing, so a config that passes would also start.

    Parameters
    ----------
    args : Namespace
        Parsed and finalized command-line arguments.
    config : ExperimentConfig
        Runtime configuration built from ``args``.

    Raises
    ------
    ConfigurationError
        When a selected problem cannot run with its effective settings or hooks.
    """
    overrides: ProblemOverrides = getattr(args, "problem_overrides", {}) or {}
    _check_effective_settings(args, config, list(resolve_problem_selections(args)), overrides)


def run_from_args(
    args: Namespace,
    config: ExperimentConfig,
    *,
    live_viz_factory: Callable[..., LiveVisualization | None] | None = None,
    plotter: Callable[..., object] | None = None,
) -> None:
    selections = list(resolve_problem_selections(args))
    multiple = len(selections) > 1
    base_variation = getattr(args, "nsgaii_variation", None)
    overrides: ProblemOverrides = getattr(args, "problem_overrides", {}) or {}
    config_source = getattr(args, "config_path", None)
    config_spec: ExperimentSpec | None = getattr(args, "config_spec", None)
    _check_effective_settings(args, config, selections, overrides)
    # With several problems, a failed run is reported after the whole series instead of stopping it.
    failures: list[str] | None = [] if multiple else None

    for idx, selection in enumerate(selections, start=1):
        override_raw = overrides.get(selection.spec.key)
        override: SpecBlock = override_raw if isinstance(override_raw, dict) else {}
        effective_selection = selection
        if override.get("n_var") is not None or override.get("n_obj") is not None:
            effective_selection = make_problem_selection(
                selection.spec.key,
                n_var=_override_value(override, "n_var", cast(int | None, selection.n_var)),
                n_obj=_override_value(override, "n_obj", cast(int | None, selection.n_obj)),
            )
        requested_seed = _override_value(override, "seed", config.seed)
        effective_config = ExperimentConfig(
            title=_override_value(override, "title", config.title),
            output_root=_override_value(override, "output_root", config.output_root),
            population_size=_override_value(override, "population_size", config.population_size),
            offspring_population_size=_override_value(override, "offspring_population_size", config.offspring_population_size),
            max_evaluations=_override_value(override, "max_evaluations", config.max_evaluations),
            seed=secrets.randbits(64) if requested_seed is None else requested_seed,
            eval_strategy=_override_value(override, "eval_strategy", getattr(config, "eval_strategy", "serial")),
            n_workers=_override_value(override, "n_workers", getattr(config, "n_workers", None)),
            live_viz=_override_value(override, "live_viz", getattr(config, "live_viz", False)),
            live_viz_interval=_override_value(override, "live_viz_interval", getattr(config, "live_viz_interval", 5)),
            live_viz_max_points=_override_value(override, "live_viz_max_points", getattr(config, "live_viz_max_points", 1000)),
        )
        effective_args = deepcopy(args)
        # What the user asked for, before main resolves a null seed: recorded as the run's request.
        effective_args.cli_request = {
            "seed": override["seed"] if "seed" in override else getattr(args, "seed", None),
            "n_var": override.get("n_var", getattr(args, "n_var", None)),
            "n_obj": override.get("n_obj", getattr(args, "n_obj", None)),
        }
        for key in (
            "algorithm",
            "engine",
            "experiment",
            "include_external",
            "external_problem_source",
        ):
            if override.get(key) is not None:
                setattr(effective_args, key, override[key])
        effective_args.selection_pressure = override.get("selection_pressure", args.selection_pressure)
        _ea_override = override.get("external_archive_size")
        _ea_pruning = normalize_prune_policy(
            str(override.get("external_archive_pruning", getattr(args, "external_archive_pruning", "crowding")))
        )
        base_external_archive = getattr(args, "external_archive", None)
        if _ea_override is not None:
            effective_args.external_archive = ExternalArchiveConfig(
                capacity=int(str(_ea_override)),
                pruning=_ea_pruning,
            )
        elif "external_archive_pruning" in override and base_external_archive is not None:
            effective_args.external_archive = ExternalArchiveConfig(
                capacity=base_external_archive.capacity,
                pruning=_ea_pruning,
            )
        else:
            effective_args.external_archive = base_external_archive
        effective_args.hv_threshold = _override_value(override, "hv_threshold", cast(float | None, args.hv_threshold))
        effective_args.hv_reference_front = _override_value(override, "hv_reference_front", cast(str | None, args.hv_reference_front))
        effective_args.n_var = override.get("n_var", args.n_var)
        effective_args.n_obj = override.get("n_obj", args.n_obj)
        effective_args.eval_strategy = override.get("eval_strategy", args.eval_strategy)
        effective_args.dask_address = override.get("dask_address", getattr(args, "dask_address", None))
        effective_args.n_workers = override.get("n_workers", args.n_workers)
        effective_args.live_viz = override.get("live_viz", args.live_viz)
        effective_args.live_viz_interval = override.get("live_viz_interval", args.live_viz_interval)
        effective_args.live_viz_max_points = override.get("live_viz_max_points", args.live_viz_max_points)
        effective_args.track_genealogy = _override_value(override, "track_genealogy", cast(bool, getattr(args, "track_genealogy", False)))
        effective_args.nsgaii_variation = merge_variation_overrides(base_variation, _override_mapping(override, "nsgaii"))
        effective_args.moead_variation = merge_variation_overrides(
            getattr(args, "moead_variation", None), _override_mapping(override, "moead")
        )
        effective_args.smsemoa_variation = merge_variation_overrides(
            getattr(args, "smsemoa_variation", None), _override_mapping(override, "smsemoa")
        )
        effective_args.nsgaiii_variation = merge_variation_overrides(
            getattr(args, "nsgaiii_variation", None), _override_mapping(override, "nsgaiii")
        )
        effective_args.spea2_variation = merge_variation_overrides(
            getattr(args, "spea2_variation", None), _override_mapping(override, "spea2")
        )
        effective_args.ibea_variation = merge_variation_overrides(
            getattr(args, "ibea_variation", None), _override_mapping(override, "ibea")
        )
        effective_args.smpso_variation = merge_variation_overrides(
            getattr(args, "smpso_variation", None), _override_mapping(override, "smpso")
        )
        effective_args.agemoea_variation = merge_variation_overrides(
            getattr(args, "agemoea_variation", None), _override_mapping(override, "agemoea")
        )
        effective_args.rvea_variation = merge_variation_overrides(
            getattr(args, "rvea_variation", None), _override_mapping(override, "rvea")
        )
        effective_args.effective_problem_override = override

        if multiple:
            _logger().info("%s", "\n" + "#" * 80)
            _logger().info(
                "Problem %s/%s: %s (%s)",
                idx,
                len(selections),
                effective_selection.spec.label,
                effective_selection.spec.key,
            )
            _logger().info("%s", "#" * 80 + "\n")

        hv_stop_config = None
        if effective_args.hv_threshold is not None:
            if effective_args.algorithm != "nsgaii":
                # Only NSGA-II stops on the hypervolume target; keep the option accepted but never silent.
                _logger().warning(
                    "--hv-threshold/hv_threshold applies only to NSGA-II; %s on %s runs its full evaluation budget.",
                    "the other algorithms" if effective_args.algorithm == "both" else effective_args.algorithm,
                    effective_selection.spec.key,
                )
            hv_stop_config = build_hv_stop_config(
                effective_args.hv_threshold,
                effective_args.hv_reference_front,
                effective_selection.spec.key,
                n_obj=effective_selection.n_obj,
            )
        variations = VariationConfigs.from_namespace(effective_args)
        execute_problem_suite(
            effective_args,
            effective_selection,
            effective_config,
            run_single_fn=run_single,
            hv_stop_config=hv_stop_config,
            variations=variations,
            include_external=effective_args.include_external,
            config_source=config_source,
            config_spec=config_spec,
            problem_override=override,
            track_genealogy=effective_args.track_genealogy,
            live_viz_factory=live_viz_factory,
            plotter=plotter,
            failures=failures,
        )
    if failures:
        raise_run_failures(failures)


__all__ = ["check_run_settings", "run_from_args"]
