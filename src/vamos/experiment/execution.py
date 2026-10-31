from __future__ import annotations

import logging
from argparse import Namespace
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from vamos.engine.algorithm.catalog import ENABLED_ALGORITHMS
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.engine.algorithm.registry import get_algorithms_registry
from vamos.engine.algorithm.variants import PERMUTATION_COMPATIBLE_ALGORITHMS
from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.config.spec import ExperimentSpec, SpecBlock
from vamos.engine.hooks import (
    HookManager,
    HookManagerConfig,
    LiveVisualization,
)
from vamos.engine.hooks.config_parse import parse_stopping_archive
from vamos.experiment._execution_support import (
    CompositeObserver,
    LiveVizAdapter,
    VariationConfigs,
)
from vamos.experiment._execution_support import (
    default_weight_path as _default_weight_path_impl,
)
from vamos.experiment.observers.console import ConsoleObserver
from vamos.experiment.observers.storage import StorageObserver
from vamos.experiment.runner_abstractions import EvaluatorSpec, resolve_evaluator, resolve_termination
from vamos.experiment.runner_utils import run_output_dir, validate_problem
from vamos.experiment.runtime.catalog import (
    EXPERIMENT_BACKENDS,
    EXPERIMENT_TYPES,
    EXTERNAL_ALGORITHM_NAMES,
    resolve_engine,
)
from vamos.foundation.core.execution import execute_algorithm
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.core.hv_stop import compute_hv_reference
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.observer import Observer, RunContext
from vamos.foundation.problem.registry import ProblemSelection
from vamos.foundation.quality_indicators.hypervolume import hypervolume


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


Metrics = dict[str, Any]


def _default_weight_path(problem_name: str, n_obj: int, pop_size: int) -> str:
    return _default_weight_path_impl(problem_name, n_obj, pop_size)


def run_single(
    engine_name: str,
    algorithm_name: str,
    selection: ProblemSelection,
    config: ExperimentConfig,
    *,
    algorithm: Any,
    cfg_data: AlgorithmConfigProtocol,
    problem: Any | None = None,
    external_archive: ExternalArchiveConfig | None = None,
    selection_pressure: int = 2,
    variations: VariationConfigs | None = None,
    hv_stop_config: dict[str, Any] | None = None,
    evaluator: Any | None = None,
    termination: tuple[str, Any] | None = None,
    config_source: str | None = None,
    config_spec: ExperimentSpec | None = None,
    problem_override: SpecBlock | None = None,
    track_genealogy: bool = False,
    live_viz: LiveVisualization | None = None,
) -> Metrics:
    problem = problem or selection.instantiate()

    # 1. Build Context
    ctx = RunContext(
        problem=problem,
        algorithm=algorithm,
        config=config,
        selection=selection,
        algorithm_name=algorithm_name,
        engine_name=engine_name,
    )

    output_dir = run_output_dir(selection, algorithm_name, engine_name, config.seed, config)
    termination_spec = resolve_termination(
        termination,
        config,
        hv_stop_config=hv_stop_config,
        algorithm_name=algorithm_name,
    )
    termination_kind, termination_payload = termination_spec
    hv_enabled = termination_kind == "hv"
    hv_termination = termination_payload if hv_enabled else None
    if hv_enabled and isinstance(hv_termination, dict):
        hv_stop_config = hv_termination

    # 2. Build Observers
    observers: list[Observer] = []

    # Console
    observers.append(ConsoleObserver())

    # Storage
    _variations = variations or VariationConfigs()
    storage = StorageObserver(
        output_dir=output_dir,
        config_source=config_source,
        config_spec=config_spec,
        problem_override=problem_override,
        hv_stop_config=hv_stop_config,
        selection_pressure=selection_pressure,
        external_archive=external_archive,
        variations=_variations.as_storage_dict(),
        termination=termination_spec,
    )
    observers.append(storage)

    # Hooks
    hook_mgr = None
    if config_spec is not None:
        try:
            hook_cfg = parse_stopping_archive(config_spec, problem_key=selection.spec.key)
            if hook_cfg["stopping_enabled"] or hook_cfg["archive_enabled"]:
                hook_mgr = HookManager(
                    cfg=HookManagerConfig(
                        stopping_enabled=hook_cfg["stopping_enabled"],
                        stop_cfg=hook_cfg["stop_cfg"],
                        archive_enabled=hook_cfg["archive_enabled"],
                        archive_cfg=hook_cfg["archive_cfg"],
                        hv_ref_point=hook_cfg["hv_ref_point"],
                    ),
                )
                observers.append(hook_mgr)
        except Exception as exc:
            _logger().warning(
                "Hook setup failed for problem '%s'; hooks disabled. Reason: %s",
                selection.spec.key,
                exc,
            )
            hook_mgr = None

    # User Viz (LiveVisualization uses RunContext)
    if live_viz:
        observers.append(live_viz)

    main_observer = CompositeObserver(observers)

    # 3. Notify Start
    main_observer.on_start(ctx)

    eval_strategy = resolve_evaluator(evaluator, config)

    kernel_backend = getattr(algorithm, "kernel", None)

    if hv_enabled and not isinstance(hv_termination, dict):
        hv_enabled = False
        hv_termination = None

    validate_problem(problem)
    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    if encoding == "permutation":
        permutation_algorithms = PERMUTATION_COMPATIBLE_ALGORITHMS
        if algorithm_name not in permutation_algorithms:
            supported = ", ".join(sorted(permutation_algorithms))
            raise ConfigurationError(
                f"Algorithm '{algorithm_name}' does not support permutation encoding.",
                suggestion=f"Use one of: {supported}.",
                details={"algorithm": algorithm_name, "encoding": "permutation", "supported": sorted(permutation_algorithms)},
            )

    # 4. Execute
    try:
        exec_result = execute_algorithm(
            algorithm,
            problem,
            termination=termination_spec,
            seed=config.seed,
            eval_strategy=eval_strategy,
            live_viz=LiveVizAdapter(main_observer),
        )
    finally:
        # Release what this run created (for example the Dask client opened for --dask-address); a backend
        # instance passed in by the caller stays open.
        if eval_strategy is not evaluator:
            eval_strategy.close()

    payload = exec_result.payload
    total_time_ms = exec_result.elapsed_ms
    F = payload.get("F")
    if F is None:
        raise RuntimeError(
            f"Algorithm '{algorithm_name}' (engine='{engine_name}') returned no objective values. The execution payload is missing key 'F'."
        )
    actual_evaluations = int(payload.get("evaluations", config.max_evaluations))

    # Termination reason determination
    termination_reason = "max_evaluations"
    if main_observer.should_stop():  # Check hooks via composite
        termination_reason = "hv_convergence"
    if hv_enabled and payload.get("hv_reached"):
        termination_reason = "hv_threshold"

    # 5. Build Final Stats / Metrics
    spread = None
    if F.size and F.shape[1] >= 1:
        spread = np.ptp(F[:, 0])
    evals_per_sec = actual_evaluations / max(1e-9, total_time_ms / 1000.0)

    metrics = {
        "algorithm": algorithm_name,
        "engine": engine_name,
        "time_ms": total_time_ms,
        "evaluations": actual_evaluations,
        "evals_per_sec": evals_per_sec,
        "spread": spread,
        "F": F,
        "X": payload.get("X"),
        "termination": termination_reason,
        "eval_strategy": getattr(config, "eval_strategy", "serial"),
        "n_workers": getattr(config, "n_workers", None),
        "config": cfg_data,
        "_kernel_backend": kernel_backend,
        "backend_device": kernel_backend.device() if kernel_backend else "external",
        "backend_capabilities": sorted(set(kernel_backend.capabilities())) if kernel_backend else [],
        "output_dir": output_dir,  # Needed?
        "payload": payload,  # Pass payload for StorageObserver to extract archives etc.
    }

    if hook_mgr is not None:
        metrics["hook_metadata"] = hook_mgr.metadata_payload()

    if hv_enabled and hv_termination:
        metrics["hv_threshold_fraction"] = hv_termination.get("threshold_fraction")
        metrics["hv_reference_point"] = hv_termination.get("reference_point")
        metrics["hv_reference_front"] = hv_termination.get("reference_front_path")

    # 6. Notify End
    main_observer.on_end(F, metrics)

    return metrics  # Return metrics for benchmark aggregation


def _print_summary(results: Iterable[Metrics], hv_ref_point: np.ndarray[Any, Any]) -> None:
    _logger().info("Experiment summary")
    _logger().info("%s", "-" * 80)
    header = f"{'Algo':<12} {'Backend':<10} {'Time (ms)':>12} {'Eval/s':>12} {'HV':>12} {'Spread f1':>12}"
    _logger().info("%s", header)
    _logger().info("%s", "-" * len(header))
    for res in results:
        spread = res["spread"]
        spread_txt = f"{spread:.6f}" if spread is not None else "-"
        hv_txt = f"{res['hv']:.6f}" if res.get("hv") is not None else "-"
        _logger().info(
            "%s",
            f"{res['algorithm']:<12} {res['engine']:<10} {res['time_ms']:>12.2f} "
            f"{res['evals_per_sec']:>12.1f} {hv_txt:>12} {spread_txt:>12}",
        )
    ref_txt = np.array2string(hv_ref_point, precision=3, suppress_small=True)
    _logger().info("Hypervolume reference point: %s", ref_txt)


def execute_problem_suite(
    args: Namespace,
    problem_selection: ProblemSelection,
    config: ExperimentConfig,
    *,
    run_single_fn: Callable[..., Metrics],
    hv_stop_config: dict[str, Any] | None = None,
    variations: VariationConfigs | None = None,
    include_external: bool = False,
    config_source: str | None = None,
    config_spec: ExperimentSpec | None = None,
    problem_override: SpecBlock | None = None,
    track_genealogy: bool = False,
    live_viz_factory: Callable[..., LiveVisualization | None] | None = None,
    plotter: Callable[..., Any] | None = None,
) -> None:
    from vamos.experiment import external  # local import to keep runner decoupled

    engines: Iterable[str | None]
    if args.experiment in EXPERIMENT_TYPES:
        engines = EXPERIMENT_BACKENDS
    else:
        engines = (getattr(args, "engine", None),)
    algorithms = list(ENABLED_ALGORITHMS) if args.algorithm == "both" else [args.algorithm]
    registered_algorithms = set(get_algorithms_registry().keys())
    use_native_external_problem = args.external_problem_source == "native"

    if include_external and problem_selection.spec.key != "zdt1":
        _logger().info("External baselines are currently available only for ZDT1; skipping external runs.")
        include_external = False

    if include_external:
        for ext in EXTERNAL_ALGORITHM_NAMES:
            if ext not in algorithms:
                algorithms.append(ext)

    runnable_algorithms = [a for a in algorithms if a in registered_algorithms]
    external_algorithms = [a for a in algorithms if a in EXTERNAL_ALGORITHM_NAMES]

    _track_genealogy: bool = getattr(args, "track_genealogy", False)
    _variations = variations or VariationConfigs.from_namespace(args)
    # The strategy and worker count come from the (per-problem) config; the Dask scheduler address only from the CLI.
    evaluator = EvaluatorSpec(dask_address=getattr(args, "dask_address", None))

    results: list[Metrics] = []
    for engine in engines:
        for algorithm_name in runnable_algorithms:
            live_viz = None
            engine_name = resolve_engine(engine, algorithm=algorithm_name)
            if live_viz_factory is not None:
                live_viz = live_viz_factory(
                    problem_selection,
                    algorithm_name,
                    engine_name,
                    config,
                )
            try:
                metrics = run_single_fn(
                    engine_name,
                    algorithm_name,
                    problem_selection,
                    config,
                    external_archive=args.external_archive,
                    selection_pressure=args.selection_pressure,
                    variations=_variations,
                    hv_stop_config=hv_stop_config if algorithm_name == "nsgaii" else None,
                    evaluator=evaluator,
                    config_source=config_source,
                    config_spec=config_spec,
                    problem_override=problem_override,
                    track_genealogy=_track_genealogy,
                    live_viz=live_viz,
                )
            except ImportError as exc:
                # Backends experiment should be resilient to missing optional deps.
                if args.experiment in EXPERIMENT_TYPES and f"Kernel '{engine_name}' requires" in str(exc):
                    _logger().warning("Skipping engine '%s': %s", engine_name, exc)
                    continue
                raise
            results.append(metrics)

    for algorithm_name in external_algorithms:
        from vamos.experiment.external.reporting import build_metrics, print_run_banner, print_run_results

        metrics = external.run_external(
            algorithm_name,
            problem_selection,
            use_native_problem=use_native_external_problem,
            config=config,
            make_metrics=build_metrics,
            print_banner=lambda problem, selection, label, backend: print_run_banner(problem, selection, label, backend, config),
            print_results=print_run_results,
        )
        if metrics is not None:
            results.append(metrics)

    if not results:
        _logger().info("No runs were executed. Check algorithm selection or install missing dependencies.")
        return

    fronts = [res["F"] for res in results if res.get("F") is not None]
    hv_ref_point = compute_hv_reference(fronts)
    for res in results:
        backend = res.pop("_kernel_backend", None)
        F_res = res.get("F")
        if F_res is None:
            res["hv"] = None
            res["hv_source"] = "none"
            continue
        if backend and backend.supports_quality_indicator("hypervolume"):
            hv_value = backend.hypervolume(F_res, hv_ref_point)
            res["hv_source"] = backend.__class__.__name__
        else:
            hv_value = hypervolume(F_res, hv_ref_point)
            res["hv_source"] = "global"
        res["hv"] = hv_value

    if len(results) == 1:
        hv_val = results[0]["hv"]
        ref_txt = np.array2string(hv_ref_point, precision=3, suppress_small=True)
        _logger().info("Hypervolume (reference %s): %.6f", ref_txt, hv_val)
    else:
        _print_summary(results, hv_ref_point)

    if plotter is not None:
        plotter(results, problem_selection, output_root=config.output_root, title=config.title)
