from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vamos.engine.tuning import Instance, TrialResult, TuningTask, available_model_based_backends
from vamos.engine.tuning.racing.eval_types import EvalFn

from ._tune_args import build_parser as _build_parser_impl
from ._tune_args import parse_args as _parse_args_impl
from ._tune_flow import (
    run_test_stage as _run_test_stage,
)
from ._tune_flow import (
    run_tuning_with_finisher as _run_tuning_with_finisher,
)
from ._tune_flow import (
    run_validation_stage as _run_validation_stage,
)
from ._tune_flow import (
    write_split_artifacts as _write_split_artifacts,
)
from ._tune_post import (
    append_summary as _append_summary,
)
from ._tune_post import (
    persist_artifacts as _persist_artifacts_impl,
)
from ._tune_post import (
    run_statistical_finisher as _run_statistical_finisher_impl,
)
from ._tune_runtime import (
    ALL_BACKENDS,
    BUILDERS,
    CLI_EXCLUDED_TUNING_PARAMS,
    CLI_TUNING_CONTRACT_REVISION,
    MODEL_BACKENDS,
    build_cli_param_space,
    make_evaluator,
)
from ._tune_runtime import (
    build_task as _build_task,
)
from ._tune_runtime import (
    run_backend as _run_backend,
)
from ._tune_runtime import (
    supports_warm_start as _supports_warm_start,
)
from ._tune_utils import (
    parse_csv_ints as _parse_csv_ints,
)
from ._tune_utils import (
    parse_csv_strings as _parse_csv_strings,
)
from ._tune_utils import (
    parse_ref_point as _parse_ref_point,
)
from ._tune_utils import (
    resolve_n_jobs as _resolve_n_jobs,
)
from ._tune_utils import (
    resolve_split_seeds as _resolve_split_seeds,
)
from ._tune_utils import (
    split_instances as _split_instances,
)

_SMOKE_BUDGET = 30
_SMOKE_TUNE_BUDGET = 4
_SMOKE_N_SEEDS = 2
_SMOKE_POP_SIZE = 16
_SMOKE_N_JOBS = 1
_CLI_EXCLUDED_TUNING_PARAM_SET = frozenset(CLI_EXCLUDED_TUNING_PARAMS)


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _configure_cli_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(level)


def _build_parser() -> argparse.ArgumentParser:
    return _build_parser_impl(builders=BUILDERS, all_backends=ALL_BACKENDS)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    return _parse_args_impl(
        argv,
        builders=BUILDERS,
        all_backends=ALL_BACKENDS,
        parse_csv_ints=_parse_csv_ints,
    )


def _clamp_positive(value: int, ceiling: int) -> int:
    return max(1, min(int(value), int(ceiling)))


def _apply_smoke_mode(args: argparse.Namespace) -> dict[str, Any]:
    args.budget = _clamp_positive(int(args.budget), _SMOKE_BUDGET)
    args.tune_budget = _clamp_positive(int(args.tune_budget), _SMOKE_TUNE_BUDGET)
    args.n_seeds = _clamp_positive(int(args.n_seeds), _SMOKE_N_SEEDS)
    args.n_jobs = _SMOKE_N_JOBS
    args.pop_size = _clamp_positive(int(args.pop_size), _SMOKE_POP_SIZE)
    args.initial_configs = _clamp_positive(int(args.initial_configs), int(args.tune_budget))
    args.validation_topk = _clamp_positive(int(args.validation_topk), int(args.tune_budget))
    args.finisher_topk = _clamp_positive(int(args.finisher_topk), int(args.tune_budget))
    args.run_validation = False
    args.run_test = False
    args.run_statistical_finisher = False
    args.multi_fidelity = False
    args.show_progress_bar = False
    if str(args.backend) in MODEL_BACKENDS and str(args.backend_fallback) == "error":
        args.backend_fallback = "random"
    return {
        "budget": int(args.budget),
        "tune_budget": int(args.tune_budget),
        "n_seeds": int(args.n_seeds),
        "n_jobs": int(args.n_jobs),
        "pop_size": int(args.pop_size),
        "initial_configs": int(args.initial_configs),
        "run_validation": bool(args.run_validation),
        "run_test": bool(args.run_test),
        "run_statistical_finisher": bool(args.run_statistical_finisher),
        "multi_fidelity": bool(args.multi_fidelity),
        "backend_fallback": str(args.backend_fallback),
    }


def _resolve_output_dir(base: Path, run_name: str, *, problem: str, algorithm: str, backend: str, seed: int) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = run_name.strip() or f"{problem}_{algorithm}_{backend}_seed{seed}_{ts}"
    out = base.expanduser().resolve() / suffix
    out.mkdir(parents=True, exist_ok=True)
    return out


def _run_statistical_finisher(
    *,
    candidates: list[dict[str, Any]],
    eval_fn: EvalFn,
    instances: list[Instance],
    seeds: list[int],
    budget: int,
    aggregator: Callable[[list[float]], float],
    alpha: float,
    min_blocks: int,
    failure_score: float,
    use_friedman: bool,
) -> dict[str, Any] | None:
    return _run_statistical_finisher_impl(
        candidates=candidates,
        eval_fn=eval_fn,
        instances=instances,
        seeds=seeds,
        budget=budget,
        aggregator=aggregator,
        alpha=alpha,
        min_blocks=min_blocks,
        failure_score=failure_score,
        use_friedman=use_friedman,
    )


def _persist_artifacts(
    output_dir: Path,
    args: argparse.Namespace,
    task: TuningTask,
    best_config: dict[str, Any],
    history: list[TrialResult],
    elapsed_seconds: float,
    resolved_jobs: int,
) -> None:
    _persist_artifacts_impl(
        output_dir=output_dir,
        args=args,
        task=task,
        best_config=best_config,
        history=history,
        elapsed_seconds=elapsed_seconds,
        resolved_jobs=resolved_jobs,
        available_backends_fn=available_model_based_backends,
    )


def _print_backend_table() -> None:
    flags = available_model_based_backends()
    print("Backend availability:")
    print("  racing       : True")
    print("  random       : True")
    for name in MODEL_BACKENDS:
        print(f"  {name:12s}: {bool(flags.get(name, False))}")


def _reject_legacy_cli_history(best_config: dict[str, Any], history: list[TrialResult]) -> None:
    configs = [best_config, *(trial.config for trial in history)]
    retired = sorted({name for config in configs for name in config if name in _CLI_EXCLUDED_TUNING_PARAM_SET})
    if retired:
        names = ", ".join(retired)
        raise RuntimeError(
            "Loaded tuning history uses archive parameters retired from the maintained CLI scoring space "
            f"({names}). Start a fresh persisted tuning study/storage or choose a new --optuna-study-name."
        )


def main(argv: Sequence[str] | None = None) -> int:
    _configure_cli_logging()
    args = _parse_args(argv)
    if bool(args.list_backends):
        _print_backend_table()
        return 0

    logger = _logger()
    smoke_profile: dict[str, Any] | None = None
    if bool(args.smoke):
        smoke_profile = _apply_smoke_mode(args)
        logger.info("Smoke mode active: %s", smoke_profile)

    requested_backend = str(args.backend)
    effective_backend = requested_backend
    availability = available_model_based_backends()
    if requested_backend in MODEL_BACKENDS and not bool(availability.get(requested_backend, False)):
        fallback = str(args.backend_fallback)
        if fallback == "error":
            raise RuntimeError(
                f"Requested backend '{requested_backend}' is not available. "
                f"Install optional dependencies or use --backend-fallback racing/random."
            )
        effective_backend = fallback
        logger.warning(
            "Backend '%s' unavailable; falling back to '%s'.",
            requested_backend,
            effective_backend,
        )
        args.backend = effective_backend

    resolved_jobs = _resolve_n_jobs(int(args.n_jobs))
    param_space = build_cli_param_space(str(args.algorithm))

    problem_names = list(_parse_csv_strings(args.instances)) or [str(args.problem)]
    all_instances = [Instance(name=name, n_var=int(args.n_var), kwargs={}) for name in problem_names]
    train_instances, validation_instances, test_instances, split_manifest = _split_instances(
        all_instances,
        train_frac=float(args.train_frac),
        validation_frac=float(args.validation_frac),
        split_seed=int(args.split_seed),
        strategy=str(args.split_strategy),
    )
    try:
        train_seeds, validation_seeds, test_seeds = _resolve_split_seeds(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    warm_start_enabled = bool(args.multi_fidelity) and bool(args.fidelity_warm_start)
    if warm_start_enabled and not _supports_warm_start(str(args.algorithm)):
        _logger().warning("Warm-start is not supported for %s; disabling it.", args.algorithm)
        warm_start_enabled = False
    if warm_start_enabled and str(args.backend) in MODEL_BACKENDS:
        _logger().warning("Warm-start is not supported for model-based backends; disabling it.")
        warm_start_enabled = False
    if warm_start_enabled and str(args.backend) == "random":
        _logger().warning("Warm-start is not supported for random backend; disabling it.")
        warm_start_enabled = False

    budget_per_run = int(args.budget)
    if bool(args.multi_fidelity) and args.fidelity_levels:
        budget_per_run = int(max(args.fidelity_levels))
    task = _build_task(
        args,
        param_space,
        budget_per_run=budget_per_run,
        instances=train_instances,
        seeds=train_seeds,
    )
    eval_fn = make_evaluator(
        problem_key=str(args.problem),
        n_var=int(args.n_var),
        n_obj=int(args.n_obj),
        algorithm_name=str(args.algorithm),
        fixed_pop_size=int(args.pop_size),
        ref_point_str=args.ref_point,
        warm_start=warm_start_enabled,
        runtime_penalty=float(args.runtime_penalty),
        failure_score=float(args.failure_score),
        logger=_logger,
    )

    out_dir = _resolve_output_dir(
        args.output_dir,
        str(args.name),
        problem=str(args.problem),
        algorithm=str(args.algorithm),
        backend=str(args.backend),
        seed=int(args.seed),
    )
    logger.info("Output: %s", out_dir)
    logger.info("Backend: requested=%s effective=%s", requested_backend, effective_backend)
    logger.info("Jobs: requested=%s resolved=%s", args.n_jobs, resolved_jobs)
    logger.info("Tune budget: %s", args.tune_budget)
    logger.info(
        "Split sizes (instances): train=%s validation=%s test=%s",
        len(train_instances),
        len(validation_instances),
        len(test_instances),
    )
    logger.info(
        "Split sizes (seeds): train=%s validation=%s test=%s",
        len(train_seeds),
        len(validation_seeds),
        len(test_seeds),
    )

    split_csv_path, split_seed_path = _write_split_artifacts(
        out_dir,
        split_manifest,
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
    )
    best_config, history, elapsed, finisher_summary_updates, finisher_artifacts = _run_tuning_with_finisher(
        args=args,
        task=task,
        eval_fn=eval_fn,
        resolved_jobs=resolved_jobs,
        train_instances=train_instances,
        train_seeds=train_seeds,
        out_dir=out_dir,
        run_backend_fn=_run_backend,
        logger=logger,
    )
    _reject_legacy_cli_history(best_config, history)

    logger.info("--- Tuning complete ---")
    logger.info("Best configuration:")
    for k, v in best_config.items():
        logger.info("  %s: %s", k, v)

    _persist_artifacts(
        output_dir=out_dir,
        args=args,
        task=task,
        best_config=best_config,
        history=history,
        elapsed_seconds=elapsed,
        resolved_jobs=resolved_jobs,
    )
    if finisher_summary_updates or finisher_artifacts:
        _append_summary(
            out_dir,
            summary_updates=finisher_summary_updates,
            artifact_updates=finisher_artifacts,
        )
    summary_updates: dict[str, Any] = {
        "backend_requested": requested_backend,
        "backend_effective": effective_backend,
        "scoring": {
            "contract_revision": CLI_TUNING_CONTRACT_REVISION,
            "metric": "hypervolume",
            "direction": "maximize",
            "reference_point": _parse_ref_point(args.ref_point, int(args.n_obj)),
            "result_source": "top_level_result_external_archive_disabled",
            "feasibility_filter": "G <= 0",
            "excluded_cli_tuning_params": list(CLI_EXCLUDED_TUNING_PARAMS),
        },
        "split": {
            "instance_counts": {
                "train": len(train_instances),
                "validation": len(validation_instances),
                "test": len(test_instances),
            },
            "seed_counts": {
                "train": len(train_seeds),
                "validation": len(validation_seeds),
                "test": len(test_seeds),
            },
            "split_seed": int(args.split_seed),
            "split_strategy": str(args.split_strategy),
            "train_frac": float(args.train_frac),
            "validation_frac": float(args.validation_frac),
        },
    }
    if smoke_profile is not None:
        summary_updates.update(
            {
                "smoke_mode": True,
                "smoke_profile": smoke_profile,
            }
        )
    _append_summary(
        out_dir,
        summary_updates=summary_updates,
        artifact_updates={
            "split_instances": split_csv_path.name,
            "split_seeds": split_seed_path.name,
        },
    )

    champions = _run_validation_stage(
        args=args,
        out_dir=out_dir,
        history=history,
        task=task,
        eval_fn=eval_fn,
        validation_instances=validation_instances,
        validation_seeds=validation_seeds,
        logger=logger,
    )
    _run_test_stage(
        args=args,
        out_dir=out_dir,
        champions=champions,
        best_config=best_config,
        task=task,
        eval_fn=eval_fn,
        test_instances=test_instances,
        test_seeds=test_seeds,
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
