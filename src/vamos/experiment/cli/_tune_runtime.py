from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np

from vamos.engine.algorithm.variants import canonical_algorithm_name
from vamos.engine.tuning import (
    AlgorithmConfigSpace,
    EvalContext,
    Instance,
    ModelBasedTuner,
    ParamSpace,
    RacingTuner,
    RandomSearchTuner,
    Scenario,
    TrialResult,
    TuningTask,
    build_agemoea_binary_config_space,
    build_agemoea_config_space,
    build_agemoea_integer_config_space,
    build_agemoea_permutation_config_space,
    build_ibea_binary_config_space,
    build_ibea_config_space,
    build_ibea_integer_config_space,
    build_ibea_permutation_config_space,
    build_moead_binary_config_space,
    build_moead_config_space,
    build_moead_integer_config_space,
    build_moead_permutation_config_space,
    build_nsgaii_binary_config_space,
    build_nsgaii_config_space,
    build_nsgaii_integer_config_space,
    build_nsgaii_mixed_config_space,
    build_nsgaii_permutation_config_space,
    build_nsgaiii_binary_config_space,
    build_nsgaiii_config_space,
    build_nsgaiii_integer_config_space,
    build_nsgaiii_permutation_config_space,
    build_rvea_binary_config_space,
    build_rvea_config_space,
    build_rvea_integer_config_space,
    build_rvea_permutation_config_space,
    build_smpso_config_space,
    build_smpso_mixed_config_space,
    build_smsemoa_binary_config_space,
    build_smsemoa_config_space,
    build_smsemoa_integer_config_space,
    build_smsemoa_permutation_config_space,
    build_spea2_binary_config_space,
    build_spea2_config_space,
    build_spea2_integer_config_space,
    build_spea2_permutation_config_space,
    config_from_assignment,
)
from vamos.engine.tuning.racing.eval_types import EvalFn
from vamos.engine.tuning.racing.warm_start import WarmStartEvaluator
from vamos.experiment.types import CheckpointPayload
from vamos.experiment.unified import optimize
from vamos.foundation.problem.registry import make_problem_selection

from . import _tune_scoring
from ._tune_utils import build_aggregator, parse_csv_strings, parse_ref_point, parse_seed_spec

BUILDERS: dict[str, Callable[[], AlgorithmConfigSpace | ParamSpace]] = {
    "nsgaii": build_nsgaii_config_space,
    "nsgaii_permutation": build_nsgaii_permutation_config_space,
    "nsgaii_mixed": build_nsgaii_mixed_config_space,
    "nsgaii_binary": build_nsgaii_binary_config_space,
    "nsgaii_integer": build_nsgaii_integer_config_space,
    "moead": build_moead_config_space,
    "moead_permutation": build_moead_permutation_config_space,
    "moead_binary": build_moead_binary_config_space,
    "moead_integer": build_moead_integer_config_space,
    "nsgaiii": build_nsgaiii_config_space,
    "nsgaiii_permutation": build_nsgaiii_permutation_config_space,
    "nsgaiii_binary": build_nsgaiii_binary_config_space,
    "nsgaiii_integer": build_nsgaiii_integer_config_space,
    "spea2": build_spea2_config_space,
    "spea2_permutation": build_spea2_permutation_config_space,
    "spea2_binary": build_spea2_binary_config_space,
    "spea2_integer": build_spea2_integer_config_space,
    "ibea": build_ibea_config_space,
    "ibea_permutation": build_ibea_permutation_config_space,
    "ibea_binary": build_ibea_binary_config_space,
    "ibea_integer": build_ibea_integer_config_space,
    "smpso": build_smpso_config_space,
    "smpso_mixed": build_smpso_mixed_config_space,
    "smsemoa": build_smsemoa_config_space,
    "smsemoa_permutation": build_smsemoa_permutation_config_space,
    "smsemoa_binary": build_smsemoa_binary_config_space,
    "smsemoa_integer": build_smsemoa_integer_config_space,
    "agemoea": build_agemoea_config_space,
    "agemoea_permutation": build_agemoea_permutation_config_space,
    "agemoea_binary": build_agemoea_binary_config_space,
    "agemoea_integer": build_agemoea_integer_config_space,
    "rvea": build_rvea_config_space,
    "rvea_permutation": build_rvea_permutation_config_space,
    "rvea_binary": build_rvea_binary_config_space,
    "rvea_integer": build_rvea_integer_config_space,
}

MODEL_BACKENDS = ("optuna", "bohb_optuna", "smac3", "bohb")
NON_MODEL_BACKENDS = ("racing", "random")
ALL_BACKENDS = NON_MODEL_BACKENDS + MODEL_BACKENDS
_ARCHIVE_TUNING_PARAMS = {"use_external_archive", "archive_unbounded", "archive_prune_policy"}
_OPTUNA_CLI_STUDY_SUFFIX = "__vamos_cli_final_population_hv_v1"


def supports_warm_start(name: str) -> bool:
    return canonical_algorithm_name(name) in {"nsgaii", "moead"}


def _without_archive_tuning_controls(param_space: ParamSpace) -> ParamSpace:
    """Remove archive controls from the maintained CLI tuning search space."""
    params = {name: spec for name, spec in param_space.params.items() if name not in _ARCHIVE_TUNING_PARAMS}
    conditions = [
        condition
        for condition in param_space.conditions
        if condition.param_name in params
        and not any(f"cfg['{name}']" in condition.expr for name in _ARCHIVE_TUNING_PARAMS)
    ]
    return ParamSpace(params=params, conditions=conditions)


def _preflight_constraint_support(args: Any) -> None:
    """Validate every selected CLI instance before any tuning backend starts."""
    problem_names = list(parse_csv_strings(getattr(args, "instances", ""))) or [str(args.problem)]
    for problem_name in problem_names:
        problem_kwargs: dict[str, Any] = {"n_var": int(args.n_var)}
        if hasattr(args, "n_obj"):
            problem_kwargs["n_obj"] = int(args.n_obj)
        problem = make_problem_selection(str(problem_name), **problem_kwargs).instantiate()
        n_constraints = int(getattr(problem, "n_constraints", 0) or 0)
        _tune_scoring._ensure_constrained_tuning_supported(str(args.algorithm), n_constraints)


def _versioned_optuna_study_name(args: Any, task: TuningTask) -> str:
    """Namespace persistent Optuna studies by the maintained CLI scoring contract."""
    raw = str(getattr(args, "optuna_study_name", "") or "").strip()
    if not raw:
        raw = f"{task.name}_{args.backend}_{int(args.seed)}"
        raw = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in raw)
    return f"{raw}{_OPTUNA_CLI_STUDY_SUFFIX}"


def make_evaluator(
    problem_key: str,
    n_var: int,
    n_obj: int,
    algorithm_name: str,
    fixed_pop_size: int,
    ref_point_str: str | None,
    warm_start: bool,
    runtime_penalty: float,
    failure_score: float,
    *,
    logger: Callable[[], Any],
) -> EvalFn:
    ref_point = parse_ref_point(ref_point_str, n_obj)

    def _score(result: Any, _ctx: EvalContext) -> float:
        return _tune_scoring._score_population_result(result, ref_point, runtime_penalty, failure_score)

    def _run_algorithm(
        config_dict: Mapping[str, object],
        ctx: EvalContext,
        checkpoint: CheckpointPayload | None,
    ) -> tuple[object, CheckpointPayload | None]:
        try:
            start_config: dict[str, Any] = dict(config_dict)
            if algorithm_name == "rvea":
                start_config["n_obj"] = n_obj
            elif "pop_size" not in start_config:
                start_config["pop_size"] = fixed_pop_size

            cfg = _tune_scoring._force_population_result_mode(config_from_assignment(algorithm_name, start_config))
            algo_name = canonical_algorithm_name(algorithm_name)
            problem_name = str(getattr(ctx.instance, "name", problem_key))
            problem_kwargs = dict(getattr(ctx.instance, "kwargs", {}) or {})
            problem_kwargs.setdefault("n_var", int(n_var))
            problem_kwargs.setdefault("n_obj", int(n_obj))
            selection = make_problem_selection(problem_name, **problem_kwargs)
            problem = selection.instantiate()
            n_constraints = int(getattr(problem, "n_constraints", 0) or 0)
            _tune_scoring._ensure_constrained_tuning_supported(algo_name, n_constraints)
            t0 = time.perf_counter()
            result = optimize(
                problem,
                algorithm=algo_name,
                algorithm_config=cfg,
                max_evaluations=int(ctx.budget),
                seed=int(ctx.seed),
                engine="numpy",
                checkpoint=checkpoint,
            )
            elapsed_s = float(time.perf_counter() - t0)
            payload = getattr(result, "data", None)
            if isinstance(payload, dict):
                payload["_elapsed_s"] = elapsed_s
                payload["_tuning_n_constraints"] = n_constraints
            checkpoint_payload = result.data.get("checkpoint")
            return result, cast(CheckpointPayload | None, checkpoint_payload)
        except _tune_scoring._UnsupportedConstrainedTuningError:
            raise
        except Exception:
            logger().warning("[tune] evaluation failed; assigning configured failure score.", exc_info=True)

            class _EmptyResult:
                data = {
                    "F": np.empty((0, n_obj), dtype=float),
                    "population": {"F": np.empty((0, n_obj), dtype=float)},
                    "_elapsed_s": 0.0,
                    "_tuning_n_constraints": 0,
                    "_tuning_failed": True,
                }

            return _EmptyResult(), None

    if warm_start:
        return WarmStartEvaluator(run_fn=_run_algorithm, score_fn=_score)

    def eval_fn(config_dict: dict[str, Any], ctx: EvalContext) -> float:
        result, _ = _run_algorithm(config_dict, ctx, None)
        return _score(result, ctx)

    return eval_fn


def build_task(
    args: Any,
    param_space: ParamSpace,
    budget_per_run: int,
    *,
    instances: list[Instance] | None = None,
    seeds: list[int] | None = None,
) -> TuningTask:
    _preflight_constraint_support(args)
    param_space = _without_archive_tuning_controls(param_space)
    if instances is None:
        problem_names = list(parse_csv_strings(args.instances)) or [str(args.problem)]
        instances = [Instance(name=name, n_var=int(args.n_var), kwargs={}) for name in problem_names]
    if seeds is None:
        seeds = parse_seed_spec(None, default_start=int(args.seed), default_count=int(args.n_seeds))
    return TuningTask(
        name=f"tune_{args.problem}_{args.algorithm}_{args.backend}",
        param_space=param_space,
        instances=instances,
        seeds=seeds,
        aggregator=build_aggregator(str(args.aggregate_mode)),
        budget_per_run=int(budget_per_run),
        maximize=True,
    )


def run_backend(
    args: Any,
    task: TuningTask,
    eval_fn: EvalFn,
    resolved_jobs: int,
) -> tuple[dict[str, Any], list[TrialResult]]:
    fidelity_levels = args.fidelity_levels
    if args.backend in MODEL_BACKENDS:
        min_seed_count = int(args.fidelity_min_seed_count)
        max_seed_count = int(args.fidelity_max_seed_count)
        optuna_study_name = str(args.optuna_study_name).strip() or None
        if str(args.optuna_storage).strip() and str(args.backend) in {"optuna", "bohb_optuna"}:
            optuna_study_name = _versioned_optuna_study_name(args, task)
            args.optuna_study_name = optuna_study_name
        model_tuner = ModelBasedTuner(
            task=task,
            max_trials=int(args.tune_budget),
            backend=str(args.backend),
            seed=int(args.seed),
            n_jobs=int(resolved_jobs),
            timeout_seconds=None if float(args.timeout_seconds) <= 0.0 else float(args.timeout_seconds),
            show_progress_bar=bool(args.show_progress_bar),
            bohb_reduction_factor=max(2, int(args.bohb_reduction_factor)),
            budget_levels=list(fidelity_levels) if fidelity_levels else None,
            fidelity_min_instance_frac=float(args.fidelity_min_instance_frac),
            fidelity_min_seed_count=(None if min_seed_count <= 0 else int(min_seed_count)),
            fidelity_max_seed_count=(None if max_seed_count <= 0 else int(max_seed_count)),
            fidelity_selection_seed=(None if int(args.fidelity_selection_seed) < 0 else int(args.fidelity_selection_seed)),
            optuna_storage_url=(str(args.optuna_storage).strip() or None),
            optuna_study_name=optuna_study_name,
            optuna_load_if_exists=bool(args.optuna_load_if_exists),
        )
        return model_tuner.run(cast(Callable[[dict[str, Any], EvalContext], float], eval_fn), verbose=True)

    if args.backend == "random":
        return RandomSearchTuner(task=task, max_trials=int(args.tune_budget), seed=int(args.seed)).run(
            cast(Callable[[dict[str, Any], EvalContext], float], eval_fn),
            verbose=True,
        )

    scenario = Scenario(
        max_experiments=int(args.tune_budget),
        elimination_fraction=float(args.elimination_fraction),
        alpha=float(args.alpha),
        min_blocks_before_elimination=int(args.min_blocks_before_elimination),
        use_statistical_tests=bool(args.use_statistical_tests),
        n_jobs=int(resolved_jobs),
        verbose=True,
        use_multi_fidelity=bool(args.multi_fidelity),
        fidelity_levels=tuple(int(v) for v in fidelity_levels) if fidelity_levels else Scenario.fidelity_levels,
        fidelity_promotion_ratio=float(args.fidelity_promotion_ratio),
        fidelity_min_configs=int(args.fidelity_min_configs),
        fidelity_warm_start=bool(args.fidelity_warm_start),
    )
    return RacingTuner(task=task, scenario=scenario, seed=int(args.seed), max_initial_configs=int(args.initial_configs)).run(
        eval_fn,
        verbose=True,
    )


__all__ = [
    "ALL_BACKENDS",
    "BUILDERS",
    "MODEL_BACKENDS",
    "build_task",
    "make_evaluator",
    "run_backend",
    "supports_warm_start",
]
