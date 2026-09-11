from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
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
from vamos.foundation.quality_indicators.hypervolume import hypervolume
from vamos.foundation.quality_indicators.pareto import pareto_filter

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
_CONSTRAINED_TUNING_UNSUPPORTED = {"agemoea", "rvea"}


class _UnsupportedConstrainedTuningError(RuntimeError):
    """Raised when maintained CLI tuning cannot score constraints safely."""


def supports_warm_start(name: str) -> bool:
    return canonical_algorithm_name(name) in {"nsgaii", "moead"}


def _ensure_constrained_tuning_supported(algorithm_name: str, n_constraints: int) -> None:
    """Reject constrained CLI tuning when population-aligned G is unavailable."""
    algo_name = canonical_algorithm_name(algorithm_name)
    if n_constraints > 0 and algo_name in _CONSTRAINED_TUNING_UNSUPPORTED:
        raise _UnsupportedConstrainedTuningError(
            f"Maintained CLI tuning does not yet support constrained {algo_name} runs because the engine does not expose "
            "population-aligned constraint values without changing its stable constraint-mode semantics. "
            "Use an explicit controlled study for this algorithm/problem combination."
        )


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


def _force_population_result_mode(config: AlgorithmConfigProtocol) -> AlgorithmConfigProtocol:
    """Prefer population result mode for tuning-compatible built-in configs.

    Most built-in result builders can expose top-level ``F/G`` aligned with the
    final population when ``result_mode='population'``. The scorer still reads
    objectives from the canonical ``population`` payload and can use constraint
    values embedded there directly, so it does not depend on top-level result
    semantics when the population payload is already self-contained.
    """
    config_obj: Any = config
    try:
        field_names = {field.name for field in fields(config_obj)}
    except TypeError as exc:
        raise RuntimeError("CLI tuning requires dataclass algorithm configs with result_mode support.") from exc
    if "result_mode" not in field_names:
        raise RuntimeError("CLI tuning algorithm config does not expose result_mode; population-aligned scoring is unavailable.")
    updated: Any = replace(config_obj, result_mode="population")
    return cast(AlgorithmConfigProtocol, updated)


def _population_front_for_scoring(result: Any) -> NDArray[np.float64]:
    payload = getattr(result, "data", None)
    if not isinstance(payload, dict):
        raise RuntimeError("Tuning evaluator requires OptimizationResult.data with a final population payload.")
    population = payload.get("population")
    if not isinstance(population, Mapping):
        raise RuntimeError("Tuning evaluator requires result.data['population'] for source-consistent scoring.")
    raw_f = population.get("F")
    if raw_f is None:
        raise RuntimeError("Tuning evaluator final population does not contain objective values 'F'.")

    F = np.asarray(raw_f, dtype=float)
    if F.ndim != 2:
        raise RuntimeError(f"Tuning evaluator expected population F to be 2-D, got shape {F.shape}.")

    raw_g = population.get("G")
    constraint_source = "population"
    if raw_g is None:
        try:
            n_constraints = int(payload.get("_tuning_n_constraints", 0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Tuning evaluator received invalid constraint-count metadata.") from exc

        if n_constraints > 0:
            raw_top_f = payload.get("F")
            if raw_top_f is None:
                raise RuntimeError("Constrained tuning requires population-aligned constraint values G.")
            top_f = np.asarray(raw_top_f, dtype=float)
            if top_f.shape != F.shape or not np.array_equal(top_f, F, equal_nan=True):
                raise RuntimeError(
                    "Constrained tuning requires top-level F to align with population F when population G is unavailable."
                )
            raw_g = payload.get("G")
            constraint_source = "top-level"
            if raw_g is None:
                raise RuntimeError("Constrained tuning result does not contain population-aligned constraint values G.")

    if raw_g is not None:
        G = np.asarray(raw_g, dtype=float)
        if G.ndim == 1:
            G = G[:, None]
        if G.ndim != 2 or G.shape[0] != F.shape[0]:
            raise RuntimeError(f"Tuning evaluator {constraint_source} G must align row-wise with population F.")
        F = F[np.all(G <= 0.0, axis=1)]

    if len(F) == 0:
        return np.empty((0, F.shape[1]), dtype=float)
    front = pareto_filter(F, return_indices=False)
    if front is None:
        return np.empty((0, F.shape[1]), dtype=float)
    return np.asarray(front, dtype=float)


def _score_population_result(
    result: Any,
    ref_point: list[float],
    runtime_penalty: float,
    failure_score: float,
) -> float:
    """Score one tuning result while distinguishing valid empty fronts from failures."""
    payload = getattr(result, "data", None)
    failed = isinstance(payload, dict) and bool(payload.get("_tuning_failed", False))
    if failed:
        base_hv = float(failure_score)
    else:
        F = _population_front_for_scoring(result)
        ref = np.asarray(ref_point, dtype=float)
        contributing = F[np.all(F <= ref, axis=1)]
        base_hv = float(hypervolume(contributing, ref)) if len(contributing) > 0 else 0.0

    elapsed_s = 0.0
    if isinstance(payload, dict):
        elapsed_raw = payload.get("_elapsed_s", 0.0)
        try:
            elapsed_s = float(elapsed_raw)
        except Exception:
            elapsed_s = 0.0
    penalized = base_hv - float(runtime_penalty) * float(np.log1p(max(0.0, elapsed_s)))
    return float(penalized)


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
        return _score_population_result(result, ref_point, runtime_penalty, failure_score)

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

            cfg = _force_population_result_mode(config_from_assignment(algorithm_name, start_config))
            algo_name = canonical_algorithm_name(algorithm_name)
            problem_name = str(getattr(ctx.instance, "name", problem_key))
            problem_kwargs = dict(getattr(ctx.instance, "kwargs", {}) or {})
            problem_kwargs.setdefault("n_var", int(n_var))
            problem_kwargs.setdefault("n_obj", int(n_obj))
            selection = make_problem_selection(problem_name, **problem_kwargs)
            problem = selection.instantiate()
            n_constraints = int(getattr(problem, "n_constraints", 0) or 0)
            _ensure_constrained_tuning_supported(algo_name, n_constraints)
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
        except _UnsupportedConstrainedTuningError:
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
            optuna_study_name=(str(args.optuna_study_name).strip() or None),
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
