from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

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
from vamos.foundation.constraints.utils import is_feasible
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.quality_indicators.hypervolume import hypervolume

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

# Output-retention policy must not change the objective set used to score an
# algorithm configuration. Programmatic experimental spaces still expose these
# controls; only the maintained CLI excludes them from its search space.
CLI_EXCLUDED_TUNING_PARAMS = (
    "archive_prune_policy",
    "archive_unbounded",
    "use_external_archive",
)
CLI_TUNING_CONTRACT_REVISION = "score_source_v2"
_CLI_EXCLUDED_TUNING_PARAM_SET = CLI_EXCLUDED_TUNING_PARAMS


def _condition_references_excluded_param(expr: str) -> bool:
    return any(f"cfg['{name}']" in expr or f'cfg["{name}"]' in expr for name in _CLI_EXCLUDED_TUNING_PARAM_SET)


def _strip_cli_excluded_params(config: Mapping[str, object]) -> dict[str, Any]:
    return {name: value for name, value in config.items() if name not in _CLI_EXCLUDED_TUNING_PARAM_SET}


def build_cli_param_space(algorithm_name: str) -> ParamSpace:
    """Return the source-consistent parameter space used by ``vamos tune``."""

    builder = BUILDERS[str(algorithm_name)]
    algo_space = builder()
    raw_space = algo_space.to_param_space() if isinstance(algo_space, AlgorithmConfigSpace) else algo_space
    params = {name: spec for name, spec in raw_space.params.items() if name not in _CLI_EXCLUDED_TUNING_PARAM_SET}
    conditions = [
        condition
        for condition in raw_space.conditions
        if condition.param_name not in _CLI_EXCLUDED_TUNING_PARAM_SET
        and not _condition_references_excluded_param(condition.expr)
    ]
    return ParamSpace(params=params, conditions=conditions)


def _score_hypervolume_result(
    result: Any,
    ref_point: NDArray[np.float64],
    failure_score: float,
) -> float:
    """Score a result with HV after applying the public ``G <= 0`` convention."""

    raw_F = getattr(result, "F", None)
    if raw_F is None:
        return float(failure_score)
    F = cast(NDArray[np.float64], np.asarray(raw_F, dtype=np.float64))
    if F.ndim != 2:
        raise ValueError(f"Tuning scorer expected F with shape (N, n_obj); got {F.shape}.")
    if F.shape[0] == 0:
        return float(failure_score)

    raw_G = getattr(result, "G", None)
    if raw_G is None:
        payload = getattr(result, "data", None)
        if isinstance(payload, Mapping):
            raw_G = payload.get("G")

    G: NDArray[np.float64] | None
    if raw_G is None:
        G = None
    else:
        G = cast(NDArray[np.float64], np.asarray(raw_G, dtype=np.float64))
        if G.ndim == 1:
            G = G.reshape(-1, 1)
        if G.ndim != 2:
            raise ValueError(f"Tuning scorer expected G with shape (N, n_constraints); got {G.shape}.")
        if G.shape[0] != F.shape[0]:
            raise ValueError(
                "Tuning scorer requires aligned F/G rows; "
                f"got F rows={F.shape[0]} and G rows={G.shape[0]}."
            )

    feasible = is_feasible(G, n=F.shape[0])
    feasible_F = F[feasible]
    if feasible_F.shape[0] == 0:
        return float(failure_score)
    return float(hypervolume(feasible_F, ref_point))


def _retired_cli_params(configs: list[Mapping[str, object]]) -> list[str]:
    return sorted({name for config in configs for name in config if name in _CLI_EXCLUDED_TUNING_PARAM_SET})


def reject_legacy_cli_history(best_config: Mapping[str, object], history: list[TrialResult]) -> None:
    """Reject histories created with the retired archive-varying CLI space."""

    configs: list[Mapping[str, object]] = [best_config]
    configs.extend(trial.config for trial in history)
    retired = _retired_cli_params(configs)
    if retired:
        names = ", ".join(retired)
        raise RuntimeError(
            "Loaded tuning history uses archive parameters retired from the maintained CLI scoring space "
            f"({names}). Start a fresh persisted tuning study/storage or choose a new --optuna-study-name."
        )


def reject_legacy_optuna_study_before_run(args: Any) -> None:
    """Reject an explicitly named legacy Optuna study before new trials run."""

    if str(args.backend) not in {"optuna", "bohb_optuna"}:
        return
    storage_url = str(args.optuna_storage or "").strip()
    study_name = str(args.optuna_study_name or "").strip()
    if not storage_url or not study_name or not bool(args.optuna_load_if_exists):
        return

    import optuna

    summaries = optuna.get_all_study_summaries(storage=storage_url)
    if not any(str(summary.study_name) == study_name for summary in summaries):
        return
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    configs: list[Mapping[str, object]] = []
    for trial in study.trials:
        stored_config = trial.user_attrs.get("config")
        if isinstance(stored_config, Mapping):
            configs.append(stored_config)
        else:
            configs.append(trial.params)
    retired = _retired_cli_params(configs)
    if retired:
        names = ", ".join(retired)
        raise RuntimeError(
            "Persisted Optuna study uses archive parameters retired from the maintained CLI scoring space "
            f"({names}). Choose a fresh --optuna-study-name or a new storage before running new trials."
        )


def tuning_scoring_summary(ref_point_str: str | None, n_obj: int) -> dict[str, Any]:
    """Return persisted provenance for the maintained CLI scoring contract."""

    return {
        "contract_revision": CLI_TUNING_CONTRACT_REVISION,
        "metric": "hypervolume",
        "direction": "maximize",
        "reference_point": parse_ref_point(ref_point_str, n_obj),
        "result_source": "top_level_result_external_archive_disabled",
        "feasibility_filter": "G <= 0",
        "excluded_cli_tuning_params": list(CLI_EXCLUDED_TUNING_PARAMS),
    }


def _validated_backend_result(
    result: tuple[dict[str, Any], list[TrialResult]],
) -> tuple[dict[str, Any], list[TrialResult]]:
    best_config, history = result
    reject_legacy_cli_history(best_config, history)
    return best_config, history


def supports_warm_start(name: str) -> bool:
    return canonical_algorithm_name(name) in {"nsgaii", "moead"}


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
    ref_point = cast(NDArray[np.float64], np.asarray(parse_ref_point(ref_point_str, n_obj), dtype=np.float64))

    def _score(result: Any, _ctx: EvalContext) -> float:
        base_hv = _score_hypervolume_result(result, ref_point, failure_score)
        elapsed_s = 0.0
        payload = getattr(result, "data", None)
        if isinstance(payload, dict):
            elapsed_raw = payload.get("_elapsed_s", 0.0)
            try:
                elapsed_s = float(elapsed_raw)
            except Exception:
                elapsed_s = 0.0
        penalized = base_hv - float(runtime_penalty) * float(np.log1p(max(0.0, elapsed_s)))
        return float(penalized)

    def _run_algorithm(
        config_dict: Mapping[str, object],
        ctx: EvalContext,
        checkpoint: CheckpointPayload | None,
    ) -> tuple[object, CheckpointPayload | None]:
        try:
            # Defensive stripping also protects evaluation of caller-supplied or
            # explicitly resumed experimental configs created before this CLI
            # scoring contract revision.
            start_config = _strip_cli_excluded_params(config_dict)
            if algorithm_name == "rvea":
                start_config["n_obj"] = n_obj
            elif "pop_size" not in start_config:
                start_config["pop_size"] = fixed_pop_size

            cfg = config_from_assignment(algorithm_name, start_config)
            algo_name = canonical_algorithm_name(algorithm_name)
            problem_name = str(getattr(ctx.instance, "name", problem_key))
            problem_kwargs = dict(getattr(ctx.instance, "kwargs", {}) or {})
            problem_kwargs.setdefault("n_var", int(n_var))
            problem_kwargs.setdefault("n_obj", int(n_obj))
            selection = make_problem_selection(problem_name, **problem_kwargs)
            t0 = time.perf_counter()
            result = optimize(
                selection.instantiate(),
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
            checkpoint_payload = result.data.get("checkpoint")
            return result, cast(CheckpointPayload | None, checkpoint_payload)
        except Exception:
            logger().warning("[tune] candidate execution failed; assigning the configured failure score.", exc_info=True)

            class _EmptyResult:
                F = None
                data = {"_elapsed_s": 0.0}

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
    if instances is None:
        problem_names = list(parse_csv_strings(args.instances)) or [str(args.problem)]
        instances = [Instance(name=name, n_var=int(args.n_var), kwargs={}) for name in problem_names]
    if seeds is None:
        seeds = parse_seed_spec(None, default_start=int(args.seed), default_count=int(args.n_seeds))
    return TuningTask(
        name=f"tune_{CLI_TUNING_CONTRACT_REVISION}_{args.problem}_{args.algorithm}_{args.backend}",
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
        reject_legacy_optuna_study_before_run(args)
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
        return _validated_backend_result(
            model_tuner.run(cast(Callable[[dict[str, Any], EvalContext], float], eval_fn), verbose=True)
        )

    if args.backend == "random":
        return _validated_backend_result(
            RandomSearchTuner(task=task, max_trials=int(args.tune_budget), seed=int(args.seed)).run(
                cast(Callable[[dict[str, Any], EvalContext], float], eval_fn),
                verbose=True,
            )
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
    return _validated_backend_result(
        RacingTuner(
            task=task,
            scenario=scenario,
            seed=int(args.seed),
            max_initial_configs=int(args.initial_configs),
        ).run(eval_fn, verbose=True)
    )


__all__ = [
    "ALL_BACKENDS",
    "BUILDERS",
    "CLI_EXCLUDED_TUNING_PARAMS",
    "CLI_TUNING_CONTRACT_REVISION",
    "MODEL_BACKENDS",
    "_score_hypervolume_result",
    "build_cli_param_space",
    "build_task",
    "make_evaluator",
    "reject_legacy_cli_history",
    "reject_legacy_optuna_study_before_run",
    "run_backend",
    "supports_warm_start",
    "tuning_scoring_summary",
]
