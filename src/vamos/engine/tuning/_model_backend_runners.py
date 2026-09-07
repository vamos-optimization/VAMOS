from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from ._model_backend_fidelity import (
    FidelityTunerLike,
    ScoreEvalFn,
    default_optuna_study_name,
    eval_config_at_budget,
    resolve_budget_levels,
    resolve_fidelity_slice,
)
from ._model_backend_utils import (
    build_configspace as _build_configspace,
)
from ._model_backend_utils import (
    build_optuna_sampler as _build_optuna_sampler,
)
from ._model_backend_utils import (
    estimate_hyperband_evals_per_iteration as _estimate_hyperband_evals_per_iteration,
)
from ._model_backend_utils import (
    sample_from_optuna_trial as _sample_from_optuna_trial,
)
from .racing.random_search_tuner import TrialResult
from .racing.tuning_task import TuningTask


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class BackendTunerLike(FidelityTunerLike, Protocol):
    task: TuningTask
    max_trials: int
    n_jobs: int
    timeout_seconds: float | None
    show_progress_bar: bool
    bohb_reduction_factor: int
    optuna_storage_url: str | None
    optuna_study_name: str | None
    optuna_load_if_exists: bool
    optuna_sampler: str

    def _worst_score(self) -> float: ...
    def _score_to_loss(self, score: float) -> float: ...


def _resolve_budget_bounds(tuner: BackendTunerLike) -> tuple[int, int]:
    levels = resolve_budget_levels(tuner)
    if tuner.budget_levels:
        min_budget_i = int(levels[0])
        max_budget_i = int(levels[-1])
    else:
        min_budget_i = 1
        max_budget_i = int(tuner.task.budget_per_run)
    min_budget_i = max(1, min_budget_i)
    max_budget_i = max(min_budget_i, max_budget_i)
    return min_budget_i, max_budget_i


def run_optuna_like(
    tuner: BackendTunerLike,
    eval_fn: ScoreEvalFn,
    *,
    bohb_mode: bool,
) -> tuple[dict[str, Any], list[TrialResult]]:
    import optuna

    levels = resolve_budget_levels(tuner)
    sampler = _build_optuna_sampler(tuner.optuna_sampler, seed=int(tuner.seed))
    pruner: Any
    if bohb_mode:
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=max(1, len(levels)),
            reduction_factor=max(2, int(tuner.bohb_reduction_factor)),
        )
    else:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=max(5, min(40, int(tuner.max_trials // 4) if tuner.max_trials > 0 else 5)),
            n_warmup_steps=max(1, len(levels) - 1),
            interval_steps=1,
        )

    storage_url = str(tuner.optuna_storage_url or "").strip() or None
    study_name_raw = str(tuner.optuna_study_name or "").strip()
    study_name = study_name_raw if study_name_raw else default_optuna_study_name(tuner)
    create_kwargs: dict[str, Any] = {
        "direction": "maximize" if tuner.task.maximize else "minimize",
        "sampler": sampler,
        "pruner": pruner,
    }
    if storage_url:
        create_kwargs["storage"] = storage_url
        create_kwargs["study_name"] = study_name
        create_kwargs["load_if_exists"] = bool(tuner.optuna_load_if_exists)

    study = optuna.create_study(**create_kwargs)

    def objective(trial: Any) -> float:
        config = _sample_from_optuna_trial(trial, tuner.task.param_space)
        tuner.task.param_space.validate(config)
        trial.set_user_attr("config", dict(config))
        final_score = tuner._worst_score()
        fidelity_trace: list[dict[str, Any]] = []
        for step_idx, budget in enumerate(levels):
            score = eval_config_at_budget(tuner, config, eval_fn, budget=int(budget))
            final_score = score
            _, _, fidelity_meta = resolve_fidelity_slice(tuner, int(budget))
            fidelity_trace.append(
                {
                    "step": int(step_idx),
                    "budget": int(budget),
                    "score": float(score),
                    "instances_used": int(fidelity_meta.get("instances_used", 0)),
                    "instances_total": int(fidelity_meta.get("instances_total", 0)),
                    "seeds_used": int(fidelity_meta.get("seeds_used", 0)),
                    "seeds_total": int(fidelity_meta.get("seeds_total", 0)),
                }
            )
            trial.report(float(score), step=step_idx)
            if trial.should_prune():
                trial.set_user_attr("fidelity_trace", fidelity_trace)
                raise optuna.TrialPruned(f"Pruned at step {step_idx} (budget={budget}).")
        trial.set_user_attr("fidelity_trace", fidelity_trace)
        return float(final_score)

    study.optimize(
        objective,
        n_trials=int(tuner.max_trials),
        n_jobs=int(tuner.n_jobs),
        timeout=float(tuner.timeout_seconds) if tuner.timeout_seconds is not None else None,
        show_progress_bar=bool(tuner.show_progress_bar),
    )

    history: list[TrialResult] = []
    for trial in study.trials:
        if trial.value is None:
            continue
        cfg = trial.user_attrs.get("config", dict(trial.params))
        details: dict[str, Any] = {"state": str(getattr(trial.state, "name", trial.state))}
        fidelity_trace = trial.user_attrs.get("fidelity_trace")
        if isinstance(fidelity_trace, list):
            details["fidelity_trace"] = fidelity_trace
        history.append(TrialResult(trial_id=int(trial.number), config=dict(cfg), score=float(trial.value), details=details))
    if not history:
        raise RuntimeError("Tuner finished without a valid configuration.")
    best_trial = study.best_trial
    best_cfg = best_trial.user_attrs.get("config", dict(best_trial.params))
    return dict(best_cfg), history


def run_smac3(tuner: BackendTunerLike, eval_fn: ScoreEvalFn) -> tuple[dict[str, Any], list[TrialResult]]:
    from smac import MultiFidelityFacade, Scenario

    cs = _build_configspace(tuner.task.param_space, seed=int(tuner.seed))
    min_budget_i, max_budget_i = _resolve_budget_bounds(tuner)

    def target(config: Mapping[str, Any], seed: int = 0, budget: float | None = None) -> float:
        _ = seed
        cfg = dict(config)
        tuner.task.param_space.validate(cfg)
        b = int(round(float(budget if budget is not None else max_budget_i)))
        b = min(int(max_budget_i), max(int(min_budget_i), b))
        score = eval_config_at_budget(tuner, cfg, eval_fn, budget=b)
        return float(tuner._score_to_loss(score))

    scenario = Scenario(
        configspace=cs,
        deterministic=False,
        n_trials=int(tuner.max_trials),
        min_budget=float(min_budget_i),
        max_budget=float(max_budget_i),
        n_workers=int(max(1, tuner.n_jobs)),
        seed=int(tuner.seed),
        walltime_limit=float(tuner.timeout_seconds) if tuner.timeout_seconds is not None else np.inf,
    )
    optimizer = MultiFidelityFacade(
        scenario=scenario,
        target_function=target,
        overwrite=True,
    )
    _ = optimizer.optimize()

    history: list[TrialResult] = []
    runhistory = optimizer.runhistory
    for trial_id, (trial_key, trial_value) in enumerate(runhistory.items()):
        cfg = runhistory.get_config(trial_key.config_id)
        if cfg is None:
            continue
        cfg_dict = dict(cfg)
        raw_cost = trial_value.cost
        if isinstance(raw_cost, list):
            if not raw_cost:
                continue
            loss = float(raw_cost[0])
        else:
            loss = float(raw_cost)
        score = float(-loss if tuner.task.maximize else loss)
        budget = None if trial_key.budget is None else int(round(float(trial_key.budget)))
        fidelity_budget = int(max_budget_i if budget is None else budget)
        _, _, fidelity_meta = resolve_fidelity_slice(tuner, int(fidelity_budget))
        details = {
            "backend": "smac3",
            "seed": None if trial_key.seed is None else int(trial_key.seed),
            "budget": budget,
            "loss": float(loss),
            "status": str(getattr(trial_value.status, "name", trial_value.status)),
            "time": float(trial_value.time),
            "cpu_time": float(trial_value.cpu_time),
            "instances_used": int(fidelity_meta.get("instances_used", 0)),
            "instances_total": int(fidelity_meta.get("instances_total", 0)),
            "seeds_used": int(fidelity_meta.get("seeds_used", 0)),
            "seeds_total": int(fidelity_meta.get("seeds_total", 0)),
        }
        history.append(TrialResult(trial_id=int(trial_id), config=cfg_dict, score=score, details=details))

    if not history:
        raise RuntimeError("Tuner finished without a valid configuration.")
    best = max(history, key=lambda h: h.score) if tuner.task.maximize else min(history, key=lambda h: h.score)
    return dict(best.config), history


def run_bohb_native(tuner: BackendTunerLike, eval_fn: ScoreEvalFn) -> tuple[dict[str, Any], list[TrialResult]]:
    import hpbandster.core.nameserver as hpns
    from hpbandster.core.worker import Worker
    from hpbandster.optimizers import BOHB

    cs = _build_configspace(tuner.task.param_space, seed=int(tuner.seed))
    eta = max(2, int(tuner.bohb_reduction_factor))
    min_budget_i, max_budget_i = _resolve_budget_bounds(tuner)
    history: list[TrialResult] = []
    lock = threading.Lock()
    trial_counter = 0

    class _BOHB(BOHB):  # type: ignore[misc]
        def _submit_job(inner_self, config_id: Any, config: Mapping[str, Any], budget: float) -> None:
            # ConfigSpace returns NumPy scalars for categorical/boolean parents;
            # HpBandSter's Pyro transport requires plain Python values.
            cfg = {name: value.item() if isinstance(value, np.generic) else value for name, value in config.items()}
            super()._submit_job(config_id, cfg, float(budget))

    class _Worker(Worker):  # type: ignore[misc]
        def compute(inner_self, config: Mapping[str, Any], budget: float | None, **kwargs: Any) -> dict[str, Any]:
            nonlocal trial_counter
            _ = inner_self, kwargs
            cfg = dict(config)
            try:
                tuner.task.param_space.validate(cfg)
            except Exception as exc:
                return {"loss": 1e9, "info": {"error": f"{type(exc).__name__}: {exc}"}}
            b = int(round(float(budget if budget is not None else max_budget_i)))
            b = min(int(max_budget_i), max(int(min_budget_i), b))
            score = eval_config_at_budget(tuner, cfg, eval_fn, budget=b)
            loss = tuner._score_to_loss(score)
            _, _, fidelity_meta = resolve_fidelity_slice(tuner, int(b))
            with lock:
                tid = int(trial_counter)
                trial_counter += 1
                history.append(
                    TrialResult(
                        trial_id=tid,
                        config=dict(cfg),
                        score=float(score),
                        details={
                            "backend": "bohb",
                            "budget": int(b),
                            "loss": float(loss),
                            "instances_used": int(fidelity_meta.get("instances_used", 0)),
                            "instances_total": int(fidelity_meta.get("instances_total", 0)),
                            "seeds_used": int(fidelity_meta.get("seeds_used", 0)),
                            "seeds_total": int(fidelity_meta.get("seeds_total", 0)),
                        },
                    )
                )
            return {"loss": float(loss), "info": {"score": float(score)}}

    run_id = f"vamos_bohb_{int(time.time())}_{int(tuner.seed)}"
    ns = None
    optimizer = None
    workers: list[Any] = []
    try:
        ns = hpns.NameServer(run_id=run_id, host="127.0.0.1", port=0)
        ns_host, ns_port = ns.start()
        for worker_id in range(max(1, int(tuner.n_jobs))):
            worker = _Worker(
                run_id=run_id,
                host="127.0.0.1",
                nameserver=ns_host,
                nameserver_port=ns_port,
                id=worker_id,
            )
            worker.run(background=True)
            workers.append(worker)

        optimizer = _BOHB(
            configspace=cs,
            run_id=run_id,
            nameserver=ns_host,
            nameserver_port=ns_port,
            min_budget=float(min_budget_i),
            max_budget=float(max_budget_i),
            eta=int(eta),
        )
        evals_per_iter = _estimate_hyperband_evals_per_iteration(max_budget=int(max_budget_i), eta=int(eta))
        n_iterations = max(1, int(math.ceil(max(1, int(tuner.max_trials)) / evals_per_iter)))
        _ = optimizer.run(n_iterations=n_iterations, min_n_workers=max(1, int(tuner.n_jobs)))
    finally:
        if optimizer is not None:
            try:
                optimizer.shutdown(shutdown_workers=True)
            except Exception:
                _logger().debug("Failed to shutdown BOHB optimizer cleanly.", exc_info=True)
        if ns is not None:
            try:
                ns.shutdown()
            except Exception:
                _logger().debug("Failed to shutdown BOHB nameserver cleanly.", exc_info=True)

    if not history:
        raise RuntimeError("Tuner finished without a valid configuration.")
    best = max(history, key=lambda h: h.score) if tuner.task.maximize else min(history, key=lambda h: h.score)
    return dict(best.config), history


__all__ = ["run_bohb_native", "run_optuna_like", "run_smac3"]
