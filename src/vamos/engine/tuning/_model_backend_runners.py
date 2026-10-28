from __future__ import annotations

import logging
import sys
import threading
import time
import types
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
    hyperband_brackets as _hyperband_brackets,
)
from ._model_backend_utils import (
    sample_from_optuna_trial as _sample_from_optuna_trial,
)
from .racing.param_space import _active_assignment
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
    """Return the smallest and the full per-run budget; they are equal without ``budget_levels``."""
    levels = resolve_budget_levels(tuner)
    min_budget_i = max(1, int(levels[0]))
    return min_budget_i, max(min_budget_i, int(levels[-1]))


def _full_budget_incumbent(history: list[TrialResult], full_budget: int, *, maximize: bool) -> TrialResult:
    """Return the best run at the full budget, as HpBandSter's ``Result.get_incumbent_id()`` does."""
    full_runs = [trial for trial in history if trial.details.get("budget") == full_budget]
    if not full_runs:
        raise RuntimeError(f"Tuner finished without a configuration evaluated at the full budget ({full_budget}); increase max_trials.")
    return max(full_runs, key=lambda h: h.score) if maximize else min(full_runs, key=lambda h: h.score)


def run_optuna_like(
    tuner: BackendTunerLike,
    eval_fn: ScoreEvalFn,
    *,
    bohb_mode: bool,
) -> tuple[dict[str, Any], list[TrialResult]]:
    import optuna

    levels = resolve_budget_levels(tuner)
    sampler = _build_optuna_sampler(tuner.optuna_sampler, seed=int(tuner.seed))
    # A trial reports the score of each level but the last at steps 1, 2, ...; the pruner may
    # stop it after any of them. With a single level there is nothing to prune.
    pruner: Any
    if len(levels) == 1:
        pruner = optuna.pruners.NopPruner()
    elif bohb_mode:
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=len(levels),
            reduction_factor=max(2, int(tuner.bohb_reduction_factor)),
        )
    else:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=max(5, min(40, int(tuner.max_trials // 4) if tuner.max_trials > 0 else 5)),
            n_warmup_steps=0,
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
            if step_idx + 1 < len(levels):
                trial.report(float(score), step=step_idx + 1)
                if trial.should_prune():
                    trial.set_user_attr("fidelity_trace", fidelity_trace)
                    raise optuna.TrialPruned(f"Pruned after level {step_idx} (budget={budget}).")
        trial.set_user_attr("fidelity_trace", fidelity_trace)
        return float(final_score)

    study.optimize(
        objective,
        n_trials=int(tuner.max_trials),
        n_jobs=int(tuner.n_jobs),
        timeout=float(tuner.timeout_seconds) if tuner.timeout_seconds is not None else None,
        show_progress_bar=bool(tuner.show_progress_bar),
    )

    def active_config(trial: Any) -> dict[str, Any]:
        # Trials resumed from a study stored before conditions were honored may record inactive parameters.
        return _active_assignment(tuner.task.param_space, dict(trial.user_attrs.get("config", trial.params)))

    history: list[TrialResult] = []
    for trial in study.trials:
        if trial.value is None:
            continue
        cfg = active_config(trial)
        details: dict[str, Any] = {"state": str(getattr(trial.state, "name", trial.state))}
        fidelity_trace = trial.user_attrs.get("fidelity_trace")
        if isinstance(fidelity_trace, list):
            details["fidelity_trace"] = fidelity_trace
        history.append(TrialResult(trial_id=int(trial.number), config=dict(cfg), score=float(trial.value), details=details))
    if not history:
        raise RuntimeError("Tuner finished without a valid configuration.")
    # A pruned trial stays in the history with its last score but never competes for the best:
    # study.best_trial considers only the trials completed at the full budget.
    if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
        raise RuntimeError("Tuner finished without a configuration evaluated at the full budget; increase max_trials.")
    return active_config(study.best_trial), history


def run_smac3(tuner: BackendTunerLike, eval_fn: ScoreEvalFn) -> tuple[dict[str, Any], list[TrialResult]]:
    from smac import HyperparameterOptimizationFacade, MultiFidelityFacade, Scenario

    cs = _build_configspace(tuner.task.param_space, seed=int(tuner.seed))
    min_budget_i, max_budget_i = _resolve_budget_bounds(tuner)
    multi_fidelity = min_budget_i < max_budget_i

    def target(config: Mapping[str, Any], seed: int = 0, budget: float | None = None) -> float:
        _ = seed
        cfg = dict(config)
        tuner.task.param_space.validate(cfg)
        b = int(round(float(budget if budget is not None else max_budget_i)))
        b = min(int(max_budget_i), max(int(min_budget_i), b))
        score = eval_config_at_budget(tuner, cfg, eval_fn, budget=b)
        return float(tuner._score_to_loss(score))

    # Hyperband between the smallest and the full budget only when fidelity levels are given;
    # otherwise every trial runs at the full budget.
    scenario = Scenario(
        configspace=cs,
        # The target ignores SMAC's seed: each evaluation already covers the task's seeds, so
        # repeating a configuration on another SMAC seed would rerun the same runs.
        deterministic=True,
        n_trials=int(tuner.max_trials),
        min_budget=float(min_budget_i) if multi_fidelity else None,
        max_budget=float(max_budget_i) if multi_fidelity else None,
        n_workers=int(max(1, tuner.n_jobs)),
        seed=int(tuner.seed),
        walltime_limit=float(tuner.timeout_seconds) if tuner.timeout_seconds is not None else np.inf,
    )
    facade = MultiFidelityFacade if multi_fidelity else HyperparameterOptimizationFacade
    optimizer = facade(
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
        # Without budgets SMAC records none: the run used the full budget.
        fidelity_budget = int(max_budget_i if trial_key.budget is None else round(float(trial_key.budget)))
        _, _, fidelity_meta = resolve_fidelity_slice(tuner, int(fidelity_budget))
        details = {
            "backend": "smac3",
            "seed": None if trial_key.seed is None else int(trial_key.seed),
            "budget": fidelity_budget,
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
    best = _full_budget_incumbent(history, max_budget_i, maximize=tuner.task.maximize)
    return dict(best.config), history


class _RandomStateCallsFromGenerator:
    """The ``RandomState``-style ``np.random`` calls of HpBandSter's BOHB generator, served by an owned ``Generator``.

    Any other ``np.random`` call fails with ``AttributeError`` instead of drawing from a global stream.
    """

    def __init__(self, generator: np.random.Generator) -> None:
        self.generator = generator

    def rand(self) -> float:
        return float(self.generator.random())

    def randint(self, low: int, high: int | None = None) -> int:
        return int(self.generator.integers(low, high))

    def choice(self, a: Any) -> Any:
        return self.generator.choice(a)


class _NumpyWithPrivateRandom:
    """The ``numpy`` module as HpBandSter's BOHB generator sees it, with ``random`` bound to a private stream."""

    def __init__(self, generator: np.random.Generator) -> None:
        self.random = _RandomStateCallsFromGenerator(generator)

    def __getattr__(self, name: str) -> Any:
        return getattr(np, name)


class _SeededDistribution:
    """A SciPy distribution whose ``rvs`` draws from a private stream unless given one."""

    def __init__(self, distribution: Any, random_state: np.random.Generator) -> None:
        self._distribution = distribution
        self._random_state = random_state

    def rvs(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("random_state", self._random_state)
        return self._distribution.rvs(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._distribution, name)


class _StatsWithPrivateRandom:
    """``scipy.stats`` as HpBandSter's BOHB generator sees it, with distributions drawing from a private stream."""

    def __init__(self, stats: Any, random_state: np.random.Generator) -> None:
        self._stats = stats
        self._random_state = random_state

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._stats, name)
        return _SeededDistribution(value, self._random_state) if hasattr(value, "rvs") else value


def _isolate_bohb_random_stream(config_generator: Any, rng: np.random.Generator) -> None:
    """Make one HpBandSter BOHB configuration generator draw from ``rng``.

    HpBandSter accepts no seed: the generator draws through the ``np`` and ``sps`` globals of
    its module (``np.random`` and SciPy ``rvs`` without a ``random_state``), that is, from
    NumPy's process-wide state. Its two sampling methods are rebuilt on this instance over a
    copy of those globals in which both draw from ``rng``. The search is then reproducible
    for a seed, and neither NumPy's global state nor HpBandSter's module is modified.
    """
    from scipy import stats  # type: ignore[import-untyped]

    scope = dict(vars(sys.modules[type(config_generator).__module__]))
    scope["np"] = _NumpyWithPrivateRandom(rng)
    scope["sps"] = _StatsWithPrivateRandom(stats, rng)
    for name in ("get_config", "impute_conditional_data"):
        function = getattr(type(config_generator), name, None)
        if not isinstance(function, types.FunctionType):
            raise RuntimeError(f"Unsupported HpBandSter version: the BOHB generator has no {name}() to seed.")
        rebuilt = types.FunctionType(function.__code__, scope, function.__name__, function.__defaults__, function.__closure__)
        setattr(config_generator, name, types.MethodType(rebuilt, config_generator))


def run_bohb_native(tuner: BackendTunerLike, eval_fn: ScoreEvalFn) -> tuple[dict[str, Any], list[TrialResult]]:
    import hpbandster.core.nameserver as hpns
    from hpbandster.core.worker import Worker
    from hpbandster.optimizers import BOHB
    from hpbandster.optimizers.iterations import SuccessiveHalving

    cs = _build_configspace(tuner.task.param_space, seed=int(tuner.seed))
    eta = max(2, int(tuner.bohb_reduction_factor))
    # Without budget_levels both bounds are the full budget: each iteration runs one configuration at it.
    min_budget_i, max_budget_i = _resolve_budget_bounds(tuner)
    history: list[TrialResult] = []
    lock = threading.Lock()
    trial_counter = 0
    brackets: list[list[int]] = []

    class _BOHB(BOHB):  # type: ignore[misc]
        def get_next_iteration(inner_self, iteration: int, iteration_kwargs: dict[str, Any] | None = None) -> Any:
            # HpBandSter's bracket for this iteration, with the planned number of configurations per rung.
            num_configs = brackets[iteration]
            return SuccessiveHalving(
                HPB_iter=iteration,
                num_configs=num_configs,
                budgets=inner_self.budgets[-len(num_configs) :],
                config_sampler=inner_self.config_generator.get_config,
                **(iteration_kwargs or {}),
            )

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
        # A stream apart from the ConfigSpace one, which is seeded with the tuner seed itself.
        _isolate_bohb_random_stream(optimizer.config_generator, np.random.default_rng([int(tuner.seed), 1]))
        brackets.extend(_hyperband_brackets(int(optimizer.max_SH_iter), int(eta), int(tuner.max_trials)))
        _ = optimizer.run(n_iterations=len(brackets), min_n_workers=max(1, int(tuner.n_jobs)))
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
    best = _full_budget_incumbent(history, max_budget_i, maximize=tuner.task.maximize)
    return dict(best.config), history


__all__ = ["run_bohb_native", "run_optuna_like", "run_smac3"]
