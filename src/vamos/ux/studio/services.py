from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.catalog import DEFAULT_ALGORITHM
from vamos.engine.algorithm.config import (
    AGEMOEAConfig,
    GenericAlgorithmConfig,
    IBEAConfig,
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    RVEAConfig,
    SMPSOConfig,
    SMSEMOAConfig,
    SPEA2Config,
)
from vamos.engine.algorithm.config.base import _SerializableConfig
from vamos.engine.algorithm.config.defaults import build_default_algorithm_config
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.engine.algorithm.registry import get_algorithms_registry, resolve_algorithm
from vamos.exceptions import InvalidAlgorithmError
from vamos.foundation.eval import EvaluationBackend
from vamos.foundation.eval.backends import resolve_eval_strategy
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.problem.types import ProblemProtocol
from vamos.study_artifacts import load_study
from vamos.ux.analysis.mcdm import reference_point_scores
from vamos.ux.studio._studio_llm import llm_generate_problem_code
from vamos.ux.studio.data import build_fronts, load_runs_from_study
from vamos.ux.studio.dm import build_decision_view

if TYPE_CHECKING:
    from vamos.ux.studio.data import FrontRecord, RunRecord
    from vamos.ux.studio.dm import DecisionView

_CONFIG_MAP: dict[str, type[_SerializableConfig]] = {
    "nsgaii": NSGAIIConfig,
    "moead": MOEADConfig,
    "spea2": SPEA2Config,
    "smsemoa": SMSEMOAConfig,
    "nsgaiii": NSGAIIIConfig,
    "ibea": IBEAConfig,
    "smpso": SMPSOConfig,
    "agemoea": AGEMOEAConfig,
    "rvea": RVEAConfig,
}


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class DynamicsCallback:
    def __init__(self) -> None:
        self.history: list[NDArray[Any]] = []
        self._generation: int | None = None
        self._recorded_generation: int | None = None

    def on_start(self, ctx: Any) -> None:
        pass

    def on_generation(self, generation: int, *, F: Any = None, **kwargs: Any) -> None:
        self._generation = generation
        try:
            if F is not None:
                self.history.append(np.array(F))
                self._recorded_generation = generation
        except Exception:
            _logger().debug("Failed to capture population snapshot in DynamicsCallback", exc_info=True)

    def on_end(self, *, final_F: Any = None) -> None:
        try:
            if final_F is not None:
                final = np.array(final_F)
                if self.history and self._recorded_generation == self._generation:
                    self.history[-1] = final
                else:
                    self.history.append(final)
                self._recorded_generation = self._generation
        except Exception:
            _logger().debug("Failed to capture final front in DynamicsCallback", exc_info=True)

    def __call__(self, algorithm: Any) -> None:
        try:
            if hasattr(algorithm, "pop") and algorithm.pop is not None:
                F = algorithm.pop.get("F")
                if F is not None:
                    self.on_generation(len(self.history), F=F)
        except Exception:
            _logger().debug("Failed to capture population snapshot in DynamicsCallback", exc_info=True)


def load_studio_data(study_dir: Path) -> tuple[list[RunRecord], list[FrontRecord]]:
    runs = load_runs_from_study(study_dir)
    fronts = build_fronts(runs)
    return runs, fronts


def discover_study_directories(base_dir: Path, *, limit: int = 8) -> list[Path]:
    """Return verified canonical study roots for the UI picker."""
    base_dir = base_dir.resolve()
    candidates: list[Path] = []

    def add_candidate(path: Path) -> None:
        path = path.resolve()
        if path in candidates or not path.exists() or not path.is_dir():
            return
        try:
            load_study(path)
        except Exception:
            return
        candidates.append(path)

    if (base_dir / "study-manifest.json").is_file():
        add_candidate(base_dir)
    for manifest_path in base_dir.rglob("study-manifest.json"):
        add_candidate(manifest_path.parent)

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[:limit]


def build_demo_study_data() -> tuple[list[RunRecord], list[FrontRecord]]:
    """Return a small built-in dataset so the Explore tab is useful on first launch."""
    from vamos.ux.studio.data import FrontRecord, RunRecord

    x = np.linspace(0.02, 1.0, 24)
    nsgaii_front = np.column_stack([x, 1.02 - np.sqrt(x)])
    moead_front = np.column_stack([x, 1.04 - np.power(x, 0.62)])
    nsgaii_front = np.clip(nsgaii_front, 0.0, None)
    moead_front = np.clip(moead_front, 0.0, None)
    points_x = np.column_stack([x, 1.0 - x])

    demo_runs = [
        RunRecord("demo", "demo/nsgaii", "Demo trade-off", "NSGA-II demo", 0, nsgaii_front, points_x, metadata={"demo": True}),
        RunRecord("demo", "demo/moead", "Demo trade-off", "MOEA/D demo", 1, moead_front, points_x, metadata={"demo": True}),
    ]
    demo_fronts = [
        FrontRecord("Demo trade-off", "NSGA-II demo", nsgaii_front, points_x, extra={"demo": True, "seeds": [0], "config": None}),
        FrontRecord("Demo trade-off", "MOEA/D demo", moead_front, points_x, extra={"demo": True, "seeds": [1], "config": None}),
    ]
    return demo_runs, demo_fronts


def build_decision_views(
    fronts: list[FrontRecord],
    weights: NDArray[Any],
    reference_point: NDArray[Any] | None,
    method: str,
) -> list[DecisionView]:
    return [
        build_decision_view(front, weights=weights, reference_point=reference_point, methods=[method, "weighted_sum", "knee"])
        for front in fronts
    ]


def _with_result_mode(cfg_data: AlgorithmConfigProtocol, result_mode: str) -> AlgorithmConfigProtocol:
    fields_map = getattr(cfg_data, "__dataclass_fields__", None)
    if not isinstance(fields_map, dict) or "result_mode" not in fields_map:
        return cfg_data
    return cast(AlgorithmConfigProtocol, replace(cast(Any, cfg_data), result_mode=result_mode))


def _build_algorithm_config(
    algorithm: str,
    *,
    pop_size: int | None,
    n_var: int | None,
    n_obj: int | None,
    encoding: str | None = None,
) -> AlgorithmConfigProtocol:
    algorithm = algorithm.lower()
    result_mode = "population"

    default_cfg = build_default_algorithm_config(
        algorithm,
        pop_size=pop_size,
        n_var=n_var,
        n_obj=n_obj,
        encoding=encoding,
    )
    if default_cfg is not None:
        return _with_result_mode(default_cfg, result_mode)

    registry = get_algorithms_registry()
    if algorithm in registry:
        base: dict[str, object] = {}
        if pop_size is not None:
            base["pop_size"] = pop_size
        if n_var is not None:
            base["n_var"] = n_var
        if n_obj is not None:
            base["n_obj"] = n_obj
        return GenericAlgorithmConfig(base)

    raise InvalidAlgorithmError(algorithm, available=sorted(registry.keys()))


def _run_algorithm(
    problem: ProblemProtocol,
    *,
    algorithm: str,
    algorithm_config: AlgorithmConfigProtocol,
    termination: tuple[str, object],
    seed: int,
    engine: str,
    eval_strategy: EvaluationBackend | str | None = None,
    live_viz: object | None = None,
) -> dict[str, Any]:
    cfg_dict = dict(algorithm_config.to_dict())
    if "engine" in cfg_dict:
        raise ValueError("engine must be configured via run arguments, not algorithm_config.")
    kernel = resolve_kernel(engine)

    if eval_strategy is not None:
        backend = resolve_eval_strategy(eval_strategy) if isinstance(eval_strategy, str) else eval_strategy
    else:
        backend_name = str(cfg_dict.get("eval_strategy", "serial"))
        backend = resolve_eval_strategy(backend_name)

    algo_ctor = resolve_algorithm(algorithm)
    algorithm_instance = algo_ctor(cfg_dict, kernel)
    result = algorithm_instance.run(
        problem=problem,
        termination=termination,
        seed=seed,
        eval_strategy=backend,
        live_viz=live_viz,
    )
    return dict(result)


def run_focused_optimization(
    problem: str,
    reference_point: NDArray[Any],
    algo: str,
    budget: int,
) -> tuple[NDArray[Any], NDArray[Any] | None]:
    selection = make_problem_selection(problem)
    instance = selection.instantiate()
    algo_name = algo or "nsgaii"
    algo_cfg = _build_algorithm_config(
        algo_name,
        pop_size=40,
        n_var=getattr(instance, "n_var", None),
        n_obj=getattr(instance, "n_obj", None),
        encoding=getattr(instance, "encoding", None),
    )
    result = _run_algorithm(
        instance,
        algorithm=algo_name,
        algorithm_config=algo_cfg,
        termination=("max_evaluations", budget),
        seed=0,
        engine="numpy",
    )
    F = result.get("F")
    if F is None:
        raise RuntimeError("Focused optimization returned no objectives.")

    scores = reference_point_scores(np.asarray(F), reference_point).scores
    order = np.argsort(scores)
    X = None
    if result.get("X") is not None:
        X = np.asarray(result["X"])[order]
    return np.asarray(F)[order], X


def run_with_history(
    problem_name: str,
    config: dict[str, Any],
    budget: int,
) -> tuple[dict[str, Any], list[NDArray[Any]]]:
    selection = make_problem_selection(problem_name)
    problem = selection.instantiate()
    algo_name = str(config.get("algorithm", DEFAULT_ALGORITHM))
    algo_cfg_raw = config.get("algorithm_config", {})

    def _coerce_algo_config(cfg: Any) -> AlgorithmConfigProtocol:
        if not cfg:
            return _build_algorithm_config(
                algo_name,
                pop_size=100,
                n_var=getattr(problem, "n_var", None),
                n_obj=getattr(problem, "n_obj", None),
                encoding=getattr(problem, "encoding", None),
            )
        if isinstance(cfg, AlgorithmConfigProtocol):
            return cfg
        if not isinstance(cfg, dict):
            raise TypeError("algorithm_config must be a config object or dict.")

        config_cls = _CONFIG_MAP.get(algo_name.lower())
        if config_cls is None:
            raise TypeError(f"Dict coercion not supported for algorithm '{algo_name}'.")
        return cast(AlgorithmConfigProtocol, config_cls.from_dict(cfg))

    algo_cfg = _coerce_algo_config(algo_cfg_raw)
    callback = DynamicsCallback()
    result = _run_algorithm(
        problem,
        algorithm=algo_name,
        algorithm_config=algo_cfg,
        termination=("max_evaluations", budget),
        seed=int(config.get("seed", 0)),
        engine=str(config.get("engine", "numpy")),
        live_viz=callback,
    )
    return result, callback.history


__all__ = [
    "DynamicsCallback",
    "build_decision_views",
    "build_demo_study_data",
    "discover_study_directories",
    "llm_generate_problem_code",
    "load_studio_data",
    "run_focused_optimization",
    "run_with_history",
]
