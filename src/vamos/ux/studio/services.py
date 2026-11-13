from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.config.defaults import build_algorithm_config
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.engine.algorithm.registry import resolve_algorithm
from vamos.foundation.eval import EvaluationBackend
from vamos.foundation.eval.backends import resolve_eval_strategy
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.types import ProblemProtocol
from vamos.study_artifacts import load_study
from vamos.ux.studio._studio_llm import llm_generate_problem_code
from vamos.ux.studio.data import build_fronts, load_runs_from_study
from vamos.ux.studio.study_search import study_roots

if TYPE_CHECKING:
    from vamos.ux.studio.data import FrontRecord, RunRecord


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
    """Verified canonical study roots for the UI picker, found by ``study_roots``; at most ``limit`` are read."""
    candidates: list[Path] = []
    for root in study_roots(base_dir.resolve()):
        if len(candidates) >= limit:
            break
        path = root.resolve()
        if path in candidates:
            continue
        try:
            load_study(path)
        except Exception:
            continue
        candidates.append(path)
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates


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


def _build_algorithm_config(
    algorithm: str,
    *,
    pop_size: int | None,
    n_var: int | None,
    n_obj: int | None,
    encoding: str | None = None,
) -> AlgorithmConfigProtocol:
    # Studio shows the final population of every algorithm, SMPSO's swarm included.
    return build_algorithm_config(algorithm, pop_size=pop_size, n_var=n_var, n_obj=n_obj, encoding=encoding, result_mode="population")


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

    algo_ctor = resolve_algorithm(algorithm)
    algorithm_instance = algo_ctor(cfg_dict, kernel)
    # A backend built here from a name belongs to this run; one the caller passes stays open.
    owns_backend = eval_strategy is None or isinstance(eval_strategy, str)
    if eval_strategy is not None:
        backend = resolve_eval_strategy(eval_strategy) if isinstance(eval_strategy, str) else eval_strategy
    else:
        backend = resolve_eval_strategy(str(cfg_dict.get("eval_strategy", "serial")))
    try:
        result = algorithm_instance.run(
            problem=problem,
            termination=termination,
            seed=seed,
            eval_strategy=backend,
            live_viz=live_viz,
        )
    finally:
        if owns_backend:
            backend.close()
    return dict(result)


__all__ = [
    "DynamicsCallback",
    "build_demo_study_data",
    "discover_study_directories",
    "llm_generate_problem_code",
    "load_studio_data",
]
