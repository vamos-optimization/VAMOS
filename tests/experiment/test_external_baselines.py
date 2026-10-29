"""External NSGA-II baselines (pymoo, jMetalPy) run as their libraries intend."""

from __future__ import annotations

import random
import sys
from statistics import median
from typing import Any

import numpy as np
import pytest

from vamos.experiment.external import run_external
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.quality_indicators.hypervolume import compute_hypervolume


def _run_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, *extra: str) -> None:
    from vamos.experiment.cli.main import main

    argv = [
        "vamos",
        "--problem",
        "zdt1",
        "--population-size",
        "20",
        "--offspring-population-size",
        "20",
        "--seed",
        "1",
        "--output-root",
        str(tmp_path / "results"),
        *extra,
    ]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    main()


def _quiet_external(name: str, *, seed: int, max_evaluations: int, use_native_problem: bool = True) -> np.ndarray:
    captured: dict[str, np.ndarray] = {}

    def make_metrics(algorithm: str, engine: str, time_ms: float, evaluations: int, F: np.ndarray) -> dict[str, Any]:
        captured["F"] = F
        return {"algorithm": algorithm, "engine": engine, "F": F}

    metrics = run_external(
        name,
        make_problem_selection("zdt1", n_var=30),
        use_native_problem=use_native_problem,
        config=ExperimentConfig(population_size=100, offspring_population_size=100, max_evaluations=max_evaluations, seed=seed),
        make_metrics=make_metrics,
        print_banner=lambda *args, **kwargs: None,
        print_results=lambda *args, **kwargs: None,
    )
    assert metrics is not None, f"{name} was skipped"
    return captured["F"]


def test_pymoo_baseline_mutates_one_variable_on_average(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    pytest.importorskip("pymoo")
    import pymoo.algorithms.moo.nsga2  # noqa: F401  (builds NSGA2's own default PM before the patch)
    from pymoo.core.population import Population
    from pymoo.core.problem import Problem
    from pymoo.operators.mutation import pm as pm_module

    built: list[Any] = []

    class _RecordingPM(pm_module.PM):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(pm_module, "PM", _RecordingPM)
    _run_cli(monkeypatch, tmp_path, "--algorithm", "pymoo_nsga2", "--max-evaluations", "200")

    assert len(built) == 1
    mutation = built[0]
    n_var = 30
    # pymoo's ``prob`` is the chance of mutating an individual; ``prob_var`` is per variable.
    assert mutation.prob.value == pytest.approx(1.0)
    assert mutation.prob_var is not None
    assert mutation.prob_var.value == pytest.approx(1.0 / n_var)

    problem = Problem(n_var=n_var, n_obj=2, xl=np.zeros(n_var), xu=np.ones(n_var))
    X = np.random.default_rng(0).random((4000, n_var))
    mutated = mutation.do(problem, Population.new(X=X.copy()), random_state=np.random.default_rng(1)).get("X")
    changed_per_individual = float(np.mean(np.sum(mutated != X, axis=1)))
    assert changed_per_individual == pytest.approx(1.0, abs=0.1)


def test_jmetalpy_vamos_problem_wrapper_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    pytest.importorskip("jmetal")
    from jmetal.algorithm.multiobjective import nsgaii as nsgaii_module

    initial: list[list[float]] = []
    create_initial_solutions = nsgaii_module.NSGAII.create_initial_solutions

    def _recording_create_initial_solutions(self: Any) -> list[Any]:
        solutions = create_initial_solutions(self)
        initial.extend(list(solution.variables) for solution in solutions)
        return solutions

    monkeypatch.setattr(nsgaii_module.NSGAII, "create_initial_solutions", _recording_create_initial_solutions)
    _run_cli(
        monkeypatch,
        tmp_path,
        "--algorithm",
        "jmetalpy_nsga2",
        "--external-problem-source",
        "vamos",
        "--max-evaluations",
        "400",
    )

    X0 = np.asarray(initial, dtype=float)
    assert X0.shape == (20, 30)
    assert np.unique(X0, axis=0).shape[0] == 20, "the initial population collapsed to repeated points"
    assert X0.min() >= 0.0 and X0.max() <= 1.0


def test_jmetalpy_baseline_restores_global_random_state() -> None:
    pytest.importorskip("jmetal")
    random.seed(123)
    np.random.seed(456)
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    first = _quiet_external("jmetalpy_nsga2", seed=7, max_evaluations=300)

    assert random.getstate() == python_state, "the jMetalPy baseline left the global random stream reseeded"
    restored = np.random.get_state()
    assert restored[0] == numpy_state[0]
    assert np.array_equal(restored[1], numpy_state[1])
    assert restored[2:] == numpy_state[2:], "the jMetalPy baseline left the global NumPy stream reseeded"

    second = _quiet_external("jmetalpy_nsga2", seed=7, max_evaluations=300)
    assert np.array_equal(first, second), "the baseline seed no longer reproduces the run"


@pytest.mark.slow
def test_external_baselines_match_vamos_nsgaii_on_zdt1() -> None:
    pytest.importorskip("pymoo")
    pytest.importorskip("jmetal")
    import vamos

    seeds = range(5)
    budget = 10_000
    reference = [1.1, 1.1]

    def median_hv(fronts: list[np.ndarray]) -> float:
        return float(median(compute_hypervolume(F, reference) for F in fronts))

    vamos_hv = median_hv(
        [np.asarray(vamos.optimize("zdt1", algorithm="nsgaii", pop_size=100, max_evaluations=budget, seed=seed).F) for seed in seeds]
    )
    pymoo_hv = median_hv([_quiet_external("pymoo_nsga2", seed=seed, max_evaluations=budget) for seed in seeds])
    jmetal_hv = median_hv([_quiet_external("jmetalpy_nsga2", seed=seed, max_evaluations=budget) for seed in seeds])
    jmetal_vamos_problem_hv = median_hv(
        [_quiet_external("jmetalpy_nsga2", seed=seed, max_evaluations=budget, use_native_problem=False) for seed in seeds]
    )

    for label, hv in (("pymoo", pymoo_hv), ("jMetalPy", jmetal_hv), ("jMetalPy (VAMOS problem)", jmetal_vamos_problem_hv)):
        assert hv == pytest.approx(vamos_hv, abs=0.03), f"{label} median HV {hv:.4f} vs VAMOS NSGA-II {vamos_hv:.4f}"
