"""The pygmo NSGA-II baseline builds its objects with the arguments pygmo accepts."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from vamos.experiment.external import run_external
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import make_problem_selection

_UINT32_LIMIT = 2**32


def _check_seed(seed: int) -> None:
    # pygmo binds seeds as unsigned 32-bit integers; pybind11 rejects anything else.
    if not isinstance(seed, int) or not 0 <= seed < _UINT32_LIMIT:
        raise TypeError("__init__(): incompatible constructor arguments.")


def _fake_pygmo(calls: dict[str, Any]) -> types.ModuleType:
    module = types.ModuleType("pygmo")

    def nsga2(gen: int = 1, cr: float = 0.95, eta_c: float = 10.0, m: float = 0.01, eta_m: float = 50.0, seed: int = 0) -> dict[str, Any]:
        _check_seed(seed)
        calls["nsga2_seed"] = seed
        return {"gen": gen}

    def zdt(prob_id: int = 1, param: int = 30) -> dict[str, Any]:
        calls["zdt"] = (prob_id, param)
        return {"n_var": param}

    class _Population:
        def __init__(self, size: int) -> None:
            self._size = size

        def get_f(self) -> np.ndarray:
            return np.zeros((self._size, 2))

    def population(problem: Any, size: int = 0, seed: int = 0) -> _Population:
        _check_seed(seed)
        calls["population_seed"] = seed
        return _Population(size)

    class _Algorithm:
        def __init__(self, uda: Any) -> None:
            self.uda = uda

        def evolve(self, pop: _Population) -> _Population:
            return pop

    module.__dict__.update(nsga2=nsga2, zdt=zdt, population=population, algorithm=_Algorithm, problem=lambda base: base)
    return module


def _run_pygmo(monkeypatch: pytest.MonkeyPatch, seed: int) -> dict[str, Any]:
    calls: dict[str, Any] = {}
    monkeypatch.setitem(sys.modules, "pygmo", _fake_pygmo(calls))
    metrics = run_external(
        "pygmo_nsga2",
        make_problem_selection("zdt1", n_var=12),
        use_native_problem=True,
        config=ExperimentConfig(population_size=10, offspring_population_size=10, max_evaluations=30, seed=seed),
        make_metrics=lambda algorithm, engine, time_ms, evaluations, F: {"algorithm": algorithm, "F": F},
        print_banner=lambda *args, **kwargs: None,
        print_results=lambda *args, **kwargs: None,
    )
    assert metrics is not None
    assert metrics["F"].shape == (10, 2)
    return calls


def test_pygmo_native_problem_uses_pygmo_dimension_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _run_pygmo(monkeypatch, seed=2)

    assert calls["zdt"] == (1, 12)
    assert calls["nsga2_seed"] == 2
    assert calls["population_seed"] == 2


def test_pygmo_folds_wide_seeds_into_32_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    # The CLI draws 64-bit seeds when --seed is omitted.
    seed = (0x1234_5678 << 32) | 0x9ABC_DEF0
    calls = _run_pygmo(monkeypatch, seed=seed)

    assert calls["nsga2_seed"] == calls["population_seed"] == 0x1234_5678 ^ 0x9ABC_DEF0
    assert _run_pygmo(monkeypatch, seed=seed + 1)["nsga2_seed"] != calls["nsga2_seed"]
