"""Tests for AGE-MOEA and RVEA ask/tell interfaces."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vamos.engine.algorithm.agemoea import AGEMOEA
from vamos.engine.algorithm.config import AGEMOEAConfig, RVEAConfig
from vamos.engine.algorithm.rvea import RVEA
from vamos.foundation.eval.backends import SerialEvalBackend
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _IdentityProblem:
    def __init__(self, n_var: int = 4) -> None:
        self.n_var = n_var
        self.n_obj = 2
        self.xl = np.zeros(n_var, dtype=float)
        self.xu = np.ones(n_var, dtype=float)
        self.encoding = "continuous"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        out["F"] = np.asarray(X[:, :2], dtype=float).copy()


def _eval(X: np.ndarray, problem: object) -> np.ndarray:
    """Evaluate offspring using the serial backend."""
    return np.asarray(SerialEvalBackend().evaluate(X, problem).F, dtype=float)


def _agemoea_cfg(pop_size: int = 12) -> dict:
    return (
        AGEMOEAConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("polynomial", prob=0.1, eta=20.0)
        .build()
        .to_dict()
    )


def _rvea_cfg(pop_size: int = 6, n_partitions: int = 5) -> dict:
    return (
        RVEAConfig.builder()
        .pop_size(pop_size)
        .n_partitions(n_partitions)
        .alpha(2.0)
        .adapt_freq(0.1)
        .crossover("sbx", prob=1.0, eta=30.0)
        .mutation("polynomial", prob=0.1, eta=20.0)
        .build()
        .to_dict()
    )


# ---------------------------------------------------------------------------
# AGE-MOEA ask/tell tests
# ---------------------------------------------------------------------------


class TestAGEMOEAAskTell:
    def test_ask_tell_produces_valid_result(self) -> None:
        pop_size = 12
        cfg = _agemoea_cfg(pop_size)
        algo = AGEMOEA(cfg, NumPyKernel())
        problem = ZDT1Problem(n_var=6)

        algo.initialize(problem, ("max_evaluations", pop_size * 3), seed=42)

        while not algo.should_terminate():
            X = algo.ask()
            assert X.shape[1] == problem.n_var
            algo.tell(_eval(X, problem))

        result = algo.result()
        assert "X" in result and "F" in result
        assert result["F"].shape[1] == problem.n_obj
        assert result["evaluations"] == pop_size * 3
        assert "n_eval" not in result
        assert "population" in result

    def test_ask_tell_matches_run(self) -> None:
        pop_size = 12
        cfg = _agemoea_cfg(pop_size)
        problem = ZDT1Problem(n_var=6)

        # Run via batch
        result_run = AGEMOEA(cfg, NumPyKernel()).run(problem, termination=("max_evaluations", pop_size * 3), seed=7)

        # Run via ask/tell
        algo = AGEMOEA(cfg, NumPyKernel())
        algo.initialize(problem, ("max_evaluations", pop_size * 3), seed=7)
        while not algo.should_terminate():
            X = algo.ask()
            algo.tell(_eval(X, problem))
        result_at = algo.result()

        np.testing.assert_array_equal(result_run["F"], result_at["F"])
        np.testing.assert_array_equal(result_run["X"], result_at["X"])
        assert result_run["evaluations"] == result_at["evaluations"]

    def test_ask_before_initialize_raises(self) -> None:
        algo = AGEMOEA(_agemoea_cfg(), NumPyKernel())
        with pytest.raises(RuntimeError, match="not initialized"):
            algo.ask()

    def test_tell_before_ask_raises(self) -> None:
        algo = AGEMOEA(_agemoea_cfg(), NumPyKernel())
        problem = ZDT1Problem(n_var=6)
        algo.initialize(problem, ("max_evaluations", 100), seed=0)
        with pytest.raises(RuntimeError, match="No pending offspring"):
            algo.tell(np.zeros((12, 2)))

    def test_double_ask_raises(self) -> None:
        algo = AGEMOEA(_agemoea_cfg(), NumPyKernel())
        problem = ZDT1Problem(n_var=6)
        algo.initialize(problem, ("max_evaluations", 100), seed=0)
        algo.ask()
        with pytest.raises(RuntimeError, match="not yet consumed"):
            algo.ask()

    def test_tell_accepts_dict(self) -> None:
        algo = AGEMOEA(_agemoea_cfg(), NumPyKernel())
        problem = _IdentityProblem()
        algo.initialize(problem, ("max_evaluations", 100), seed=0)
        X = algo.ask()
        F = X[:, :2].copy()
        algo.tell({"F": F})
        assert algo.state is not None
        assert algo.state.generation == 1

    def test_tell_accepts_object_with_F(self) -> None:
        algo = AGEMOEA(_agemoea_cfg(), NumPyKernel())
        problem = _IdentityProblem()
        algo.initialize(problem, ("max_evaluations", 100), seed=0)
        X = algo.ask()
        F = X[:, :2].copy()
        algo.tell(SimpleNamespace(F=F))
        assert algo.state is not None

    def test_state_accessible(self) -> None:
        algo = AGEMOEA(_agemoea_cfg(), NumPyKernel())
        assert algo.state is None
        problem = ZDT1Problem(n_var=6)
        algo.initialize(problem, ("max_evaluations", 100), seed=0)
        assert algo.state is not None
        assert algo.state.pop_size == 12

    def test_n_gen_termination(self) -> None:
        pop_size = 12
        cfg = _agemoea_cfg(pop_size)
        algo = AGEMOEA(cfg, NumPyKernel())
        problem = _IdentityProblem()
        algo.initialize(problem, ("n_gen", 3), seed=0)
        assert algo.state is not None
        assert algo.state.max_evals == 3 * pop_size

    def test_numba_same_seed_reproducible(self) -> None:
        pop_size = 12
        cfg = _agemoea_cfg(pop_size)
        problem = ZDT1Problem(n_var=6)

        result_a = AGEMOEA(cfg, NumbaKernel()).run(problem, termination=("max_evaluations", pop_size * 3), seed=9)
        result_b = AGEMOEA(cfg, NumbaKernel()).run(problem, termination=("max_evaluations", pop_size * 3), seed=9)

        np.testing.assert_array_equal(result_a["F"], result_b["F"])
        np.testing.assert_array_equal(result_a["X"], result_b["X"])


# ---------------------------------------------------------------------------
# RVEA ask/tell tests
# ---------------------------------------------------------------------------


class TestRVEAAskTell:
    def test_ask_tell_produces_valid_result(self) -> None:
        pop_size = 6
        cfg = _rvea_cfg(pop_size, n_partitions=5)
        algo = RVEA(cfg, NumPyKernel())
        problem = ZDT1Problem(n_var=6)

        algo.initialize(problem, ("max_evaluations", pop_size * 3), seed=42)

        while not algo.should_terminate():
            X = algo.ask()
            assert X.shape[1] == problem.n_var
            algo.tell(_eval(X, problem))

        result = algo.result()
        assert "X" in result and "F" in result
        assert result["F"].shape[1] == problem.n_obj
        assert result["evaluations"] == pop_size * 3
        assert "n_eval" not in result
        assert "population" in result

    def test_ask_tell_matches_run(self) -> None:
        pop_size = 6
        cfg = _rvea_cfg(pop_size, n_partitions=5)
        problem = ZDT1Problem(n_var=6)

        # Run via batch
        result_run = RVEA(cfg, NumPyKernel()).run(problem, termination=("max_evaluations", pop_size * 3), seed=7)

        # Run via ask/tell
        algo = RVEA(cfg, NumPyKernel())
        algo.initialize(problem, ("max_evaluations", pop_size * 3), seed=7)
        while not algo.should_terminate():
            X = algo.ask()
            algo.tell(_eval(X, problem))
        result_at = algo.result()

        np.testing.assert_array_equal(result_run["F"], result_at["F"])
        np.testing.assert_array_equal(result_run["X"], result_at["X"])
        assert result_run["evaluations"] == result_at["evaluations"]

    def test_ask_before_initialize_raises(self) -> None:
        algo = RVEA(_rvea_cfg(), NumPyKernel())
        with pytest.raises(RuntimeError, match="not initialized"):
            algo.ask()

    def test_tell_before_ask_raises(self) -> None:
        algo = RVEA(_rvea_cfg(), NumPyKernel())
        problem = ZDT1Problem(n_var=6)
        algo.initialize(problem, ("max_evaluations", 100), seed=0)
        with pytest.raises(RuntimeError, match="No pending offspring"):
            algo.tell(np.zeros((6, 2)))

    def test_double_ask_raises(self) -> None:
        algo = RVEA(_rvea_cfg(), NumPyKernel())
        problem = ZDT1Problem(n_var=6)
        algo.initialize(problem, ("max_evaluations", 100), seed=0)
        algo.ask()
        with pytest.raises(RuntimeError, match="not yet consumed"):
            algo.ask()

    def test_tell_accepts_dict(self) -> None:
        algo = RVEA(_rvea_cfg(), NumPyKernel())
        problem = _IdentityProblem()
        algo.initialize(problem, ("max_evaluations", 100), seed=0)
        X = algo.ask()
        F = X[:, :2].copy()
        algo.tell({"F": F})
        assert algo.state is not None
        assert algo.state.generation == 2  # RVEA starts at 1

    def test_tell_accepts_object_with_F(self) -> None:
        algo = RVEA(_rvea_cfg(), NumPyKernel())
        problem = _IdentityProblem()
        algo.initialize(problem, ("max_evaluations", 100), seed=0)
        X = algo.ask()
        F = X[:, :2].copy()
        algo.tell(SimpleNamespace(F=F))
        assert algo.state is not None

    def test_state_accessible(self) -> None:
        algo = RVEA(_rvea_cfg(), NumPyKernel())
        assert algo.state is None
        problem = ZDT1Problem(n_var=6)
        algo.initialize(problem, ("max_evaluations", 100), seed=0)
        assert algo.state is not None
        assert algo.state.pop_size == 6

    def test_n_gen_termination(self) -> None:
        pop_size = 6
        cfg = _rvea_cfg(pop_size, n_partitions=5)
        algo = RVEA(cfg, NumPyKernel())
        problem = _IdentityProblem()
        algo.initialize(problem, ("n_gen", 5), seed=0)
        assert algo.state is not None
        assert algo.state.max_gen == 5

    def test_reference_vector_adaptation(self) -> None:
        """Verify periodic reference-vector adaptation runs without error."""
        pop_size = 6
        cfg = _rvea_cfg(pop_size, n_partitions=5)
        cfg["adapt_freq"] = 1.0  # trigger adaptation every generation
        algo = RVEA(cfg, NumPyKernel())
        problem = ZDT1Problem(n_var=6)

        algo.initialize(problem, ("max_evaluations", pop_size * 5), seed=42)
        while not algo.should_terminate():
            X = algo.ask()
            algo.tell(_eval(X, problem))

        result = algo.result()
        assert np.isfinite(result["F"]).all()

    def test_numba_same_seed_reproducible(self) -> None:
        pop_size = 6
        cfg = _rvea_cfg(pop_size, n_partitions=5)
        problem = ZDT1Problem(n_var=6)

        result_a = RVEA(cfg, NumbaKernel()).run(problem, termination=("max_evaluations", pop_size * 3), seed=9)
        result_b = RVEA(cfg, NumbaKernel()).run(problem, termination=("max_evaluations", pop_size * 3), seed=9)

        np.testing.assert_array_equal(result_a["F"], result_b["F"])
        np.testing.assert_array_equal(result_a["X"], result_b["X"])


# ---------------------------------------------------------------------------
# Probability expressions (review finding VR-021)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["agemoea", "rvea"])
@pytest.mark.parametrize(
    ("problem_name", "n_obj", "encoding", "operators"),
    [
        ("zdt1", 2, "real", {"mutation": ("pm", {"prob": "1/n", "eta": 20.0})}),
        ("dtlz2", 3, "real", {"mutation": ("pm", {"prob": "1/n", "eta": 20.0})}),
        ("bin_knapsack", 2, "binary", {"crossover": ("hux", {"prob": 0.9}), "mutation": ("bitflip", {"prob": "1/n"})}),
    ],
)
def test_agemoea_and_rvea_resolve_one_over_n(algorithm, problem_name, n_obj, encoding, operators) -> None:
    from dataclasses import replace

    import vamos
    from vamos.engine.algorithm.config.defaults import build_default_algorithm_config

    problem = vamos.make_problem_selection(problem_name).instantiate()
    config = build_default_algorithm_config(algorithm, pop_size=None, n_var=problem.n_var, n_obj=n_obj, encoding=encoding)
    config = replace(config, **operators)

    result = vamos.optimize(problem, algorithm=algorithm, algorithm_config=config, max_evaluations=3 * config.pop_size, seed=1)

    assert result.data["evaluations"] == 3 * config.pop_size
    assert np.isfinite(result.F).all()
