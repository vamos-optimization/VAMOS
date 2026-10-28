"""MOEA/D's compiled one-offspring step (numba engine) must reproduce the reference step bit for bit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("numba")

import vamos
from vamos.engine.algorithm.config import MOEADConfig
from vamos.engine.algorithm.moead import MOEAD, steady_state
from vamos.engine.operators.policies.moead import build_variation_operators, one_offspring_real_variation
from vamos.foundation.eval import EvaluationResult
from vamos.foundation.eval.backends import SerialEvalBackend
from vamos.foundation.kernel._numba_moead_steady_state import clip_row
from vamos.foundation.kernel.moead_kernels import AGG_MODIFIED_TCHEBYCHEFF, AGG_PBI, AGG_TCHEBYCHEFF, AGG_WEIGHTED_SUM
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.constrained_many import CDTLZProblem
from vamos.foundation.problem.dtlz import DTLZ2Problem
from vamos.foundation.problem.zdt1 import ZDT1Problem

pytestmark = pytest.mark.numba

# name -> (problem factory, population size); 3-objective problems use the 91-vector Das-Dennis lattice.
PROBLEMS: dict[str, tuple[Any, int]] = {
    "zdt1": (lambda: ZDT1Problem(n_var=30), 100),
    "dtlz2": (lambda: DTLZ2Problem(n_var=12, n_obj=3), 91),
    "c1dtlz1": (lambda: CDTLZProblem("c1dtlz1", n_var=7, n_obj=3), 91),
    "c3dtlz1": (lambda: CDTLZProblem("c3dtlz1", n_var=7, n_obj=3), 91),
}
CROSSOVERS = {
    "de": ("de", {"cr": 1.0, "f": 0.5}),
    "sbx": ("sbx", {"prob": 0.9, "eta": 20.0}),
    "sbx-all-vars": ("sbx", {"prob": 0.9, "eta": 15.0, "prob_var": 1.0}),
}
AGGREGATIONS = {
    "tchebycheff": ("tchebycheff", {}),
    "pbi": ("pbi", {"theta": 5.0}),
    "weighted_sum": ("weighted_sum", {}),
    "modified_tchebycheff": ("modified_tchebycheff", {"rho": 0.001}),
}


def _compiled_variation_available() -> bool:
    return NumbaKernel().variation_kernels() is not None


requires_compiled = pytest.mark.skipif(
    not _compiled_variation_available(), reason="compiled pow differs from np.power here; the reference step runs instead"
)


def _config(problem: str, **changes: Any) -> dict[str, Any]:
    factory, pop_size = PROBLEMS[problem]
    sample = factory()
    cfg = MOEADConfig.default(pop_size=pop_size, n_var=sample.n_var, n_obj=sample.n_obj).to_dict()
    cfg.update(changes)
    return cfg


def _disable_compiled_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(steady_state.SteadyStateStep, "build", classmethod(lambda cls, *args, **kwargs: None))


def _reference_calls_fail(algorithm: MOEAD) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the reference MOEA/D step ran instead of the compiled one")

    algorithm.ask = unexpected  # type: ignore[method-assign]
    algorithm.tell = unexpected  # type: ignore[method-assign]


class _CountingProblem:
    """Delegate to a problem and count evaluated rows."""

    def __init__(self, problem: Any) -> None:
        self._problem = problem
        self.n_var, self.n_obj, self.xl, self.xu = problem.n_var, problem.n_obj, problem.xl, problem.xu
        self.n_constraints = getattr(problem, "n_constraints", 0)
        self.encoding = getattr(problem, "encoding", "real")
        self.rows = 0

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        self.rows += X.shape[0]
        self._problem.evaluate(X, out)


def _run(cfg: dict[str, Any], problem: Any, kernel: Any, *, seed: int, evals: int, **kwargs: Any) -> tuple[MOEAD, dict[str, Any]]:
    algorithm = MOEAD(dict(cfg), kernel=kernel)
    termination = kwargs.pop("termination", ("max_evaluations", evals))
    return algorithm, algorithm.run(problem, termination=termination, seed=seed, **kwargs)


def _assert_same_run(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    for key in ("X", "F"):
        assert actual[key].tobytes() == expected[key].tobytes(), key
    for key in ("X", "F"):
        assert actual["population"][key].tobytes() == expected["population"][key].tobytes(), key
    assert actual["evaluations"] == expected["evaluations"]
    assert actual["hv_reached"] == expected["hv_reached"]
    left, right = expected["checkpoint"], actual["checkpoint"]
    assert right["rng_state"] == left["rng_state"]
    assert right["generation"] == left["generation"] and right["n_eval"] == left["n_eval"]
    for key in ("subproblem_order", "ideal"):
        assert np.asarray(right["extra"][key]).tobytes() == np.asarray(left["extra"][key]).tobytes(), key
    assert right["extra"]["subproblem_cursor"] == left["extra"]["subproblem_cursor"]
    if left["G"] is None:
        assert right["G"] is None
    else:
        assert right["G"].tobytes() == left["G"].tobytes()
    if "archive" in expected:
        for key in ("X", "F"):
            assert actual["archive"][key].tobytes() == expected["archive"][key].tobytes()


@requires_compiled
@pytest.mark.parametrize("seed", [1, 7])
@pytest.mark.parametrize("problem", ["zdt1", "dtlz2", "c1dtlz1"])
def test_default_steady_state_matches_numpy_engine_with_exact_budget(problem: str, seed: int) -> None:
    factory, _ = PROBLEMS[problem]
    cfg = _config(problem)
    reference_problem = _CountingProblem(factory())
    _, expected = _run(cfg, reference_problem, NumPyKernel(), seed=seed, evals=2003)

    algorithm = MOEAD(dict(cfg), kernel=NumbaKernel())
    _reference_calls_fail(algorithm)
    compiled_problem = _CountingProblem(factory())
    actual = algorithm.run(compiled_problem, termination=("max_evaluations", 2003), seed=seed)

    assert algorithm._steady is not None
    _assert_same_run(expected, actual)
    assert actual["evaluations"] == compiled_problem.rows == reference_problem.rows == 2003
    state = algorithm._st
    assert state is not None and state.generation == expected["checkpoint"]["generation"]
    pending = (state.pending_offspring, state.pending_active_indices, state.pending_parent_pairs, state.pending_use_neighbors)
    assert pending == (None, None, None, None)


@requires_compiled
@pytest.mark.parametrize("aggregation", sorted(AGGREGATIONS))
@pytest.mark.parametrize("crossover", sorted(CROSSOVERS))
@pytest.mark.parametrize("problem", ["zdt1", "c3dtlz1"])
def test_operators_and_aggregations_match_numpy_engine(problem: str, crossover: str, aggregation: str) -> None:
    factory, _ = PROBLEMS[problem]
    cfg = _config(problem, crossover=CROSSOVERS[crossover], aggregation=AGGREGATIONS[aggregation])
    _, expected = _run(cfg, factory(), NumPyKernel(), seed=3, evals=1000)
    algorithm = MOEAD(dict(cfg), kernel=NumbaKernel())
    _reference_calls_fail(algorithm)
    actual = algorithm.run(factory(), termination=("max_evaluations", 1000), seed=3)
    assert algorithm._steady is not None
    _assert_same_run(expected, actual)


class _Frames:
    """A live callback that is not a no-op, so MOEA/D reports every generation."""

    def __init__(self) -> None:
        self.frames: list[tuple[int, bytes, int]] = []

    def on_start(self, ctx: Any = None) -> None:
        return None

    def on_generation(self, generation: int, F: Any = None, X: Any = None, stats: Any = None) -> None:
        self.frames.append((generation, np.asarray(F).tobytes(), int((stats or {}).get("evals", -1))))

    def on_end(self, final_F: Any = None, final_stats: Any = None) -> None:
        return None


@requires_compiled
@pytest.mark.parametrize("problem", ["zdt1", "c1dtlz1"])
@pytest.mark.parametrize("crossover", ["de", "sbx"])
def test_observed_generations_match_numpy_engine(problem: str, crossover: str) -> None:
    factory, _ = PROBLEMS[problem]
    cfg = _config(problem, crossover=CROSSOVERS[crossover])
    expected_frames, actual_frames = _Frames(), _Frames()
    _, expected = _run(cfg, factory(), NumPyKernel(), seed=11, evals=500, live_viz=expected_frames)
    algorithm, actual = _run(cfg, factory(), NumbaKernel(), seed=11, evals=500, live_viz=actual_frames)
    assert algorithm._steady is not None
    assert len(actual_frames.frames) == len(expected_frames.frames) > 1
    assert actual_frames.frames == expected_frames.frames
    _assert_same_run(expected, actual)


@requires_compiled
@pytest.mark.parametrize("crossover", ["de", "sbx"])
def test_hv_termination_and_external_archive_match_reference_step(crossover: str, monkeypatch: pytest.MonkeyPatch) -> None:
    factory, _ = PROBLEMS["zdt1"]
    cfg = _config("zdt1", crossover=CROSSOVERS[crossover], external_archive={"capacity": 40, "pruning": "crowding"})
    hv = ("hv", {"target_value": 100.0, "reference_point": [11.0, 11.0], "max_evaluations": 3000})
    compiled_algorithm, actual = _run(cfg, factory(), NumbaKernel(), seed=5, evals=0, termination=hv)
    assert compiled_algorithm._steady is not None
    _disable_compiled_step(monkeypatch)
    reference_algorithm, expected = _run(cfg, factory(), NumbaKernel(), seed=5, evals=0, termination=hv)
    assert reference_algorithm._steady is None
    assert "archive" in expected
    assert expected["hv_reached"] and 100 < expected["evaluations"] < 3000
    _assert_same_run(expected, actual)


@requires_compiled
@pytest.mark.parametrize("crossover", ["de", "sbx"])
def test_ask_tell_interface_matches_numpy_engine(crossover: str) -> None:
    factory, _ = PROBLEMS["c1dtlz1"]
    cfg = _config("c1dtlz1", crossover=CROSSOVERS[crossover])
    problem = factory()
    algorithms = [MOEAD(dict(cfg), kernel=kernel) for kernel in (NumPyKernel(), NumbaKernel())]
    for algorithm in algorithms:
        algorithm._initialize_run(problem, ("max_evaluations", 400), seed=21)
    assert algorithms[0]._steady is None and algorithms[1]._steady is not None
    backend = SerialEvalBackend()
    for _ in range(400 - 91):
        children = [algorithm.ask() for algorithm in algorithms]
        assert children[1].tobytes() == children[0].tobytes()
        for algorithm, child in zip(algorithms, children):
            algorithm.tell(backend.evaluate(child, problem), problem)
    expected, actual = (algorithm._st for algorithm in algorithms)
    assert expected is not None and actual is not None
    for name in ("X", "F", "G", "ideal", "subproblem_order"):
        assert getattr(actual, name).tobytes() == getattr(expected, name).tobytes(), name
    assert actual.rng.bit_generator.state == expected.rng.bit_generator.state
    assert actual.n_eval == expected.n_eval == 400
    with pytest.raises(RuntimeError, match="no remaining evaluation budget"):
        algorithms[1].ask()


UNSUPPORTED: dict[str, dict[str, Any]] = {
    "genealogy": {"track_genealogy": True},
    "batch": {"batch_size": 5},
    "reflect-repair": {"repair": ("reflect", {})},
    "gaussian-mutation": {"mutation": ("gaussian", {"prob": 0.1, "sigma": 0.1})},
    "blx-crossover": {"crossover": ("blx_alpha", {"prob": 0.9, "alpha": 0.5})},
}


@pytest.mark.parametrize("case", sorted(UNSUPPORTED))
def test_unsupported_configurations_keep_reference_step(case: str) -> None:
    factory, _ = PROBLEMS["zdt1"]
    cfg = _config("zdt1", **UNSUPPORTED[case])
    numba_algorithm, actual = _run(cfg, factory(), NumbaKernel(), seed=2, evals=400)
    assert numba_algorithm._steady is None
    _, expected = _run(cfg, factory(), NumPyKernel(), seed=2, evals=400)
    _assert_same_run(expected, actual)


def test_numpy_engine_and_disabled_neighbourhood_kernel_keep_reference_step(monkeypatch: pytest.MonkeyPatch) -> None:
    factory, _ = PROBLEMS["zdt1"]
    cfg = _config("zdt1")
    numpy_algorithm, _ = _run(cfg, factory(), NumPyKernel(), seed=2, evals=200)
    assert numpy_algorithm._steady is None
    monkeypatch.setattr(steady_state, "get_update_neighborhood_numba", lambda: None)
    numba_algorithm, _ = _run(cfg, factory(), NumbaKernel(), seed=2, evals=200)
    assert numba_algorithm._steady is None


class _SignedZeroProblem:
    """Objectives hit 0.0 and -0.0, so the ideal point sees signed-zero ties."""

    n_var, n_obj, n_constraints, encoding = 6, 2, 0, "real"
    xl, xu = np.zeros(6), np.ones(6)

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        out["F"][:, 0] = np.where(X[:, 1] < 0.5, -0.0, 0.0)
        out["F"][:, 1] = 1.0 - X[:, 0]


class _OddResults:
    """Serial evaluation that returns float32 objectives and, once, a NaN (no validation)."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, X: np.ndarray, problem: Any) -> EvaluationResult:
        result = SerialEvalBackend().evaluate(X, problem)
        self.calls += 1
        F = result.F.copy()
        if self.calls == 301:
            F[0, 0] = np.nan
        return EvaluationResult(F=F.astype(np.float32) if self.calls % 3 == 0 else F, G=result.G)


@requires_compiled
def test_signed_zero_ties_and_unexpected_results_match_numpy_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {"numpy_ideal": 0, "reference_tell": 0}
    numpy_ideal = steady_state.SteadyStateStep._replace_with_numpy_ideal
    reference_tell = MOEAD.tell

    def counting_numpy_ideal(self: Any, *args: Any) -> None:
        counts["numpy_ideal"] += 1
        numpy_ideal(self, *args)

    def counting_tell(self: Any, *args: Any) -> Any:
        counts["reference_tell"] += 1
        return reference_tell(self, *args)

    monkeypatch.setattr(steady_state.SteadyStateStep, "_replace_with_numpy_ideal", counting_numpy_ideal)
    monkeypatch.setattr(MOEAD, "tell", counting_tell)

    cfg = MOEADConfig.default(pop_size=20, n_var=6, n_obj=2).to_dict()
    _, expected = _run(cfg, _SignedZeroProblem(), NumPyKernel(), seed=4, evals=600)
    counts.update(numpy_ideal=0, reference_tell=0)
    algorithm, actual = _run(cfg, _SignedZeroProblem(), NumbaKernel(), seed=4, evals=600)
    assert algorithm._steady is not None
    assert counts == {"numpy_ideal": counts["numpy_ideal"], "reference_tell": 0} and counts["numpy_ideal"] > 0
    _assert_same_run(expected, actual)

    factory, _ = PROBLEMS["zdt1"]
    cfg = _config("zdt1", aggregation=AGGREGATIONS["tchebycheff"])
    _, expected = _run(cfg, factory(), NumPyKernel(), seed=4, evals=900, eval_strategy=_OddResults())
    counts.update(numpy_ideal=0, reference_tell=0)
    algorithm, actual = _run(cfg, factory(), NumbaKernel(), seed=4, evals=900, eval_strategy=_OddResults())
    assert algorithm._steady is not None
    assert counts["reference_tell"] > 0 and counts["numpy_ideal"] > 0
    assert np.isnan(expected["checkpoint"]["extra"]["ideal"]).any()
    _assert_same_run(expected, actual)


def test_optimize_facade_uses_compiled_step_and_matches_numpy_engine() -> None:
    runs = [
        vamos.optimize("zdt1", algorithm="moead", pop_size=100, engine=engine, seed=9, max_evaluations=1500)
        for engine in ("numpy", "numba")
    ]
    assert runs[1].X.tobytes() == runs[0].X.tobytes()
    assert runs[1].F.tobytes() == runs[0].F.tobytes()
    assert runs[0].data["evaluations"] == runs[1].data["evaluations"] == 1500


def test_clip_row_matches_numpy_clip() -> None:
    values = np.array([-0.0, 0.0, -0.0, 0.0, 2.0, -1.0, np.nan, np.inf, -np.inf, 0.5, 1.0, 0.0, 3.0])
    lower = np.array([0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, np.nan])
    upper = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -0.0, 4.0])
    expected = np.clip(values, lower, upper)
    actual = values.copy()
    clip_row(actual, lower, upper)
    assert actual.tobytes() == expected.tobytes()
    rng = np.random.default_rng(0)
    for _ in range(50):
        values = rng.normal(size=40) * 2.0
        values[rng.random(40) < 0.2] = rng.choice([0.0, -0.0, 1.0, np.nan], size=1)
        lower, upper = np.minimum(0.0, -rng.random(40)), rng.random(40)
        lower[rng.random(40) < 0.3] = -0.0
        actual = values.copy()
        clip_row(actual, lower, upper)
        assert actual.tobytes() == np.clip(values, lower, upper).tobytes()


def test_variation_description_reads_the_built_operators() -> None:
    xl, xu = np.zeros(5), np.ones(5)
    kernels = NumbaKernel().variation_kernels()
    if kernels is None:
        pytest.skip("compiled variation kernels are disabled on this platform")
    rng = np.random.default_rng(0)
    cfg = {"crossover": ("de", {"cr": 0.7, "f": 0.4}), "mutation": ("pm", {"prob": 0.3, "eta": 15.0})}
    variation = one_offspring_real_variation(*build_variation_operators(cfg, "real", 5, xl, xu, rng, variation_kernels=kernels), 5)
    assert variation is not None
    assert (variation.crossover, variation.de_f, variation.de_cr, variation.pm_prob, variation.pm_eta) == ("de", 0.4, 0.7, 0.3, 15.0)
    assert np.array_equal(variation.clip_lower, xl) and np.array_equal(variation.pm_upper, xu)
    sbx_cfg = {"crossover": ("sbx", {"prob": 0.8, "eta": 12.0, "prob_var": 0.6}), "mutation": ("pm", {})}
    sbx = one_offspring_real_variation(*build_variation_operators(sbx_cfg, "real", 5, xl, xu, rng, variation_kernels=kernels), 5)
    assert sbx is not None and (sbx.crossover, sbx.sbx_prob, sbx.sbx_eta, sbx.sbx_prob_var) == ("sbx", 0.8, 12.0, 0.6)
    assert one_offspring_real_variation(*build_variation_operators(cfg, "real", 5, xl, xu, rng), 5) is None
    reflect = dict(cfg, repair=("reflect", {}))
    assert one_offspring_real_variation(*build_variation_operators(reflect, "real", 5, xl, xu, rng, variation_kernels=kernels), 5) is None


def test_backends_expose_the_compiled_step_only_for_numba() -> None:
    assert NumPyKernel().moead_steady_state_kernels() is None
    kernels = NumbaKernel().moead_steady_state_kernels()
    if NumbaKernel().variation_kernels() is None:
        assert kernels is None
    else:
        assert kernels is not None and kernels.name == "numba"
        workspace = kernels.workspace(4, replace_limit=2, aggregation_id=AGG_PBI)
        assert workspace.pairs.base is not None and workspace.active.base is not None
        assert np.shares_memory(workspace.sbx_draws, workspace.mutation_draws)
    from vamos.engine.algorithm.moead.aggregation import resolve_aggregation_spec

    ids = [resolve_aggregation_spec(name, {})[0] for name in ("tchebycheff", "weighted_sum", "pbi", "modified_tchebycheff")]
    assert ids == [AGG_TCHEBYCHEFF, AGG_WEIGHTED_SUM, AGG_PBI, AGG_MODIFIED_TCHEBYCHEFF]


def test_moead_package_reaches_compiled_code_only_through_the_backend() -> None:
    import ast
    from pathlib import Path

    import vamos.engine.algorithm.moead as package

    for path in Path(package.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                assert not name.startswith("numba"), (path.name, name)
                assert "._numba" not in name and not name.startswith("vamos.foundation.kernel.numba"), (path.name, name)
