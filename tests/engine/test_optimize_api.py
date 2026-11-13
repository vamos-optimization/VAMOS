import numpy as np
import pytest

from vamos import OptimizationResult, StudyResult, make_problem, optimize
from vamos.engine.algorithm.config import MOEADConfig, NSGAIIConfig
from vamos.foundation.exceptions import ConfigurationError, InvalidAlgorithmError
from vamos.foundation.problem.binary import BinaryKnapsackProblem
from vamos.foundation.problem.tsp import TSPProblem
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _nsgaii_cfg():
    return (
        NSGAIIConfig.builder()
        .pop_size(6)
        .offspring_size(6)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .result_mode("population")
        .build()
    )


def test_optimize_explicit_algorithm_nsga2():
    problem = ZDT1Problem(n_var=6)
    cfg = _nsgaii_cfg()
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        max_evaluations=12,
        seed=1,
        engine="numpy",
    )
    assert isinstance(result, OptimizationResult)
    assert result.F.shape[1] == problem.n_obj
    assert result.X.shape[0] == cfg.pop_size


def test_optimize_explicit_algorithm_moead():
    problem = ZDT1Problem(n_var=6)
    cfg_data = (
        MOEADConfig.builder()
        .pop_size(8)
        .neighbor_size(3)
        .delta(0.9)
        .replace_limit(1)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .build()
    )
    result = optimize(
        problem,
        algorithm="moead",
        algorithm_config=cfg_data,
        max_evaluations=8,
        seed=2,
        engine="numpy",
    )
    assert result.F.shape[0] > 0
    assert result.F.shape[1] == problem.n_obj


def test_optimize_unknown_algorithm_errors():
    problem = ZDT1Problem(n_var=4)
    with pytest.raises(InvalidAlgorithmError, match="Unknown algorithm"):
        optimize(problem, algorithm="unknown_algo", max_evaluations=4, pop_size=6)


def test_optimize_resolves_pop_size_consistently() -> None:
    problem = ZDT1Problem(n_var=6)
    pop_size = 10
    max_evaluations = 12

    result_direct = optimize(
        problem,
        algorithm="nsgaii",
        max_evaluations=max_evaluations,
        pop_size=pop_size,
        seed=1,
        engine="numpy",
    )
    cfg = NSGAIIConfig.default(pop_size=pop_size, n_var=problem.n_var)
    result_cfg = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        max_evaluations=max_evaluations,
        seed=1,
        engine="numpy",
    )

    direct_defaults = result_direct.explain_defaults()
    cfg_defaults = result_cfg.explain_defaults()
    assert "resolved_spec" in direct_defaults
    assert "resolved_spec" in cfg_defaults
    assert direct_defaults["resolved_spec"]["population"]["initial_size"] == pop_size
    assert cfg_defaults["resolved_spec"]["population"]["initial_size"] == pop_size
    assert direct_defaults["resolved_spec"]["termination"]["config"]["max_evaluations"] == max_evaluations
    assert cfg_defaults["resolved_spec"]["termination"]["config"]["max_evaluations"] == max_evaluations


def test_optimize_accepts_max_evaluations() -> None:
    problem = ZDT1Problem(n_var=6)
    result = optimize(
        problem,
        algorithm="nsgaii",
        max_evaluations=12,
        pop_size=6,
        seed=1,
        engine="numpy",
    )
    defaults = result.explain_defaults()
    assert defaults["resolved_spec"]["termination"]["config"]["max_evaluations"] == 12


def test_optimize_records_backend_resolution_metadata() -> None:
    problem = ZDT1Problem(n_var=6)
    result = optimize(
        problem,
        algorithm="nsgaii",
        max_evaluations=12,
        pop_size=6,
        seed=1,
    )

    defaults = result.explain_defaults()
    assert defaults["resolved_spec"]["backend"]["kernel"]["resolution"]["name"] == "numpy"
    assert result.meta["engine_source"] == "default"
    assert result.meta["kernel_backend"] == "numpy"


def test_optimize_accepts_termination_without_algorithm_config() -> None:
    # VAMOS 1.1.0 raised a false TypeError for a keyword that optimize() declares.
    problem = ZDT1Problem(n_var=6)
    result = optimize(problem, algorithm="nsgaii", pop_size=6, termination=("max_evaluations", 12), seed=1, engine="numpy")
    assert result.meta["run_artifact_resolved_spec"]["termination"] is not None


def test_optimize_auto_rejects_single_objective_problem() -> None:
    problem = make_problem(
        lambda x: [x[0] ** 2],
        n_var=2,
        n_obj=1,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        encoding="real",
    )

    with pytest.raises(ConfigurationError, match="multi-objective"):
        optimize(problem, algorithm="auto", max_evaluations=12, pop_size=6, seed=1)


def test_optimize_permutation_problem_uses_encoding_defaults() -> None:
    problem = TSPProblem(n_cities=8)
    result = optimize(
        problem,
        algorithm="nsgaii",
        max_evaluations=16,
        pop_size=8,
        seed=1,
        engine="numpy",
    )
    assert isinstance(result, OptimizationResult)
    assert result.F is not None
    assert result.X is not None
    assert result.X.shape[1] == problem.n_var


@pytest.mark.parametrize(
    ("problem", "pop_size"),
    [
        (TSPProblem(n_cities=8), 8),
        (BinaryKnapsackProblem(n_var=16), 16),
    ],
)
def test_optimize_moead_non_real_problem_uses_encoding_defaults(problem, pop_size: int) -> None:
    result = optimize(
        problem,
        algorithm="moead",
        max_evaluations=2 * pop_size,
        pop_size=pop_size,
        seed=1,
        engine="numpy",
    )
    assert isinstance(result, OptimizationResult)
    assert result.F is not None
    assert result.X is not None
    assert result.X.shape[1] == problem.n_var


def test_optimize_multi_seed_returns_study_result() -> None:
    problem = ZDT1Problem(n_var=6)

    result = optimize(
        problem,
        algorithm="nsgaii",
        max_evaluations=12,
        pop_size=6,
        seed=[1, 2, 3],
        engine="numpy",
    )

    assert isinstance(result, StudyResult)
    assert len(result) == 3
    assert all(isinstance(run, OptimizationResult) for run in result)
    assert np.all(result.metric_values("evaluations") == 12)
    assert result.mean("evaluations") == pytest.approx(12.0)
    assert result.std("evaluations") == pytest.approx(0.0)
    assert result.best_run("evaluations").meta["seed"] == 1


def test_optimize_multi_seed_study_result_is_sliceable() -> None:
    problem = ZDT1Problem(n_var=6)
    result = optimize(
        problem,
        algorithm="nsgaii",
        max_evaluations=12,
        pop_size=6,
        seed=[1, 2, 3],
        engine="numpy",
    )

    sliced = result[:2]

    assert isinstance(sliced, StudyResult)
    assert len(sliced) == 2
