import pytest

from vamos import optimize
from vamos.engine.algorithm.config import NSGAIIIConfig
from vamos.foundation.problem.dtlz import DTLZ2Problem
from vamos.foundation.quality_indicators.hypervolume import hypervolume


@pytest.mark.reference
def test_dtlz2_nsgaii_convergence():
    """
    Verify NSGA-III solves DTLZ2 (3 objectives).
    Baseline: HV > 0.67 (seed 42 reaches 0.7192; the front's maximum is 0.8074)
    Budget: 5,000 evals
    """
    # 3-objective problem
    problem = DTLZ2Problem(n_var=12, n_obj=3)

    # Reference point for 3-obj normalized space (slightly beyond ideal [1,1,1])
    ref_point = [1.1, 1.1, 1.1]

    # NSGA-III Config
    # Uses reference directions (Das-Dennis)
    # n_obj=3, p=12 => 91 points
    pop_size = 91

    algo_cfg = (
        NSGAIIIConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=0.9, eta=30.0)
        .mutation("polynomial", prob=1.0 / 12, eta=20.0)
        .selection("random")
        .reference_directions(divisions=12)
        .build()
    )

    result = optimize(
        problem,
        algorithm="nsgaiii",
        algorithm_config=algo_cfg,
        max_evaluations=5000,
        seed=42,
        engine="numpy",
    )

    hv = hypervolume(result.F, ref_point)
    assert hv > 0.67, f"DTLZ2 NSGA-III failed to converge. HV={hv:.4f} < 0.67"
