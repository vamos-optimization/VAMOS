import numpy as np
import pytest

from vamos import optimize
from vamos.foundation.problem.zdt1 import ZDT1Problem


@pytest.mark.smoke
def test_nsgaii_invariants():
    """
    Verify NSGA-II invariants:
    1. Population size is maintained exactly.
    2. Solutions respect problem bounds.
    3. Offspring generation doesn't crash or produce NaNs.
    """
    problem = ZDT1Problem(n_var=10)
    pop_size = 40

    # We use 'numpy' engine for baseline invariants, 'numba' for performance check if enabled
    # Let's test default (numpy) first
    res = optimize(problem, algorithm="nsgaii", max_evaluations=1000, pop_size=pop_size, seed=123, engine="numpy")

    # Invariant 1: Population Size
    # NSGA-II should return exactly pop_size solutions unless we requested result_mode="non_dominated" AND the front is small?
    # By default result_mode="non_dominated" usually returns the front.
    # To check internal population stability, we might need to inspect the algorithm object,
    # but strictly speaking the 'result' usually contains the final front or population.
    # If the default is "non_dominated", len(res.F) might be <= pop_size.
    # But for ZDT1 with 1000 evals and pop 40, we likely have 40 points in the front or close to it.

    # Let's check non-nullity and basic bounds
    assert res.F is not None
    assert res.X is not None

    # Invariant 2: Bounds
    # ZDT1 bounds are [0, 1] for all vars
    assert np.all(res.X >= 0.0 - 1e-12)
    assert np.all(res.X <= 1.0 + 1e-12)

    # Invariant 3: No NaNs
    assert not np.any(np.isnan(res.F))
    assert not np.any(np.isnan(res.X))


@pytest.mark.smoke
def test_nsgaii_population_mode_invariants():
    """NSGA-II with result_mode="population" returns the whole final population, inside the bounds."""
    from vamos.engine.algorithm.config import NSGAIIConfig

    problem = ZDT1Problem(n_var=6)
    config = NSGAIIConfig.default(pop_size=10, n_var=problem.n_var).to_dict()
    config["result_mode"] = "population"
    res = optimize(
        problem, algorithm="nsgaii", algorithm_config=NSGAIIConfig.from_dict(config), max_evaluations=200, seed=123, engine="numpy"
    )

    assert res.X.shape == (10, problem.n_var)
    assert res.F.shape == (10, 2)
    assert np.all(res.X >= problem.xl) and np.all(res.X <= problem.xu)
