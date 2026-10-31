import numpy as np
import pytest

from vamos import optimize
from vamos.engine.algorithm.config import AGEMOEAConfig
from vamos.foundation.problem.zdt1 import ZDT1Problem
from vamos.foundation.quality_indicators.hypervolume import hypervolume

# jMetal 7.9 AGEMOEABuilder with its default selection, seeds 0 to 29: population
# 100, SBX (probability 0.9, eta 15), polynomial mutation (1/n, eta 20), ZDT1
# with 30 variables and 10 000 evaluations, hypervolume with reference (1.1, 1.1).
JMETAL_ZDT1_HV_MEAN = 0.8545
JMETAL_ZDT1_HV_STD = 0.0040
JMETAL_RUNS = 30


@pytest.mark.reference
def test_agemoea_zdt1_hypervolume_matches_jmetal():
    """AGE-MOEA's mean ZDT1 hypervolume lies within two standard errors of jMetal's."""
    problem = ZDT1Problem(n_var=30)
    reference = np.array([1.1, 1.1])
    config = AGEMOEAConfig.default(pop_size=100, n_var=problem.n_var)
    values = []
    for seed in range(10):
        result = optimize(problem, algorithm="agemoea", algorithm_config=config, max_evaluations=10000, seed=seed, engine="numpy")
        values.append(hypervolume(result.F[np.all(result.F <= reference, axis=1)], reference))

    mean = float(np.mean(values))
    standard_error = float(np.sqrt(JMETAL_ZDT1_HV_STD**2 / JMETAL_RUNS + np.var(values, ddof=1) / len(values)))
    assert abs(mean - JMETAL_ZDT1_HV_MEAN) <= 2.0 * standard_error, (
        f"mean HV {mean:.4f} over {len(values)} seeds; jMetal 7.9 {JMETAL_ZDT1_HV_MEAN:.4f} (2 SE = {2.0 * standard_error:.4f})"
    )
