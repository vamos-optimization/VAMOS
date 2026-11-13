import numpy as np
import pytest

from vamos import optimize
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.foundation.problem.zdt1 import ZDT1Problem


def test_optimize_reproducible_with_seed():
    problem = ZDT1Problem(n_var=8)
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(10)
        .offspring_size(10)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    )
    res1 = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("max_evaluations", 30),
        seed=42,
        engine="numpy",
    )
    res2 = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("max_evaluations", 30),
        seed=42,
        engine="numpy",
    )
    assert np.array_equal(res1.F, res2.F)


@pytest.mark.numba
def test_optimize_numba_reproducible_with_seed():
    problem = ZDT1Problem(n_var=8)
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(10)
        .offspring_size(10)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    )
    res1 = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("max_evaluations", 30),
        seed=42,
        engine="numba",
    )
    res2 = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("max_evaluations", 30),
        seed=42,
        engine="numba",
    )
    assert np.array_equal(res1.F, res2.F)


def _stochastic_components():
    from vamos.engine.algorithm.components.selection import (
        BoltzmannSelection,
        RandomSelection,
        RankingSelection,
        SUSSelection,
        TournamentSelection,
    )
    from vamos.engine.algorithm.components.subset_selection import select_top_k_kmeans
    from vamos.engine.operators.impl.real.initialize import (
        HaltonInitializer,
        LatinHypercubeInitializer,
        OppositionBasedInitializer,
        ScatterSearchInitializer,
        SobolInitializer,
    )
    from vamos.engine.operators.impl.real.mutation import LinkedPolynomialMutation, UniformMutation
    from vamos.engine.tuning.racing.param_space import ParamSpace, Real

    lower, upper = np.zeros(3), np.ones(3)
    X = np.full((4, 3), 0.5)
    F = np.random.default_rng(0).random((12, 2))
    components = {
        "TournamentSelection": lambda: TournamentSelection(2, lambda a, b: a - b),
        "RandomSelection": lambda: RandomSelection(),
        "BoltzmannSelection": lambda: BoltzmannSelection(),
        "RankingSelection": lambda: RankingSelection(),
        "SUSSelection": lambda: SUSSelection(),
        "LatinHypercubeInitializer": lambda: LatinHypercubeInitializer(4, lower, upper),
        "ScatterSearchInitializer": lambda: ScatterSearchInitializer(4, lower, upper),
        "SobolInitializer": lambda: SobolInitializer(4, lower, upper),
        "HaltonInitializer": lambda: HaltonInitializer(4, lower, upper),
        "OppositionBasedInitializer": lambda: OppositionBasedInitializer(4, lower, upper, problem=ZDT1Problem(n_var=3)),
        "UniformMutation": lambda: UniformMutation(1.0, lower=lower, upper=upper)(X),
        "LinkedPolynomialMutation": lambda: LinkedPolynomialMutation(1.0, lower=lower, upper=upper)(X),
        "ParamSpace.sample": lambda: ParamSpace(params={"x": Real("x", 0.0, 1.0)}).sample(),
        "select_top_k_kmeans": lambda: select_top_k_kmeans(F, 3),
    }
    try:
        from vamos.foundation.kernel.numba_backend import NumbaKernel
    except ImportError:  # pragma: no cover - numba is optional
        return components
    params = {"prob": 1.0, "eta": 20.0}
    components["NumbaKernel.sbx_crossover"] = lambda: NumbaKernel().sbx_crossover(np.full((4, 3), 0.5), params, None, 0.0, 1.0)
    components["NumbaKernel.polynomial_mutation"] = lambda: NumbaKernel().polynomial_mutation(X.copy(), params, None, 0.0, 1.0)
    return components


@pytest.mark.parametrize("name", sorted(_stochastic_components()))
def test_stochastic_components_require_a_generator(name: str) -> None:
    from vamos.foundation.exceptions import ConfigurationError

    # AGENTS.md: stochastic production paths receive and own an explicit numpy.random.Generator.
    with pytest.raises(ConfigurationError, match="numpy.random.Generator"):
        _stochastic_components()[name]()


def test_kmeans_top_k_is_reproducible() -> None:
    result = optimize("zdt1", algorithm="nsgaii", max_evaluations=400, pop_size=40, seed=2)

    first = result.top_k(5, method="kmeans")["indices"]
    second = result.top_k(5, method="kmeans")["indices"]

    np.testing.assert_array_equal(first, second)
