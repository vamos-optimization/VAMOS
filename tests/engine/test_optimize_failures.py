import numpy as np
import pytest

from vamos import optimize
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.foundation.exceptions import ConfigurationError, InvalidAlgorithmError
from vamos.foundation.problem.zdt1 import ZDT1Problem


def test_unknown_algorithm_name_errors():
    with pytest.raises(InvalidAlgorithmError, match="Unknown algorithm"):
        optimize(ZDT1Problem(n_var=4), algorithm="does_not_exist", max_evaluations=4, pop_size=4, seed=0)


def test_unknown_problem_selection_errors():
    with pytest.raises(Exception):
        optimize("nonexistent_problem", algorithm="nsgaii", max_evaluations=4, pop_size=4, seed=0)


def test_invalid_max_evaluations_errors():
    with pytest.raises(ConfigurationError, match="max_evaluations must be a positive integer"):
        optimize(ZDT1Problem(n_var=4), algorithm="nsgaii", max_evaluations=0, pop_size=4, seed=0)


def test_invalid_pop_size_errors():
    with pytest.raises(ConfigurationError, match="pop_size must be a positive integer"):
        optimize(ZDT1Problem(n_var=4), algorithm="nsgaii", max_evaluations=4, pop_size=-1, seed=0)


def test_invalid_algorithm_config_pop_size_errors():
    # The builder rejects a non-positive population as soon as it is set.
    with pytest.raises(ConfigurationError, match="pop_size must be a positive integer"):
        NSGAIIConfig.builder().pop_size(-4)


class _NeverEvaluated:
    """A duck-typed problem whose attributes the test overrides; evaluating it is a test failure."""

    encoding = "real"
    n_constraints = 0

    def __init__(self, **attributes: object) -> None:
        self.n_var = 2
        self.n_obj = 2
        self.xl = np.zeros(2)
        self.xu = np.ones(2)
        for name, value in attributes.items():
            if value is _MISSING:
                delattr(self, name)
            else:
                setattr(self, name, value)

    def evaluate(self, X, out):  # pragma: no cover - reaching it is the failure
        raise AssertionError("the invalid problem was evaluated")


_MISSING = object()


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        (None, "problem must be"),
        (123, "problem must be"),
        (_NeverEvaluated(xl=np.array([0.0, 2.0]), xu=np.array([1.0, 1.0])), r"xl must not exceed xu"),
        (_NeverEvaluated(xl=np.zeros(3)), r"xl must be a scalar or have shape \(2,\)"),
        (_NeverEvaluated(xu=np.array([1.0, np.inf])), "xu must be finite"),
        (_NeverEvaluated(n_obj=_MISSING), "n_obj"),
        (_NeverEvaluated(n_var=2.5), "n_var must be a positive integer"),
        (_NeverEvaluated(n_obj=0), "n_obj must be a positive integer"),
        (_NeverEvaluated(encoding="quantum"), "Unknown encoding"),
    ],
    ids=["none", "int", "xl-above-xu", "xl-length", "infinite-xu", "missing-n_obj", "float-n_var", "zero-n_obj", "bad-encoding"],
)
def test_optimize_rejects_invalid_problem_objects(problem: object, message: str):
    with pytest.raises(ConfigurationError, match=message):
        optimize(problem, algorithm="nsgaii", max_evaluations=8, pop_size=4, seed=0)


def test_invalid_eval_strategy_errors():
    with pytest.raises(ConfigurationError, match="eval_strategy must be one of"):
        optimize(
            ZDT1Problem(n_var=4),
            algorithm="nsgaii",
            max_evaluations=4,
            pop_size=4,
            seed=0,
            eval_strategy="parallel",
        )


def _single_objective_problem():
    from vamos import make_problem

    return make_problem(lambda x: [float(np.sum(x**2))], n_var=3, n_obj=1, xl=0.0, xu=1.0, encoding="real")


@pytest.mark.parametrize("algorithm", ["nsgaii", "ibea", "agemoea", "moead"])
def test_optimize_rejects_a_population_below_the_algorithm_minimum(algorithm: str):
    with pytest.raises(ConfigurationError, match=r"needs pop_size >= 2"):
        optimize(ZDT1Problem(n_var=4), algorithm=algorithm, max_evaluations=8, pop_size=1, seed=0)


@pytest.mark.parametrize("algorithm", ["smsemoa", "agemoea", "nsgaiii", "rvea"])
def test_optimize_rejects_fewer_objectives_than_the_algorithm_needs(algorithm: str):
    with pytest.raises(ConfigurationError, match=r"needs at least 2 objectives; the problem has 1"):
        optimize(_single_objective_problem(), algorithm=algorithm, max_evaluations=16, pop_size=8, seed=0)


def test_optimize_checks_the_population_of_an_algorithm_config():
    config = NSGAIIConfig.default(pop_size=4, n_var=4).to_dict()
    config["pop_size"] = 1
    with pytest.raises(ConfigurationError, match=r"needs pop_size >= 2"):
        optimize(ZDT1Problem(n_var=4), algorithm="nsgaii", algorithm_config=NSGAIIConfig.from_dict(config), max_evaluations=8, seed=0)


def test_algorithm_minimums_cover_every_enabled_algorithm():
    from vamos.engine.algorithm.catalog import ALGORITHM_MINIMUMS, ENABLED_ALGORITHMS

    assert sorted(ALGORITHM_MINIMUMS) == sorted(ENABLED_ALGORITHMS)
    for minimums in ALGORITHM_MINIMUMS.values():
        assert set(minimums) == {"pop_size", "n_obj"}
        assert all(isinstance(value, int) and value >= 1 for value in minimums.values())


@pytest.mark.parametrize("algorithm", ["spea2", "smsemoa", "smpso"])
def test_algorithms_without_a_tournament_still_run_one_individual(algorithm: str):
    result = optimize(ZDT1Problem(n_var=4), algorithm=algorithm, max_evaluations=8, pop_size=1, seed=0)
    assert result.F.shape[1] == 2
