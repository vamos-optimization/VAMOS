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
