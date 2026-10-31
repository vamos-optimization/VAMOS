"""Tests for probability expression parsing."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import vamos
from vamos.engine.algorithm.config.defaults import build_default_algorithm_config
from vamos.engine.variation.helpers import prepare_mutation_params, resolve_prob_expression

N_VAR = 30

VALID = [
    ("1/n", 1 / N_VAR),
    ("1/N", 1 / N_VAR),
    (" 1 / n ", 1 / N_VAR),
    ("2/n", 2 / N_VAR),
    (0.5, 0.5),
    ("0.5", 0.5),
    (0, 0.0),
    (1, 1.0),
]
INVALID = ["abc", "0,5", "1/nvar", 1.7, -0.2, float("nan"), True]


@pytest.mark.parametrize(("value", "expected"), VALID)
def test_valid_probability_expressions(value, expected) -> None:
    assert resolve_prob_expression(value, N_VAR, default=0.1) == pytest.approx(expected)


@pytest.mark.parametrize("value", INVALID)
def test_invalid_probability_expressions_are_rejected(value) -> None:
    with pytest.raises(ValueError, match="probability"):
        resolve_prob_expression(value, N_VAR, default=0.1)


def test_missing_probability_uses_the_default_and_expressions_are_capped() -> None:
    assert resolve_prob_expression(None, N_VAR, default=0.2) == 0.2
    assert resolve_prob_expression("2/n", 1, default=0.2) == 1.0


def _front_with_mutation_prob(algorithm: str, prob: object) -> np.ndarray:
    """Objectives after two generations on ZDT1 with a polynomial mutation of probability ``prob``."""
    problem = vamos.make_problem_selection("zdt1", n_var=N_VAR).instantiate()
    config = build_default_algorithm_config(algorithm, pop_size=20, n_var=N_VAR, n_obj=2, encoding="real")
    config = replace(config, mutation=("pm", {"prob": prob, "eta": 20.0}))
    result = vamos.optimize(problem, algorithm=algorithm, algorithm_config=config, max_evaluations=3 * config.pop_size, seed=1)
    return np.asarray(result.F)


@pytest.mark.parametrize("algorithm", ["nsgaii", "moead"])
def test_probability_expressions_parse_identically_for_every_algorithm(algorithm) -> None:
    for value, expected in VALID[:5]:
        # An expression and its numeric value give the same run, bit for bit.
        assert np.array_equal(_front_with_mutation_prob(algorithm, value), _front_with_mutation_prob(algorithm, expected)), (
            algorithm,
            value,
        )
    for value in ("abc", "0,5", 1.7, -0.2):
        with pytest.raises(ValueError, match="probability"):
            _front_with_mutation_prob(algorithm, value)


def test_prob_factor_is_consumed_before_operator_construction() -> None:
    params = prepare_mutation_params({"prob_factor": 2.0, "eta": 20.0}, "real", 10)
    assert params == {"prob": pytest.approx(0.2), "eta": 20.0}
    assert prepare_mutation_params({"prob_factor": 2.0}, "real", 1)["prob"] == 1.0
    for factor in (0.0, -1.0):
        with pytest.raises(ValueError, match="prob_factor"):
            prepare_mutation_params({"prob_factor": factor}, "real", 10)

    config = replace(
        build_default_algorithm_config("nsgaii", pop_size=20, n_var=10, n_obj=2, encoding="real"),
        mutation=("pm", {"prob_factor": 2.0, "eta": 20.0}),
    )
    problem = vamos.make_problem_selection("zdt1", n_var=10).instantiate()
    result = vamos.optimize(problem, algorithm="nsgaii", algorithm_config=config, max_evaluations=40, seed=1)
    assert np.isfinite(result.F).all()
