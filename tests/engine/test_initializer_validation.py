from __future__ import annotations

import numpy as np
import pytest

from vamos import make_problem
from vamos.engine.algorithm.components.population import initialize_population
from vamos.engine.operators.impl.permutation import (
    order_crossover,
    swap_mutation,
    validate_permutation_population,
)


def _init(raw: np.ndarray, encoding: str, *, xl: np.ndarray, xu: np.ndarray) -> np.ndarray:
    return initialize_population(
        pop_size=raw.shape[0],
        n_var=raw.shape[1],
        xl=xl,
        xu=xu,
        encoding=encoding,
        rng=np.random.default_rng(0),
        initializer=lambda: raw,
    )


def test_custom_initializer_preserves_real_dtype_contract() -> None:
    raw = np.array([[0, 1], [1, 0]], dtype=np.int64)

    X = _init(raw, "real", xl=np.zeros(2), xu=np.ones(2))

    assert np.issubdtype(X.dtype, np.floating)


def test_custom_initializer_validates_binary_values_and_dtype() -> None:
    raw = np.array([[0.0, 1.0], [1.0, 0.0]])

    X = _init(raw, "binary", xl=np.zeros(2), xu=np.ones(2))

    assert X.dtype == np.int8
    with pytest.raises(ValueError, match="0/1"):
        _init(np.array([[0, 2], [1, 0]]), "binary", xl=np.zeros(2), xu=np.ones(2))


def test_custom_initializer_validates_integer_values_and_bounds() -> None:
    xl = np.array([0, 1])
    xu = np.array([3, 4])

    X = _init(np.array([[0.0, 1.0], [3.0, 4.0]]), "integer", xl=xl, xu=xu)

    assert np.issubdtype(X.dtype, np.integer)
    with pytest.raises(ValueError, match="integer-valued"):
        _init(np.array([[0.5, 1.0], [3.0, 4.0]]), "integer", xl=xl, xu=xu)
    with pytest.raises(ValueError, match="bounds"):
        _init(np.array([[0, 5], [3, 4]]), "integer", xl=xl, xu=xu)


def test_custom_initializer_validates_permutation_domain() -> None:
    raw = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])

    X = _init(raw, "permutation", xl=np.zeros(3), xu=np.full(3, 2))

    assert np.issubdtype(X.dtype, np.integer)
    for invalid in (
        np.array([[1, 2, 3], [3, 2, 1]]),
        np.array([[0, 1, 1], [2, 1, 0]]),
        np.array([[0.0, 1.5, 2.0], [2.0, 1.0, 0.0]]),
    ):
        with pytest.raises(ValueError, match="permutation"):
            _init(invalid, "permutation", xl=np.zeros(3), xu=np.full(3, 2))


def test_permutation_operators_reject_invalid_labels_before_crossover() -> None:
    rng = np.random.default_rng(0)
    invalid = np.array([[1, 2, 3], [3, 2, 1]])

    with pytest.raises(ValueError, match="permutation"):
        validate_permutation_population(invalid)
    with pytest.raises(ValueError, match="permutation"):
        order_crossover(invalid, prob=1.0, rng=rng)
    with pytest.raises(ValueError, match="permutation"):
        swap_mutation(np.array([[0.0, 1.5, 2.0]]), prob=1.0, rng=rng)


@pytest.mark.parametrize("encoding", ["binary", "integer", "permutation"])
def test_functional_problem_preserves_discrete_dtype(encoding: str) -> None:
    seen: list[np.dtype] = []

    def fn(x: np.ndarray) -> list[float]:
        seen.append(x.dtype)
        return [float(np.sum(x)), float(np.sum(x))]

    problem = make_problem(
        fn,
        n_var=3,
        n_obj=2,
        xl=0,
        xu=3,
        encoding=encoding,
    )
    out: dict[str, np.ndarray] = {}
    problem.evaluate(np.array([[0, 1, 2]], dtype=np.int64), out)

    assert np.issubdtype(seen[0], np.integer)
    assert np.issubdtype(out["F"].dtype, np.floating)


@pytest.mark.parametrize("problem_name", ["zdt5", "tsp6", "int_alloc", "mixed_design"])
@pytest.mark.parametrize("initializer", ["lhs", "scatter", "sobol", "halton", "obl"])
def test_real_initializers_reject_non_real_encodings(problem_name: str, initializer: str) -> None:
    from vamos.engine.algorithm.components.population import resolve_bounds
    from vamos.foundation.problem.registry import make_problem_selection

    problem = make_problem_selection(problem_name).instantiate()
    encoding = getattr(problem, "encoding", "real")
    xl, xu = resolve_bounds(problem, encoding)
    with pytest.raises(ValueError, match="real encoding"):
        initialize_population(
            pop_size=8,
            n_var=problem.n_var,
            xl=xl,
            xu=xu,
            encoding=encoding,
            rng=np.random.default_rng(0),
            problem=problem,
            initializer={"type": initializer},
        )


def test_binary_run_with_real_initializer_fails_before_evaluation() -> None:
    from vamos import optimize
    from vamos.engine.algorithm.config import NSGAIIConfig

    cfg = NSGAIIConfig.builder().pop_size(10).crossover("one_point", prob=0.9).mutation("bitflip", prob="1/n").initializer("lhs").build()
    with pytest.raises(ValueError, match="real encoding"):
        optimize("zdt5", algorithm="nsgaii", algorithm_config=cfg, max_evaluations=40, seed=1, engine="numpy")
