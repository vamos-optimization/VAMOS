"""Tests for the make_problem() convenience builder."""

from __future__ import annotations

import numpy as np
import pytest

from vamos.foundation.exceptions import BoundsError, ConfigurationError, ProblemDimensionError
from vamos.foundation.problem.builder import FunctionalProblem
from vamos.foundation.problem.builder import make_problem as _make_problem


def make_problem(*args: object, **kwargs: object) -> FunctionalProblem:
    """Test helper: keep test fixtures concise while make_problem requires encoding."""
    kwargs.setdefault("encoding", "real")
    return _make_problem(*args, **kwargs)


# ======================================================================
# Construction – happy paths
# ======================================================================


class TestMakeProblemConstruction:
    """Verify that make_problem creates a valid ProblemProtocol."""

    def test_minimal_with_bounds(self) -> None:
        problem = make_problem(
            lambda x: [x[0], 1 - x[0]],
            n_var=2,
            n_obj=2,
            bounds=[(0, 1), (0, 1)],
        )
        assert isinstance(problem, FunctionalProblem)
        assert problem.n_var == 2
        assert problem.n_obj == 2
        assert problem.encoding == "real"
        np.testing.assert_array_equal(problem.xl, [0.0, 0.0])
        np.testing.assert_array_equal(problem.xu, [1.0, 1.0])

    def test_scalar_xl_xu(self) -> None:
        problem = make_problem(
            lambda x: [x[0]],
            n_var=3,
            n_obj=1,
            xl=0.0,
            xu=5.0,
        )
        np.testing.assert_array_equal(problem.xl, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(problem.xu, [5.0, 5.0, 5.0])

    def test_array_xl_xu(self) -> None:
        problem = make_problem(
            lambda x: [x[0]],
            n_var=2,
            n_obj=1,
            xl=[0.0, -1.0],
            xu=[1.0, 2.0],
        )
        np.testing.assert_array_equal(problem.xl, [0.0, -1.0])
        np.testing.assert_array_equal(problem.xu, [1.0, 2.0])

    def test_defaults_xl_xu(self) -> None:
        """When neither bounds nor xl/xu given, defaults to [0, 1]."""
        problem = make_problem(
            lambda x: [x[0]],
            n_var=2,
            n_obj=1,
        )
        np.testing.assert_array_equal(problem.xl, [0.0, 0.0])
        np.testing.assert_array_equal(problem.xu, [1.0, 1.0])

    def test_named_function_uses_name(self) -> None:
        def my_cool_function(x: np.ndarray) -> list[float]:
            return [x[0]]

        problem = make_problem(my_cool_function, n_var=1, n_obj=1)
        assert problem.name == "my_cool_function"

    def test_lambda_gets_default_name(self) -> None:
        problem = make_problem(lambda x: [x[0]], n_var=1, n_obj=1)
        assert problem.name == "custom_problem"

    def test_explicit_name(self) -> None:
        problem = make_problem(lambda x: [x[0]], n_var=1, n_obj=1, name="my_problem")
        assert problem.name == "my_problem"

    def test_encoding_alias(self) -> None:
        problem = make_problem(lambda x: [x[0]], n_var=1, n_obj=1, encoding="continuous")
        assert problem.encoding == "real"

    def test_repr(self) -> None:
        problem = make_problem(lambda x: [x[0]], n_var=2, n_obj=1, name="test")
        r = repr(problem)
        assert "test" in r
        assert "n_var=2" in r
        assert "n_obj=1" in r


# ======================================================================
# Evaluation – scalar mode
# ======================================================================


class TestScalarEvaluation:
    """Scalar (non-vectorized) evaluation via auto-vectorization."""

    def test_basic_evaluation(self) -> None:
        problem = make_problem(
            lambda x: [x[0], 1.0 - x[0]],
            n_var=2,
            n_obj=2,
            bounds=[(0, 1), (0, 1)],
        )
        X = np.array([[0.3, 0.5], [0.7, 0.2]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert "F" in out
        assert out["F"].shape == (2, 2)
        np.testing.assert_allclose(out["F"][:, 0], [0.3, 0.7])
        np.testing.assert_allclose(out["F"][:, 1], [0.7, 0.3])

    def test_single_objective(self) -> None:
        problem = make_problem(
            lambda x: [sum(x)],
            n_var=3,
            n_obj=1,
            xl=0.0,
            xu=1.0,
        )
        X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert out["F"].shape == (2, 1)
        np.testing.assert_allclose(out["F"][:, 0], [0.6, 1.5])

    def test_writes_to_preallocated_buffer(self) -> None:
        problem = make_problem(
            lambda x: [x[0], 1 - x[0]],
            n_var=1,
            n_obj=2,
            xl=0.0,
            xu=1.0,
        )
        X = np.array([[0.5]])
        F = np.zeros((1, 2))
        out: dict[str, np.ndarray] = {"F": F}
        problem.evaluate(X, out)

        # Must have written to the same buffer
        assert out["F"] is F
        np.testing.assert_allclose(F[0], [0.5, 0.5])


# ======================================================================
# Evaluation – vectorized mode
# ======================================================================


class TestVectorizedEvaluation:
    """Vectorized (batch) evaluation."""

    def test_basic_vectorized(self) -> None:
        def vec_fn(X: np.ndarray) -> np.ndarray:
            return np.column_stack([X[:, 0], 1 - X[:, 0]])

        problem = make_problem(
            vec_fn,
            n_var=2,
            n_obj=2,
            bounds=[(0, 1), (0, 1)],
            vectorized=True,
        )

        X = np.array([[0.2, 0.0], [0.8, 0.0]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert out["F"].shape == (2, 2)
        np.testing.assert_allclose(out["F"][:, 0], [0.2, 0.8])
        np.testing.assert_allclose(out["F"][:, 1], [0.8, 0.2])

    def test_vectorized_flat_output_is_rejected(self) -> None:
        # np.hstack gives (N * n_obj,): reshaping it to (N, n_obj) would mix the objectives of different solutions.
        problem = make_problem(
            lambda X: np.hstack([X[:, 0], 1 - X[:, 0]]),
            n_var=2,
            n_obj=2,
            bounds=[(0, 1), (0, 1)],
            vectorized=True,
        )
        X = np.random.default_rng(0).random((10, 2))

        with pytest.raises(ValueError, match=r"returned shape \(20,\), expected \(10, 2\).*np\.column_stack"):
            problem.evaluate(X, {})

    def test_vectorized_flat_constraints_are_rejected(self) -> None:
        problem = make_problem(
            lambda X: np.column_stack([X[:, 0], X[:, 1]]),
            n_var=2,
            n_obj=2,
            bounds=[(0, 1), (0, 1)],
            vectorized=True,
            constraints=lambda X: np.hstack([X[:, 0] - 0.5, X[:, 1] - 0.5]),
            n_constraints=2,
        )
        X = np.random.default_rng(0).random((6, 2))

        with pytest.raises(ValueError, match=r"constraints fn returned shape \(12,\), expected \(6, 2\)"):
            problem.evaluate(X, {})

    def test_vectorized_one_dimensional_output_is_kept_when_unambiguous(self) -> None:
        single = make_problem(lambda X: X[:, 0] + X[:, 1], n_var=2, n_obj=1, xl=0.0, xu=1.0, vectorized=True)
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        out: dict[str, np.ndarray] = {}
        single.evaluate(X, out)
        np.testing.assert_allclose(out["F"], [[0.3], [0.7], [1.1]])

        one_row = make_problem(lambda X: np.array([X[0, 0], 1 - X[0, 0]]), n_var=1, n_obj=2, xl=0.0, xu=1.0, vectorized=True)
        out = {}
        one_row.evaluate(np.array([[0.25]]), out)
        np.testing.assert_allclose(out["F"], [[0.25, 0.75]])

    def test_problem_subclass_flat_objectives_are_rejected(self) -> None:
        from vamos.foundation.problem.base import Problem

        class FlatObjectives(Problem):
            def __init__(self) -> None:
                self.n_var = 2
                self.n_obj = 2
                self.xl = np.zeros(2)
                self.xu = np.ones(2)

            def objectives(self, X: np.ndarray) -> np.ndarray:
                return np.hstack([X[:, 0], 1 - X[:, 0]])

        with pytest.raises(ValueError, match=r"objectives\(\) returned shape \(8,\), expected \(4, 2\)"):
            FlatObjectives().evaluate(np.full((4, 2), 0.5), {})


# ======================================================================
# Constraints
# ======================================================================


class TestConstraints:
    """Constraint function handling."""

    def test_scalar_constraints(self) -> None:
        problem = make_problem(
            lambda x: [x[0], x[1]],
            n_var=2,
            n_obj=2,
            bounds=[(0, 5), (0, 5)],
            constraints=lambda x: [x[0] + x[1] - 4],
            n_constraints=1,
        )

        X = np.array([[1.0, 2.0], [3.0, 3.0]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert "F" in out
        assert "G" in out
        assert out["G"].shape == (2, 1)
        np.testing.assert_allclose(out["G"][:, 0], [-1.0, 2.0])

    def test_vectorized_constraints(self) -> None:
        problem = make_problem(
            lambda X: np.column_stack([X[:, 0], X[:, 1]]),
            n_var=2,
            n_obj=2,
            bounds=[(0, 5), (0, 5)],
            vectorized=True,
            constraints=lambda X: (X[:, 0] + X[:, 1] - 4).reshape(-1, 1),
            n_constraints=1,
        )

        X = np.array([[1.0, 2.0], [3.0, 3.0]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert out["G"].shape == (2, 1)
        np.testing.assert_allclose(out["G"][:, 0], [-1.0, 2.0])


# ======================================================================
# Validation errors
# ======================================================================


class TestValidation:
    """Input validation should raise helpful errors."""

    def test_fn_not_callable(self) -> None:
        with pytest.raises(ConfigurationError, match="callable"):
            make_problem("not_a_function", n_var=2, n_obj=2)  # type: ignore[arg-type]

    def test_n_var_zero(self) -> None:
        with pytest.raises(ProblemDimensionError, match="n_var"):
            make_problem(lambda x: [x[0]], n_var=0, n_obj=1)

    def test_n_obj_zero(self) -> None:
        with pytest.raises(ProblemDimensionError, match="n_obj"):
            make_problem(lambda x: [x[0]], n_var=1, n_obj=0)

    def test_bounds_and_xl_xu_exclusive(self) -> None:
        with pytest.raises(BoundsError, match="either"):
            make_problem(
                lambda x: [x[0]],
                n_var=1,
                n_obj=1,
                bounds=[(0, 1)],
                xl=0.0,
            )

    def test_bounds_length_mismatch(self) -> None:
        with pytest.raises(BoundsError, match="n_var=2"):
            make_problem(lambda x: [x[0]], n_var=2, n_obj=1, bounds=[(0, 1)])

    def test_bounds_inverted(self) -> None:
        with pytest.raises(BoundsError, match="lower bound"):
            make_problem(lambda x: [x[0]], n_var=1, n_obj=1, bounds=[(5, 0)])

    def test_xl_xu_inverted(self) -> None:
        with pytest.raises(BoundsError, match="Lower bounds must not exceed upper bounds"):
            make_problem(lambda x: [x[0]], n_var=2, n_obj=1, xl=[1.0, 0.0], xu=[0.0, 1.0])

    def test_bounds_malformed(self) -> None:
        with pytest.raises(BoundsError, match="pair"):
            make_problem(
                lambda x: [x[0]],
                n_var=1,
                n_obj=1,
                bounds=[(1,)],  # type: ignore[arg-type]
            )

    def test_constraints_not_callable(self) -> None:
        with pytest.raises(ConfigurationError, match="callable"):
            make_problem(
                lambda x: [x[0]],
                n_var=1,
                n_obj=1,
                constraints="bad",  # type: ignore[arg-type]
                n_constraints=1,
            )

    def test_xl_xu_broadcast_shape_error(self) -> None:
        with pytest.raises(BoundsError, match="broadcast xl/xu"):
            make_problem(lambda x: [x[0]], n_var=3, n_obj=1, xl=[0.0, 1.0], xu=1.0)

    def test_constraints_without_n_constraints(self) -> None:
        with pytest.raises(ProblemDimensionError, match="n_constraints"):
            make_problem(
                lambda x: [x[0]],
                n_var=1,
                n_obj=1,
                constraints=lambda x: [0],
                n_constraints=0,
            )

    def test_bad_encoding(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown encoding"):
            make_problem(lambda x: [x[0]], n_var=1, n_obj=1, encoding="quantum")

    def test_encoding_required(self) -> None:
        with pytest.raises(TypeError, match="encoding"):
            _make_problem(lambda x: [x[0]], n_var=1, n_obj=1)
