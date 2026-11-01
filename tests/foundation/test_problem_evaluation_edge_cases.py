"""Tests for problem evaluation edge cases in the VAMOS framework.

Covers shape-handling logic in both ``Problem.evaluate()`` and
``FunctionalProblem.evaluate()`` for scalar, 1-D, and 2-D objective
returns, wrong input shapes, constraint edge cases, and encoding
validation through ``make_problem``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vamos.foundation.problem.base import Problem
from vamos.foundation.problem.builder import make_problem

# ======================================================================
# Minimal Problem subclasses used by the tests
# ======================================================================


class ScalarObjectiveProblem(Problem):
    """Single-objective problem whose objectives() returns a 1-D array."""

    def __init__(self) -> None:
        self.n_var = 2
        self.n_obj = 1
        self.xl = np.zeros(2)
        self.xu = np.ones(2)

    def objectives(self, X: np.ndarray) -> np.ndarray:
        return np.sum(X**2, axis=1)  # returns 1D


class TwoObjectiveProblem(Problem):
    """Multi-objective problem that returns a correct 2-D array."""

    def __init__(self) -> None:
        self.n_var = 3
        self.n_obj = 2
        self.xl = np.zeros(3)
        self.xu = np.ones(3)

    def objectives(self, X: np.ndarray) -> np.ndarray:
        f1 = np.sum(X**2, axis=1)
        f2 = np.sum((X - 1) ** 2, axis=1)
        return np.column_stack([f1, f2])


class ZeroDScalarProblem(Problem):
    """Problem that returns a 0-D numpy scalar from objectives().

    This simulates the edge case of evaluating a single solution on a
    single objective where the user accidentally reduces to a scalar.
    """

    def __init__(self) -> None:
        self.n_var = 2
        self.n_obj = 1
        self.xl = np.zeros(2)
        self.xu = np.ones(2)

    def objectives(self, X: np.ndarray) -> np.ndarray:
        # Single row -> sum returns a 0-D scalar
        return np.float64(np.sum(X**2))


class ConstrainedScalarProblem(Problem):
    """Problem with one constraint that returns a 0-D scalar."""

    n_constraints = 1

    def __init__(self) -> None:
        self.n_var = 2
        self.n_obj = 1
        self.xl = np.zeros(2)
        self.xu = np.ones(2)

    def objectives(self, X: np.ndarray) -> np.ndarray:
        return np.sum(X**2, axis=1)

    def constraints(self, X: np.ndarray) -> np.ndarray:
        # 0-D scalar: simulates single-solution single-constraint edge case
        return np.float64(np.sum(X) - 1.0)


class WrongShapeObjectiveProblem(Problem):
    """Problem whose objectives() returns the wrong shape."""

    def __init__(self) -> None:
        self.n_var = 2
        self.n_obj = 2
        self.xl = np.zeros(2)
        self.xu = np.ones(2)

    def objectives(self, X: np.ndarray) -> np.ndarray:
        # Returns (N, 3) instead of (N, 2) -- wrong column count
        return np.column_stack([X[:, 0], X[:, 1], X[:, 0] + X[:, 1]])


# ======================================================================
# Problem.evaluate() — 0-D scalar return
# ======================================================================


class TestProblemEvaluate0DScalar:
    """When objectives() returns a 0-D numpy scalar (single solution,
    single objective), evaluate() should reshape it to (1, 1)."""

    def test_0d_scalar_reshaped_to_1x1(self) -> None:
        problem = ZeroDScalarProblem()
        X = np.array([[0.3, 0.4]])  # single solution
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert "F" in out
        assert out["F"].shape == (1, 1)
        expected = 0.3**2 + 0.4**2
        np.testing.assert_allclose(out["F"][0, 0], expected)

    def test_0d_scalar_zero_input(self) -> None:
        problem = ZeroDScalarProblem()
        X = np.array([[0.0, 0.0]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert out["F"].shape == (1, 1)
        np.testing.assert_allclose(out["F"][0, 0], 0.0)


# ======================================================================
# Problem.evaluate() — correct 2-D return
# ======================================================================


class TestProblemEvaluateCorrect2D:
    """When objectives() returns a proper (N, n_obj) array, evaluate()
    should pass it through without issues."""

    def test_two_solutions_two_objectives(self) -> None:
        problem = TwoObjectiveProblem()
        X = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
        )
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert "F" in out
        assert out["F"].shape == (2, 2)
        # f1 = sum(x^2), f2 = sum((x-1)^2)
        np.testing.assert_allclose(
            out["F"][0, 0],
            0.1**2 + 0.2**2 + 0.3**2,
        )
        np.testing.assert_allclose(
            out["F"][0, 1],
            (0.1 - 1) ** 2 + (0.2 - 1) ** 2 + (0.3 - 1) ** 2,
        )

    def test_single_solution_two_objectives(self) -> None:
        problem = TwoObjectiveProblem()
        X = np.array([[0.5, 0.5, 0.5]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert out["F"].shape == (1, 2)

    def test_writes_into_preallocated_buffer(self) -> None:
        problem = TwoObjectiveProblem()
        X = np.array([[0.5, 0.5, 0.5]])
        F_buf = np.zeros((1, 2))
        out: dict[str, np.ndarray] = {"F": F_buf}
        problem.evaluate(X, out)

        # Must reuse the same buffer object
        assert out["F"] is F_buf
        assert not np.allclose(F_buf, 0.0)

    def test_many_solutions(self) -> None:
        problem = TwoObjectiveProblem()
        rng = np.random.default_rng(42)
        X = rng.random((50, 3))
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert out["F"].shape == (50, 2)


# ======================================================================
# Problem.evaluate() — wrong input shape (ValueError)
# ======================================================================


class TestProblemEvaluateWrongInputShape:
    """evaluate() must raise ValueError when X has the wrong shape."""

    def test_1d_input_raises(self) -> None:
        problem = ScalarObjectiveProblem()
        X = np.array([0.5, 0.5])  # 1-D instead of 2-D
        out: dict[str, np.ndarray] = {}
        with pytest.raises(ValueError, match="Expected decision matrix"):
            problem.evaluate(X, out)

    def test_wrong_n_var_raises(self) -> None:
        problem = ScalarObjectiveProblem()
        # Problem expects n_var=2 but we pass 3 columns
        X = np.array([[0.1, 0.2, 0.3]])
        out: dict[str, np.ndarray] = {}
        with pytest.raises(ValueError, match="Expected decision matrix"):
            problem.evaluate(X, out)

    def test_3d_input_raises(self) -> None:
        problem = ScalarObjectiveProblem()
        X = np.ones((2, 2, 2))  # 3-D tensor
        out: dict[str, np.ndarray] = {}
        with pytest.raises(ValueError, match="Expected decision matrix"):
            problem.evaluate(X, out)

    def test_empty_batch_wrong_cols_raises(self) -> None:
        problem = ScalarObjectiveProblem()
        X = np.empty((0, 5))  # 0 rows but wrong column count
        out: dict[str, np.ndarray] = {}
        with pytest.raises(ValueError, match="Expected decision matrix"):
            problem.evaluate(X, out)


# ======================================================================
# Problem.evaluate() — 1-D return from objectives (auto-reshape)
# ======================================================================


class TestProblemEvaluate1DAutoReshape:
    """When objectives() returns a 1-D array of length N (common for
    single-objective problems), evaluate() should reshape to (N, 1)."""

    def test_single_solution_1d(self) -> None:
        problem = ScalarObjectiveProblem()
        X = np.array([[0.3, 0.4]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert out["F"].shape == (1, 1)
        np.testing.assert_allclose(out["F"][0, 0], 0.3**2 + 0.4**2)

    def test_multiple_solutions_1d(self) -> None:
        problem = ScalarObjectiveProblem()
        X = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert out["F"].shape == (4, 1)
        np.testing.assert_allclose(out["F"][:, 0], [0.0, 1.0, 1.0, 2.0])

    def test_1d_reshape_boundary_values(self) -> None:
        """Verify auto-reshape at boundary (lower/upper bounds)."""
        problem = ScalarObjectiveProblem()
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert out["F"].shape == (2, 1)
        np.testing.assert_allclose(out["F"][0, 0], 0.0)
        np.testing.assert_allclose(out["F"][1, 0], 2.0)


# ======================================================================
# Problem.evaluate() — wrong objective output shape
# ======================================================================


class TestProblemEvaluateWrongObjectiveShape:
    """evaluate() must raise ValueError when objectives() produces a
    shape that cannot match (N, n_obj)."""

    def test_too_many_columns(self) -> None:
        problem = WrongShapeObjectiveProblem()
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        out: dict[str, np.ndarray] = {}
        with pytest.raises(ValueError, match="returned shape"):
            problem.evaluate(X, out)


# ======================================================================
# Problem.evaluate() — constraint edge cases
# ======================================================================


class TestProblemConstraintEdgeCases:
    """Edge cases for constraint evaluation in the base Problem class."""

    def test_0d_scalar_constraint_reshaped(self) -> None:
        """A 0-D scalar constraint return should be reshaped to (1, 1)."""
        problem = ConstrainedScalarProblem()
        X = np.array([[0.3, 0.4]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert "G" in out
        assert out["G"].shape == (1, 1)
        expected_g = (0.3 + 0.4) - 1.0
        np.testing.assert_allclose(out["G"][0, 0], expected_g)

    def test_constraint_preallocated_buffer(self) -> None:
        problem = ConstrainedScalarProblem()
        X = np.array([[0.3, 0.4]])
        G_buf = np.zeros((1, 1))
        out: dict[str, np.ndarray] = {"G": G_buf}
        problem.evaluate(X, out)

        assert out["G"] is G_buf

    def test_constraint_none_when_declared_raises(self) -> None:
        """If n_constraints > 0 but constraints() returns None, raise."""

        class BadConstraintProblem(Problem):
            n_constraints = 1

            def __init__(self) -> None:
                self.n_var = 2
                self.n_obj = 1
                self.xl = np.zeros(2)
                self.xu = np.ones(2)

            def objectives(self, X: np.ndarray) -> np.ndarray:
                return np.sum(X**2, axis=1)

            # constraints() not overridden -- returns None by default

        problem = BadConstraintProblem()
        X = np.array([[0.5, 0.5]])
        out: dict[str, np.ndarray] = {}
        with pytest.raises(ValueError, match="returned None"):
            problem.evaluate(X, out)


# ======================================================================
# FunctionalProblem via make_problem — scalar mode
# ======================================================================


class TestFunctionalProblemScalarMode:
    """FunctionalProblem in scalar (non-vectorized) mode."""

    def test_scalar_single_objective(self) -> None:
        problem = make_problem(
            lambda x: x[0] ** 2 + x[1] ** 2,
            n_var=2,
            n_obj=1,
            xl=0.0,
            xu=1.0,
            encoding="real",
        )
        X = np.array([[0.3, 0.4]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert out["F"].shape == (1, 1)
        np.testing.assert_allclose(out["F"][0, 0], 0.3**2 + 0.4**2)

    def test_scalar_multi_objective(self) -> None:
        problem = make_problem(
            lambda x: [x[0], 1 - x[0]],
            n_var=2,
            n_obj=2,
            bounds=[(0, 1), (0, 1)],
            encoding="real",
        )
        X = np.array([[0.3, 0.5], [0.7, 0.2]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert out["F"].shape == (2, 2)
        np.testing.assert_allclose(out["F"][0], [0.3, 0.7])
        np.testing.assert_allclose(out["F"][1], [0.7, 0.3])

    def test_scalar_mode_returns_pure_float(self) -> None:
        """A scalar function returning a plain Python float (0-D) should
        be handled correctly for a single solution."""
        problem = make_problem(
            lambda x: float(x[0] + x[1]),
            n_var=2,
            n_obj=1,
            xl=0.0,
            xu=1.0,
            encoding="real",
        )
        X = np.array([[0.25, 0.75]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert out["F"].shape == (1, 1)
        np.testing.assert_allclose(out["F"][0, 0], 1.0)

    def test_scalar_mode_multiple_solutions(self) -> None:
        problem = make_problem(
            lambda x: [x[0] ** 2, x[1] ** 2],
            n_var=2,
            n_obj=2,
            xl=0.0,
            xu=1.0,
            encoding="real",
        )
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert out["F"].shape == (3, 2)
        np.testing.assert_allclose(out["F"][0], [0.01, 0.04])
        np.testing.assert_allclose(out["F"][2], [0.25, 0.36])


# ======================================================================
# FunctionalProblem via make_problem — vectorized mode
# ======================================================================


class TestFunctionalProblemVectorizedMode:
    """FunctionalProblem in vectorized (batch) mode."""

    def test_vectorized_two_objectives(self) -> None:
        def vec_fn(X: np.ndarray) -> np.ndarray:
            f1 = np.sum(X**2, axis=1)
            f2 = np.sum((X - 1) ** 2, axis=1)
            return np.column_stack([f1, f2])

        problem = make_problem(
            vec_fn,
            n_var=3,
            n_obj=2,
            xl=0.0,
            xu=1.0,
            vectorized=True,
            encoding="real",
        )
        X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert out["F"].shape == (2, 2)
        np.testing.assert_allclose(out["F"][0], [0.0, 3.0])
        np.testing.assert_allclose(out["F"][1], [3.0, 0.0])

    def test_vectorized_single_objective_1d_return(self) -> None:
        """Vectorized function returning 1-D (N,) for single objective
        should be auto-reshaped to (N, 1)."""
        problem = make_problem(
            lambda X: np.sum(X**2, axis=1),  # returns (N,)
            n_var=2,
            n_obj=1,
            xl=0.0,
            xu=1.0,
            vectorized=True,
            encoding="real",
        )
        X = np.array([[0.3, 0.4], [0.5, 0.6]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert out["F"].shape == (2, 1)
        np.testing.assert_allclose(out["F"][0, 0], 0.3**2 + 0.4**2)
        np.testing.assert_allclose(out["F"][1, 0], 0.5**2 + 0.6**2)

    def test_vectorized_wrong_input_shape_raises(self) -> None:
        problem = make_problem(
            lambda X: X[:, 0],
            n_var=2,
            n_obj=1,
            xl=0.0,
            xu=1.0,
            vectorized=True,
            encoding="real",
        )
        X = np.array([0.5, 0.5])  # 1-D instead of 2-D
        out: dict[str, np.ndarray] = {}
        with pytest.raises(ValueError, match="Expected decision matrix"):
            problem.evaluate(X, out)

    def test_vectorized_preallocated_buffer(self) -> None:
        problem = make_problem(
            lambda X: np.column_stack([X[:, 0], X[:, 1]]),
            n_var=2,
            n_obj=2,
            xl=0.0,
            xu=1.0,
            vectorized=True,
            encoding="real",
        )
        X = np.array([[0.1, 0.2]])
        F_buf = np.zeros((1, 2))
        out: dict[str, np.ndarray] = {"F": F_buf}
        problem.evaluate(X, out)

        assert out["F"] is F_buf
        np.testing.assert_allclose(F_buf[0], [0.1, 0.2])


# ======================================================================
# FunctionalProblem — 0-D scalar constraint return
# ======================================================================


class TestFunctionalProblemScalarConstraint:
    """FunctionalProblem with a constraint function that returns a 0-D
    scalar (single solution, single constraint)."""

    def test_scalar_constraint_0d_reshaped(self) -> None:
        problem = make_problem(
            lambda x: [x[0] ** 2],
            n_var=2,
            n_obj=1,
            xl=0.0,
            xu=5.0,
            encoding="real",
            constraints=lambda x: x[0] + x[1] - 4.0,  # returns scalar
            n_constraints=1,
        )
        X = np.array([[1.0, 2.0]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert "G" in out
        assert out["G"].shape == (1, 1)
        np.testing.assert_allclose(out["G"][0, 0], -1.0)

    def test_scalar_constraint_multiple_solutions(self) -> None:
        problem = make_problem(
            lambda x: [x[0] ** 2],
            n_var=2,
            n_obj=1,
            xl=0.0,
            xu=5.0,
            encoding="real",
            constraints=lambda x: [x[0] + x[1] - 4.0],
            n_constraints=1,
        )
        X = np.array([[1.0, 2.0], [3.0, 3.0]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert out["G"].shape == (2, 1)
        np.testing.assert_allclose(out["G"][:, 0], [-1.0, 2.0])

    def test_vectorized_constraint_0d_scalar(self) -> None:
        """Vectorized constraint returning a 0-D scalar for one solution."""
        problem = make_problem(
            lambda X: X[:, 0].reshape(-1, 1),
            n_var=2,
            n_obj=1,
            xl=0.0,
            xu=5.0,
            vectorized=True,
            encoding="real",
            constraints=lambda X: np.float64(np.sum(X) - 4.0),
            n_constraints=1,
        )
        X = np.array([[1.0, 2.0]])
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)

        assert out["G"].shape == (1, 1)
        np.testing.assert_allclose(out["G"][0, 0], -1.0)

    def test_constraint_preallocated_buffer(self) -> None:
        problem = make_problem(
            lambda x: [x[0]],
            n_var=2,
            n_obj=1,
            xl=0.0,
            xu=5.0,
            encoding="real",
            constraints=lambda x: [x[0] + x[1] - 4.0],
            n_constraints=1,
        )
        X = np.array([[1.0, 2.0]])
        G_buf = np.zeros((1, 1))
        out: dict[str, np.ndarray] = {"G": G_buf}
        problem.evaluate(X, out)

        assert out["G"] is G_buf
        np.testing.assert_allclose(G_buf[0, 0], -1.0)


# ======================================================================
# make_problem with encoding="real" — parameter validation
# ======================================================================


class TestMakeProblemEncodingValidation:
    """Encoding parameter validation through make_problem."""

    def test_encoding_real_accepted(self) -> None:
        problem = make_problem(
            lambda x: [x[0]],
            n_var=1,
            n_obj=1,
            encoding="real",
        )
        assert problem.encoding == "real"

    def test_encoding_continuous_alias(self) -> None:
        """'continuous' should normalize to 'real'."""
        problem = make_problem(
            lambda x: [x[0]],
            n_var=1,
            n_obj=1,
            encoding="continuous",
        )
        assert problem.encoding == "real"

    def test_encoding_float_alias(self) -> None:
        """'float' should normalize to 'real'."""
        problem = make_problem(
            lambda x: [x[0]],
            n_var=1,
            n_obj=1,
            encoding="float",
        )
        assert problem.encoding == "real"

    def test_encoding_case_insensitive(self) -> None:
        problem = make_problem(
            lambda x: [x[0]],
            n_var=1,
            n_obj=1,
            encoding="REAL",
        )
        assert problem.encoding == "real"

    def test_encoding_with_whitespace(self) -> None:
        problem = make_problem(
            lambda x: [x[0]],
            n_var=1,
            n_obj=1,
            encoding="  real  ",
        )
        assert problem.encoding == "real"

    def test_encoding_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown encoding"):
            make_problem(
                lambda x: [x[0]],
                n_var=1,
                n_obj=1,
                encoding="quantum",
            )

    def test_encoding_empty_string_defaults_to_real(self) -> None:
        """An empty encoding string should fall back to the default."""
        problem = make_problem(
            lambda x: [x[0]],
            n_var=1,
            n_obj=1,
            encoding="",
        )
        assert problem.encoding == "real"

    def test_encoding_binary(self) -> None:
        problem = make_problem(
            lambda x: [x[0]],
            n_var=1,
            n_obj=1,
            encoding="binary",
        )
        assert problem.encoding == "binary"

    def test_encoding_integer(self) -> None:
        problem = make_problem(
            lambda x: [x[0]],
            n_var=1,
            n_obj=1,
            encoding="integer",
        )
        assert problem.encoding == "integer"

    def test_encoding_int_alias(self) -> None:
        problem = make_problem(
            lambda x: [x[0]],
            n_var=1,
            n_obj=1,
            encoding="int",
        )
        assert problem.encoding == "integer"

    def test_encoding_permutation(self) -> None:
        problem = make_problem(
            lambda x: [x[0]],
            n_var=1,
            n_obj=1,
            encoding="permutation",
        )
        assert problem.encoding == "permutation"

    def test_encoding_perm_alias(self) -> None:
        problem = make_problem(
            lambda x: [x[0]],
            n_var=1,
            n_obj=1,
            encoding="perm",
        )
        assert problem.encoding == "permutation"

    def test_encoding_mixed(self) -> None:
        problem = make_problem(
            lambda x: [x[0]],
            n_var=1,
            n_obj=1,
            encoding="mixed",
        )
        assert problem.encoding == "mixed"

    def test_encoding_is_required(self) -> None:
        """make_problem must require the encoding keyword argument."""
        with pytest.raises(TypeError, match="encoding"):
            make_problem(lambda x: [x[0]], n_var=1, n_obj=1)  # type: ignore[call-arg]


class _ProtocolProblemWithUndeclaredG:
    """A pymoo-style problem: it names its constraints n_ieq_constr, which VAMOS does not read."""

    n_var = 2
    n_obj = 2
    n_ieq_constr = 1
    xl = np.zeros(2)
    xu = np.ones(2)
    encoding = "real"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        out["F"] = X.copy()
        out["G"] = X[:, :1] - 0.5


class _ConstraintsWithoutDeclaration(Problem):
    def __init__(self) -> None:
        self.n_var = 2
        self.n_obj = 2
        self.xl = np.zeros(2)
        self.xu = np.ones(2)

    def objectives(self, X: np.ndarray) -> np.ndarray:
        return X.copy()

    def constraints(self, X: np.ndarray) -> np.ndarray:
        return X[:, :1] - 0.5


def test_undeclared_constraint_output_raises() -> None:
    from vamos.foundation.eval.population import evaluate_population_with_constraints
    from vamos.foundation.exceptions import EvaluationError

    X = np.full((3, 2), 0.25)
    with pytest.raises(EvaluationError, match=r"n_constraints.*n_ieq_constr"):
        evaluate_population_with_constraints(_ProtocolProblemWithUndeclaredG(), X)
    with pytest.raises(EvaluationError, match="n_constraints"):
        evaluate_population_with_constraints(_ConstraintsWithoutDeclaration(), X)


def test_non_finite_output_message_names_the_possible_causes() -> None:
    from vamos.foundation.eval.population import evaluate_population_with_constraints
    from vamos.foundation.exceptions import EvaluationError

    class NaNProblem(_ConstraintsWithoutDeclaration):
        def constraints(self, X: np.ndarray) -> None:
            return None

        def objectives(self, X: np.ndarray) -> np.ndarray:
            return np.full_like(X, np.nan)

    with pytest.raises(EvaluationError, match=r"finite.*penalty.*operator"):
        evaluate_population_with_constraints(NaNProblem(), np.full((2, 2), 0.5))
