"""Tests for VAMOS exception hierarchy."""

from __future__ import annotations

import pytest


class TestVAMOSError:
    """Test base VAMOSError class."""

    def test_basic_error(self):
        """VAMOSError should work with just a message."""
        from vamos.foundation.exceptions import VAMOSError

        err = VAMOSError("Something went wrong")
        assert "Something went wrong" in str(err)
        assert err.message == "Something went wrong"
        assert err.suggestion is None

    def test_error_with_suggestion(self):
        """VAMOSError should include suggestion in message."""
        from vamos.foundation.exceptions import VAMOSError

        err = VAMOSError("Something went wrong", suggestion="Try this instead")
        assert "Something went wrong" in str(err)
        assert "Suggestion: Try this instead" in str(err)
        assert err.suggestion == "Try this instead"

    def test_error_with_details(self):
        """VAMOSError should store details."""
        from vamos.foundation.exceptions import VAMOSError

        err = VAMOSError("Error", details={"key": "value"})
        assert err.details == {"key": "value"}


class TestConfigurationErrors:
    """Test configuration-related errors."""

    def test_invalid_algorithm_error(self):
        """InvalidAlgorithmError should list available algorithms."""
        from vamos.foundation.exceptions import InvalidAlgorithmError

        err = InvalidAlgorithmError("unknown_algo")
        assert "unknown_algo" in str(err)
        assert "nsgaii" in str(err)  # Default available

    def test_invalid_algorithm_error_custom_available(self):
        """InvalidAlgorithmError should accept custom available list."""
        from vamos.foundation.exceptions import InvalidAlgorithmError

        err = InvalidAlgorithmError("bad", available=["algo1", "algo2"])
        assert "algo1" in str(err)
        assert "algo2" in str(err)

    def test_invalid_engine_error(self):
        """InvalidEngineError should suggest installation."""
        from vamos.foundation.exceptions import InvalidEngineError

        err = InvalidEngineError("bad_engine")
        assert "bad_engine" in str(err)
        assert "pip install" in str(err)

    def test_invalid_operator_error(self):
        """InvalidOperatorError should describe operator type."""
        from vamos.foundation.exceptions import InvalidOperatorError

        err = InvalidOperatorError("crossover", "bad_cx", available=["sbx", "blx"])
        assert "crossover" in str(err)
        assert "bad_cx" in str(err)
        assert "sbx" in str(err)

    def test_missing_config_error(self):
        """MissingConfigError should suggest fix."""
        from vamos.foundation.exceptions import MissingConfigError

        err = MissingConfigError("pop_size", config_class="NSGAIIConfig")
        assert "pop_size" in str(err)
        assert "NSGAIIConfig.default()" in str(err)


class TestProblemErrors:
    """Test problem-related errors."""

    def test_invalid_problem_error(self):
        """InvalidProblemError should suggest available_problem_names()."""
        from vamos.foundation.exceptions import InvalidProblemError

        err = InvalidProblemError("unknown_problem")
        assert "unknown_problem" in str(err)
        assert "available_problem_names()" in str(err)

    def test_invalid_problem_error_with_examples(self):
        """InvalidProblemError should show example problems."""
        from vamos.foundation.exceptions import InvalidProblemError

        err = InvalidProblemError("bad", available=["zdt1", "zdt2", "dtlz1"])
        assert "zdt1" in str(err)

    def test_problem_dimension_error(self):
        """ProblemDimensionError should include dimension info."""
        from vamos.foundation.exceptions import ProblemDimensionError

        err = ProblemDimensionError("Invalid dimensions", n_var=10, n_obj=2)
        assert "Invalid dimensions" in str(err)
        assert err.details["n_var"] == 10
        assert err.details["n_obj"] == 2

    def test_bounds_error(self):
        """BoundsError should suggest checking bounds."""
        from vamos.foundation.exceptions import BoundsError

        err = BoundsError("Lower bound exceeds upper bound")
        assert "Lower bound" in str(err)
        assert "xl <= xu" in str(err)


class TestOptimizationErrors:
    """Test runtime optimization errors."""

    def test_convergence_error(self):
        """ConvergenceError should suggest increasing evaluations."""
        from vamos.foundation.exceptions import ConvergenceError

        err = ConvergenceError("Failed to converge", evaluations=1000)
        assert "converge" in str(err)
        assert "max_evaluations" in str(err)
        assert err.details["evaluations"] == 1000

    def test_evaluation_error(self):
        """EvaluationError should suggest checking evaluate function."""
        from vamos.foundation.exceptions import EvaluationError

        err = EvaluationError("NaN in objectives")
        assert "NaN" in str(err)
        assert "evaluate()" in str(err)

    def test_constraint_violation_error(self):
        """ConstraintViolationError should suggest checking constraints."""
        from vamos.foundation.exceptions import ConstraintViolationError

        err = ConstraintViolationError("All solutions infeasible")
        assert "infeasible" in str(err)
        assert "constraint" in str(err).lower()


class TestDataErrors:
    """Test data/IO errors."""

    def test_results_not_found_error(self):
        """ResultsNotFoundError should include path."""
        from vamos.foundation.exceptions import ResultsNotFoundError

        err = ResultsNotFoundError("/path/to/results")
        assert "/path/to/results" in str(err)
        assert "not found" in str(err).lower()

    def test_invalid_results_error(self):
        """InvalidResultsError should suggest re-running."""
        from vamos.foundation.exceptions import InvalidResultsError

        err = InvalidResultsError("Corrupted data", path="/some/path")
        assert "Corrupted" in str(err)
        assert "re-running" in str(err)

    def test_result_selection_error_is_value_error_compatible(self):
        """ResultSelectionError should remain ValueError-compatible for callers."""
        from vamos.foundation.exceptions import ResultSelectionError

        err = ResultSelectionError("Unknown method")
        assert isinstance(err, ValueError)
        assert "Unknown method" in str(err)

    def test_no_solutions_error(self):
        """NoSolutionsError should provide a user-facing suggestion."""
        from vamos.foundation.exceptions import NoSolutionsError

        err = NoSolutionsError()
        assert "No solutions" in str(err)
        assert "Run the optimization longer" in str(err)


class TestDependencyErrors:
    """Test dependency errors."""

    def test_dependency_error(self):
        """DependencyError should suggest installation command."""
        from vamos.foundation.exceptions import DependencyError

        err = DependencyError("pandas", "to_dataframe()", "pip install pandas")
        assert "pandas" in str(err)
        assert "to_dataframe()" in str(err)
        assert "pip install pandas" in str(err)

    def test_backend_not_available_error(self):
        """BackendNotAvailableError should suggest vamos[compute]."""
        from vamos.foundation.exceptions import BackendNotAvailableError

        err = BackendNotAvailableError("numba")
        assert "numba" in str(err)
        assert "vamos[compute]" in str(err)


class TestExceptionHierarchy:
    """Test exception inheritance."""

    def test_all_inherit_from_vamos_error(self):
        """All custom exceptions should inherit from VAMOSError."""
        from vamos.foundation.exceptions import (
            ConfigurationError,
            DataError,
            DependencyError,
            InvalidAlgorithmError,
            NoSolutionsError,
            OptimizationError,
            ProblemError,
            ResultSelectionError,
            VAMOSError,
        )

        assert issubclass(ConfigurationError, VAMOSError)
        assert issubclass(InvalidAlgorithmError, ConfigurationError)
        assert issubclass(ProblemError, VAMOSError)
        assert issubclass(OptimizationError, VAMOSError)
        assert issubclass(DataError, VAMOSError)
        assert issubclass(DependencyError, VAMOSError)
        assert issubclass(ResultSelectionError, DataError)
        assert issubclass(NoSolutionsError, ResultSelectionError)

    def test_catch_all_vamos_errors(self):
        """Should be able to catch all VAMOS errors with VAMOSError."""
        from vamos.foundation.exceptions import InvalidAlgorithmError, VAMOSError

        with pytest.raises(VAMOSError):
            raise InvalidAlgorithmError("test")


class TestExceptionUsage:
    """Test exceptions in actual usage."""

    @pytest.mark.smoke
    def test_optimize_invalid_algorithm(self):
        """optimize() should raise InvalidAlgorithmError."""
        from vamos import optimize
        from vamos.foundation.exceptions import InvalidAlgorithmError
        from vamos.foundation.problem.zdt1 import ZDT1Problem as ZDT1

        problem = ZDT1(n_var=10)
        with pytest.raises(InvalidAlgorithmError) as exc_info:
            optimize(problem, algorithm="invalid_algo", max_evaluations=100, pop_size=20)

        assert "invalid_algo" in str(exc_info.value)
        assert exc_info.value.suggestion is not None

    @pytest.mark.smoke
    def test_all_exceptions_importable(self):
        """All exceptions should be importable from foundation.exceptions."""
        from vamos.foundation.exceptions import (
            BackendNotAvailableError,
            BoundsError,
            ConfigurationError,
            ConstraintViolationError,
            ConvergenceError,
            DataError,
            DependencyError,
            EvaluationError,
            InvalidAlgorithmError,
            InvalidEngineError,
            InvalidOperatorError,
            InvalidProblemError,
            InvalidResultsError,
            MissingConfigError,
            NoSolutionsError,
            OptimizationError,
            ProblemDimensionError,
            ProblemError,
            ResultSelectionError,
            ResultsNotFoundError,
            VAMOSError,
        )

        # Just verify all are not None
        assert all(
            [
                VAMOSError,
                ConfigurationError,
                InvalidAlgorithmError,
                InvalidEngineError,
                InvalidOperatorError,
                MissingConfigError,
                ProblemError,
                InvalidProblemError,
                ProblemDimensionError,
                BoundsError,
                OptimizationError,
                ConvergenceError,
                EvaluationError,
                ConstraintViolationError,
                DataError,
                ResultsNotFoundError,
                InvalidResultsError,
                ResultSelectionError,
                NoSolutionsError,
                DependencyError,
                BackendNotAvailableError,
            ]
        )


def test_artifact_and_study_errors_are_vamos_errors():
    """Run-artifact and study errors share the VAMOSError base and the public namespace (VR-090)."""
    import vamos.exceptions as public
    from vamos.experiment.artifacts.errors import OutputCollisionError, RunArtifactError
    from vamos.experiment.study.errors import InvalidStudySpecError, StudyError

    assert issubclass(RunArtifactError, public.VAMOSError) and issubclass(StudyError, public.VAMOSError)
    assert public.RunArtifactError is RunArtifactError and public.StudyError is StudyError
    assert {"RunArtifactError", "StudyError"} <= set(public.__all__)

    error = OutputCollisionError(
        operation="save result", reason="already exists", expected="empty", actual="run", action="Choose another path."
    )
    assert (
        str(error)
        == error.message
        == "Cannot save result: run artifact already exists. Expected 'empty'; received 'run'. Choose another path."
    )
    assert error.suggestion is None and error.as_dict()["reason"] == "already exists"
    study_error = InvalidStudySpecError(operation="plan study", reason="bad", expected="a", actual="b", action="Fix it.")
    assert str(study_error) == study_error.message and study_error.as_dict()["category"] == "study_spec"


def test_unknown_problem_raises_invalid_problem_error():
    """An unknown problem name raises InvalidProblemError, still a KeyError (VR-090)."""
    from vamos import optimize
    from vamos.exceptions import InvalidProblemError
    from vamos.foundation.problem.registry import available_problem_names, make_problem_selection

    with pytest.raises(InvalidProblemError) as caught:
        make_problem_selection("zdt7")
    assert isinstance(caught.value, KeyError)
    assert str(caught.value).startswith("Unknown problem 'zdt7'.")
    assert "Did you mean one of: 'zdt6'" in str(caught.value)
    assert caught.value.details["available"] == sorted(available_problem_names())

    with pytest.raises(KeyError, match="Unknown problem 'zdt7'"):
        optimize("zdt7", max_evaluations=100)
