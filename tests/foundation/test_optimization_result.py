"""Tests for OptimizationResult enriched methods."""

from __future__ import annotations

import logging

import numpy as np
import pytest


def _has_matplotlib() -> bool:
    """Check if matplotlib is available."""
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


def _has_pandas() -> bool:
    """Check if pandas is available."""
    try:
        import pandas  # noqa: F401

        return True
    except ImportError:
        return False


class TestOptimizationResultBasics:
    """Test basic OptimizationResult functionality."""

    def test_result_len(self):
        """Result should support len()."""
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        result = OptimizationResult({"F": F, "X": None})

        assert len(result) == 3

    def test_result_repr(self):
        """Result should have informative repr."""
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.5, 0.5]])
        result = OptimizationResult({"F": F})

        repr_str = repr(result)
        assert "2 solutions" in repr_str
        assert "2 objectives" in repr_str

    def test_n_objectives(self):
        """n_objectives property should work."""
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9, 0.5], [0.5, 0.5, 0.5]])
        result = OptimizationResult({"F": F})

        assert result.n_objectives == 3


class TestOptimizationResultSummary:
    """Test summary() method."""

    def test_summary_runs(self, caplog):
        """summary() should log without error."""
        from vamos.experiment.optimization_result import OptimizationResult
        from vamos.ux.api import log_result_summary

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        result = OptimizationResult({"F": F})

        caplog.set_level(logging.INFO, logger="vamos.ux.results")
        log_result_summary(result)
        assert "Optimization Result" in caplog.text
        assert "Solutions: 3" in caplog.text
        assert "Objectives: 2" in caplog.text


class TestOptimizationResultDefaults:
    """Test explain_defaults() metadata helper."""

    def test_explain_defaults_returns_meta(self):
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.5, 0.5]])
        meta = {
            "run_artifact_resolved_spec": {"problem": {"component_id": "vamos.problem:zdt1@1"}},
            "default_sources": {"algorithm": "explicit", "max_evaluations": "auto"},
        }
        result = OptimizationResult({"F": F}, meta=meta)

        explained = result.explain_defaults()

        assert explained["resolved_spec"]["problem"]["component_id"] == "vamos.problem:zdt1@1"
        assert explained["default_sources"]["max_evaluations"] == "auto"

    def test_explain_defaults_handles_missing_meta(self):
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9]])
        result = OptimizationResult({"F": F})

        assert result.explain_defaults() == {}


class TestOptimizationResultBest:
    """Test best() method."""

    def test_best_knee(self):
        """best('knee') should return a solution."""
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        X = np.array([[1, 2], [3, 4], [5, 6]])
        result = OptimizationResult({"F": F, "X": X})

        with pytest.warns(FutureWarning, match="alias of 'balanced_sum'"):
            best = result.best("knee")

        assert "F" in best
        assert "X" in best
        assert "index" in best
        assert 0 <= best["index"] < 3

    def test_best_balanced_sum_is_default_name_for_normalized_sum(self):
        """best('balanced_sum') should match the default normalized-sum selector."""
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        result = OptimizationResult({"F": F})

        named = result.best("balanced_sum")
        default = result.best()

        assert named["index"] == default["index"]

    def test_best_knee_uses_pareto_front_scaling(self):
        """best('knee') should pick from the Pareto front with front-based normalization."""
        from vamos.experiment.optimization_result import OptimizationResult

        # Three non-dominated points plus an extreme dominated outlier.
        F = np.array(
            [
                [0.0, 100.0],
                [50.0, 0.0],
                [25.0, 25.0],
                [10000.0, 1.0],
            ]
        )
        X = np.arange(F.shape[0] * 2).reshape(F.shape[0], 2)
        result = OptimizationResult({"F": F, "X": X})

        with pytest.warns(FutureWarning):
            best = result.best("knee")

        assert best["index"] == 2
        assert np.allclose(best["F"], F[2])

    def test_best_min_f1(self):
        """best('min_f1') should return min first objective."""
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        result = OptimizationResult({"F": F})

        best = result.best("min_f1")

        assert best["F"][0] == 0.1
        assert best["index"] == 0

    def test_best_min_f2(self):
        """best('min_f2') should return min second objective."""
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        result = OptimizationResult({"F": F})

        best = result.best("min_f2")

        assert best["F"][1] == 0.1
        assert best["index"] == 2

    def test_best_balanced(self):
        """best('balanced') should return a solution."""
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        result = OptimizationResult({"F": F})

        best = result.best("balanced")

        assert "index" in best
        assert 0 <= best["index"] < 3

    def test_best_invalid_method(self):
        """best() with invalid method should raise."""
        from vamos.experiment.optimization_result import OptimizationResult
        from vamos.foundation.exceptions import ResultSelectionError

        F = np.array([[0.1, 0.9], [0.5, 0.5]])
        result = OptimizationResult({"F": F})

        with pytest.raises(ResultSelectionError, match="Unknown method"):
            result.best("invalid")


class TestOptimizationResultPlot:
    """Test plot() method."""

    @pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
    def test_plot_2d(self):
        """plot() should work for 2D fronts."""
        from vamos.experiment.optimization_result import OptimizationResult
        from vamos.ux.api import plot_result_front

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        result = OptimizationResult({"F": F})

        ax = plot_result_front(result, show=False)
        assert ax is not None

    @pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
    def test_plot_3d(self):
        """plot() should work for 3D fronts."""
        from vamos.experiment.optimization_result import OptimizationResult
        from vamos.ux.api import plot_result_front

        F = np.array([[0.1, 0.8, 0.1], [0.5, 0.3, 0.2], [0.8, 0.1, 0.1]])
        result = OptimizationResult({"F": F})

        ax = plot_result_front(result, show=False)
        assert ax is not None

    def test_plot_no_matplotlib(self):
        """plot() should raise if matplotlib not available."""
        # This is hard to test without mocking, skip for now
        pass


class TestOptimizationResultDataFrame:
    """Test to_dataframe() method."""

    @pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
    def test_to_dataframe_basic(self):
        """to_dataframe() should create valid DataFrame."""
        from vamos.experiment.optimization_result import OptimizationResult
        from vamos.ux.api import result_to_dataframe

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        X = np.array([[1, 2], [3, 4], [5, 6]])
        result = OptimizationResult({"F": F, "X": X})

        df = result_to_dataframe(result)

        assert len(df) == 3
        assert "f1" in df.columns
        assert "f2" in df.columns
        assert "x1" in df.columns
        assert "x2" in df.columns

    @pytest.mark.skipif(not _has_pandas(), reason="pandas not installed")
    def test_to_dataframe_no_x(self):
        """to_dataframe() should work without X."""
        from vamos.experiment.optimization_result import OptimizationResult
        from vamos.ux.api import result_to_dataframe

        F = np.array([[0.1, 0.9], [0.5, 0.5]])
        result = OptimizationResult({"F": F})

        df = result_to_dataframe(result)

        assert len(df) == 2
        assert "f1" in df.columns
        assert "x1" not in df.columns


class TestOptimizationResultSave:
    """Test save() method."""

    def test_save_creates_files(self, tmp_path):
        """save() should create expected files."""
        import vamos

        result = vamos.optimize("zdt1", pop_size=4, max_evaluations=4, seed=3)

        out_dir = tmp_path / "test_results"
        vamos.save_result(result, out_dir)

        assert {path.name for path in out_dir.iterdir()} == {"manifest.json", "result.npz", "environment.json"}


class TestOptimizationResultTopK:
    """Test top-k ranking/report helpers."""

    def test_top_k_archive_knee(self):
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.3, 0.7], [0.5, 0.5]])
        archive_F = np.array([[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]])
        archive_X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = OptimizationResult({"F": F, "archive": {"F": archive_F, "X": archive_X}})

        with pytest.warns(FutureWarning):
            top = result.top_k(k=2, source="archive", method="knee", nondominated_only=True)

        assert top["F"].shape == (2, 2)
        assert top["X"] is not None and top["X"].shape == (2, 2)
        assert top["indices"].shape == (2,)
        assert top["scores"].shape == (2,)
        assert np.all(np.diff(top["scores"]) >= 0.0)

    def test_top_k_population_source(self):
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.3, 0.7], [0.5, 0.5]])
        pop_F = np.array([[0.8, 0.2], [0.2, 0.8], [0.4, 0.4]])
        pop_X = np.array([[1, 1], [2, 2], [3, 3]])
        result = OptimizationResult({"F": F, "population": {"F": pop_F, "X": pop_X}})

        top = result.top_k(k=1, source="population", method="balanced", nondominated_only=False)

        assert top["F"].shape == (1, 2)
        assert int(top["indices"][0]) in {0, 1, 2}
        assert top["source"] == "population"
        assert top["method"] == "balanced"

    def test_top_k_filters_dominated_when_requested(self):
        from vamos.experiment.optimization_result import OptimizationResult

        # Third point is dominated by first two trade-off points.
        archive_F = np.array([[0.0, 1.0], [1.0, 0.0], [1.2, 1.2]])
        result = OptimizationResult({"F": archive_F, "archive": {"F": archive_F, "X": None}})

        top = result.top_k(k=3, source="archive", method="balanced_sum", nondominated_only=True)

        assert top["F"].shape[0] == 2
        assert set(top["indices"].tolist()) == {0, 1}

    def test_top_k_weighted_sum_validation(self):
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.9, 0.1]])
        result = OptimizationResult({"F": F})

        with pytest.raises(ValueError, match="weights must be a 1D array"):
            result.top_k(k=1, source="result", method="weighted_sum", weights=np.array([1.0, 1.0, 1.0]))

    def test_top_k_invalid_source(self):
        from vamos.experiment.optimization_result import OptimizationResult
        from vamos.foundation.exceptions import ResultSelectionError

        F = np.array([[0.1, 0.9], [0.9, 0.1]])
        result = OptimizationResult({"F": F})

        with pytest.raises(ResultSelectionError, match="source must be one of"):
            result.top_k(k=1, source="invalid", method="balanced_sum")

    def test_best_empty_result_raises_no_solutions(self):
        from vamos.experiment.optimization_result import OptimizationResult
        from vamos.foundation.exceptions import NoSolutionsError

        result = OptimizationResult({"F": np.empty((0, 2))})

        with pytest.raises(NoSolutionsError, match="No solutions available"):
            result.best("balanced_sum")

    def test_top_k_crowding_method(self):
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.0, 1.0], [1.0, 0.0], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3]])
        result = OptimizationResult({"F": F, "archive": {"F": F, "X": None}})

        top = result.top_k(k=3, source="archive", method="crowding", nondominated_only=False)

        assert top["F"].shape[0] == 3
        assert top["method"] == "crowding"
        # Extremes should always be preserved
        assert 0 in top["indices"] and 1 in top["indices"]

    def test_top_k_report_rows(self):
        from vamos.experiment.optimization_result import OptimizationResult

        archive_F = np.array([[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]])
        archive_X = np.array([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]])
        result = OptimizationResult({"F": archive_F, "archive": {"F": archive_F, "X": archive_X}})

        rows = result.top_k_report(k=2, source="archive", method="balanced_sum", nondominated_only=False)

        assert len(rows) == 2
        assert rows[0]["rank"] == 1
        assert "score" in rows[0]
        assert "f1" in rows[0] and "f2" in rows[0]
        assert "x1" in rows[0] and "x2" in rows[0]


def _constrained_population_result():
    import vamos
    from vamos import make_problem

    problem = make_problem(
        lambda x: [x[0], x[1]],
        n_var=2,
        n_obj=2,
        xl=0,
        xu=1,
        constraints=lambda x: [0.8 - (x[0] + x[1])],
        n_constraints=1,
        encoding="real",
    )
    # One generation: the returned population still mixes feasible and infeasible rows.
    return vamos.optimize(problem, algorithm="nsgaii", pop_size=20, max_evaluations=20, seed=7)


class TestOptimizationResultFeasibility:
    """Selection helpers rank feasible solutions first (review finding VR-015)."""

    def test_front_and_best_prefer_feasible_solutions(self):
        result = _constrained_population_result()
        feasible = np.all(result.data["G"] <= 0.0, axis=1)
        assert 0 < feasible.sum() < len(result), "the scenario needs both feasible and infeasible rows"

        front_F, front_idx = result.front(return_indices=True)
        assert feasible[front_idx].all()
        assert np.array_equal(front_F, result.F[front_idx])
        assert np.array_equal(result.front(), front_F)
        assert feasible[result.best()["index"]]
        assert feasible[result.top_k(k=5)["indices"]].all()
        assert all(feasible[row["index"]] for row in result.top_k_report(k=5))

        # feasible_only=False restores the objective-only selection.
        _, unfiltered_idx = result.front(return_indices=True, feasible_only=False)
        assert not feasible[unfiltered_idx].all()

    def test_selection_without_feasible_rows_uses_the_least_violated(self):
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.2]])
        G = np.array([[0.3], [0.1], [0.1], [0.4]])
        result = OptimizationResult({"F": F, "X": None, "G": G})

        _, front_idx = result.front(return_indices=True)
        assert sorted(front_idx.tolist()) == [1, 2]
        assert result.best()["index"] in {1, 2}

    def test_result_without_constraints_is_unchanged(self):
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.6, 0.6]])
        result = OptimizationResult({"F": F, "X": None})

        _, front_idx = result.front(return_indices=True)
        assert sorted(front_idx.tolist()) == [0, 1, 2]

    def test_top_k_defaults_work_without_archive(self):
        result = _constrained_population_result()
        assert "archive" not in result.data

        top = result.top_k(k=3)
        rows = result.top_k_report(k=3)

        assert top["source"] == "result"
        assert len(rows) == 3 and rows[0]["source"] == "result"

    def test_top_k_auto_source_prefers_the_archive(self):
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.3, 0.7], [0.5, 0.5]])
        archive_F = np.array([[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]])
        result = OptimizationResult({"F": F, "archive": {"F": archive_F, "X": None}})

        top = result.top_k(k=3)

        assert top["source"] == "archive"
        assert top["F"].shape == (3, 2)

    def test_knee_is_a_deprecated_alias_of_balanced_sum(self):
        from vamos.experiment.optimization_result import OptimizationResult

        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        result = OptimizationResult({"F": F, "archive": {"F": F, "X": None}})

        with pytest.warns(FutureWarning, match="balanced_sum"):
            knee = result.best("knee")
        assert knee["index"] == result.best("balanced_sum")["index"]
        with pytest.warns(FutureWarning, match="knee_point_scores"):
            result.top_k(k=2, method="knee")
