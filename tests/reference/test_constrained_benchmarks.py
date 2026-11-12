"""
Benchmark tests for constrained multi-objective optimization.

These tests validate that algorithms correctly handle constraints
and produce feasible solutions on standard constrained test problems.
"""

from __future__ import annotations

import numpy as np
import pytest


class SimpleConstrainedProblem:
    """A simple constrained bi-objective problem for testing."""

    def __init__(self):
        self.n_var = 2
        self.n_obj = 2
        self.n_constraints = 2
        self.xl = np.array([0.0, 0.0])
        self.xu = np.array([1.0, 1.0])

    def evaluate(self, X, out):
        # Objectives: conflicting linear functions
        f1 = X[:, 0]
        f2 = 1 - X[:, 0] + X[:, 1]

        # Constraints: g(x) <= 0 is feasible
        g1 = X[:, 0] + X[:, 1] - 1.5  # x0 + x1 <= 1.5
        g2 = 0.5 - X[:, 0]  # x0 >= 0.5

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])


class OSYProblem:
    """
    Osyczka-Kundu (OSY) problem - classic constrained MO problem.

    6 variables, 2 objectives, 6 inequality constraints.
    """

    def __init__(self):
        self.n_var = 6
        self.n_obj = 2
        self.n_constraints = 6
        self.xl = np.array([0, 0, 1, 0, 1, 0], dtype=float)
        self.xu = np.array([10, 10, 5, 6, 5, 10], dtype=float)

    def evaluate(self, X, out):
        x1, x2, x3, x4, x5, x6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]

        # Objectives
        f1 = -(25 * (x1 - 2) ** 2 + (x2 - 2) ** 2 + (x3 - 1) ** 2 + (x4 - 4) ** 2 + (x5 - 1) ** 2)
        f2 = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2

        # Constraints (g <= 0 is feasible)
        g1 = -(x1 + x2 - 2)
        g2 = -(6 - x1 - x2)
        g3 = -(2 - x2 + x1)
        g4 = -(2 - x1 + 3 * x2)
        g5 = -(4 - (x3 - 3) ** 2 - x4)
        g6 = -((x5 - 3) ** 2 + x6 - 4)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6])


def compute_feasibility_rate(G: np.ndarray | None) -> float:
    """Compute fraction of solutions that satisfy all constraints."""
    if G is None or len(G) == 0:
        return 1.0

    # A solution is feasible if all constraints <= 0
    feasible_mask = (G <= 0).all(axis=1)
    return feasible_mask.mean()


class TestConstrainedBenchmarks:
    """Test suite for constrained optimization benchmarks."""

    def test_simple_constrained_nsga2(self):
        """Test NSGA-II on simple constrained problem."""
        import vamos

        problem = SimpleConstrainedProblem()

        result = vamos.optimize(problem, algorithm="nsgaii", max_evaluations=1000, pop_size=50, seed=42)

        # Should find solutions
        assert len(result) > 0, "Should find at least some solutions"

        # Check objectives are reasonable
        assert result.F is not None
        assert result.F.shape[1] == 2

    def test_osy_nsga2(self):
        """Test NSGA-II on OSY problem."""
        import vamos

        problem = OSYProblem()

        result = vamos.optimize(problem, algorithm="nsgaii", max_evaluations=2000, pop_size=100, seed=42)

        # Should find solutions
        assert len(result) > 0, "Should find solutions on OSY"
        assert result.F is not None

    def test_constrained_moead(self):
        """Test MOEA/D on constrained problem."""
        import vamos

        problem = SimpleConstrainedProblem()

        result = vamos.optimize(problem, algorithm="moead", max_evaluations=1000, seed=42)

        assert len(result) > 0, "MOEA/D should find solutions"

    def test_feasibility_maintained(self):
        """Verify algorithms produce mostly feasible solutions."""
        import vamos

        problem = SimpleConstrainedProblem()

        # Run optimization
        result = vamos.optimize(problem, algorithm="nsgaii", max_evaluations=2000, pop_size=100, seed=42)

        # Constraints re-evaluated on the returned solutions: every one is feasible.
        assert result.X is not None and len(result.X) > 0
        out: dict[str, np.ndarray] = {}
        problem.evaluate(result.X, out)
        assert out.get("G") is not None
        feasibility = compute_feasibility_rate(out["G"])
        assert feasibility == 1.0, f"Low feasibility: {feasibility:.2%}"


@pytest.mark.parametrize("algorithm", ["nsgaii", "moead", "smsemoa"])
def test_constrained_algorithm_comparison(algorithm):
    """Compare algorithms on constrained problem."""
    import vamos

    problem = SimpleConstrainedProblem()

    result = vamos.optimize(problem, algorithm=algorithm, max_evaluations=500, seed=42)

    # The returned solutions, re-evaluated, are all feasible.
    assert len(result.X) > 0, f"{algorithm} returned no solution"
    out: dict[str, np.ndarray] = {}
    problem.evaluate(result.X, out)
    assert compute_feasibility_rate(out["G"]) == 1.0, f"{algorithm} returned infeasible solutions"
