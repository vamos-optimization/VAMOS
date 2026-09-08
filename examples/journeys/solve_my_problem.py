"""Journey 2: define and solve a small custom bi-objective problem.

Run from the repository root after installing VAMOS:

    python examples/journeys/solve_my_problem.py
"""

from __future__ import annotations

from vamos import make_problem, optimize


def objectives(x):
    return [
        x[0] ** 2 + x[1] ** 2,
        (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2,
    ]


def main() -> None:
    problem = make_problem(
        objectives,
        n_var=2,
        n_obj=2,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        encoding="real",
    )

    result = optimize(
        problem,
        algorithm="nsgaii",
        max_evaluations=200,
        pop_size=40,
        engine="numpy",
        seed=42,
    )

    print(f"objective_matrix={result.F.shape}")
    print(f"decision_matrix={result.X.shape}")
    print(f"evaluations={result.data['evaluations']}")


if __name__ == "__main__":
    main()
