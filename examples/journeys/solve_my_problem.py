"""Journey 2: translate and solve a small domain model with VAMOS.

The equations are deliberately simple teaching surrogates, not a validated
physical process model. Run from the repository root after installing VAMOS:

    python examples/journeys/solve_my_problem.py
"""

from __future__ import annotations

from vamos import make_problem, optimize


def objectives(x):
    """Return the two scores to minimize for one candidate design."""
    temperature, residence_time = x

    energy_score = ((temperature - 60.0) / 40.0) ** 2 + 0.25 * (residence_time / 10.0)
    conversion_shortfall = (100.0 - temperature) / 40.0 + 2.0 / residence_time

    return [energy_score, conversion_shortfall]


def constraints(x):
    """Encode temperature * residence_time >= 450 as g(x) <= 0."""
    temperature, residence_time = x
    return [450.0 - temperature * residence_time]


def build_problem():
    return make_problem(
        objectives,
        n_var=2,
        n_obj=2,
        bounds=[(60.0, 100.0), (2.0, 10.0)],
        encoding="real",
        constraints=constraints,
        n_constraints=1,
        name="process_design_surrogate",
    )


def solve():
    return optimize(
        build_problem(),
        algorithm="nsgaii",
        max_evaluations=400,
        pop_size=40,
        engine="numpy",
        seed=42,
    )


def main() -> None:
    result = solve()
    if result.X is None or result.F is None:
        raise RuntimeError("Expected decision and objective matrices from the custom-problem journey.")

    front_F, front_indices = result.front(return_indices=True)
    front_X = result.X[front_indices]

    print(f"objective_matrix={result.F.shape}")
    print(f"decision_matrix={result.X.shape}")
    print(f"non_dominated_subset={front_F.shape}")
    print(f"matching_decisions={front_X.shape}")
    print(f"evaluations={result.data['evaluations']}")


if __name__ == "__main__":
    main()
