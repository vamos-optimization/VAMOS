"""
Constrained welded beam design example with feasibility filtering.

Shows how to enable constraint handling in NSGA-II and plot feasible vs. infeasible solutions.

Usage:
    python examples/problems/welded_beam_constraints.py

Requirements:
    pip install -e ".[examples]"  # matplotlib for plotting
"""

from __future__ import annotations

import numpy as np

from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import WeldedBeamDesignProblem


def main() -> None:
    problem = WeldedBeamDesignProblem()
    # VAMOS 1.0.1 requires this metadata to activate the four emitted constraints.
    problem.n_constraints = 4

    cfg = (
        NSGAIIConfig.builder()
        .pop_size(40)
        .offspring_size(40)
        .crossover("mixed", prob=0.9)
        .mutation("mixed", prob="1/n")
        .selection("tournament", size=2)
        .constraint_mode("feasibility")  # respect problem-provided G <= 0 constraints
        .build()
    )

    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("max_evaluations", 400),
        seed=5,
        engine="numpy",
    )

    F = result.F
    G = result.data.get("G")
    feasible_mask = np.ones(F.shape[0], dtype=bool)
    if G is not None:
        feasible_mask = (G <= 0).all(axis=1)

    print(f"Solutions found: {len(F)}")
    print(f"Feasible solutions: {int(feasible_mask.sum())}")
    if G is not None and (~feasible_mask).any():
        agg_violation = np.maximum(G, 0.0).sum(axis=1)
        worst = float(agg_violation[~feasible_mask].max())
        print(f"Worst aggregated violation among infeasible points: {worst:.4f}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.scatter(F[feasible_mask, 0], F[feasible_mask, 1], c="teal", label="Feasible", alpha=0.85)
        if (~feasible_mask).any():
            plt.scatter(
                F[~feasible_mask, 0],
                F[~feasible_mask, 1],
                c="lightgray",
                label="Infeasible",
                alpha=0.6,
            )
        plt.xlabel("Fabrication cost")
        plt.ylabel("Deflection proxy")
        plt.title("Welded beam design (constraints handled by NSGA-II)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - plotting optional
        print("Plotting skipped:", exc)


if __name__ == "__main__":
    main()
