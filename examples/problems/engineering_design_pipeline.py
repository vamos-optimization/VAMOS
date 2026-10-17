"""
Mixed-encoding engineering design example (welded beam surrogate).

Usage:
    python examples/problems/engineering_design_pipeline.py

Requirements:
    pip install -e ".[examples]"  # matplotlib
"""

from __future__ import annotations

from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import WeldedBeamDesignProblem
from vamos.ux.api import plot_pareto_front_2d


def build_config(pop_size: int = 30) -> NSGAIIConfig:
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("mixed")
        .mutation("mixed")
        .selection("tournament", size=2)
        .constraint_mode("feasibility")
        .build()
    )
    return cfg


def main(seed: int = 11) -> None:
    problem = WeldedBeamDesignProblem()
    # VAMOS 1.0.1 requires this metadata to activate the four emitted constraints.
    problem.n_constraints = 4
    cfg = build_config(pop_size=24)
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("max_evaluations", 200),
        seed=seed,
        engine="numpy",
    )
    F = result.F
    G = result.data.get("G")
    print(f"Collected {F.shape[0]} design candidates.")

    feasible_mask = None
    if G is not None:
        feasible_mask = (G <= 0).all(axis=1)
        print(f"Feasible designs: {int(feasible_mask.sum())}/{F.shape[0]}")

    try:
        import matplotlib.pyplot as plt

        to_plot = F if feasible_mask is None else F[feasible_mask]
        plot_pareto_front_2d(
            to_plot,
            labels=("Fabrication cost", "Deflection proxy"),
            title="Welded beam design trade-offs",
        )
        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - plotting optional in CI
        print("Plotting skipped:", exc)


if __name__ == "__main__":
    main()
