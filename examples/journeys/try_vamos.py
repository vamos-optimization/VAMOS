"""Journey 1: run a first deterministic VAMOS optimization.

Run from the repository root after installing VAMOS:

    python examples/journeys/try_vamos.py
"""

from __future__ import annotations

from vamos import optimize


def main() -> None:
    result = optimize(
        "zdt1",
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
