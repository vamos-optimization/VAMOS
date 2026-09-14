"""Experimental Dask distributed-evaluation example for VAMOS.

Dask support is an optional third-party integration outside the VAMOS 1.0
stable compatibility surface. This example deliberately uses only the public
VAMOS optimization API: create an active Dask ``Client`` and pass
``eval_strategy="dask"`` to ``optimize``.

Requirements:
    pip install "vamos-optimization[compute]"

Usage:
    # Local cluster (for testing)
    python dask_cluster.py

    # Compare serial and Dask evaluation
    python dask_cluster.py --compare

    # Connect to an existing cluster
    python dask_cluster.py --scheduler tcp://scheduler.example.com:8786
"""

from __future__ import annotations

import argparse
import time

import numpy as np


class ExpensiveProblem:
    """Simple ZDT1-like problem with an artificial per-individual delay.

    Distributed evaluation requires the problem to be serializable. Keeping the
    class at module scope provides the most portable Dask/cloudpickle behavior.
    """

    encoding = "continuous"

    def __init__(self, n_var: int = 10, delay: float = 0.01) -> None:
        self.n_var = n_var
        self.n_obj = 2
        self.n_constraints = 0
        self.xl = np.zeros(n_var, dtype=float)
        self.xu = np.ones(n_var, dtype=float)
        self.delay = float(delay)

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        time.sleep(self.delay * float(X.shape[0]))

        f1 = X[:, 0]
        g = 1.0 + 9.0 * X[:, 1:].mean(axis=1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        out["F"] = np.column_stack([f1, f2])


def run_serial(problem: ExpensiveProblem, budget: int) -> float:
    """Run the same optimization with serial evaluation."""
    from vamos import optimize
    from vamos.algorithms import NSGAIIConfig

    print("Running SERIAL evaluation...")
    config = NSGAIIConfig.default(pop_size=50, n_var=problem.n_var)
    start = time.perf_counter()
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=config,
        max_evaluations=budget,
        seed=42,
        engine="numpy",
        eval_strategy="serial",
    )
    elapsed = time.perf_counter() - start
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Solutions: {len(result)}")
    return elapsed


def _run_with_active_client(problem: ExpensiveProblem, budget: int) -> float:
    """Run VAMOS while the caller owns an active Dask Client."""
    from vamos import optimize
    from vamos.algorithms import NSGAIIConfig

    print("Running EXPERIMENTAL DASK evaluation...")
    config = NSGAIIConfig.default(pop_size=50, n_var=problem.n_var)
    start = time.perf_counter()
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=config,
        max_evaluations=budget,
        seed=42,
        engine="numpy",
        eval_strategy="dask",
    )
    elapsed = time.perf_counter() - start
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Solutions: {len(result)}")
    return elapsed


def run_distributed(problem: ExpensiveProblem, budget: int, scheduler: str | None = None) -> float | None:
    """Run optimization through the experimental public Dask path."""
    try:
        from dask.distributed import Client, LocalCluster
    except ImportError:
        print('ERROR: Dask not installed. Run: pip install "vamos-optimization[compute]"')
        return None

    if scheduler:
        print(f"Connecting to scheduler: {scheduler}")
        with Client(scheduler) as client:
            print(f"  Dashboard: {client.dashboard_link}")
            return _run_with_active_client(problem, budget)

    print("Creating local Dask cluster...")
    with LocalCluster(n_workers=4, threads_per_worker=1) as cluster:
        with Client(cluster) as client:
            print(f"  Dashboard: {client.dashboard_link}")
            return _run_with_active_client(problem, budget)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental Dask distributed VAMOS example")
    parser.add_argument("--scheduler", "-s", help="Dask scheduler address (for example tcp://scheduler:8786)")
    parser.add_argument("--budget", "-b", type=int, default=500, help="Evaluation budget (default: 500)")
    parser.add_argument("--delay", type=float, default=0.01, help="Artificial per-individual evaluation delay (seconds)")
    parser.add_argument("--compare", action="store_true", help="Compare serial vs experimental Dask evaluation")
    args = parser.parse_args()

    problem = ExpensiveProblem(delay=args.delay)

    if args.compare:
        serial_time = run_serial(problem, args.budget)
        dist_time = run_distributed(problem, args.budget, args.scheduler)

        if dist_time and serial_time:
            speedup = serial_time / dist_time
            print(f"\nSpeedup: {speedup:.2f}x")
    else:
        run_distributed(problem, args.budget, args.scheduler)


if __name__ == "__main__":
    main()
