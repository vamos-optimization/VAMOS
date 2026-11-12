#!/usr/bin/env python
"""
Benchmark script for SWEVO paper: VAMOS vs jMetalPy vs pymoo.

This script compares execution time across frameworks for solving
multi-objective optimization problems.

Usage:
    python benchmark_frameworks.py --problems zdt1 zdt2 dtlz2 --evals 100000 --seeds 30
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    framework: str
    problem: str
    algorithm: str
    n_evals: int
    seed: int
    runtime_seconds: float
    n_solutions: int
    hv: float | None = None


def run_vamos_benchmark(problem: str, n_evals: int, seed: int, engine: str = "numpy") -> BenchmarkResult:
    """Run VAMOS benchmark."""
    from vamos.api import optimize
    from vamos.foundation.problem.registry import make_problem_selection

    # Resolve problem
    prob_selection = make_problem_selection(problem)
    prob_instance = prob_selection.instantiate()

    start = time.perf_counter()
    result = optimize(
        prob_instance,
        algorithm="nsgaii",
        max_evaluations=n_evals,
        pop_size=100,
        engine=engine,
        seed=seed,
    )
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        framework=f"VAMOS ({engine})",
        problem=problem,
        algorithm="NSGA-II",
        n_evals=n_evals,
        seed=seed,
        runtime_seconds=elapsed,
        n_solutions=len(result.F),
    )


def run_pymoo_benchmark(problem: str, n_evals: int, seed: int) -> BenchmarkResult:
    """Run pymoo benchmark."""
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.problems import get_problem
        from pymoo.termination import get_termination
    except ImportError:
        return BenchmarkResult(
            framework="pymoo",
            problem=problem,
            algorithm="NSGA-II",
            n_evals=n_evals,
            seed=seed,
            runtime_seconds=-1,
            n_solutions=0,
        )

    prob = get_problem(problem)
    algorithm = NSGA2(pop_size=100)
    termination = get_termination("n_eval", n_evals)

    start = time.perf_counter()
    res = minimize(prob, algorithm, termination, seed=seed, verbose=False)
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        framework="pymoo",
        problem=problem,
        algorithm="NSGA-II",
        n_evals=n_evals,
        seed=seed,
        runtime_seconds=elapsed,
        n_solutions=len(res.F) if res.F is not None else 0,
    )


def run_jmetalpy_benchmark(problem: str, n_evals: int, seed: int) -> BenchmarkResult:
    """Run jMetalPy benchmark."""
    try:
        import random

        from jmetal.algorithm.multiobjective.nsgaii import NSGAII
        from jmetal.operator import PolynomialMutation, SBXCrossover
        from jmetal.problem import DTLZ1, DTLZ2, ZDT1, ZDT2, ZDT3
        from jmetal.util.termination_criterion import StoppingByEvaluations
    except ImportError:
        return BenchmarkResult(
            framework="jMetalPy",
            problem=problem,
            algorithm="NSGA-II",
            n_evals=n_evals,
            seed=seed,
            runtime_seconds=-1,
            n_solutions=0,
        )

    # Map problem names
    problem_map = {
        "zdt1": ZDT1,
        "zdt2": ZDT2,
        "zdt3": ZDT3,
        "dtlz1": lambda: DTLZ1(number_of_variables=7, number_of_objectives=3),
        "dtlz2": lambda: DTLZ2(number_of_variables=12, number_of_objectives=3),
    }

    if problem.lower() not in problem_map:
        return BenchmarkResult(
            framework="jMetalPy",
            problem=problem,
            algorithm="NSGA-II",
            n_evals=n_evals,
            seed=seed,
            runtime_seconds=-1,
            n_solutions=0,
        )

    random.seed(seed)
    np.random.seed(seed)

    prob_class = problem_map[problem.lower()]
    prob = prob_class() if not callable(prob_class) or problem.lower().startswith("dtlz") else prob_class()

    algorithm = NSGAII(
        problem=prob,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / prob.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=0.9, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=n_evals),
    )

    start = time.perf_counter()
    algorithm.run()
    elapsed = time.perf_counter() - start

    solutions = algorithm.get_result()

    return BenchmarkResult(
        framework="jMetalPy",
        problem=problem,
        algorithm="NSGA-II",
        n_evals=n_evals,
        seed=seed,
        runtime_seconds=elapsed,
        n_solutions=len(solutions),
    )


def run_pygmo_benchmark(problem: str, n_evals: int, seed: int) -> BenchmarkResult:
    """Run pygmo benchmark."""
    try:
        import pygmo as pg
    except ImportError:
        return BenchmarkResult(
            framework="pygmo",
            problem=problem,
            algorithm="NSGA-II",
            n_evals=n_evals,
            seed=seed,
            runtime_seconds=-1,
            n_solutions=0,
        )

    problem_map = {
        "zdt1": lambda: pg.problem(pg.zdt(prob_id=1, param=30)),
        "zdt2": lambda: pg.problem(pg.zdt(prob_id=2, param=30)),
        "zdt3": lambda: pg.problem(pg.zdt(prob_id=3, param=30)),
        "dtlz1": lambda: pg.problem(pg.dtlz(prob_id=1, dim=7, fdim=3)),
        "dtlz2": lambda: pg.problem(pg.dtlz(prob_id=2, dim=12, fdim=3)),
    }

    prob_key = problem.lower()
    if prob_key not in problem_map:
        return BenchmarkResult(
            framework="pygmo",
            problem=problem,
            algorithm="NSGA-II",
            n_evals=n_evals,
            seed=seed,
            runtime_seconds=-1,
            n_solutions=0,
        )

    prob = problem_map[prob_key]()
    pop_size = 100
    gen = max(1, (n_evals - pop_size) // pop_size)

    algo = pg.algorithm(
        pg.nsga2(
            gen=gen,
            cr=0.9,
            eta_c=20.0,
            m=1.0 / prob.get_nx(),
            eta_m=20.0,
            seed=seed,
        )
    )

    start = time.perf_counter()
    pop = pg.population(prob, size=pop_size, seed=seed)
    pop = algo.evolve(pop)
    elapsed = time.perf_counter() - start

    fits = pop.get_f()

    return BenchmarkResult(
        framework="pygmo",
        problem=problem,
        algorithm="NSGA-II",
        n_evals=n_evals,
        seed=seed,
        runtime_seconds=elapsed,
        n_solutions=len(fits),
    )


def run_benchmarks(
    problems: list[str],
    n_evals: int,
    seeds: list[int],
    frameworks: list[str],
) -> pd.DataFrame:
    """Run all benchmarks and return results as DataFrame."""
    results: list[BenchmarkResult] = []

    for problem in problems:
        for seed in seeds:
            print(f"Running {problem} with seed {seed}...")

            if "vamos-numpy" in frameworks:
                print("  VAMOS (NumPy)...", end=" ", flush=True)
                res = run_vamos_benchmark(problem, n_evals, seed, "numpy")
                print(f"{res.runtime_seconds:.2f}s")
                results.append(res)

            if "vamos-numba" in frameworks:
                print("  VAMOS (Numba)...", end=" ", flush=True)
                res = run_vamos_benchmark(problem, n_evals, seed, "numba")
                print(f"{res.runtime_seconds:.2f}s")
                results.append(res)

            if "vamos-moocore" in frameworks:
                print("  VAMOS (moocore)...", end=" ", flush=True)
                try:
                    res = run_vamos_benchmark(problem, n_evals, seed, "moocore")
                    print(f"{res.runtime_seconds:.2f}s")
                    results.append(res)
                except Exception as e:
                    print(f"SKIPPED ({e})")

            if "pymoo" in frameworks:
                print("  pymoo...", end=" ", flush=True)
                res = run_pymoo_benchmark(problem, n_evals, seed)
                print(f"{res.runtime_seconds:.2f}s" if res.runtime_seconds > 0 else "SKIPPED")
                results.append(res)

            if "jmetalpy" in frameworks:
                print("  jMetalPy...", end=" ", flush=True)
                res = run_jmetalpy_benchmark(problem, n_evals, seed)
                print(f"{res.runtime_seconds:.2f}s" if res.runtime_seconds > 0 else "SKIPPED")
                results.append(res)

            if "pygmo" in frameworks:
                print("  pygmo...", end=" ", flush=True)
                res = run_pygmo_benchmark(problem, n_evals, seed)
                print(f"{res.runtime_seconds:.2f}s" if res.runtime_seconds > 0 else "SKIPPED")
                results.append(res)

    return pd.DataFrame([r.__dict__ for r in results])


def main():
    parser = argparse.ArgumentParser(description="Benchmark MOEA frameworks")
    parser.add_argument("--problems", nargs="+", default=["zdt1", "zdt2", "dtlz2"])
    parser.add_argument("--evals", type=int, default=100000)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--frameworks", nargs="+", default=["vamos-numba", "pymoo", "jmetalpy", "pygmo"])
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/benchmark_results.csv"))
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output CSV.")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing benchmark output: {output}. Pass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seeds))

    print(f"Running benchmarks: {args.problems}")
    print(f"Evaluations: {args.evals}, Seeds: {seeds}")
    print(f"Frameworks: {args.frameworks}")
    print("-" * 50)

    df = run_benchmarks(args.problems, args.evals, seeds, args.frameworks)

    # Save raw results
    df.to_csv(output, index=False)
    print(f"\nResults saved to {output}")

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY (median runtime in seconds)")
    print("=" * 50)

    summary = (
        df[df.runtime_seconds > 0]
        .groupby(["framework", "problem"])
        .agg({"runtime_seconds": ["median", "std"], "n_solutions": "median"})
        .round(3)
    )

    print(summary)

    # Print speedup vs pymoo
    print("\n" + "=" * 50)
    print("SPEEDUP vs pymoo")
    print("=" * 50)

    pymoo_times = df[df.framework == "pymoo"].groupby("problem")["runtime_seconds"].median()
    for fw in df.framework.unique():
        if fw == "pymoo":
            continue
        fw_times = df[df.framework == fw].groupby("problem")["runtime_seconds"].median()
        speedup = pymoo_times / fw_times
        print(f"\n{fw}:")
        for prob, spd in speedup.items():
            print(f"  {prob}: {spd:.2f}x")


if __name__ == "__main__":
    main()
