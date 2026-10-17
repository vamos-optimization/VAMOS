"""Verified VAMOS 1.0.1 lesson: performance. See coverage-setup for dependencies."""


def new_output(output):
    from pathlib import Path

    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_observations(output, data):
    import json

    (output / "observations.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return data


def run_performance_example(output, budget=400, repeats=3, backends=("numpy", "numba", "moocore")):
    import importlib.metadata
    import os
    import platform
    import statistics
    import time

    from moocore import hypervolume

    from vamos import optimize
    from vamos.algorithms import NSGAIIConfig
    from vamos.problems import ZDT1

    if budget < 40 or repeats < 1 or not backends or len(set(backends)) != len(backends):
        raise ValueError("Use budget >= 40, positive repeats and distinct backends.")
    output = new_output(output)
    config = NSGAIIConfig.default(pop_size=40, n_var=10)
    data = {
        "kind": "performance",
        "budget": budget,
        "repeats": repeats,
        "reference_point": [1.1, 11.0],
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "packages": {p: importlib.metadata.version(p) for p in ("vamos-optimization", "numpy", "scipy", "numba", "moocore")},
            "threads": {
                k: os.environ.get(k, "unset") for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
            },
        },
        "rows": [],
    }
    # First measured call in this process is not a guaranteed empty OS/JIT disk cache.
    schedule = [("first", backend, 42) for backend in backends]
    for seed in range(repeats):
        order = backends[seed % len(backends) :] + backends[: seed % len(backends)]
        schedule.extend(("subsequent", backend, seed) for backend in order)
    for phase, backend, seed in schedule:
        problem = ZDT1(n_var=10)
        started = time.perf_counter()
        result = optimize(problem, algorithm="nsgaii", algorithm_config=config, max_evaluations=budget, seed=seed, engine=backend)
        elapsed = time.perf_counter() - started
        data["rows"].append(
            {
                "phase": phase,
                "backend": backend,
                "seed": seed,
                "seconds": elapsed,
                "evaluations": int(result.data["evaluations"]),
                "hv": float(hypervolume(result.front(), ref=data["reference_point"])),
            }
        )
    for backend in backends:
        first = next(r["seconds"] for r in data["rows"] if r["backend"] == backend and r["phase"] == "first")
        samples = [r["seconds"] for r in data["rows"] if r["backend"] == backend and r["phase"] == "subsequent"]
        print(
            f"{backend}: first {first:.6f}s; subsequent median {statistics.median(samples):.6f}s; "
            f"range [{min(samples):.6f}, {max(samples):.6f}]s"
        )
    print(f"Actual evaluations: {sum(r['evaluations'] for r in data['rows'])}; fresh optimizer state for every call.")
    print("Times describe this machine and tiny workload; no universal backend ranking follows.")
    return write_observations(output, data)


def advanced_cli(function):
    import argparse

    parser = argparse.ArgumentParser(description="Run a bounded workflow in a new output directory.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    function(args.output)


if __name__ == "__main__":
    advanced_cli(run_performance_example)
