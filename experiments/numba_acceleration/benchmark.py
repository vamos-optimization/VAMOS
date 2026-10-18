"""Reproducible NumPy/Numba engine benchmarks for VAMOS algorithms and kernels.

The script measures whichever ``vamos`` package is importable, so the same file
can time two source trees by changing ``PYTHONPATH``. Subcommands:

``runs``
    Full optimizations with the JIT already compiled in the process (one
    warm-up run per case is excluded). Repeated over seeds; wall time per run
    and, in a separate pass, Python peak allocation via ``tracemalloc``.
``startup``
    One fresh process per measurement: import plus a short first run, either
    with an empty Numba cache (``cold``) or with the cache written by the
    previous process (``cached``).
``kernels``
    Per-operation timings of ranking, survival, tournament selection, SBX and
    polynomial mutation for each backend.
``report``
    Merge JSON-lines results into a Markdown summary (median and interquartile
    range per configuration).

Results are JSON lines that contain no absolute paths.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "VAMOS_USE_NUMBA_MOEAD")

# (case id, algorithm, problem, problem kwargs, population, evaluations, config kind)
RUN_CASES: tuple[tuple[str, str, str, dict[str, int], int, int, str], ...] = (
    ("nsgaii-zdt1-m2-p100", "nsgaii", "zdt1", {}, 100, 20000, "generational"),
    ("nsgaii-zdt1-m2-p1000", "nsgaii", "zdt1", {}, 1000, 20000, "generational"),
    ("nsgaii-dtlz2-m3-p100", "nsgaii", "dtlz2", {"n_obj": 3}, 100, 20000, "generational"),
    ("nsgaii-dtlz2-m3-p500", "nsgaii", "dtlz2", {"n_obj": 3}, 500, 50000, "generational"),
    ("nsgaii-dtlz2-m5-p200", "nsgaii", "dtlz2", {"n_obj": 5}, 200, 20000, "generational"),
    ("nsgaii-dtlz2-m10-p200", "nsgaii", "dtlz2", {"n_obj": 10}, 200, 20000, "generational"),
    ("nsgaii-c1dtlz1-m3-p100", "nsgaii", "c1dtlz1", {"n_obj": 3}, 100, 20000, "generational"),
    ("nsgaii-expensive-m3-p100", "nsgaii", "expensive-dtlz2", {"n_obj": 3}, 100, 5000, "generational"),
    ("nsgaii-ss-zdt1-m2-p100", "nsgaii", "zdt1", {}, 100, 5000, "steady"),
    ("nsgaii-ss-dtlz2-m3-p100", "nsgaii", "dtlz2", {"n_obj": 3}, 100, 5000, "steady"),
    ("moead-zdt1-m2-p100", "moead", "zdt1", {}, 100, 20000, "default"),
    ("moead-dtlz2-m3-p91", "moead", "dtlz2", {"n_obj": 3}, 91, 20000, "default"),
    ("moead-c1dtlz1-m3-p91", "moead", "c1dtlz1", {"n_obj": 3}, 91, 10000, "default"),
    ("nsgaiii-dtlz2-m3-p91", "nsgaiii", "dtlz2", {"n_obj": 3}, 91, 20000, "default"),
    ("smsemoa-zdt1-m2-p100", "smsemoa", "zdt1", {}, 100, 5000, "default"),
    ("smsemoa-dtlz2-m3-p100", "smsemoa", "dtlz2", {"n_obj": 3}, 100, 5000, "default"),
    ("spea2-dtlz2-m3-p100", "spea2", "dtlz2", {"n_obj": 3}, 100, 10000, "default"),
    ("ibea-zdt1-m2-p100", "ibea", "zdt1", {}, 100, 10000, "default"),
    ("smpso-dtlz2-m3-p100", "smpso", "dtlz2", {"n_obj": 3}, 100, 10000, "default"),
    ("agemoea-dtlz2-m3-p100", "agemoea", "dtlz2", {"n_obj": 3}, 100, 10000, "default"),
    ("rvea-dtlz2-m3-p91", "rvea", "dtlz2", {"n_obj": 3}, 91, 10000, "default"),
)


class ExpensiveDTLZ2:
    """DTLZ2 plus one deterministic 256x256 matrix product per evaluated solution (an expensive evaluation stand-in)."""

    def __init__(self, n_obj: int = 3) -> None:
        from vamos.foundation.problem.registry import make_problem_selection

        self._inner = make_problem_selection("dtlz2", n_obj=n_obj).instantiate()
        self.n_var = self._inner.n_var
        self.n_obj = self._inner.n_obj
        self.n_constr = 0
        self.xl = self._inner.xl
        self.xu = self._inner.xu
        self.encoding = "real"
        self._work = np.linspace(0.0, 1.0, 256 * 256).reshape(256, 256)

    def evaluate(self, X: np.ndarray, out: dict[str, Any]) -> None:
        self._inner.evaluate(X, out)
        for _ in range(X.shape[0]):
            np.dot(self._work, self._work)


def _pin_to_cpu(cpu: int | None) -> None:
    """Pin the process to one logical processor (hybrid CPUs otherwise mix core types)."""
    if cpu is None:
        return
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
        return
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), 1 << cpu):
        raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")


def _environment() -> dict[str, Any]:
    import vamos

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "vamos": getattr(vamos, "__version__", "unknown"),
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "cpu_affinity": os.environ.get("VAMOS_BENCHMARK_CPU"),
    }
    try:
        import numba

        info["numba"] = numba.__version__
    except ImportError:
        info["numba"] = None
    try:
        import moocore

        info["moocore"] = getattr(moocore, "__version__", "unknown")
    except ImportError:
        info["moocore"] = None
    source = Path(vamos.__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(source), "status", "--porcelain", "src"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = "unknown", None
    info["source_git_sha"] = sha
    info["source_dirty"] = dirty
    return info


def _algorithm_config(algorithm: str, population: int, kind: str) -> Any:
    from vamos.algorithms import NSGAIIConfig

    if algorithm == "nsgaii":
        builder = (
            NSGAIIConfig.builder()
            .pop_size(population)
            .crossover("sbx", prob=0.9, eta=20.0)
            .mutation("pm", prob="1/n", eta=20.0)
            .selection("tournament", size=2)
        )
        builder = builder.offspring_size(1) if kind == "steady" else builder.offspring_size(population)
        return builder.build()
    return None


def _optimize(case: tuple[str, str, str, dict[str, int], int, int, str], engine: str, seed: int, evaluations: int | None = None) -> Any:
    import vamos

    _, algorithm, problem, kwargs, population, budget, kind = case
    config = _algorithm_config(algorithm, population, kind)
    problem_arg: Any = ExpensiveDTLZ2(**kwargs) if problem == "expensive-dtlz2" else problem
    problem_kwargs = {} if problem == "expensive-dtlz2" else kwargs
    budget = budget if evaluations is None else evaluations
    if config is not None:
        return vamos.optimize(
            problem_arg,
            algorithm=algorithm,
            algorithm_config=config,
            termination=("max_evaluations", budget),
            engine=engine,
            seed=seed,
            **problem_kwargs,
        )
    return vamos.optimize(
        problem_arg, algorithm=algorithm, pop_size=population, max_evaluations=budget, engine=engine, seed=seed, **problem_kwargs
    )


def _write(lines: Iterable[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, sort_keys=True) + "\n")


def command_runs(args: argparse.Namespace) -> None:
    env = _environment()
    selected = [case for case in RUN_CASES if not args.cases or case[0] in args.cases]
    for case in selected:
        _optimize(case, args.engine, seed=10_000, evaluations=max(4 * case[4], 400))  # JIT warm-up, excluded
        records = []
        for seed in args.seeds:
            for repeat in range(args.repeat):
                start = time.perf_counter()
                result = _optimize(case, args.engine, seed)
                elapsed = time.perf_counter() - start
                records.append(
                    {
                        "kind": "run",
                        "label": args.label,
                        "engine": args.engine,
                        "case": case[0],
                        "seed": seed,
                        "repeat": repeat,
                        "seconds": elapsed,
                        "evaluations": int(result.data["evaluations"]),
                        "final_F_sha": _array_digest(np.asarray(result.F)),
                    }
                )
        tracemalloc.start()
        _optimize(case, args.engine, args.seeds[0])
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        for record in records:
            record["tracemalloc_peak_bytes"] = peak
            record["environment"] = env
        _write(records, args.output)
        seconds = [record["seconds"] for record in records]
        print(
            f"{args.label:12s} {case[0]:28s} median={statistics.median(seconds):8.3f}s n={len(seconds)} peak={peak / 1e6:7.1f}MB",
            flush=True,
        )


def _array_digest(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()[:16]


_STARTUP_SNIPPET = """
import time
t0 = time.perf_counter()
import vamos
t1 = time.perf_counter()
vamos.optimize("{problem}", algorithm="{algorithm}", engine="{engine}", seed=1, pop_size={population}, max_evaluations={evaluations}{extra})
t2 = time.perf_counter()
vamos.optimize("{problem}", algorithm="{algorithm}", engine="{engine}", seed=2, pop_size={population}, max_evaluations={evaluations}{extra})
t3 = time.perf_counter()
print(t1 - t0, t2 - t1, t3 - t2)
"""


def command_startup(args: argparse.Namespace) -> None:
    env = _environment()
    cases = (("nsgaii", "zdt1", 100, 2000, ""), ("nsgaii", "dtlz2", 100, 2000, ", n_obj=3"), ("moead", "zdt1", 100, 2000, ""))
    records = []
    with tempfile.TemporaryDirectory() as cache_root:
        for algorithm, problem, population, evaluations, extra in cases:
            cache_dir = Path(cache_root) / f"{algorithm}-{problem}"
            cache_dir.mkdir()
            snippet = _STARTUP_SNIPPET.format(
                algorithm=algorithm, problem=problem, engine=args.engine, population=population, evaluations=evaluations, extra=extra
            )
            for phase in ["cold"] + ["cached"] * args.repeat:
                child_env = dict(os.environ, NUMBA_CACHE_DIR=str(cache_dir))
                start = time.perf_counter()
                completed = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, env=child_env, check=True)
                process_seconds = time.perf_counter() - start
                import_s, first_run_s, second_run_s = (float(value) for value in completed.stdout.split()[-3:])
                record = {
                    "kind": "startup",
                    "label": args.label,
                    "engine": args.engine,
                    "case": f"{algorithm}-{problem}",
                    "phase": phase,
                    "process_seconds": process_seconds,
                    "import_seconds": import_s,
                    "first_run_seconds": first_run_s,
                    "second_run_seconds": second_run_s,
                    "environment": env,
                }
                records.append(record)
                print(
                    f"{args.label:12s} {algorithm}-{problem:6s} {phase:6s} process={process_seconds:6.2f}s import={import_s:5.2f}s first={first_run_s:6.2f}s second={second_run_s:6.2f}s",
                    flush=True,
                )
    _write(records, args.output)


def _apply_to_copy(operator: Callable[[np.ndarray, np.random.Generator], object], values: np.ndarray, rng: np.random.Generator) -> object:
    return operator(values.copy(), rng)


def _timeit(fn: Callable[[], object], repeat: int) -> list[float]:
    fn()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return samples


def command_kernels(args: argparse.Namespace) -> None:
    from vamos.engine.operators.impl.real import PolynomialMutation, SBXCrossover, VariationWorkspace
    from vamos.foundation.kernel.registry import resolve_kernel

    env = _environment()
    kernel = resolve_kernel(args.engine)
    variation_kernels = getattr(kernel, "variation_kernels", lambda: None)()
    rng = np.random.default_rng(0)
    records = []
    for n_obj in (2, 3, 5, 10):
        for population in (50, 100, 200, 500, 1000):
            F = rng.random((2 * population, n_obj))
            F[:, -1] += 0.3 * F[:, 0]
            X = rng.random((2 * population, 30))
            repeat = max(5, args.repeat if population <= 200 else args.repeat // 4)
            operations: dict[str, Callable[[], object]] = {
                "nsga2_ranking": partial(kernel.nsga2_ranking, F),
                "nsga2_survival": partial(
                    kernel.nsga2_survival, X[:population], F[:population], X[population:], F[population:], population
                ),
            }
            for name, fn in operations.items():
                samples = _timeit(fn, repeat)
                records.append(
                    {
                        "kind": "kernel",
                        "label": args.label,
                        "engine": args.engine,
                        "operation": name,
                        "n_obj": n_obj,
                        "population": population,
                        "seconds": samples,
                        "environment": env,
                    }
                )
    for population in (1, 50, 100, 200, 500, 1000):
        for n_var in (12, 30):
            lower, upper = np.zeros(n_var), np.ones(n_var)
            parents = rng.random((population, 2, n_var))
            offspring = rng.random((max(1, 2 * population if population > 1 else 1), n_var))
            # Older source trees have no variation-kernel hook; their operators take no such argument.
            extra: dict[str, Any] = {} if variation_kernels is None else {"variation_kernels": variation_kernels}
            sbx = SBXCrossover(
                prob_crossover=0.9, eta=20.0, lower=lower, upper=upper, workspace=VariationWorkspace(), allow_inplace=True, **extra
            )
            pm = PolynomialMutation(1.0 / n_var, 20.0, lower=lower, upper=upper, workspace=VariationWorkspace(), **extra)
            generator = np.random.default_rng(1)
            ranks = rng.integers(0, 5, size=2 * max(population, 2))
            crowding = rng.random(2 * max(population, 2))
            operations = {
                "sbx_crossover": partial(_apply_to_copy, sbx, parents, generator),
                "polynomial_mutation": partial(pm, offspring, generator),
                "tournament_selection": partial(kernel.tournament_selection, ranks, crowding, 2, generator, 2 * max(population, 2)),
            }
            for name, fn in operations.items():
                if name == "tournament_selection" and n_var != 12:
                    continue
                samples = _timeit(fn, args.repeat)
                records.append(
                    {
                        "kind": "kernel",
                        "label": args.label,
                        "engine": args.engine,
                        "operation": name,
                        "n_var": n_var,
                        "population": population,
                        "seconds": samples,
                        "environment": env,
                    }
                )
    _write(records, args.output)
    print(f"{args.label}: {len(records)} kernel measurements written", flush=True)


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


def command_report(args: argparse.Namespace) -> None:
    records = [json.loads(line) for path in args.inputs for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = ["# Numba acceleration benchmark summary", ""]
    runs: dict[tuple[str, str], list[float]] = defaultdict(list)
    peaks: dict[tuple[str, str], int] = {}
    digests: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    labels: list[str] = []
    for record in records:
        if record["kind"] == "run":
            key = (record["case"], record["label"])
            runs[key].append(record["seconds"])
            peaks[key] = record["tracemalloc_peak_bytes"]
            digests[(record["case"], record["label"], record["seed"])].add(record["final_F_sha"])
            if record["label"] not in labels:
                labels.append(record["label"])
    if runs:
        lines += ["## Full optimizations (warm JIT, median [IQR] seconds; Python peak MB)", ""]
        lines.append("| case | " + " | ".join(labels) + " |")
        lines.append("|---|" + "---|" * len(labels))
        for case in dict.fromkeys(case for case, _ in runs):
            cells = []
            for label in labels:
                values = runs.get((case, label))
                if not values:
                    cells.append("-")
                    continue
                median, q1, q3 = _quartiles(values)
                cells.append(f"{median:.3f} [{q1:.3f}-{q3:.3f}]; {peaks[(case, label)] / 1e6:.1f}")
            lines.append(f"| {case} | " + " | ".join(cells) + " |")
        lines.append("")
        lines += [
            "## Final-front agreement",
            "",
            "Configurations whose final objective matrices are bitwise identical for every seed (digest of `result.F`).",
            "",
            "| case | identical groups |",
            "|---|---|",
        ]
        for case in dict.fromkeys(case for case, _ in runs):
            seeds = sorted({seed for (c, _, seed) in digests if c == case})
            signature: dict[str, tuple[str, ...]] = {}
            for label in labels:
                signature[label] = tuple(sorted(next(iter(digests.get((case, label, seed), {"-"}))) for seed in seeds))
            groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
            for label, key in signature.items():
                groups[key].append(label)
            lines.append(f"| {case} | " + "; ".join(" = ".join(group) for group in groups.values()) + " |")
        lines.append("")
    startups = [record for record in records if record["kind"] == "startup"]
    if startups:
        lines += [
            "## Startup (fresh process; median seconds)",
            "",
            "| label | case | phase | process | import | first run | second run |",
            "|---|---|---|---|---|---|---|",
        ]
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in startups:
            grouped[(record["label"], record["case"], record["phase"])].append(record)
        for (label, case, phase), items in grouped.items():

            def med(field: str, items: list[dict[str, Any]] = items) -> float:
                return float(statistics.median(item[field] for item in items))

            lines.append(
                f"| {label} | {case} | {phase} | {med('process_seconds'):.2f} | {med('import_seconds'):.2f} | {med('first_run_seconds'):.2f} | {med('second_run_seconds'):.2f} |"
            )
        lines.append("")
    kernels = [record for record in records if record["kind"] == "kernel"]
    if kernels:
        lines += ["## Kernels (median microseconds)", ""]
        table: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        kernel_labels: list[str] = []
        for record in kernels:
            size = f"M={record['n_obj']}" if "n_obj" in record else f"n_var={record['n_var']}"
            key = (record["operation"], f"{size} N={record['population']}")
            table[key][record["label"]] = statistics.median(record["seconds"]) * 1e6
            if record["label"] not in kernel_labels:
                kernel_labels.append(record["label"])
        lines.append("| operation | size | " + " | ".join(kernel_labels) + " |")
        lines.append("|---|---|" + "---|" * len(kernel_labels))
        for (operation, size), values in table.items():
            lines.append(
                f"| {operation} | {size} | "
                + " | ".join(f"{values[label]:.1f}" if label in values else "-" for label in kernel_labels)
                + " |"
            )
        lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("runs", "startup", "kernels"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--label", required=True)
        cmd.add_argument("--engine", choices=("numpy", "numba"), required=True)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--repeat", type=int, default=3 if name != "kernels" else 40)
        cmd.add_argument("--cpu", type=int, default=None, help="Pin to one logical processor.")
        if name == "runs":
            cmd.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
            cmd.add_argument("--cases", nargs="*", default=[])
    report = sub.add_parser("report")
    report.add_argument("--inputs", nargs="+", required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command != "report":
        _pin_to_cpu(args.cpu)
        if args.cpu is not None:
            os.environ["VAMOS_BENCHMARK_CPU"] = str(args.cpu)
    {"runs": command_runs, "startup": command_startup, "kernels": command_kernels, "report": command_report}[args.command](args)


if __name__ == "__main__":
    main()
