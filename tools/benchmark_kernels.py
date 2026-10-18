from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vamos.engine.algorithm.components.archive_core import _unique_rows_with_tolerance
from vamos.engine.algorithm.moead.aggregation import AGG_TCHEBYCHEFF
from vamos.engine.algorithm.moead.neighborhood_kernels import (
    dummy_buffers,
    get_update_neighborhood_batch_numba,
    update_neighborhood_batch_python,
)
from vamos.foundation.kernel.numpy_backend import NumPyKernel

try:
    from vamos.foundation.kernel.numba_backend import NumbaKernel
except ImportError:  # pragma: no cover - optional dependency
    NumbaKernel = None

BENCHMARK_THRESHOLDS = {
    "archive_deduplication": 0.15,
    "default": 0.10,
}


def _benchmark(fn: Any, *, warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {
        "median_seconds": float(statistics.median(samples)),
        "min_seconds": float(min(samples)),
        "max_seconds": float(max(samples)),
    }


def _mutation_cases(smoke: bool) -> dict[str, Any]:
    rng = np.random.default_rng(7)
    n_ind, n_var = (128, 32) if smoke else (768, 64)
    X = rng.random((n_ind, n_var), dtype=np.float64)
    params = {"prob": 1.0 / n_var, "eta": 20.0}
    numpy_kernel = NumPyKernel()

    def _run_numpy() -> None:
        work = X.copy()
        numpy_kernel.polynomial_mutation(work, params, np.random.default_rng(123), 0.0, 1.0)

    cases = {
        "polynomial_mutation.numpy": {"fn": _run_numpy, "shape": [n_ind, n_var]},
    }

    if NumbaKernel is not None:
        numba_kernel = NumbaKernel()

        def _run_numba() -> None:
            work = X.copy()
            numba_kernel.polynomial_mutation(work, params, np.random.default_rng(123), 0.0, 1.0)

        cases["polynomial_mutation.numba"] = {"fn": _run_numba, "shape": [n_ind, n_var]}

    return cases


def _selection_cases(smoke: bool) -> dict[str, Any]:
    rng = np.random.default_rng(11)
    n_candidates = 256 if smoke else 1024
    n_parents = 384 if smoke else 1536
    pressure = 4
    ranks = rng.integers(0, 8, size=n_candidates, dtype=np.int64)
    crowding = rng.normal(size=n_candidates)
    crowding[::17] = np.nan
    numpy_kernel = NumPyKernel()

    def _run_numpy() -> None:
        numpy_kernel.tournament_selection(ranks, crowding, pressure, np.random.default_rng(321), n_parents)

    cases = {
        "tournament_selection.numpy": {
            "fn": _run_numpy,
            "shape": [n_candidates],
            "pressure": pressure,
            "n_parents": n_parents,
        },
    }

    if NumbaKernel is not None:
        numba_kernel = NumbaKernel()

        def _run_numba() -> None:
            numba_kernel.tournament_selection(ranks, crowding, pressure, np.random.default_rng(321), n_parents)

        cases["tournament_selection.numba"] = {
            "fn": _run_numba,
            "shape": [n_candidates],
            "pressure": pressure,
            "n_parents": n_parents,
        }

    return cases


def _ranking_cases(smoke: bool) -> dict[str, Any]:
    rng = np.random.default_rng(13)
    n_points, n_obj = (256, 3) if smoke else (1000, 3)
    F = rng.random((n_points, n_obj), dtype=np.float64)
    F[:, -1] += 0.3 * F[:, 0]
    X = rng.random((n_points, 16), dtype=np.float64)
    half = n_points // 2
    kernels: dict[str, Any] = {"numpy": NumPyKernel()}
    if NumbaKernel is not None:
        kernels["numba"] = NumbaKernel()

    cases: dict[str, Any] = {}
    for label, kernel in kernels.items():

        def _run_ranking(kernel: Any = kernel) -> None:
            kernel.nsga2_ranking(F)

        def _run_survival(kernel: Any = kernel) -> None:
            kernel.nsga2_survival(X[:half], F[:half], X[half:], F[half:], half)

        cases[f"nsga2_ranking.{label}"] = {"fn": _run_ranking, "shape": [n_points, n_obj]}
        cases[f"nsga2_survival.{label}"] = {"fn": _run_survival, "shape": [n_points, n_obj]}
    return cases


def _crossover_cases(smoke: bool) -> dict[str, Any]:
    rng = np.random.default_rng(19)
    n_parents, n_var = (128, 32) if smoke else (768, 64)
    parents = rng.random((n_parents, n_var), dtype=np.float64)
    params = {"prob": 0.9, "eta": 20.0}
    kernels: dict[str, Any] = {"numpy": NumPyKernel()}
    if NumbaKernel is not None:
        kernels["numba"] = NumbaKernel()

    cases: dict[str, Any] = {}
    for label, kernel in kernels.items():

        def _run(kernel: Any = kernel) -> None:
            kernel.sbx_crossover(parents, params, np.random.default_rng(29), 0.0, 1.0)

        cases[f"sbx_crossover.{label}"] = {"fn": _run, "shape": [n_parents, n_var]}
    return cases


def _archive_case(smoke: bool) -> dict[str, Any]:
    rng = np.random.default_rng(17)
    n_rows, n_obj = (384, 3) if smoke else (2048, 4)
    values = rng.random((n_rows, n_obj), dtype=np.float64)
    values[1::9] = values[0::9][: values[1::9].shape[0]]
    tol = 1e-8

    def _run_archive() -> None:
        _unique_rows_with_tolerance(values, tol)

    return {
        "archive_deduplication": {
            "fn": _run_archive,
            "shape": [n_rows, n_obj],
            "tolerance": tol,
        }
    }


def _moead_cases(smoke: bool) -> dict[str, Any]:
    rng = np.random.default_rng(23)
    pop_size, n_var, n_obj, batch_size, neighbor_size = (64, 12, 3, 16, 8) if smoke else (256, 24, 3, 64, 16)
    X = rng.random((pop_size, n_var), dtype=np.float64)
    F = rng.random((pop_size, n_obj), dtype=np.float64)
    weights = rng.random((pop_size, n_obj), dtype=np.float64)
    weights_safe = np.maximum(weights, 1e-12)
    weights_norm = np.linalg.norm(weights_safe, axis=1, keepdims=True)
    weights_unit = weights_safe / np.where(weights_norm == 0.0, 1.0, weights_norm)
    ideal = F.min(axis=0)
    children = rng.random((batch_size, n_var), dtype=np.float64)
    children_f = rng.random((batch_size, n_obj), dtype=np.float64)
    candidate_orders = np.tile(np.arange(neighbor_size, dtype=np.int64), (batch_size, 1))
    candidate_lengths = np.full(batch_size, neighbor_size, dtype=np.int64)
    dummy_g, dummy_cv, dummy_child_g = dummy_buffers()
    numba_batch = get_update_neighborhood_batch_numba()

    def _run_python() -> None:
        update_neighborhood_batch_python(
            X.copy(),
            F.copy(),
            dummy_g,
            dummy_cv,
            weights,
            weights_safe,
            weights_unit,
            ideal,
            children,
            children_f,
            np.empty((batch_size, 0), dtype=float),
            np.zeros(batch_size, dtype=float),
            candidate_orders,
            candidate_lengths,
            2,
            AGG_TCHEBYCHEFF,
            5.0,
            1e-6,
            0,
        )

    cases: dict[str, Any] = {
        "moead_neighborhood.python": {
            "fn": _run_python,
            "shape": [pop_size, n_var, n_obj],
            "batch_size": batch_size,
            "neighbor_size": neighbor_size,
        }
    }

    if numba_batch is not None:
        numba_batch(
            X.copy(),
            F.copy(),
            dummy_g,
            dummy_cv,
            weights,
            weights_safe,
            weights_unit,
            ideal,
            children,
            children_f,
            np.empty((batch_size, 0), dtype=float),
            np.zeros(batch_size, dtype=float),
            candidate_orders,
            candidate_lengths,
            2,
            AGG_TCHEBYCHEFF,
            5.0,
            1e-6,
            0,
        )

        def _run_numba() -> None:
            assert numba_batch is not None
            numba_batch(
                X.copy(),
                F.copy(),
                dummy_g,
                dummy_cv,
                weights,
                weights_safe,
                weights_unit,
                ideal,
                children,
                children_f,
                np.empty((batch_size, 0), dtype=float),
                np.zeros(batch_size, dtype=float),
                candidate_orders,
                candidate_lengths,
                2,
                AGG_TCHEBYCHEFF,
                5.0,
                1e-6,
                0,
            )

        cases["moead_neighborhood.numba"] = {
            "fn": _run_numba,
            "shape": [pop_size, n_var, n_obj],
            "batch_size": batch_size,
            "neighbor_size": neighbor_size,
        }

    return cases


def _build_cases(smoke: bool) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    cases.update(_mutation_cases(smoke))
    cases.update(_crossover_cases(smoke))
    cases.update(_selection_cases(smoke))
    cases.update(_ranking_cases(smoke))
    cases.update(_archive_case(smoke))
    cases.update(_moead_cases(smoke))
    return cases


def _allowed_regression(name: str) -> float:
    if name.startswith("archive_deduplication"):
        return BENCHMARK_THRESHOLDS["archive_deduplication"]
    return BENCHMARK_THRESHOLDS["default"]


def _compare_against_baseline(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    current_cases = current["benchmarks"]
    baseline_cases = baseline["benchmarks"]
    for name, current_case in current_cases.items():
        if name not in baseline_cases:
            continue
        baseline_case = baseline_cases[name]
        current_median = float(current_case["timing"]["median_seconds"])
        baseline_median = float(baseline_case["timing"]["median_seconds"])
        if baseline_median <= 0.0:
            continue
        allowed = _allowed_regression(name)
        limit = baseline_median * (1.0 + allowed)
        if current_median > limit:
            errors.append(f"{name}: {current_median:.6f}s exceeds baseline {baseline_median:.6f}s by more than {allowed:.0%}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark core VAMOS kernels and write a seeded JSON report.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output JSON report.")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline JSON report to compare against.")
    parser.add_argument("--smoke", action="store_true", help="Run reduced benchmark sizes suitable for CI smoke checks.")
    parser.add_argument("--repeat", type=int, default=None, help="Override benchmark repeat count.")
    parser.add_argument("--warmup", type=int, default=None, help="Override warmup count.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output report.")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing benchmark output: {args.output}. Pass --overwrite to replace it.")

    repeat = int(args.repeat if args.repeat is not None else (3 if args.smoke else 7))
    warmup = int(args.warmup if args.warmup is not None else (1 if args.smoke else 2))
    cases = _build_cases(args.smoke)

    report: dict[str, Any] = {
        "meta": {
            "smoke": bool(args.smoke),
            "repeat": repeat,
            "warmup": warmup,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "benchmarks": {},
    }

    for name, case in cases.items():
        timing = _benchmark(case["fn"], warmup=warmup, repeat=repeat)
        entry = {key: value for key, value in case.items() if key != "fn"}
        entry["timing"] = timing
        report["benchmarks"][name] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if args.baseline is None or not args.baseline.exists():
        return 0

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    errors = _compare_against_baseline(report, baseline)
    if errors:
        for error in errors:
            print(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
