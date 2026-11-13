"""Seeded VAMOS vs pymoo comparisons for publication-facing benchmark evidence."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vamos import optimize
from vamos.algorithms import MOEADConfig, NSGAIIConfig
from vamos.foundation.problem.resolver import resolve_reference_front_path
from vamos.foundation.quality_indicators.moocore_indicators import get_indicator, has_moocore
from vamos.foundation.quality_indicators.pareto import pareto_filter
from vamos.resources import weight_path


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    problem: str
    n_var: int
    n_obj: int
    pop_size: int
    max_evaluations: int
    divisions: int


FULL_CASES: dict[str, BenchmarkCase] = {
    "zdt1": BenchmarkCase("zdt1", "zdt1", n_var=30, n_obj=2, pop_size=100, max_evaluations=10_000, divisions=99),
    "dtlz2": BenchmarkCase("dtlz2", "dtlz2", n_var=12, n_obj=3, pop_size=91, max_evaluations=9_100, divisions=12),
    "wfg1": BenchmarkCase("wfg1", "wfg1", n_var=24, n_obj=3, pop_size=91, max_evaluations=9_100, divisions=12),
}

SMOKE_CASES: dict[str, BenchmarkCase] = {
    "zdt1": BenchmarkCase("zdt1", "zdt1", n_var=30, n_obj=2, pop_size=40, max_evaluations=400, divisions=39),
    "dtlz2": BenchmarkCase("dtlz2", "dtlz2", n_var=12, n_obj=3, pop_size=91, max_evaluations=910, divisions=12),
    "wfg1": BenchmarkCase("wfg1", "wfg1", n_var=24, n_obj=3, pop_size=91, max_evaluations=910, divisions=12),
}

DEFAULT_SEEDS = (42, 1337, 2024)
DEFAULT_ENGINE = "numba"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run seeded VAMOS vs pymoo comparisons on fixed benchmark cases. "
            "The VAMOS side defaults to the Numba backend because this script is aimed at "
            "publication-facing performance evidence rather than minimal-install smoke tests."
        )
    )
    parser.add_argument("--cases", nargs="+", choices=sorted(FULL_CASES), default=sorted(FULL_CASES), help="Benchmark cases to run.")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("nsgaii", "moead"),
        default=("nsgaii", "moead"),
        help="Algorithms to compare across both frameworks.",
    )
    parser.add_argument("--engine", default=DEFAULT_ENGINE, help="VAMOS engine to benchmark (default: numba).")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), help="Seeds used for repeated runs.")
    parser.add_argument("--smoke", action="store_true", help="Use reduced budgets suitable for quick validation.")
    parser.add_argument(
        "--output",
        default="artifacts/performance/pymoo_comparison.json",
        help="Path to the JSON report.",
    )
    parser.add_argument(
        "--markdown",
        default="artifacts/performance/pymoo_comparison.md",
        help="Optional Markdown summary path.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    return parser


def _require_optional_deps() -> None:
    if not has_moocore():
        raise SystemExit("This benchmark requires 'moocore'. Install it with `pip install -e \".[compute]\"`.")
    try:
        import pymoo  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "This benchmark requires 'pymoo', which VAMOS does not depend on. Install it with `pip install -c constraints/ci.txt pymoo`."
        ) from exc


def _reference_front(case: BenchmarkCase) -> np.ndarray:
    ref_path = resolve_reference_front_path(case.problem, explicit_path=None, n_obj=case.n_obj)
    if ref_path is not None:
        front = np.loadtxt(ref_path, delimiter=",", dtype=float)
        if front.ndim == 1:
            front = front.reshape(-1, case.n_obj)
        front_arr = np.asarray(front, dtype=float)
        if front_arr.ndim == 2 and front_arr.shape[1] == case.n_obj:
            return front_arr
    return _pymoo_reference_front(case)


def _pymoo_reference_front(case: BenchmarkCase) -> np.ndarray:
    from pymoo.problems import get_problem

    problem_kwargs: dict[str, Any] = {"n_var": case.n_var}
    if not case.problem.startswith("zdt"):
        problem_kwargs["n_obj"] = case.n_obj
    problem = get_problem(case.problem, **problem_kwargs)
    front = np.asarray(problem.pareto_front(), dtype=float)
    if front.ndim == 1:
        front = front.reshape(-1, case.n_obj)
    if front.ndim != 2 or front.shape[1] != case.n_obj:
        raise ValueError(f"Unable to obtain a {case.n_obj}-objective reference front for case '{case.name}'.")
    return front


def _reference_point(reference_front: np.ndarray) -> np.ndarray:
    span = np.ptp(reference_front, axis=0)
    padding = np.where(span > 0.0, 0.1 * span, 1.0)
    return np.asarray(np.max(reference_front, axis=0) + padding, dtype=float)


def _normalize_front(front: np.ndarray) -> np.ndarray:
    front_arr = np.asarray(front, dtype=float)
    if front_arr.ndim == 1:
        front_arr = front_arr.reshape(1, -1)
    filtered = pareto_filter(front_arr, return_indices=False)
    if filtered is None or filtered.size == 0:
        raise ValueError("Benchmark run produced an empty Pareto front.")
    return np.asarray(filtered, dtype=float)


def _front_metrics(front: np.ndarray, reference_front: np.ndarray) -> dict[str, float]:
    ref_point = _reference_point(reference_front)
    hv = float(get_indicator("hv", reference_point=ref_point).compute(front).value)
    igd_plus = float(get_indicator("igd_plus", reference_front=reference_front).compute(front).value)
    epsilon_add = float(get_indicator("epsilon_additive", reference_front=reference_front).compute(front).value)
    return {
        "hv": hv,
        "igd_plus": igd_plus,
        "epsilon_additive": epsilon_add,
        "n_solutions": float(front.shape[0]),
    }


def _vamos_config(case: BenchmarkCase, algorithm: str) -> NSGAIIConfig | MOEADConfig:
    if algorithm == "nsgaii":
        return (
            NSGAIIConfig.builder()
            .pop_size(case.pop_size)
            .selection("tournament")
            .crossover("sbx", prob=1.0, eta=20.0)
            .mutation("pm", prob="1/n", eta=20.0)
            .build()
        )
    if algorithm == "moead":
        builder = (
            MOEADConfig.builder()
            .pop_size(case.pop_size)
            .batch_size(1)
            .neighbor_size(20)
            .delta(0.9)
            .replace_limit(2)
            .crossover("sbx", prob=1.0, eta=20.0)
            .mutation("pm", prob="1/n", eta=20.0)
            .aggregation("pbi", theta=5.0)
        )
        if case.n_obj > 2:
            try:
                weights_dir = weight_path(f"W{case.n_obj}D_{case.pop_size}.dat").parent
            except ValueError:
                builder = builder.weight_vectors(divisions=case.divisions)
            else:
                builder = builder.weight_vectors(path=str(weights_dir))
        else:
            builder = builder.weight_vectors(divisions=case.divisions)
        return builder.build()
    raise ValueError(f"Unsupported algorithm '{algorithm}'.")


def _run_vamos(case: BenchmarkCase, algorithm: str, *, engine: str, seed: int) -> dict[str, Any]:
    cfg = _vamos_config(case, algorithm)
    start = time.perf_counter()
    result = optimize(
        case.problem,
        algorithm=algorithm,
        algorithm_config=cfg,
        max_evaluations=case.max_evaluations,
        engine=engine,
        seed=seed,
        n_var=case.n_var,
        n_obj=case.n_obj,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    front = _normalize_front(np.asarray(result.F, dtype=float))
    meta = dict(result.meta)
    return {
        "framework": "vamos",
        "algorithm": algorithm,
        "case": case.name,
        "problem": case.problem,
        "seed": seed,
        "runtime_ms": elapsed_ms,
        "front": front,
        "engine": meta.get("engine", engine),
        "kernel_backend": meta.get("kernel_backend", engine),
        "engine_source": meta.get("engine_source", "explicit"),
        "max_evaluations": case.max_evaluations,
        "pop_size": case.pop_size,
    }


def _run_pymoo(case: BenchmarkCase, algorithm: str, *, seed: int) -> dict[str, Any]:
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.decomposition.pbi import PBI
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    from pymoo.problems import get_problem
    from pymoo.util.ref_dirs import get_reference_directions

    problem_kwargs: dict[str, Any] = {"n_var": case.n_var}
    if not case.problem.startswith("zdt"):
        problem_kwargs["n_obj"] = case.n_obj
    problem = get_problem(case.problem, **problem_kwargs)
    mutation_prob = 1.0 / float(case.n_var)

    if algorithm == "nsgaii":
        algo = NSGA2(
            pop_size=case.pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=1.0, eta=20.0),
            mutation=PM(prob=mutation_prob, eta=20.0),
            eliminate_duplicates=True,
        )
    elif algorithm == "moead":
        ref_dirs = get_reference_directions("uniform", case.n_obj, n_partitions=case.divisions)
        algo = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=20,
            decomposition=PBI(theta=5.0),
            prob_neighbor_mating=0.9,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=1.0, eta=20.0),
            mutation=PM(prob=mutation_prob, eta=20.0),
        )
    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}'.")

    start = time.perf_counter()
    res = minimize(problem, algo, ("n_eval", case.max_evaluations), seed=seed, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    front = _normalize_front(np.asarray(res.F, dtype=float))
    return {
        "framework": "pymoo",
        "algorithm": algorithm,
        "case": case.name,
        "problem": case.problem,
        "seed": seed,
        "runtime_ms": elapsed_ms,
        "front": front,
        "engine": "pymoo",
        "kernel_backend": "pymoo",
        "engine_source": "external",
        "max_evaluations": case.max_evaluations,
        "pop_size": case.pop_size,
    }


def _with_metrics(run: dict[str, Any], reference_front: np.ndarray) -> dict[str, Any]:
    front = np.asarray(run.pop("front"), dtype=float)
    metrics = _front_metrics(front, reference_front)
    return {
        **run,
        **metrics,
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for run in runs:
        key = (str(run["case"]), str(run["algorithm"]), str(run["framework"]))
        groups.setdefault(key, []).append(run)

    summary: list[dict[str, Any]] = []
    metric_keys = ("runtime_ms", "hv", "igd_plus", "epsilon_additive", "n_solutions")
    for case_name, algorithm, framework in sorted(groups):
        bucket = groups[(case_name, algorithm, framework)]
        entry: dict[str, Any] = {
            "case": case_name,
            "algorithm": algorithm,
            "framework": framework,
            "engine": bucket[0]["engine"],
            "kernel_backend": bucket[0]["kernel_backend"],
            "engine_source": bucket[0]["engine_source"],
            "runs": len(bucket),
            "max_evaluations": bucket[0]["max_evaluations"],
            "pop_size": bucket[0]["pop_size"],
        }
        for metric in metric_keys:
            values = np.asarray([float(item[metric]) for item in bucket], dtype=float)
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_std"] = float(values.std(ddof=0))
        summary.append(entry)
    return summary


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# VAMOS vs pymoo Benchmark Report",
        "",
        "This report uses a common seeded recipe across frameworks: matching population sizes, evaluation budgets, and operator settings for NSGA-II and MOEA/D.",
        "",
        f"- Generated: `{payload['generated_at_utc']}`",
        f"- VAMOS engine: `{payload['vamos_engine']}`",
        f"- Seeds: `{', '.join(str(seed) for seed in payload['seeds'])}`",
        f"- Cases: `{', '.join(payload['cases'])}`",
        "",
        "| Case | Algorithm | Framework | Engine | Runtime ms | HV | IGD+ | Epsilon+ | Solutions |",
        "|------|-----------|-----------|--------|-----------:|---:|-----:|---------:|----------:|",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {case} | {algorithm} | {framework} | {engine} | {runtime_ms_mean:.2f} | {hv_mean:.6f} | "
            "{igd_plus_mean:.6f} | {epsilon_additive_mean:.6f} | {n_solutions_mean:.1f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Lower is better for runtime, IGD+, and additive epsilon. Higher is better for hypervolume.",
            "",
            "The JSON companion includes per-seed runs and standard deviations.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _selected_cases(names: Iterable[str], *, smoke: bool) -> list[BenchmarkCase]:
    catalog = SMOKE_CASES if smoke else FULL_CASES
    return [catalog[name] for name in names]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _require_optional_deps()

    cases = _selected_cases(args.cases, smoke=bool(args.smoke))
    seeds = tuple(int(seed) for seed in args.seeds)

    runs: list[dict[str, Any]] = []
    for case in cases:
        reference_front = _reference_front(case)
        for algorithm in args.algorithms:
            for seed in seeds:
                runs.append(_with_metrics(_run_vamos(case, algorithm, engine=str(args.engine), seed=seed), reference_front))
                runs.append(_with_metrics(_run_pymoo(case, algorithm, seed=seed), reference_front))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "vamos_engine": str(args.engine),
        "cases": [case.name for case in cases],
        "algorithms": [str(name) for name in args.algorithms],
        "seeds": list(seeds),
        "smoke": bool(args.smoke),
        "summary": _aggregate_runs(runs),
        "runs": runs,
        "case_configs": [asdict(case) for case in cases],
    }

    output_path = Path(args.output).expanduser().resolve()
    markdown_path = Path(args.markdown).expanduser().resolve() if args.markdown else None
    collisions = [path for path in (output_path, markdown_path) if path is not None and path.exists()]
    if collisions and not args.overwrite:
        names = ", ".join(str(path) for path in collisions)
        raise FileExistsError(f"Refusing to overwrite existing benchmark output(s): {names}. Pass --overwrite to replace them.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if markdown_path is not None:
        _write_text(markdown_path, _markdown_report(payload))

    print(f"Wrote JSON report to {output_path}")
    if markdown_path is not None:
        print(f"Wrote Markdown report to {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
