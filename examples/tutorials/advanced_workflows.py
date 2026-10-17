"""Bounded tuning, controlled ablation and backend measurements for VAMOS 1.0.1."""

from __future__ import annotations


def new_output(output):
    from pathlib import Path

    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_observations(output, data):
    import json

    (output / "observations.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return data


def run_tuning_example(output):
    import csv
    import json
    import os
    import subprocess
    import sysconfig
    from pathlib import Path

    output = new_output(output)
    command = [
        str(Path(sysconfig.get_path("scripts")) / ("vamos.exe" if os.name == "nt" else "vamos")),
        "tune",
        "--algorithm",
        "nsgaii",
        "--problem",
        "zdt1",
        "--backend",
        "random",
        "--budget",
        "400",
        "--tune-budget",
        "2",
        "--pop-size",
        "20",
        "--seed",
        "7",
        "--n-seeds",
        "1",
        "--n-jobs",
        "1",
        "--n-var",
        "10",
        "--ref-point",
        "1.1,11",
        "--aggregate-mode",
        "median",
        "--split-seed",
        "42",
        "--run-validation",
        "--run-test",
        "--validation-seeds",
        "101,102",
        "--test-seeds",
        "201,202",
        "--validation-topk",
        "1",
        "--no-run-statistical-finisher",
        "--output-dir",
        str(output),
        "--name",
        "training",
    ]
    # The CLI remains Experimental; the wrapper preserves its original artifacts.
    with (output / "console.txt").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    root = output / "training"
    summary = json.loads((root / "tuning_summary.json").read_text(encoding="utf-8"))
    history = json.loads((root / "tuning_history.json").read_text(encoding="utf-8"))
    splits = json.loads((root / "split_seeds.json").read_text(encoding="utf-8"))
    stages = {}
    for stage in ("validation", "test"):
        with (root / f"{stage}_metrics.csv").open(encoding="utf-8", newline="") as stream:
            stages[stage] = [
                {key: row[key] for key in ("score_agg", "rows_total", "rows_ok", "fail_rate")} for row in csv.DictReader(stream)
            ]
    data = {
        "kind": "tuning",
        "status": "Experimental",
        "budget_per_run": 400,
        "reference_point": [1.1, 11.0],
        "seeds": splits,
        "backend": summary["backend_effective"],
        "trials": history,
        "best_training_score": summary["best_score"],
        "stages": stages,
    }
    print(f"Experimental tuner: {data['backend']}; trials: {len(history)}; budget per run: 400")
    print(f"Best training HV: {summary['best_score']:.6f}")
    print(f"Seeds: {splits}")
    for stage, rows in stages.items():
        print(
            f"{stage}: {len(rows)} candidate summaries; successful evaluations of candidates: "
            f"{sum(int(row['rows_ok']) for row in rows)}/{sum(int(row['rows_total']) for row in rows)}"
        )
    print("The same ZDT1 instance is reused across splits; held-out seeds are not held-out problems.")
    return write_observations(output, data)


def run_ablation_example(output, budget=400, seeds=(11, 22, 33)):
    from pathlib import Path

    from moocore import hypervolume

    from vamos import StudySpec, create_study, load_result, plan_study, verify_run
    from vamos.algorithms import NSGAIIConfig

    if budget < 40 or budget % 40 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use distinct seeds and a positive budget divisible by population 40.")
    output = new_output(output)
    specs = {}
    for name, probability in (("baseline", 0.1), ("without-mutation", 0.0)):
        config = (
            NSGAIIConfig.builder()
            .pop_size(40)
            .offspring_size(40)
            .crossover("sbx", prob=0.9, eta=20.0)
            .mutation("pm", prob=probability, eta=20.0)
            .selection("tournament", size=2)
            .result_mode("population")
            .build()
        )
        specs[name] = StudySpec(
            problems=["zdt1"],
            algorithms=["nsgaii"],
            seeds=seeds,
            n_var=10,
            pop_size=40,
            max_evaluations=budget,
            engine="numpy",
            algorithm_configs={"nsgaii": config.to_dict()},
            labels={"workflow": "controlled-ablation", "variant": name},
            on_error="continue",
        )
    plans = {name: plan_study(spec, output=output / name) for name, spec in specs.items()}
    if any(plan.status != "ready" for plan in plans.values()):
        raise ValueError("An ablation plan is not ready.")
    print(
        f"Planned tasks: {sum(p.task_count for p in plans.values())}; evaluations: {sum(p.total_evaluation_budget for p in plans.values())}"
    )
    data = {"kind": "ablation", "reference_point": [1.1, 11.0], "budget": budget, "rows": []}
    for name, spec in specs.items():
        study = create_study(spec, output=output / name).run()
        if study.plan_id != plans[name].plan_id:
            raise RuntimeError("Created plan differs from the reviewed plan.")
        for row in study.summarize().rows:
            record = {
                "variant": name,
                "seed": row.seed,
                "study_id": study.study_id,
                "task_id": row.task_id,
                "run_id": row.selected_run_id,
                "state": row.state,
                "hv": None,
            }
            if row.state == "succeeded":
                run_path = study.root / Path(row.run_manifest_path).parent
                verification = verify_run(run_path)
                if verification.run_id != row.selected_run_id or verification.numerical_bundle_safety != "valid":
                    raise ValueError("A successful ablation run failed verification.")
                result = load_result(run_path)
                front = result.front()
                record.update(
                    evaluations=int(result.data["evaluations"]),
                    front=front.tolist(),
                    hv=float(hypervolume(front, ref=data["reference_point"])),
                )
            data["rows"].append(record)
    write_observations(output, data)
    if any(row["state"] != "succeeded" for row in data["rows"]):
        raise RuntimeError("Failures retained in observations.json; do not compare incomplete variants.")
    for row in data["rows"]:
        print(f"{row['variant']}: seed {row['seed']}; evaluations {row['evaluations']}; HV {row['hv']:.6f}")
    print("One removed component, three seeds, one problem: descriptive evidence only.")
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
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lesson", choices=("tuning", "ablation", "performance"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    globals()[f"run_{args.lesson}_example"](args.output)
