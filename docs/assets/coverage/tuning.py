"""Verified VAMOS 1.0.1 lesson: tuning. See coverage-setup for dependencies."""


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


def advanced_cli(function):
    import argparse

    parser = argparse.ArgumentParser(description="Run a bounded workflow in a new output directory.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    function(args.output)


if __name__ == "__main__":
    advanced_cli(run_tuning_example)
