"""T07: generated from examples/tutorials/study_workflow.py; see study-setup for dependencies."""


def metric_policy():
    return {
        "objectives": "minimize both; original units; no per-run normalization",
        "hv_reference": [1.1, 11.0],
        "igd_plus_reference": "201 analytical points, f1 uniformly spaced on [0,1]",
        "zdt1_f2": "1 - sqrt(f1)",
        "zdt2_f2": "1 - f1**2",
        "aggregation": "median and linear-interpolation quartiles within problem/algorithm/budget",
        "budget_axis": "separate fresh studies; not checkpoints or a continuous trajectory",
    }


def measure_front(problem, front):
    import numpy as np
    from moocore import hypervolume, igd_plus

    values = np.asarray(front, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite two-objective front.")
    if problem not in {"zdt1", "zdt2"}:
        raise ValueError("This tutorial defines references only for zdt1 and zdt2.")
    reference_point = np.array(metric_policy()["hv_reference"])
    if np.any(values >= reference_point):
        raise ValueError("A point lies outside the declared HV reference box; review the protocol.")
    x = np.linspace(0.0, 1.0, 201)
    y = 1.0 - np.sqrt(x) if problem == "zdt1" else 1.0 - x**2
    reference_front = np.column_stack([x, y])
    return {
        "hv": float(hypervolume(values, ref=reference_point)),
        "igd_plus": float(igd_plus(values, ref=reference_front)),
    }


def measure_studies(study_paths, output):
    import hashlib
    import importlib.metadata
    import json
    import platform
    from pathlib import Path

    from vamos import load_result, load_study, verify_run

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new measurement directory: {output}")
    dataset = {
        "schema": "vamos.documentation.study-measurements.v1",
        "purpose": "Derived teaching observations, not a StudyManifest or RunManifest",
        "policy": metric_policy(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("vamos-optimization", "numpy", "scipy", "joblib", "moocore")},
        "studies": [],
        "rows": [],
    }
    common_design, budgets = None, set()
    for study_path in study_paths:
        study = load_study(study_path)
        report, summary = study.inspect(), study.summarize()
        intent = study.spec.as_intent_dict()
        budget = intent["run_defaults"].pop("max_evaluations")
        if common_design is None:
            common_design = intent
        if intent != common_design or budget in budgets:
            raise ValueError("Compare the same design at distinct budgets; do not pool incompatible or duplicate studies.")
        budgets.add(budget)
        if tuple(study.spec.problems) != ("zdt1", "zdt2") or study.spec.n_var != 10:
            raise ValueError("This teaching measurement protocol requires ZDT1/ZDT2 with 10 variables.")
        dataset["studies"].append(
            {
                "study_id": study.study_id,
                "plan_id": study.plan_id,
                "budget": budget,
                "state": report.state,
                "task_count": len(summary.rows),
                "root_manifest_sha256": summary.root_manifest_sha256,
                "issues": [issue.as_dict() for issue in report.issues],
                "intent": study.spec.as_intent_dict(),
            }
        )
        for row in summary.rows:
            record = row.as_dict()
            record.pop("metrics")
            record.update(
                {
                    "problem": row.problem_id.split(":", 1)[1].split("@", 1)[0],
                    "algorithm": row.algorithm_id.split(":", 1)[1].split("@", 1)[0],
                    "hv": None,
                    "igd_plus": None,
                    "front": None,
                }
            )
            if row.state == "succeeded" and row.run_manifest_path is not None:
                run_path = study.root / Path(row.run_manifest_path).parent
                verified = verify_run(run_path)
                if verified.run_id != row.selected_run_id or verified.numerical_bundle_safety != "valid":
                    raise ValueError("Numerical run evidence does not match the study summary.")
                result = load_result(run_path)
                front = result.front()
                record["front"] = front.tolist()
                record["front_sha256"] = hashlib.sha256(json.dumps(record["front"], separators=(",", ":")).encode()).hexdigest()
                record.update(measure_front(record["problem"], front))
            dataset["rows"].append(record)
    if not dataset["studies"]:
        raise ValueError("Supply at least one persisted study.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "measurements.json").write_text(json.dumps(dataset, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    measured = sum(row["hv"] is not None for row in dataset["rows"])
    print(f"Persisted studies: {len(dataset['studies'])}")
    print(f"Planned rows retained: {len(dataset['rows'])}")
    print(f"Measured successful rows: {measured}")
    print(f"Unavailable rows (not zero-filled): {len(dataset['rows']) - measured}")
    print("HV: higher is better; IGD+: lower is better. References fixed before comparison.")
    return dataset


def main_t07(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Measure verified stored runs; no optimization is executed.")
    parser.add_argument("--studies", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    measure_studies(args.studies, args.output)


if __name__ == "__main__":
    main_t07()
