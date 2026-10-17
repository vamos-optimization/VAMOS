"""Verified VAMOS 1.0.1 lesson: ablation. See coverage-setup for dependencies."""


def new_output(output):
    from pathlib import Path

    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_observations(output, data):
    import json

    (output / "observations.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return data


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


def advanced_cli(function):
    import argparse

    parser = argparse.ArgumentParser(description="Run a bounded workflow in a new output directory.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    function(args.output)


if __name__ == "__main__":
    advanced_cli(run_ablation_example)
