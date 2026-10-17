"""T06: generated from examples/tutorials/study_workflow.py; see study-setup for dependencies."""


def study_spec(profile="smoke", budget=200):
    from vamos import StudySpec

    if profile not in {"smoke", "recorded"}:
        raise ValueError("Choose smoke or recorded.")
    population, repeats = (20, 2) if profile == "smoke" else (40, 10)
    if budget < population or budget % population:
        raise ValueError("Use a budget at least one population and divisible by it.")
    return StudySpec(
        problems=["zdt1", "zdt2"],
        algorithms=["nsgaii", "moead"],
        seeds=list(range(repeats)),
        max_evaluations=budget,
        pop_size=population,
        n_var=10,
        engine="numpy",
        eval_strategy="serial",
        on_error="continue",
        max_attempts_per_task=2,
    )


def plan_or_execute_study(output, profile="smoke", budget=200, execute=False):
    from pathlib import Path

    from vamos import create_study, plan_study

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new study directory: {output}")
    spec = study_spec(profile, budget)
    plan = plan_study(spec, output=output)
    if plan.status != "ready":
        raise ValueError(f"Plan is not ready: {plan.errors}")
    print(f"Planned tasks: {plan.task_count}")
    print(f"Planned evaluations (first attempts): {plan.total_evaluation_budget}")
    if not execute:
        print("Plan only: no directory created and no optimization executed.")
        return plan
    study = create_study(spec, output=output)
    if study.plan_id != plan.plan_id:
        raise RuntimeError("Created plan differs from the reviewed plan.")
    completed = study.run()
    report = completed.inspect()
    print(f"Study state: {report.state}")
    print(f"Succeeded tasks: {report.counts['succeeded']}/{plan.task_count}")
    print(f"Verified run metadata: {report.verified_run_count}")
    print(f"Inspection issues: {len(report.issues)}")
    return completed


def main_t06(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Plan, then optionally execute a small durable study.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("smoke", "recorded"), default="smoke")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--run", action="store_true", help="Execute after checking the plan")
    args = parser.parse_args(argv)
    plan_or_execute_study(args.output, args.profile, args.budget, execute=args.run)


if __name__ == "__main__":
    main_t06()
