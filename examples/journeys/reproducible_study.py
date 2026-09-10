"""Journey 3: plan, run, and trace a small durable VAMOS study.

Run from the repository root after installing VAMOS:

    python examples/journeys/reproducible_study.py --output results/journeys/study

The output directory must not already exist. VAMOS never silently overwrites an
existing durable study.

This is a bounded teaching study. Two seeds and a tiny evaluation budget are
useful for demonstrating the workflow, not for supporting a scientific claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vamos import (
    Study,
    StudyPlanReport,
    StudyReport,
    StudySpec,
    StudySummary,
    create_study,
    plan_study,
)

PROBLEMS = ("zdt1", "zdt2")
ALGORITHMS = ("nsgaii", "moead")
SEEDS = (0, 1)
MAX_EVALUATIONS = 80
POP_SIZE = 20


def build_spec() -> StudySpec:
    """Define the complete experimental matrix before any task runs."""
    return StudySpec(
        problems=PROBLEMS,
        algorithms=ALGORITHMS,
        seeds=[0, 1],
        max_evaluations=MAX_EVALUATIONS,
        pop_size=POP_SIZE,
        engine="numpy",
        eval_strategy="serial",
        on_error="continue",
    )


def run(output: Path) -> tuple[StudyPlanReport, Study, StudyReport, StudySummary]:
    if output.exists():
        raise FileExistsError(f"Study output already exists: {output}")

    spec = build_spec()
    expected_tasks = len(PROBLEMS) * len(ALGORITHMS) * len(SEEDS)

    # Planning is read-only: inspect the resolved matrix and total budget first.
    preview = plan_study(spec, output=str(output))
    if preview.status != "ready":
        raise RuntimeError(f"Study plan is not ready: {preview.errors}")
    if preview.task_count != expected_tasks:
        raise RuntimeError(f"Expected {expected_tasks} tasks, got {preview.task_count}")

    # Creation publishes the immutable plan; execution happens only in run().
    study = create_study(spec, output=str(output))
    if study.plan_id != preview.plan_id:
        raise RuntimeError("Created study does not match the plan that was reviewed.")
    completed = study.run()

    report = completed.inspect()
    summary = completed.summarize()

    print(f"plan_id={preview.plan_id}")
    print(f"study_id={completed.study_id}")
    print(f"planned_tasks={preview.task_count}")
    print(f"planned_evaluations={preview.total_evaluation_budget}")
    print(f"state={report.state}")
    print(f"verified_runs={report.verified_run_count}")

    # One summary row corresponds to one planned problem/algorithm/seed task.
    for row in summary.rows:
        print(
            "task "
            f"problem={row.problem_id} "
            f"algorithm={row.algorithm_id} "
            f"seed={row.seed} "
            f"state={row.state} "
            f"evaluations={row.evaluations} "
            f"run_id={row.selected_run_id} "
            f"manifest={row.run_manifest_path}"
        )

    return preview, completed, report, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/journeys/reproducible-study"),
        help="New directory for the durable study",
    )
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
