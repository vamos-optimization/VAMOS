"""Journey 3: plan and execute a small durable VAMOS study.

Run from the repository root after installing VAMOS:

    python examples/journeys/reproducible_study.py --output results/journeys/study

The output directory must not already exist. VAMOS never silently overwrites an
existing durable study.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vamos import StudySpec, create_study, plan_study


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Study output already exists: {output}")

    spec = StudySpec(
        problems=["zdt1"],
        algorithms=["nsgaii"],
        seeds=[0, 1],
        max_evaluations=80,
        pop_size=20,
        on_error="continue",
    )

    preview = plan_study(spec, output=str(output))
    completed = create_study(spec, output=str(output)).run()
    report = completed.inspect()
    summary = completed.summarize()

    print(f"plan_id={preview.plan_id}")
    print(f"study_id={completed.study_id}")
    print(f"state={report.state}")
    print(f"tasks={len(summary.rows)}")


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
