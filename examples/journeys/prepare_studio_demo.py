"""Create a real, reproducible ZDT1 study for the Studio walkthrough.

After installing VAMOS, run this script from any working directory:

    python prepare_studio_demo.py --output results/studio-demo

The default four runs are a UI demonstration, not a comparative benchmark.
The output directory must not already exist.
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

from vamos import StudySpec, create_study, load_result, plan_study


def run(output: Path, *, max_evaluations: int = 10_000, pop_size: int = 100, plot: Path | None = None) -> None:
    if output.exists():
        raise FileExistsError(f"Choose a new study output directory: {output}")

    spec = StudySpec(
        problems=("zdt1",),
        algorithms=("nsgaii", "moead"),
        seeds=[42, 43],
        max_evaluations=max_evaluations,
        pop_size=pop_size,
        engine="numpy",
        eval_strategy="serial",
    )
    preview = plan_study(spec, output=str(output))
    if preview.status != "ready":
        raise RuntimeError(f"Study plan is not ready: {preview.errors}")
    print(f"Planned: {preview.task_count} runs, {preview.total_evaluation_budget} evaluations")

    study = create_study(spec, output=str(output))
    completed = study.run()
    report = completed.inspect()
    if report.state != "completed" or report.verified_run_count != 4 or report.issues:
        raise RuntimeError(f"Study did not finish cleanly: {report}")

    print(f"Verified runs: {report.verified_run_count}")
    print(f"Study directory: {output.resolve()}")
    print("In Studio, open Explore Results and load this exact directory.")

    if plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["svg.hashsalt"] = "vamos-studio-study"
        fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
        colors = {"vamos.algorithm:nsgaii@1": "#0d9488", "vamos.algorithm:moead@1": "#315a94"}
        labels = {"vamos.algorithm:nsgaii@1": "NSGA-II", "vamos.algorithm:moead@1": "MOEA/D"}
        plotted: set[str] = set()
        for row in completed.summarize().rows:
            if row.run_manifest_path is None:
                raise RuntimeError("A completed run is missing its manifest path.")
            result = load_result(output / Path(row.run_manifest_path).parent)
            algorithm = row.algorithm_id
            label = None if algorithm in plotted else labels[algorithm]
            ax.scatter(result.F[:, 0], result.F[:, 1], s=19, alpha=0.65, color=colors[algorithm], label=label)
            plotted.add(algorithm)
        ax.set(xlabel="Objective 1 (minimize)", ylabel="Objective 2 (minimize)", title="ZDT1 · real saved objective values")
        ax.grid(alpha=0.18)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, title="Seeds 42 and 43 pooled")
        fig.set_facecolor("#f7f9fc")
        plot.parent.mkdir(parents=True, exist_ok=True)
        if plot.suffix.lower() == ".svg":
            buffer = StringIO()
            fig.savefig(buffer, format="svg", metadata={"Date": None})
            plot.write_text("\n".join(line.rstrip() for line in buffer.getvalue().splitlines()) + "\n", encoding="utf-8")
        else:
            fig.savefig(plot, metadata={"Date": None})
        plt.close(fig)
        print(f"Result figure: {plot.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/studio-demo"))
    parser.add_argument("--max-evaluations", type=int, default=10_000)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--plot", type=Path, help="Optional SVG output (requires the analysis extra)")
    args = parser.parse_args()
    run(args.output, max_evaluations=args.max_evaluations, pop_size=args.pop_size, plot=args.plot)


if __name__ == "__main__":
    main()
