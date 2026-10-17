"""Verified VAMOS 1.0.1 lesson: spea2. See coverage-setup for dependencies."""


def execute_example(problem, algorithm, config, budget, seed=42, engine="numpy"):
    import numpy as np

    from vamos import OptimizationResult, optimize

    result = optimize(problem, algorithm=algorithm, algorithm_config=config, max_evaluations=budget, seed=seed, engine=engine)
    G = result.data.get("G")
    if getattr(problem, "n_constraints", 0) and G is None:
        raise ValueError("Declared constraints are missing from the returned result.")
    feasible_rows = np.arange(len(result.F)) if G is None else np.flatnonzero(np.all(G <= 0, axis=1))
    if len(feasible_rows):
        feasible = OptimizationResult({"X": result.X[feasible_rows], "F": result.F[feasible_rows]})
        front, local = feasible.front(return_indices=True)
        front_rows = feasible_rows[local]
    else:
        front, front_rows = np.empty((0, problem.n_obj)), np.empty(0, dtype=int)
    print(f"Algorithm: {algorithm}; encoding: {problem.encoding}; backend: {engine}; seed: {seed}")
    print(f"Returned X: {result.X.shape}; F: {result.F.shape}; result mode: population")
    print(f"Feasible rows: {len(feasible_rows)}/{len(result.F)}; feasible non-dominated rows: {len(front)}")
    print(f"Evaluations: {result.data['evaluations']}")
    return {
        "problem": problem,
        "algorithm": algorithm,
        "config": config,
        "result": result,
        "feasible_rows": feasible_rows,
        "front": front,
        "front_rows": front_rows,
    }


def run_spea2(budget=400):
    from vamos.algorithms import SPEA2Config
    from vamos.problems import ZDT1

    problem = ZDT1(n_var=10)
    config = (
        SPEA2Config.builder()
        .pop_size(40)
        .archive_size(40)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("pm", prob=1.0 / problem.n_var, eta=20.0)
        .selection("tournament")
        .result_mode("population")
        .build()
    )
    return execute_example(problem, "spea2", config, budget)


def plot_example(data, output):
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new figure filename: {output}")
    if output.suffix.lower() != ".svg":
        raise ValueError("Use an .svg output filename.")
    f = data["result"].F
    with plt.rc_context({"svg.hashsalt": "vamos-coverage", "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(7, 4.5), layout="constrained")
        if f.shape[1] == 3:
            ax = fig.add_subplot(projection="3d")
            ax.scatter(*f.T, c="#007f7a", s=22)
            ax.set(xlabel="f1", ylabel="f2", zlabel="f3")
            ax.set_box_aspect((1, 1, 0.85), zoom=0.85)
        else:
            ax = fig.add_subplot()
            ax.scatter(*f.T, c="#b6bec8", s=25, label="Returned population")
            if len(data["front"]):
                ax.scatter(*data["front"].T, c="#007f7a", s=30, label="Feasible non-dominated rows")
            labels = {
                "WeldedBeamDesignProblem": ("Cost (minimize)", "Deflection proxy (minimize)"),
                "FeatureSelectionProblem": ("Validation error (minimize)", "Feature count (minimize)"),
                "HyperparameterTuningProblem": ("Validation error (minimize)", "Complexity proxy (minimize)"),
            }
            xlabel, ylabel = labels.get(type(data["problem"]).__name__, ("f1 (minimize)", "f2 (minimize)"))
            ax.set(xlabel=xlabel, ylabel=ylabel)
            ax.legend()
        if data["problem"].__class__.__name__ in {"ZDT1", "ZDT1Problem"}:
            x = np.linspace(0, 1, 101)
            ax.plot(x, 1 - np.sqrt(x), "--", c="#7c8795", label="Analytical front")
            ax.legend()
        ax.set_title(f"{data['algorithm'].upper()} · {data['result'].data['evaluations']} evaluations · seed 42")
        fig.savefig(output, metadata={"Date": None}, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
    output.write_text("\n".join(line.rstrip() for line in output.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")


def lesson_cli(function, default_budget=400):
    import argparse

    parser = argparse.ArgumentParser(description="Run a bounded VAMOS example and optionally plot its actual result.")
    parser.add_argument("--budget", type=int, default=default_budget)
    parser.add_argument("--output", help="New SVG filename; requires Matplotlib")
    args = parser.parse_args()
    data = function(budget=args.budget)
    if args.output:
        plot_example(data, args.output)


if __name__ == "__main__":
    lesson_cli(run_spea2, default_budget=400)
