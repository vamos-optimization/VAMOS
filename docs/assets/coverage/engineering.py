"""Verified VAMOS 1.0.1 lesson: engineering. See coverage-setup for dependencies."""


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


def run_engineering(budget=400):
    import numpy as np

    from vamos.algorithms import NSGAIIConfig
    from vamos.problems import WeldedBeamDesignProblem

    problem = WeldedBeamDesignProblem()
    # This 1.0.1 class emits four G columns but omits their protocol metadata.
    problem.n_constraints = 4
    config = (
        NSGAIIConfig.builder()
        .pop_size(40)
        .offspring_size(40)
        .crossover("mixed", prob=0.9)
        .mutation("mixed", prob=1.0 / problem.n_var)
        .selection("tournament", size=2)
        .constraint_mode("feasibility")
        .result_mode("population")
        .build()
    )
    data = execute_example(problem, "nsgaii", config, budget)
    if not len(data["front_rows"]):
        raise ValueError("No feasible design; inspect the model and budget before selecting a candidate.")
    # An explicit preference: minimum cost among the feasible non-dominated rows.
    row = int(data["front_rows"][np.argmin(data["front"][:, 0])])
    data["selected_row"] = row
    print(f"Minimum-cost feasible front row: {row}")
    print("Decision (h, l, t, b, material, stiffeners):", np.round(data["result"].X[row], 6).tolist())
    print("Objectives (cost, deflection proxy):", np.round(data["result"].F[row], 6).tolist())
    print("Constraint residuals G <= 0:", np.round(data["result"].data["G"][row], 6).tolist())
    return data


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
    lesson_cli(run_engineering, default_budget=400)
