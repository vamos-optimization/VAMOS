# Examples

VAMOS keeps executable learning material in `examples/` and `notebooks/`, while the documentation explains the supported workflow around that code. This separation avoids maintaining several slightly different copies of the same example.

## Three executable journeys

The three homepage journeys each have one deliberately small script. They use only public VAMOS facades, explicit evaluation budgets, and explicit random seeds. The documentation smoke suite executes these scripts, so they are checked as runnable learning paths rather than illustrative pseudocode.

Run the commands from the repository root after installing VAMOS.

### Try VAMOS

```bash
python examples/journeys/try_vamos.py
```

This runs NSGA-II on ZDT1 with the NumPy reference backend and prints the shapes of the objective and decision matrices together with the evaluation count. It is the smallest end-to-end optimization path.

[View the executable source](https://github.com/vamos-optimization/VAMOS/blob/main/examples/journeys/try_vamos.py) · [Read the Quickstart](guide/zero_to_hero.md)

### Solve my problem

```bash
python examples/journeys/solve_my_problem.py
```

This wraps a plain two-objective Python function with `make_problem(...)`, optimizes it through the same `optimize(...)` facade used for built-in benchmarks, and reports the resulting matrix shapes and evaluation count.

[View the executable source](https://github.com/vamos-optimization/VAMOS/blob/main/examples/journeys/solve_my_problem.py) · [Read Solve your own problem](guide/custom-problem.md)

### Run a reproducible study

```bash
python examples/journeys/reproducible_study.py --output results/journeys/study
```

This plans a two-seed study before execution, creates the immutable study plan, runs the bounded tasks, and inspects the persisted study summary. Use a new output directory: durable studies are not silently overwritten.

[View the executable source](https://github.com/vamos-optimization/VAMOS/blob/main/examples/journeys/reproducible_study.py) · [Read Durable studies](guide/studies.md)

## Choose by task

| Goal | Start here | Repository material |
| --- | --- | --- |
| Run a first optimization | [Quickstart](guide/zero_to_hero.md) | `examples/journeys/try_vamos.py` |
| Compare built-in algorithms | [Algorithms & Backends](reference/algorithms.md) | `examples/basics/algorithm_showcase.py` |
| Define your own objectives | [Solve your own problem](guide/custom-problem.md) | `examples/journeys/solve_my_problem.py` and `examples/problems/` |
| Work with constraints | [Constraints](reference/constraints.md) | `notebooks/1_intermediate/11_constrained_optimization.ipynb` |
| Persist and inspect runs | [Run artifacts & replay](guide/run-artifacts.md) | canonical run-artifact examples referenced by that guide |
| Run a persistent experiment matrix | [Durable studies](guide/studies.md) | `examples/journeys/reproducible_study.py` |
| Tune an algorithm | [Hyperparameter tuning](topics/tuning.md) | `examples/tuning/` and tuning notebooks |
| Use distributed evaluation | [Scaling with Dask](scaling/dask.md) | `examples/distributed/` |
| Build a plugin | [Plugin Guide](topics/plugin_guide.md) | `examples/plugins/` |

## Notebooks

The notebook suite is organized by learning level:

- `notebooks/0_basic/` — first runs, API comparisons, and guided learning.
- `notebooks/1_intermediate/` — discrete problems, constraints, MCDM, and interactive analysis.
- `notebooks/2_advanced/` — tuning, performance, extension workflows, ablations, and publication-oriented benchmarking.

`notebooks/INDEX.ipynb` is the maintained notebook catalog in a repository checkout. The CI smoke suite executes selected notebooks; a notebook being present in the repository is not, by itself, a claim that every cell is part of the stable public API.

## Cookbook

Use the [Cookbook](guide/cookbook.md) for short task-oriented recipes. When a recipe establishes supported behavior, its public imports should come from the curated facades (`vamos`, `vamos.algorithms`, `vamos.problems`, or `vamos.ux.api`) rather than implementation modules.

## Reproducibility rule

Examples intended to support scientific results should make the evaluation budget and random seed explicit. Paper-grade comparisons should also record the environment used for the run; see [Run artifacts & replay](guide/run-artifacts.md) and the repository's pinned publication environment where applicable.
