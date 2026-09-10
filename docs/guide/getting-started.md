# Get started

Choose the shortest route that matches what you want to accomplish. Installation, first-run code, result interpretation, custom-problem guidance, and durable-study guidance each have one canonical page; this page only routes between them.

## Try VAMOS

1. [Install VAMOS](installation.md) from PyPI in an isolated Python environment.
2. Run the [Quickstart](zero_to_hero.md) to solve a built-in benchmark with the public `optimize(...)` facade.
3. Read [Understanding optimization results](understanding-results.md) to connect decision rows (`X`) with objective rows (`F`) and distinguish a returned set from its non-dominated subset.
4. Use the [API reference](../reference/api_reference.md) when you need exact signatures rather than tutorial guidance.

If Python itself is still new to you, begin with the [Minimal Python Track](minimal-python.md).

## Solve your own problem

Use [Solve your own problem](custom-problem.md) to define objectives, bounds, encodings, vectorized evaluation, and constraints with `make_problem(...)`.

Use `vamos create-problem` when you prefer a generated file scaffold. Adding a reusable built-in problem to VAMOS is a contributor workflow and is documented separately in [Adding a problem](../dev/add_problem.md).

## Run a reproducible study

Use [Durable studies](studies.md) when you need a persistent problem-by-algorithm-by-seed matrix with explicit task state, resume, retry, inspection, and summaries.

Use [Run artifacts & replay](run-artifacts.md) for individual persisted runs and for the distinction between loading, verification, and executable replay.

## Choose an interface

| Need | Recommended surface |
| --- | --- |
| A Python script or notebook | `vamos.optimize(...)` |
| Interpret `X`, `F`, and a non-dominated subset | `OptimizationResult` + [Understanding results](understanding-results.md) |
| A plain Python objective function | `vamos.make_problem(...)` + `vamos.optimize(...)` |
| A guided command-line workflow | `vamos quickstart` or the stable CLI commands documented in [CLI & Config](cli.md) |
| Multiple seeds in one small call | `optimize(..., seed=[...])` returning `StudyResult` |
| A persistent experiment matrix | `StudySpec`, `plan_study`, `create_study`, and `Study.run()` |
| Exact parameters for a built-in algorithm | Public configuration objects from `vamos.algorithms` |

VAMOS 1.0 uses NumPy as its deterministic reference backend. Reproducibility is a same-environment promise, not a cross-platform or cross-backend bitwise guarantee. See [Stability and versioning](../project/stability-and-versioning.md) and [Known limitations](../project/known-limitations.md) for the exact supported surface.

## Continue from here

- [Examples](../examples.md) — choose maintained scripts, notebooks, or task-oriented guides.
- [Troubleshooting](troubleshooting.md) — installation, dependency, configuration, and runtime issues.
- [Algorithms & Backends](../reference/algorithms.md) — algorithm-specific parameters and backend notes.
- [Analysis & Visualization](../topics/analysis.md) — inspect and visualize optimization results.
