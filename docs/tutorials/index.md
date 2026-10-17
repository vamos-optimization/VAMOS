# Tutorials

Learn a workflow from its first run to an interpretable result. Start with the
route that matches your question; each link below leads to an existing worked
example. For a single operation, use the [How-to Guides](../guide/index.md).

## Choose a learning path

| Your goal | Follow this route | What you will understand |
| --- | --- | --- |
| Try VAMOS | [Install](../guide/installation.md) → [Quickstart](../guide/zero_to_hero.md) → [Understand the result](../guide/understanding-results.md) | How to run a built-in problem, relate decisions to objectives, and distinguish a population from its non-dominated subset. |
| Solve your own problem | [Define and check a constrained problem](../guide/custom-problem.md) → [Inspect the result](../guide/understanding-results.md#inspect-before-selecting-one-solution) → [Select one solution](../guide/cookbook.md#14-select-one-solution) | How to express bounds and requirements, check feasibility, and select a candidate using explicit preferences. |
| Run a reproducible study | [Plan and run a study](../guide/studies.md) → [Trace its results](../guide/studies.md#5-trace-every-summary-row-to-its-run) → [Verify and replay a run](../guide/run-artifacts.md) | How to make the experiment matrix and budget explicit and trace each result to preserved evidence. |

New to the Python workflow? The [Minimal Python Track](../guide/minimal-python.md)
provides a command-line entry point. Prefer executable files? The
[Examples gallery](../examples.md#three-executable-journeys) has a small script
for each of the three routes, with setup and source links.

## Learn an algorithm

After a first run, choose a guide by the mechanism you want to understand.
Each guide includes runnable code, an actual result, and its limits.

| Guide | Focus | Before you start |
| --- | --- | --- |
| [NSGA-II](../algorithms/nsgaii.md) | Non-dominated sorting, diversity, and explicit configuration on ZDT1. | Read a two-objective result. |
| [MOEA/D](../algorithms/moead.md) | Decomposition and the role of weight vectors and neighborhoods on ZDT1. | Understand objective trade-offs and evaluation budgets. |
| [NSGA-III](../algorithms/nsgaiii.md) | Reference directions and population size on three-objective DTLZ2. | Understand a Pareto front and why more objectives change the problem. |

Use the [algorithm capability matrix](../reference/algorithms.md) to check
encoding, constraint, archive, and backend support before choosing a family.
The [configuration reference](../reference/api/algorithms/index.md) gives the
exact fields for all built-in algorithms.

## Continue with a specific task

- [Save, load, verify, or replay](../guide/run-artifacts.md) an individual run.
- [Resume or retry](../guide/studies.md#resume-and-retry) a durable study.
- [Plot and analyze results](../topics/analysis.md) with the **Experimental** analysis helpers.
- [Tune a configuration](../topics/tuning.md) through the **Experimental** tuning workflow.
- Browse the [Examples gallery](../examples.md) for notebooks by learning level.

The small study demonstrates the workflow; it does not establish an algorithm
ranking. Preserve that distinction when moving from a tutorial to your own
experiment. Check the [stability policy](../project/stability-and-versioning.md)
and the dependencies stated by each destination before using optional features.
