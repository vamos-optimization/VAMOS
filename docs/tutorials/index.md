# Tutorials

Learn a workflow from its first run to an interpretable result. Start with the
route that matches your question; each link below leads to an existing worked
example. For a single operation, use the [How-to Guides](../guide/index.md).

## Choose a learning path

Start with the three fundamental lessons. Each has complete downloadable code,
actual output and a figure, an interpretation, and one guided change. They
are verified against the published **VAMOS 1.0.1** wheel; use the
[shared setup](setup.md) for their exact dependency recipe.

| Lesson | Outcome |
| --- | --- |
| [T01. Your first meaningful result](t01-first-result.md) | Run ZDT1 and distinguish the population from its non-dominated subset. |
| [T02. Solve your own constrained problem](t02-constrained-problem.md) | Validate the evaluator, check feasibility, and preserve row alignment. |
| [T03. From a Pareto set to a decision](t03-pareto-decision.md) | State scales and preferences, select a feasible alternative, and recover its decisions. |

Continue with a complete experiment. The [study setup](study-setup.md) provides
three standalone scripts, a small functional run, and a larger recorded example.
The [recorded report](recorded-study-report.md) exposes every measured row.

| Lesson | Outcome |
| --- | --- |
| [T06. Run a fair comparative study](t06-fair-study.md) | Fix the matrix, seeds, budget and incident policy before execution. |
| [T07. Measure quality and progress with budget](t07-quality-and-budget.md) | Measure saved fronts using fixed references and compare fresh runs at increasing budgets. |
| [T08. Turn study results into a report](t08-study-report.md) | Rebuild tables and figures from retained observations and qualify the conclusions. |

| Your goal | Follow this route | What you will understand |
| --- | --- | --- |
| Try VAMOS | [Set up](setup.md) → [T01](t01-first-result.md) → [Result reference guide](../guide/understanding-results.md) | How to run a built-in problem, relate decisions to objectives, and distinguish a population from its non-dominated subset. |
| Solve your own problem | [T02](t02-constrained-problem.md) → [T03](t03-pareto-decision.md) → [Full modeling guide](../guide/custom-problem.md) | How to express bounds and requirements, check feasibility, and select a candidate using explicit preferences. |
| Run a reproducible study | [T06: plan and run](t06-fair-study.md) → [T07: measure](t07-quality-and-budget.md) → [T08: report](t08-study-report.md) | How to make the design explicit and trace each table, figure and conclusion to preserved evidence. |

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
