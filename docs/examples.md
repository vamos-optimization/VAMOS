# Examples

VAMOS keeps executable learning material in `examples/` and `notebooks/`, while the documentation explains the supported workflow around that code. This separation avoids maintaining several slightly different copies of the same example.

## Choose by task

| Goal | Start here | Repository material |
| --- | --- | --- |
| Run a first optimization | [Quickstart](guide/zero_to_hero.md) | `examples/basics/quickstart.py` |
| Compare built-in algorithms | [Algorithms & Backends](reference/algorithms.md) | `examples/basics/algorithm_showcase.py` |
| Define your own objectives | [Solve your own problem](guide/custom-problem.md) | `examples/problems/` |
| Work with constraints | [Constraints](reference/constraints.md) | `notebooks/1_intermediate/11_constrained_optimization.ipynb` |
| Persist and inspect runs | [Run artifacts & replay](guide/run-artifacts.md) | canonical run-artifact examples referenced by that guide |
| Run a persistent experiment matrix | [Durable studies](guide/studies.md) | study examples referenced by that guide |
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
