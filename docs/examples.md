# Examples

VAMOS keeps executable learning material in `examples/` and `notebooks/`, while the documentation explains the supported workflow around that code. This separation avoids maintaining several slightly different copies of the same example.

Use this gallery to choose runnable material by task or learning level. Follow
[Tutorials](tutorials/index.md) for a connected learning path, or use
[How-to Guides](guide/index.md) for a single operation.

## Obtain the example files

For the new **T01–T03** lessons, use their
[standalone downloads and tested setup](tutorials/setup.md). These documentation
downloads are verified with VAMOS 1.0.1 and do not require cloning the repository;
they are not included in the release-tag ZIP described below.

The PyPI package provides the Python API and CLI. The scripts and notebooks on
this page live in the [source repository](https://github.com/vamos-optimization/VAMOS).
Install the [published package](guide/installation.md), then download the
[VAMOS 1.0.1 source ZIP](https://github.com/vamos-optimization/VAMOS/archive/refs/tags/v1.0.1.zip)
and extract it, or use Git:

```bash
git clone --branch v1.0.1 --depth 1 https://github.com/vamos-optimization/VAMOS.git
cd VAMOS
```

Run the commands below from that folder using your existing VAMOS environment.
An editable install is unnecessary when you only want to run an example against
the published package. Links on this page use the matching `v1.0.1` source tag.
The three journeys below need only the core package; plotting and notebooks
require the additional dependencies described below. Record the package version,
example revision, and installed dependencies with your experiment.

For development work, obtain `main` separately and follow
[Install from a source checkout](guide/installation.md#install-from-a-source-checkout).
Its examples may require changes beyond the published package.

## Three executable journeys

The three homepage journeys each have one deliberately small script. They use only **Stable** public VAMOS facades, explicit evaluation budgets, and explicit random seeds. The documentation smoke suite executes these scripts, so they are checked as runnable learning paths rather than illustrative pseudocode.

Run the commands from the repository root after the setup above.

### Try VAMOS

```bash
python examples/journeys/try_vamos.py
```

This runs NSGA-II on ZDT1 with the NumPy reference backend and prints the shapes of the objective and decision matrices together with the evaluation count. It is the smallest end-to-end optimization path.

[View the executable source](https://github.com/vamos-optimization/VAMOS/blob/v1.0.1/examples/journeys/try_vamos.py) · [Read the Quickstart](guide/zero_to_hero.md)

### Solve my problem

```bash
python examples/journeys/solve_my_problem.py
```

This maps two domain variables, two objective scores, bounds, and one inequality requirement into `make_problem(...)`, then optimizes the resulting teaching surrogate through the same `optimize(...)` facade used for built-in benchmarks. It also extracts the matching decision rows for the non-dominated subset.

[View the executable source](https://github.com/vamos-optimization/VAMOS/blob/v1.0.1/examples/journeys/solve_my_problem.py) · [Read Solve your own problem](guide/custom-problem.md)

### Run a reproducible study

```bash
python examples/journeys/reproducible_study.py --output results/journeys/study
```

This builds a bounded `2 problems × 2 algorithms × 2 seeds` matrix, inspects the planned task count and evaluation budget before execution, publishes the exact reviewed plan, runs it, and prints the canonical run identity and manifest path for every summary row. The tiny budget and two-seed schedule are for teaching the workflow, not for making comparative performance claims.

Use a new output directory: durable studies are not silently overwritten.

[View the executable source](https://github.com/vamos-optimization/VAMOS/blob/v1.0.1/examples/journeys/reproducible_study.py) · [Read Run a reproducible study](guide/studies.md)

## Choose by task

| Goal | Start here | Status | Repository material |
| --- | --- | --- | --- |
| Run a first optimization | [Quickstart](guide/zero_to_hero.md) | Stable | [`examples/journeys/try_vamos.py`](https://github.com/vamos-optimization/VAMOS/blob/v1.0.1/examples/journeys/try_vamos.py) and [`examples/basics/quickstart.py`](https://github.com/vamos-optimization/VAMOS/blob/v1.0.1/examples/basics/quickstart.py) |
| Compare built-in algorithms | [Algorithms & Backends](reference/algorithms.md) | Stable built-in algorithms | [`examples/basics/algorithm_showcase.py`](https://github.com/vamos-optimization/VAMOS/blob/v1.0.1/examples/basics/algorithm_showcase.py) |
| Define your own objectives | [Solve your own problem](guide/custom-problem.md) | Stable problem definition | [`examples/journeys/solve_my_problem.py`](https://github.com/vamos-optimization/VAMOS/blob/v1.0.1/examples/journeys/solve_my_problem.py) and [`examples/problems/`](https://github.com/vamos-optimization/VAMOS/tree/v1.0.1/examples/problems/) |
| Work with constraints | [Constraints](reference/constraints.md) | Stable problem definition; check notebook imports | [`notebooks/1_intermediate/11_constrained_optimization.ipynb`](https://github.com/vamos-optimization/VAMOS/blob/v1.0.1/notebooks/1_intermediate/11_constrained_optimization.ipynb) |
| Persist and inspect runs | [Run artifacts & replay](guide/run-artifacts.md) | Stable | canonical run-artifact examples referenced by that guide |
| Plan and trace a persistent experiment matrix | [Run a reproducible study](guide/studies.md) | Stable, single-owner | [`examples/journeys/reproducible_study.py`](https://github.com/vamos-optimization/VAMOS/blob/v1.0.1/examples/journeys/reproducible_study.py) |
| Tune an algorithm configuration | [Hyperparameter tuning](topics/tuning.md) | Experimental | `vamos tune` CLI workflow and advanced tuning notebooks |
| Optimize model hyperparameters as decision variables | [Hyperparameter tuning](topics/tuning.md) | Example-specific custom problem | [`examples/tuning/hyperparam_tuning.py`](https://github.com/vamos-optimization/VAMOS/blob/v1.0.1/examples/tuning/hyperparam_tuning.py) |
| Use distributed evaluation | [Scaling with Dask](scaling/dask.md) | Experimental | [`examples/distributed/`](https://github.com/vamos-optimization/VAMOS/tree/v1.0.1/examples/distributed/) |
| Build a plugin | [Plugin Guide](topics/plugin_guide.md) | Experimental | [`examples/plugins/`](https://github.com/vamos-optimization/VAMOS/tree/v1.0.1/examples/plugins/) |

## Notebooks

The notebook suite is organized by learning level:

- [`notebooks/0_basic/`](https://github.com/vamos-optimization/VAMOS/tree/v1.0.1/notebooks/0_basic/) — first runs, API comparisons, and guided learning.
- [`notebooks/1_intermediate/`](https://github.com/vamos-optimization/VAMOS/tree/v1.0.1/notebooks/1_intermediate/) — discrete problems, constraints, MCDM, and interactive analysis.
- [`notebooks/2_advanced/`](https://github.com/vamos-optimization/VAMOS/tree/v1.0.1/notebooks/2_advanced/) — tuning, performance, extension workflows, ablations, and publication-oriented benchmarking.

[`notebooks/INDEX.ipynb`](https://github.com/vamos-optimization/VAMOS/blob/v1.0.1/notebooks/INDEX.ipynb) is the maintained notebook catalog in a repository checkout. The CI smoke suite executes selected notebooks; a notebook being present in the repository is not, by itself, a claim that every cell is part of the stable public API.

The [stability policy](project/stability-and-versioning.md) defines these labels.
Analysis, visualization, tuning, and plugin notebooks may use Experimental or
Internal interfaces even when the optimization call itself is Stable. Install
the [analysis extra](guide/installation.md#optional-extras) for notebooks and
any additional dependencies named by the selected example.

## Cookbook

Use the [Cookbook](guide/cookbook.md) for short task-oriented recipes. When a recipe establishes supported behavior, its public imports should come from the curated facades (`vamos`, `vamos.algorithms`, `vamos.problems`, or `vamos.ux.api`) rather than implementation modules.

## Reproducibility rule

Examples intended to support scientific results should make the evaluation budget and random seed explicit. For comparative studies, preserve the complete problem–algorithm–seed design and the run provenance behind every reported row. Paper-grade comparisons should also record the environment used for the run and define the statistical analysis separately; see [Run artifacts & replay](guide/run-artifacts.md) and the repository's pinned publication environment where applicable.
