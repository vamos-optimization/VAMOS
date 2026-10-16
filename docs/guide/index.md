# How-to Guides

Find the operation you need once you have a problem or a result. For a guided
learning sequence, start with [Tutorials](../tutorials/index.md); for exact
parameters and guarantees, use [Reference](../reference/index.md).

## Define and configure a run

| Task | Guide |
| --- | --- |
| Express objectives, bounds, and constraints | [Define your problem](custom-problem.md#translate-your-model-into-vamos) and [check the evaluator](custom-problem.md#check-your-evaluator-before-spending-a-budget). |
| Check whether an algorithm supports your encoding | [Algorithm capability matrix](../reference/algorithms.md) and [problem catalog](../reference/problems.md). |
| Supply an algorithm configuration | [Configuration recipe](cookbook.md#10-algorithm-config-objects) and [configuration fields](../reference/api/algorithms/index.md). |
| Run from the command line or validate a configuration file | [CLI & Config](cli.md) and [validation recipe](cookbook.md#16-validate-a-config-file). |
| Use Numba or evaluate in multiple processes | [Numba recipe](cookbook.md#5-using-numba) and [multiprocessing recipe](cookbook.md#11-multiprocessing-evaluation). |
| Configure stopping and an external archive | [Stopping & Archive](../experiment/stopping_and_archive.md). |

## Preserve and recover work

| Task | Guide |
| --- | --- |
| Save or load an individual run | [Run artifacts](run-artifacts.md). |
| Verify an artifact or reproduce a supported built-in run | [Inspect and verify](run-artifacts.md#inspect-and-verify), then [exact built-in replay](run-artifacts.md#exact-built-in-replay). |
| Plan a persistent experiment matrix | [Plan the study budget](studies.md#2-plan-before-spending-the-budget). |
| Recover interrupted work or retry failed study tasks | [Resume and retry](studies.md#resume-and-retry). |
| Diagnose a refused replay, invalid value, or missing dependency | [Troubleshooting](troubleshooting.md). |

Loading reads stored data, verification checks an artifact, and replay executes
the recorded run. Study resume and retry operate on task state. Follow the
linked guide for the guarantees of the operation you need.

## Inspect and analyze results

| Task | Guide |
| --- | --- |
| Match decision rows to objective rows | [Understanding results](understanding-results.md#start-with-x-and-f). |
| Select one feasible solution using preferences | [Selection recipe](cookbook.md#14-select-one-solution). |
| Export a table or combine saved fronts | [DataFrame export](cookbook.md#17-export-a-dataframe) and [combined fronts](cookbook.md#18-combine-fronts-from-multiple-runs). |
| Compute hypervolume with an explicit reference point | [Hypervolume recipe](cookbook.md#19-compute-hypervolume) and [reference-relative ZDT1 example](cookbook.md#20-reference-relative-hypervolume-for-zdt1). |
| Use statistical or plotting helpers | [Analysis & Visualization](../topics/analysis.md), **Experimental**. |

## Optional experimental workflows

These destinations state their own dependencies and support boundaries:

- [Hyperparameter tuning](../topics/tuning.md): define the scoring, budget, and validation split before using the tuning CLI.
- [Scaling with Dask](../scaling/dask.md): distribute problem evaluation when its cost justifies coordination overhead.
- [VAMOS Studio](studio.md): explore stored results and use trusted local Python workflows.
- [VAMOS Assist](../assist.md): prepare specifications with the template-first flow or an optional provider.

The [Cookbook](cookbook.md) remains the complete recipe index. For runnable
scripts and notebooks, use the [Examples gallery](../examples.md).
