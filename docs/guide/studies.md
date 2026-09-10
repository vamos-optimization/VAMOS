# Run a reproducible study

A durable study freezes a problem-by-algorithm-by-seed matrix before any task runs. It preserves task identity, attempts, state transitions, and verified references to each canonical run.

The important step is not merely calling `StudySpec(...)`: it is deciding **what comparison you intend to make**, fixing that design before execution, checking its cost, and keeping every reported row traceable to the run that produced it.

For a bounded repository-checkout example that you can run unchanged, use the [Run a reproducible study executable journey](../examples.md#run-a-reproducible-study).

## Start from an experimental question

Suppose you want to compare NSGA-II and MOEA/D on ZDT1 and ZDT2 under the same evaluation budget and the same explicit seed schedule.

The maintained teaching journey uses this matrix:

| Factor | Values |
| --- | --- |
| problems | `zdt1`, `zdt2` |
| algorithms | `nsgaii`, `moead` |
| seeds | `0`, `1` |
| maximum evaluations per task | `80` |
| population size | `20` |
| kernel backend | `numpy` |

That is `2 × 2 × 2 = 8` planned tasks and a maximum planned budget of `8 × 80 = 640` objective evaluations.

The numbers are intentionally tiny so the example remains runnable in documentation CI. **Two seeds and 80 evaluations are not evidence that one algorithm is better than another.** For a scientific study, choose the replication count, evaluation budget, indicators, tuning protocol, and statistical analysis according to the question and observed variability rather than copying tutorial values.

## 1. Define the complete matrix explicitly

```python
from vamos import StudySpec

spec = StudySpec(
    problems=["zdt1", "zdt2"],
    algorithms=["nsgaii", "moead"],
    seeds=[0, 1],
    max_evaluations=80,
    pop_size=20,
    engine="numpy",
    eval_strategy="serial",
    on_error="continue",
)
```

`StudySpec` requires the seed sequence explicitly. All defaults selected while resolving the study are frozen into each task's resolved run specification before the study is published.

Using the same seed list for every algorithm makes the random-seed schedule explicit and prevents accidentally comparing different sets of runs. It does **not** by itself create common random numbers or a statistically paired experiment: different algorithms can consume randomness differently.

The example holds population size constant for clarity. That is not a universal rule for algorithm comparison. If algorithms need different scientifically justified settings, freeze those settings or their tuning protocol explicitly rather than forcing an inappropriate common configuration.

## 2. Plan before spending the budget

`plan_study(...)` resolves the matrix without creating a study or executing an optimization:

```python
from vamos import plan_study

preview = plan_study(spec, output="studies/comparison")

print(preview.status)
print(preview.task_count)
print(preview.total_evaluation_budget)
print(preview.problem_ids)
print(preview.algorithm_ids)
print(preview.seeds)
```

For the teaching matrix, `task_count` is `8` and `total_evaluation_budget` is `640`.

Use this stage to catch accidental scope growth before a campaign begins. If you add one more algorithm to a study with 10 problems and 30 seeds, you have added 300 tasks; the plan makes that multiplication visible before execution.

`plan_study(...)` is read-only. Its output-path check is advisory and does not reserve the destination, so another process could still occupy that path before creation.

## 3. Publish the plan, then execute it

```python
from vamos import create_study

study = create_study(spec, output="studies/comparison")
assert study.plan_id == preview.plan_id

completed = study.run()
```

`create_study(...)` publishes the durable study and its immutable resolved plan; it does not run an optimization. `run()` is the separate execution step.

Checking the `plan_id` is useful when a workflow has a review or approval step between planning and execution: it confirms that the study you created is the same resolved plan you inspected.

VAMOS 1.0.0 executes a durable study sequentially under a single mutating owner. `eval_strategy` belongs to evaluation inside an individual optimization task; it does not make the study itself a concurrent multi-owner scheduler.

## 4. Inspect operational state before interpreting results

```python
report = completed.inspect()

print(report.state)
print(report.counts)
print(report.verified_run_count)
print(report.issues)
```

`inspect()` answers operational questions: which tasks succeeded, which failed or remain unfinished, whether referenced run metadata could be verified, and whether recovery work is required.

A completed study is not automatically a scientifically adequate study. Operational success means the planned tasks and their evidence are in a coherent durable state; it does not establish convergence, sufficient replication, a meaningful indicator, fair tuning, statistical significance, or practical importance.

## 5. Trace every summary row to its run

`StudySummary` has one deterministic row for every planned task. The row keeps the experimental factors and the run evidence together:

```python
summary = completed.summarize()

for row in summary.rows:
    print(
        row.problem_id,
        row.algorithm_id,
        row.seed,
        row.evaluation_budget,
        row.state,
        row.evaluations,
        row.selected_run_id,
        row.run_manifest_path,
    )
```

For a successful task, `selected_run_id` identifies the selected successful run and `run_manifest_path` points to its canonical manifest inside the study directory. The summary also carries fields such as `run_manifest_sha256`, `run_metadata_available`, termination information, runtime, and failure information when applicable.

This is the key provenance chain:

```text
experimental factor combination
        ↓
planned task_id
        ↓
selected attempt
        ↓
canonical run_id
        ↓
run manifest and numerical artifacts
```

Do not discard `problem_id`, `algorithm_id`, `seed`, or the run identifiers when exporting results for downstream analysis. A table of indicator values without provenance is much harder to audit or reproduce.

## 6. Separate durable evidence from statistical conclusions

A study summary is a deterministic projection of the planned tasks and canonical run evidence. It is not an automatic leaderboard and does not decide that one algorithm "wins".

Before making comparative claims, define the analysis separately. Typical questions include:

- which performance indicator is appropriate for the problem and objective count;
- whether all algorithms received comparable computational budgets;
- how algorithm-specific hyperparameters were chosen;
- how many independent runs are needed to characterize stochastic variability;
- what statistical comparison and effect-size reporting are appropriate across problems;
- whether failed or incomplete tasks must be resolved before analysis.

The durable study layer gives that analysis a stable evidence base. It deliberately does not replace experimental-design judgment.

## Python lifecycle in one block

```python
from vamos import StudySpec, create_study, load_study, plan_study

spec = StudySpec(
    problems=["zdt1", "zdt2"],
    algorithms=["nsgaii", "moead"],
    seeds=[0, 1],
    max_evaluations=80,
    pop_size=20,
    engine="numpy",
    eval_strategy="serial",
    on_error="continue",
)

preview = plan_study(spec, output="studies/example")  # read-only
study = create_study(spec, output="studies/example")  # no optimization yet
assert study.plan_id == preview.plan_id
completed = study.run()                               # sequential study execution

report = completed.inspect()
summary = completed.summarize()
print(preview.task_count, preview.total_evaluation_budget)
print(report.state, report.counts, report.verified_run_count)
print(len(summary.rows))

loaded = load_study("studies/example")                # data-only
```

`Study` is an immutable snapshot. A mutating method returns a newly loaded snapshot; the previous object does not update in place. Inspection and summary verify and project persisted metadata without materializing result arrays or executing components.

## CLI lifecycle

The configuration file contains one JSON object whose keys match `StudySpec`:

```json
{
  "problems": ["zdt1", "zdt2"],
  "algorithms": ["nsgaii", "moead"],
  "seeds": [0, 1],
  "max_evaluations": 80,
  "pop_size": 20,
  "engine": "numpy",
  "eval_strategy": "serial",
  "on_error": "continue"
}
```

```bash
vamos study plan study.json --output studies/example --json
vamos study create study.json --output studies/example --json
vamos study run studies/example --json
vamos study inspect studies/example --json
vamos study summarize studies/example --format csv --output artifacts/studies/example.csv --json
```

Plan first and review the task count and total budget before `create` or `run`. JSON mode writes one `vamos.study-command-result` document to stdout; diagnostics and warnings go to stderr.

## Resume and retry

After interruption, reload the directory and resume eligible work:

```python
resumed = load_study("studies/example").resume()
retried = resumed.retry(failed_only=True)
```

The CLI equivalents are:

```bash
vamos study resume studies/example --json
vamos study resume studies/example --retry-failed --json
vamos study retry studies/example --failed --json
```

Retry is explicit and bounded by `max_attempts_per_task`. VAMOS does not mutate the immutable study plan or silently substitute current convenience defaults.

A retry creates new attempt evidence for the same planned task; it does not quietly turn the study into a different experimental matrix. When analyzing a retried study, retain the selected-attempt and run provenance recorded by the summary.

## Ownership boundary

VAMOS 1.0.0 permits one mutating owner per study. Do not run `run`, `resume`, or `retry` concurrently against the same directory. Execution is sequential; distributed workers, multiprocess ownership, and cross-process cancellation are unsupported. See [Known limitations](../project/known-limitations.md).

## Next steps

- [Run artifacts & replay](run-artifacts.md) — inspect and verify the canonical evidence behind an individual task.
- [Understanding optimization results](understanding-results.md) — interpret `X`, `F`, fronts, populations, and archives inside a run.
- [Algorithms & Backends](../reference/algorithms.md) — make algorithm/backend choices explicit before a comparison.
- [Analysis & Visualization](../topics/analysis.md) — move from durable execution into downstream result analysis.
- [Stability and versioning](../project/stability-and-versioning.md) — understand the 1.x compatibility surface.
