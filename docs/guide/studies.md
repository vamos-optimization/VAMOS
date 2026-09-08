# Durable studies

A durable study freezes a problem-by-algorithm-by-seed matrix before any task
runs. It preserves task identity, attempts, state transitions, and verified
references to each canonical run.

For a bounded repository-checkout example that you can run unchanged, use the [Run a reproducible study executable journey](../examples.md#run-a-reproducible-study).

## Python lifecycle

```python
from vamos import StudySpec, create_study, load_study, plan_study

spec = StudySpec(
    problems=["zdt1", "zdt2"],
    algorithms=["nsgaii"],
    seeds=[0, 1],
    max_evaluations=400,
    pop_size=40,
    on_error="continue",
)

preview = plan_study(spec, output="studies/example")  # read-only
study = create_study(spec, output="studies/example")  # no optimization yet
completed = study.run()                               # sequential execution

report = completed.inspect()
summary = completed.summarize()
print(preview.plan_id, completed.study_id)
print(report.state, report.counts)
print(len(summary.rows))

loaded = load_study("studies/example")                # data-only
```

`Study` is an immutable snapshot. A mutating method returns a newly loaded
snapshot; the previous object does not update in place. Inspection and summary
verify and project persisted metadata without materializing result arrays or
executing components.

## CLI lifecycle

The configuration file contains one JSON object whose keys match `StudySpec`:

```json
{
  "problems": ["zdt1", "zdt2"],
  "algorithms": ["nsgaii"],
  "seeds": [0, 1],
  "max_evaluations": 400,
  "pop_size": 40,
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

JSON mode writes one `vamos.study-command-result` document to stdout;
diagnostics and warnings go to stderr.

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

Retry is explicit and bounded by `max_attempts_per_task`. VAMOS does not mutate
the immutable study plan or silently substitute current convenience defaults.

## Ownership boundary

VAMOS 1.0.0 permits one mutating owner per study. Do not run `run`, `resume`,
or `retry` concurrently against the same directory. Execution is sequential;
distributed workers, multiprocess ownership, and cross-process cancellation
are unsupported. See [Known limitations](../project/known-limitations.md).
