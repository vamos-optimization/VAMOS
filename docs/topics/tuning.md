# Hyperparameter Tuning

VAMOS can tune **algorithm configuration**: operator probabilities, distribution
indices, population-related settings, and other parameters that control a MOEA.
The configuration tuner evaluates candidate settings across explicit problem and
seed blocks, then compares their scalar quality scores.

!!! warning "Experimental surface in VAMOS 1.0.0"
    The `vamos.engine.tuning` / racing APIs and the `vamos tune` and
    `vamos ablation` commands are **experimental**. They are supported for
    evaluation but may change incompatibly in a minor release. The stable
    `optimize(...)` and algorithm-configuration facades used to evaluate a
    candidate remain governed by the [stability policy](../project/stability-and-versioning.md).

This page is about configuring optimization algorithms. If the variables in
*your optimization problem* are model hyperparameters (for example SVM `C` and
`gamma`), that is a different workflow; see
`examples/tuning/hyperparam_tuning.py` for that formulation.

## Install

The built-in `random` and `racing` tuners use the core installation.

Model-based CLI backends (`optuna`, `smac3`, `bohb`, and `bohb_optuna`) require
the optional tuning dependencies:

```bash
pip install "vamos-optimization[tuning]"
```

From a local checkout:

```bash
pip install -e ".[tuning]"
```

## What a tuning experiment must define

Before tuning, fix the experimental protocol just as you would for an algorithm
comparison:

1. **Parameter space** — what may change and its legal range.
2. **Training instances** — the problems on which candidate configurations are
   selected.
3. **Algorithm seeds** — stochastic replicates used for every candidate.
4. **Per-run evaluation budget** — the optimization budget given to each
   candidate/problem/seed run.
5. **Quality metric and direction** — for example IGD (lower is better) or HV
   (higher is better).
6. **Aggregation rule** — how repeated problem/seed scores become one scalar
   tuning score.

Keep the tuner seed separate from the algorithm seeds. The tuner seed controls
which configurations are proposed; the algorithm seeds control the stochastic
optimization runs used to evaluate each proposal. Report both.

For scientific use, reserve **held-out problems and/or seeds** for validation or
final testing. Do not choose a configuration on the same test blocks used for
the final claim.

## Programmatic random search

The maintained runnable example is:

```bash
python examples/tuning/random_search_nsgaii.py
```

It deliberately stays small for documentation smoke tests: two candidate
configurations are evaluated on `ZDT1 × ZDT2 × seeds {0, 1}` with 80 function
evaluations per run. The score is a self-contained IGD estimate against the
analytic ZDT fronts, aggregated with the median. This is a **teaching design**,
not a recommended sample size for a publication.

The essential mapping is:

```python
from typing import Any

from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.engine.tuning import EvalContext


def evaluate_config(config: dict[str, Any], ctx: EvalContext) -> float:
    algorithm_config = (
        NSGAIIConfig.builder()
        .pop_size(20)
        .selection("tournament")
        .crossover("sbx", prob=float(config["crossover_prob"]), eta=20.0)
        .mutation(
            "pm",
            prob=1.0 / ctx.instance.n_var,
            eta=float(config["mutation_eta"]),
        )
        .build()
    )
    result = optimize(
        ctx.instance.name,
        algorithm="nsgaii",
        algorithm_config=algorithm_config,
        max_evaluations=ctx.budget,
        n_var=ctx.instance.n_var,
        seed=ctx.seed,
        engine="numpy",
    )
    if result.F is None or result.F.size == 0:
        raise RuntimeError("NSGA-II returned no objective vectors.")
    return igd(result.F, ctx.instance.name)  # lower is better
```

Here `EvalContext` is important: the tuner chooses the candidate configuration,
while the tuning task supplies the problem, seed, and evaluation budget for the
current experimental block. The complete example defines the small `igd(...)`
helper, parameter space, `TuningTask`, and `RandomSearchTuner` around this
function.

A compact task looks like this:

```python
import numpy as np

from vamos.engine.tuning import Instance, ParamSpace, RandomSearchTuner, Real, TuningTask

space = ParamSpace(
    params={
        "crossover_prob": Real("crossover_prob", 0.7, 1.0),
        "mutation_eta": Real("mutation_eta", 10.0, 40.0),
    }
)

task = TuningTask(
    name="nsgaii_zdt_training_demo",
    param_space=space,
    instances=[Instance(name="zdt1", n_var=30), Instance(name="zdt2", n_var=30)],
    seeds=[0, 1],
    budget_per_run=80,
    maximize=False,
    aggregator=lambda values: float(np.median(values)),
)

tuner = RandomSearchTuner(task=task, max_trials=2, seed=7)
best_config, history = tuner.run(evaluate_config, verbose=False)
```

Do not interpret the best score from a tiny search as evidence that the
configuration is generally superior. The selection process itself creates
optimism; validate the selected configuration on blocks that were not used to
choose it.

## Racing tuner

`RacingTuner` uses the same `TuningTask` and evaluation function, but can stop
spending evaluations on weak candidates as evidence accumulates.

```python
from vamos.engine.tuning import RacingTuner, Scenario

tuner = RacingTuner(
    task,
    scenario=Scenario(
        max_experiments=50,
        min_survivors=3,
        n_jobs=4,
    ),
    seed=7,
)
best_config, history = tuner.run(evaluate_config, verbose=False)
```

`Scenario.max_experiments` counts configuration-evaluation blocks
(`configuration × instance × seed`), not objective-function evaluations inside
the MOEA. The latter are controlled by `TuningTask.budget_per_run` (or the
configured fidelity schedule). Keep those two budgets distinct when estimating
compute cost.

Racing can use paired statistical elimination after enough blocks have been
observed. That is an allocation mechanism inside the experimental tuner; it is
not a substitute for an independently designed final comparison on held-out
blocks.

## Command line: `vamos tune`

`vamos tune` is the experimental orchestration path for larger tuning runs. Its
**current default backend is `optuna`**, which requires the optional `tuning`
extra. For a core-installation command that is safe to copy and paste, select a
built-in backend explicitly:

```bash
vamos tune \
  --algorithm nsgaii \
  --problem zdt1 \
  --backend random \
  --budget 1000 \
  --tune-budget 20 \
  --n-jobs 1
```

Available backend families are:

- `random` — built-in random search;
- `racing` — built-in racing;
- `optuna` — current CLI default; requires the tuning extra;
- `bohb_optuna`, `smac3`, and `bohb` — optional model-based backends from the
  tuning extra.

### Cheap verification path

Use the explicit built-in backend plus `--smoke` when you only want to verify
the CLI and artifact path:

```bash
vamos tune \
  --instances zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1 \
  --algorithm nsgaii \
  --backend random \
  --smoke \
  --output-dir results/tuning_smoke
```

`--smoke` clamps budgets and workers and disables validation, test, and
statistical-finisher stages. It is a real execution path, but its tiny budget
is not intended for scientific conclusions.

### Larger split-based example

```bash
vamos tune \
  --instances zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1 \
  --algorithm nsgaii \
  --backend optuna \
  --backend-fallback random \
  --split-strategy suite_stratified \
  --budget 5000 \
  --tune-budget 200 \
  --n-jobs -1
```

The key controls include:

- `--algorithm`: algorithm family to tune;
- `--instances`: comma-separated problem list; overrides `--problem`;
- `--backend`: `racing`, `random`, `optuna`, `bohb_optuna`, `smac3`, or `bohb`;
- `--backend-fallback`: behavior when an optional model backend is unavailable;
- `--split-strategy`: `suite_stratified` or `random` instance splitting;
- `--budget`: per-run algorithm evaluation budget;
- `--tune-budget`: racing experiments or model trials;
- `--n-seeds`: algorithm seeds per candidate configuration;
- `--aggregate-mode`: aggregation across instance/seed scores;
- `--n-jobs`: parallel workers (`-1` means CPU cores minus one);
- `--run-validation`, `--run-test`: optional post-tuning evaluation stages;
- `--run-statistical-finisher`: optional paired-test selection on the training
  split top-k.

Output artifacts include:

- `best_config_raw.json` and `best_config_active.json`;
- `tuning_history.json` and `tuning_history.csv`;
- `tuning_summary.json`;
- `split_instances.csv` and `split_seeds.json`;
- optional finisher/validation/test artifacts when those stages are enabled.

The tuning spaces can also include external-archive controls. When an external
archive is enabled, remember that archive/result semantics affect the quality
metric you are tuning; define the result source consistently across candidate
configurations.

## Ablation planning

`vamos ablation` is also experimental. For a durable scientific comparison of
already chosen variants, the stable study lifecycle is preferable: represent
each scientifically distinct configuration explicitly, plan the matrix before
execution, and retain the canonical run provenance.

```python
from pathlib import Path

from vamos import StudySpec, create_study, plan_study

output_root = Path("results/ablation_demo")
summaries = {}
for variant, population_size in {"baseline": 50, "tuned": 80}.items():
    spec = StudySpec(
        problems=["zdt1", "dtlz2"],
        algorithms=["nsgaii"],
        seeds=[1, 2, 3],
        max_evaluations=20_000,
        pop_size=population_size,
        algorithm_configs={"nsgaii": {"pop_size": population_size}},
        labels={"workflow": "ablation", "variant": variant},
    )
    planned = plan_study(spec)
    completed = create_study(spec, output=output_root / variant).run()
    assert completed.plan_id == planned.plan_id
    summaries[variant] = completed.summarize()
```

Build caller-specific analysis tables from `StudySummary.rows`, but retain the
canonical provenance fields (`study_id`, `plan_id`, `task_id`,
`selected_run_id`, `run_manifest_path`, and `run_manifest_sha256`). A derived
table is an analysis artifact, not resume authority.

For executable ablation material, see:

- `examples/tuning/ablation_runner.py`;
- `notebooks/2_advanced/32_ablation_planning.ipynb`;
- `examples/configs/study_nsgaii.json` for a configuration whose keys map to
  `StudySpec`.

When interpreting variant contributions, define the metric, replicate
aggregation, and statistical procedure before comparing deltas against the
baseline. Tuning, validation, final testing, and ablation answer different
questions; keep their data partitions and provenance explicit.
