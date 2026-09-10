# Hyperparameter Tuning

VAMOS can tune **algorithm configuration**: operator probabilities, distribution
indices, population-related settings, archive controls, and other parameters
that control a MOEA. A tuning campaign evaluates candidate configurations over
explicit problem and seed blocks and ranks them by a scalar quality score.

!!! warning "Experimental surface in VAMOS 1.0.0"
    The tuning/racing implementation and the `vamos tune` and `vamos ablation`
    commands are **experimental**. They are supported for evaluation but may
    change incompatibly in a minor release. Stable optimization calls used by
    the tuning runtime remain governed by the
    [stability policy](../project/stability-and-versioning.md).

This page is about configuring optimization algorithms. If the variables in
*your optimization problem* are model hyperparameters (for example SVM `C` and
`gamma`), that is a different workflow; see
`examples/tuning/hyperparam_tuning.py` for that formulation.

## Programmatic API status

VAMOS 1.0.0 does **not** expose a curated public programmatic facade for the
algorithm-configuration tuner. Maintained user workflows should therefore use
`vamos tune` rather than import deep `vamos.engine.*` implementation modules.

The implementation-level tuning and racing classes remain documented for
advanced evaluation and contributors under the
[experimental tuning API reference](../reference/api/experimental/tuning.md),
but those deep imports are not a compatibility commitment or a copy-paste
learning path.

## What a tuning experiment must define

Before tuning, fix the experimental protocol just as you would for an algorithm
comparison:

1. **Parameter space** — what may change and its legal range.
2. **Training instances** — the problems on which candidate configurations are
   selected.
3. **Algorithm seeds** — stochastic replicates used for every candidate.
4. **Per-run evaluation budget** — the optimization budget given to each
   candidate/problem/seed run.
5. **Quality metric and direction** — for example IGD+ (lower is better) or HV
   (higher is better).
6. **Aggregation rule** — how repeated problem/seed scores become one scalar
   tuning score.

The current CLI has one important seed-coupling constraint: `--seed` is the
global tuning seed **and** the base used to derive the training algorithm-seed
schedule. `--n-seeds` changes how many training seeds are derived; it does not
let you provide an independent training-seed list. Therefore changing
`--seed` changes both configuration-search randomness and training evaluation
randomness. Record `--seed` and `--n-seeds` together and do not interpret them
as independently controlled factors. `--split-seed` separately controls the
problem split, while validation/test seed lists can be overridden with
`--validation-seeds` and `--test-seeds`.

For scientific use, reserve **held-out problems and/or seeds** for validation or
final testing. Do not choose a configuration on the same test blocks used for
the final claim.

## Install

The built-in `random` and `racing` backends use the core installation.

Model-based backends (`optuna`, `smac3`, `bohb`, and `bohb_optuna`) require the
optional tuning dependencies:

```bash
pip install "vamos-optimization[tuning]"
```

From a local checkout:

```bash
pip install -e ".[tuning]"
```

## Start with a core-installation run

The `vamos tune` CLI currently defaults to **`optuna`**, so omitting
`--backend` requires the optional tuning extra. For a command that works with a
core installation, choose a built-in backend explicitly:

```bash
vamos tune \
  --algorithm nsgaii \
  --problem zdt1 \
  --backend random \
  --budget 1000 \
  --tune-budget 20 \
  --n-jobs 1
```

The important budgets are different quantities:

- `--budget` is the MOEA objective-evaluation budget for each candidate run;
- `--tune-budget` is the configuration-search budget (trials/experiments,
  depending on backend).

Do not report only `--tune-budget` when estimating compute cost. Candidate
configurations are evaluated across the selected instances and seeds, and each
of those runs consumes its own algorithm budget.

## Cheap verification path

Use `--smoke` when you only want to verify the CLI, evaluator, and artifact
pipeline:

```bash
vamos tune \
  --instances zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1 \
  --algorithm nsgaii \
  --backend random \
  --smoke \
  --output-dir results/tuning_smoke
```

`--smoke` clamps budgets and workers and disables validation, test, and
statistical-finisher stages. It is a real execution path and is exercised by
the documentation smoke suite, but its tiny design is **not** intended for
scientific conclusions.

## Choose a backend deliberately

The current backend families are:

- `random` — built-in random search;
- `racing` — built-in racing that progressively allocates experiments and can
  eliminate weak candidates;
- `optuna` — current CLI default; requires the tuning extra;
- `bohb_optuna`, `smac3`, and `bohb` — optional model-based backends from the
  tuning extra.

A racing run uses the same scientific ingredients as any other tuning run:
problem blocks, algorithm seeds, a per-run budget, a metric, and an aggregation
rule. Its statistical elimination is an **allocation mechanism during tuning**;
it is not a substitute for an independently designed final comparison on
held-out blocks.

## Split-based tuning

For a larger campaign, use explicit instance splitting and post-tuning stages:

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
- `--backend-fallback`: fallback when an optional model backend is unavailable;
- `--split-strategy`: `suite_stratified` or `random` instance splitting;
- `--split-seed`: random seed for the instance split;
- `--seed`: coupled global tuner seed and base for training algorithm seeds;
- `--n-seeds`: number of training algorithm seeds derived from `--seed`;
- `--validation-seeds`, `--test-seeds`: optional explicit post-selection seed
  schedules;
- `--budget`: per-run algorithm evaluation budget;
- `--tune-budget`: racing experiments or model trials;
- `--aggregate-mode`: aggregation across instance/seed scores;
- `--n-jobs`: parallel workers (`-1` means CPU cores minus one);
- `--run-validation`, `--run-test`: optional post-tuning evaluation stages;
- `--run-statistical-finisher`: optional paired-test selection on the training
  split top-k.

Define the split before interpreting results. Tuning chooses among candidate
configurations; validation can support model-selection decisions; the final
test split should answer the pre-specified performance question without being
fed back into another tuning round.

## Artifacts and provenance

Tuning output includes:

- `best_config_raw.json` and `best_config_active.json`;
- `tuning_history.json` and `tuning_history.csv`;
- `tuning_summary.json`;
- `split_instances.csv` and `split_seeds.json`;
- optional finisher/validation/test artifacts when those stages are enabled.

Keep these together with the command/configuration and environment used for the
campaign. A best configuration without its search space, seeds, metric,
aggregation rule, budgets, and split is not a reproducible tuning result.

The tuning spaces can include external-archive controls. When an external
archive is enabled, archive/result semantics affect the metric being tuned;
keep the result source consistent across candidate configurations.

## Ablation planning

`vamos ablation` is also experimental. For a durable scientific comparison of
already chosen variants, the stable study lifecycle is preferable: represent
each scientifically distinct configuration explicitly, plan the matrix before
execution, and retain canonical run provenance.

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
