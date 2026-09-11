# Hyperparameter Tuning

VAMOS can tune **algorithm configuration**: operator probabilities, distribution
indices, population-related settings, and other parameters that control a MOEA.
A tuning campaign evaluates candidate configurations over explicit problem and
seed blocks and ranks them by a scalar quality score.

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
4. **Evaluation-budget schedule** — the optimization budget or fidelity levels
   given to each candidate/problem/seed run.
5. **Scoring contract** — the metric, reference point, result source, direction,
   runtime penalty, and failure handling actually used by the selected tuning
   interface.
6. **Aggregation rule** — how repeated problem/seed scores become one scalar
   tuning score.

### Current CLI scoring contract: hypervolume

The maintained `vamos tune` CLI currently has **no metric selector**. It scores
candidate runs with **hypervolume (HV) and maximizes that score**.

The score source is fixed: VAMOS takes the **final population**, removes
infeasible rows when constraint values are present, Pareto-filters the remaining
objective vectors, and computes HV on that feasible non-dominated population.
It does not score top-level `result.F`, because that field can refer to an
external archive for archive-enabled algorithm configurations.

The maintained CLI also removes `use_external_archive`, `archive_unbounded`, and
`archive_prune_policy` from its built-in search space. This keeps the optimized
parameter space aligned with the set actually used for scoring. Archive design
can still be studied explicitly in a separate controlled experiment, but it is
not mixed into ordinary `vamos tune` comparisons.

!!! warning "Constrained AGE-MOEA and RVEA"
    The maintained experimental `vamos tune` CLI does not currently support
    constrained AGE-MOEA and RVEA runs. Those engines do not expose the
    population-aligned constraint values required by this scorer without
    changing their stable constraint-mode semantics. The CLI therefore fails
    explicitly for those algorithm/problem combinations instead of silently
    treating infeasible rows as feasible. Use an explicit controlled study when
    constrained AGE-MOEA or RVEA is the research target.

- `--ref-point` supplies one HV reference point for the tuning run.
- If it is omitted (or cannot be parsed with the required dimensionality), the
  current default is `[10.0, ..., 10.0]`, one value per objective.
- `--runtime-penalty` changes the scalar score to
  `HV - lambda * log1p(runtime_seconds)`; its default is `0.0`, so the default
  score is plain HV.
- `--failure-score` has a narrower scope than its name may suggest. Its default
  is `0.0`, and it is used when the evaluator catches a failure while running
  the candidate algorithm and therefore has no usable result. It is **not a
  universal failure policy**: exceptions raised later while scoring HV can
  propagate in the `random` backend, while racing may substitute its own
  backend-level sentinel instead of `--failure-score`.
- `--aggregate-mode` then combines the per-block scores using `mean`, `median`,
  `p25`, or `p10`.

Choose and report a reference point that is meaningful for **all** problems in
the tuning campaign. The CLI uses the same supplied reference point across the
selected instances. In particular, validate that the reference point is valid
for the objective values produced by every selected problem before a long run;
`--failure-score` should not be relied on to rescue an invalid HV scoring
setup. If you need IGD, IGD+, epsilon indicators, or another selection metric,
the current maintained CLI cannot select it; that requires a custom/experimental
workflow rather than a `vamos tune` flag.

### Current CLI seed coupling

The current CLI also has an important seed-coupling constraint: `--seed` is the
global tuning seed **and** the base used to derive the training algorithm-seed
schedule. `--n-seeds` changes how many training seeds are derived; it does not
let you provide an independent training-seed list.

Therefore changing `--seed` changes both configuration-search randomness and
training evaluation randomness. Record `--seed` and `--n-seeds` together and
do not interpret them as independently controlled factors. `--split-seed`
separately controls the problem split, while validation/test seed lists can be
overridden with `--validation-seeds` and `--test-seeds`.

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

- `--budget` is the candidate-run MOEA evaluation budget for `random` and the
  ordinary non-racing path; it also supplies the baseline budget used by
  downstream validation/test stages unless those stages override it.
- `--tune-budget` is the configuration-search budget (trials/experiments,
  depending on backend).

Do not report only `--tune-budget` when estimating compute cost. Candidate
configurations are evaluated across the selected instances and seeds, and each
of those runs consumes its own algorithm budget.

### Racing has its own fidelity-budget schedule

`racing` is a special case. The CLI enables multi-fidelity racing by default.
If `--fidelity-levels` is omitted, the current racing schedule is
**`1000,3000,10000` evaluations**. Those values are passed directly as the
candidate-run budgets at successive fidelity levels; they are not capped by
`--budget`. Therefore, for example, `--backend racing --budget 5000` can still
execute promoted candidate blocks with a 10,000-evaluation budget.

For a fixed-budget racing experiment, disable multi-fidelity explicitly:

```bash
vamos tune \
  --algorithm nsgaii \
  --problem zdt1 \
  --backend racing \
  --budget 5000 \
  --no-multi-fidelity
```

For multi-fidelity racing, specify the schedule explicitly and include it in the
experimental record. If 5000 evaluations is intended to be the maximum
fidelity, for example:

```bash
vamos tune \
  --algorithm nsgaii \
  --problem zdt1 \
  --backend racing \
  --budget 5000 \
  --fidelity-levels 1000,3000,5000
```

Treat `--fidelity-levels`, rather than `--budget`, as the authoritative tuning
budget schedule while multi-fidelity racing is enabled. `--fidelity-promotion-ratio`
and `--fidelity-min-configs` additionally affect how many configurations reach
each level, so compute estimates should account for the schedule and promotion
policy together.

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
  eliminate weak candidates; the CLI enables its multi-fidelity schedule by
  default;
- `optuna` — current CLI default; requires the tuning extra;
- `bohb_optuna`, `smac3`, and `bohb` — optional model-based backends from the
  tuning extra.

A racing run uses the same scientific ingredients as any other tuning run:
problem blocks, algorithm seeds, an explicit fidelity/budget schedule, the CLI
HV score, and an aggregation rule. Its statistical elimination is an
**allocation mechanism during tuning**; it is not a substitute for an
independently designed final comparison on held-out blocks.

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
- `--ref-point`: global HV reference point used by the CLI scorer;
- `--budget`: ordinary candidate-run budget; not a cap on enabled racing
  multi-fidelity levels;
- `--tune-budget`: racing experiments or model trials;
- `--multi-fidelity` / `--no-multi-fidelity`: enable or disable racing
  multi-fidelity execution;
- `--fidelity-levels`: explicit increasing candidate-run budgets for the
  multi-fidelity schedule;
- `--aggregate-mode`: aggregation across instance/seed scores;
- `--n-jobs`: parallel workers (`-1` means CPU cores minus one);
- `--run-validation`, `--run-test`: optional post-tuning evaluation stages;
- `--run-statistical-finisher`: optional paired-test selection on the training
  split top-k.

Define the split and budget schedule before interpreting results. Tuning chooses
among candidate configurations; validation can support model-selection
decisions; the final test split should answer the pre-specified performance
question without being fed back into another tuning round.

## Artifacts and provenance

Tuning output includes:

- `best_config_raw.json` and `best_config_active.json`;
- `tuning_history.json` and `tuning_history.csv`;
- `tuning_summary.json`;
- `split_instances.csv` and `split_seeds.json`;
- optional finisher/validation/test artifacts when those stages are enabled.

Keep these together with the command/configuration and environment used for the
campaign. A best configuration without its search space, seeds, HV reference
point, fixed final-population score source, aggregation rule, budget/fidelity
schedule, and split is not a reproducible tuning result.

## Archive studies are separate from ordinary CLI tuning

The maintained CLI intentionally excludes external-archive controls from its
ordinary algorithm-configuration search so every candidate is compared on the
same final-population basis. If archive capacity, pruning, or archive-enabled
result semantics are themselves the research question, define them explicitly
as experimental variants and compare them under a pre-specified, common metric
and result-source protocol. For publication-grade claims, run the selected
variants on held-out blocks through the stable Study lifecycle and retain the
canonical run provenance.

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
