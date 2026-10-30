# Hyperparameter Tuning

VAMOS can tune **algorithm configuration**: operator probabilities, distribution
indices, population-related settings, and other parameters that control a MOEA.
A tuning campaign evaluates candidate configurations over explicit problem and
seed blocks and ranks them by a scalar quality score.

!!! warning "Experimental surface in VAMOS 1.1.0"
    The tuning/racing implementation and the `vamos tune` and `vamos ablation`
    commands are **experimental**. They are supported for evaluation but may
    change incompatibly in a minor release. Stable optimization calls used by
    the tuning runtime remain governed by the
    [stability policy](../project/stability-and-versioning.md).

This page is about configuring optimization algorithms. If the variables in
*your optimization problem* are model hyperparameters (for example SVM `C` and
`gamma`), that is a different workflow; see
[`examples/tuning/hyperparam_tuning.py`](https://github.com/vamos-optimization/VAMOS/blob/main/examples/tuning/hyperparam_tuning.py) for that formulation.

## Changes included in 1.0.1

The install command below selects **VAMOS 1.1.0**, which keeps the corrected
scoring, archive-search, and persistence behavior introduced in 1.0.1 and
described on this page:

| Behavior | VAMOS 1.0.0 | VAMOS 1.0.1 |
| --- | --- | --- |
| HV input | Top-level `result.F`; may be population, archive, or SMPSO leaders depending on the algorithm/configuration | Feasible non-dominated final population |
| External-archive search parameters | Included where defined by the search-space builder | Removed from ordinary CLI tuning |
| Optuna persistent study name | User-supplied name, without the scoring-contract suffix | Adds `__vamos_cli_final_population_hv_v1` |
| AGE-MOEA/RVEA constraint handling | Constraints are not used in survival | Feasibility-aware behavior when enabled |

Record the package version and any source commit with a tuning campaign. Do not
mix persisted trials from the two scoring contracts: their result sources and
search spaces differ. See also the
[algorithm changes](../reference/algorithms.md#changes-included-in-101).

## Programmatic API status

VAMOS 1.1.0 does **not** expose a curated public programmatic facade for the
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

### CLI scoring contract: hypervolume

In VAMOS 1.1.0, the maintained `vamos tune` CLI has **no metric selector**. It scores
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

!!! note "Constrained AGE-MOEA and RVEA (1.0.1)"
    Since VAMOS 1.0.1, AGE-MOEA and RVEA preserve population-aligned constraint values whenever
    constraint handling is active, so constrained runs can use the same
    final-population HV scoring contract as the other maintained algorithms.
    Their stable `constraint_mode="none"` setting remains an explicit opt-out:
    in that mode constraints are ignored and no `G` values participate in
    survival, archives, or population results.

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
python -m pip install "vamos-optimization[tuning]==1.1.0"
```

For an editable development installation, follow
[Install from a source checkout](../guide/installation.md#install-from-a-source-checkout)
and select the `tuning` extra from the repository root.

The `irace` backend needs no Python extra. It needs R with `Rscript` and the
CRAN package irace 4.4.4 or newer (`install.packages("irace")` in R); see
[The irace backend](#the-irace-backend).

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

### Model-based backends run one fidelity unless given levels

**Development after 1.1.0:** without `--fidelity-levels` (`budget_levels=None`
in Python), `optuna`, `bohb_optuna`, `smac3` and `bohb` evaluate every
candidate once, at `--budget`, on every training instance and seed. A trial
then costs one run per instance/seed pair, and `optuna` and `bohb_optuna` prune
nothing. VAMOS 1.1.0 and earlier invented a schedule instead: Optuna evaluated
each trial at `b/3`, `2b/3` and `b` (about `2b` evaluations per pair), and
SMAC3 and BOHB ran Hyperband from 1 evaluation, so most of their runs had too
few evaluations for any population and failed.

With `--fidelity-levels`, the last level is always `--budget`:

- `optuna` and `bohb_optuna` evaluate a trial at each level in turn, each from
  scratch, so a trial that is never pruned costs the sum of the levels per
  instance/seed pair. The median pruner (`optuna`) or the Hyperband pruner
  (`bohb_optuna`) can stop a trial after any level but the last. A pruned trial
  stays in the history with state `PRUNED` and its last score, but it is never
  returned as the best configuration.
- `smac3` and `bohb` run Hyperband between the smallest and the largest level;
  the intermediate budgets are the largest level divided by powers of η (3 for
  SMAC3, `--bohb-reduction-factor` for BOHB), not the intermediate levels.
  `--tune-budget` counts SMAC3 runs (one per configuration and budget) and the
  configurations that BOHB samples.
- `smac3` and `bohb` return the best configuration evaluated at the full
  budget, and fail with an error when the search ends before any configuration
  reaches it. VAMOS 1.1.0 and earlier returned the best score of any budget, and
  BOHB ignored `--tune-budget`.

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
  tuning extra;
- `irace` — iterated racing with the R package irace, run through `Rscript`.

A racing run uses the same scientific ingredients as any other tuning run:
problem blocks, algorithm seeds, an explicit fidelity/budget schedule, the CLI
HV score, and an aggregation rule. Its statistical elimination is an
**allocation mechanism during tuning**; it is not a substitute for an
independently designed final comparison on held-out blocks.

**Development after 1.1.0:** racing returns the best of the configurations
still alive when the race ends, aggregated over the problem/seed blocks that
all of them were evaluated on; with multi-fidelity racing, over the last
fidelity level they reached. VAMOS 1.1.0 and earlier compared every evaluated
configuration on its own blocks, so a configuration the race had already
eliminated could be returned.

### Conditional parameters

A trial contains only the parameters that are active for the rest of its
assignment: `crossover_eta` only for SBX, `archive_prune_policy` only when the
external archive is enabled and bounded, and, since VAMOS 1.1.0 for real-valued
operator spaces, `mutation_eta` only for polynomial and linked polynomial
mutation. The evaluation callback, the trial history, and the returned best
configuration contain that active assignment. Before VAMOS 1.1.0, the
model-based backends kept every parameter, and `random` and `racing` could keep
a parameter whose own conditions held while its parent was inactive. They now
still draw every value in declaration order, so their seeded values do not
change, and keep a parameter only when its parents are kept.

`optuna` and `bohb_optuna` suggest parents before the parameters that depend on
them, whatever their declaration order, and record only active suggestions in
`trial.params`. `smac3` and `bohb` receive the dependencies as ConfigSpace
conditions, so their models see them too. Several conditions on one parameter
are combined with AND, and descendants of an inactive parameter stay inactive.
The ConfigSpace translation supports comparisons of `cfg['parent']` with a
scalar literal (`==`, `!=`, `<`, `<=`, `>`, `>=`), boolean parents, and
`and`/`or`/`not`; any other condition, an unknown parent, or a dependency cycle
fails with an explicit error before sampling starts. `AlgorithmConfigSpace`
samples and encodes parameters in the same dependency order, so a parameter
may be declared before its parent. `irace` receives the same subset as
irace conditions; see [The irace backend](#the-irace-backend).

HpBandSter's native BOHB accepts no seed, and its configuration generator draws
from NumPy's global random state. The `bohb` backend gives the generator of
each search a private random stream derived from the tuning seed instead.
NumPy's global state therefore remains entirely the caller's, including inside
the evaluation callback. With one worker (`--n-jobs 1`) a native BOHB search is
reproducible for a seed; with several workers, the order in which results
arrive can still change it.

Inactive parameters never changed how a candidate algorithm was built, so the
scores of studies persisted by earlier versions remain comparable and keep
their namespace. Their stored trials may record inactive values; when such a
study is resumed, the history and the best configuration leave them out. Seeded
`random` and `racing` searches propose the same candidates as VAMOS 1.0.2,
without an inactive `mutation_eta`, except for MOEA/D: its `mutation_eta` moved
into the polynomial-mutation block, so a seeded MOEA/D search proposes
different candidates.

### The irace backend

`vamos tune --backend irace` runs the R package irace. In the CLI,
`--tune-budget` is irace's `maxExperiments`, as for `racing`; `--n-jobs`
becomes irace's `parallel`; `--irace-rscript` selects `Rscript`;
`--irace-instance-seeds` chooses who provides the seeds (below);
`--irace-option KEY=VALUE` adds a scenario option and may be repeated; and
the scenario, `irace.log` and `irace.Rdata` are kept in `<output>/irace`
unless `--irace-work-dir` names another directory. The CLI rejects
`--timeout-seconds` and `--fidelity-levels` with this backend, and falls
back through `--backend-fallback` only when the configured `Rscript` is
missing. irace needs many experiments even for a short race: the built-in
real-valued NSGA-II space requires at least 728 with its default iterations,
so `--smoke` uses 64 experiments and one iteration (`nbIterations=1`) unless
that option is given.

In Python, `ModelBasedTuner(backend="irace")` belongs to the
implementation-level API described in
[Programmatic API status](#programmatic-api-status). It keeps the contract of
the other model-based backends: `tuner.run(eval_fn)` calls
`eval_fn(assignment, ctx)` in the calling process and returns the best
configuration and the history.

VAMOS writes an irace scenario in `irace_work_dir` (a temporary directory by
default) and runs irace with `irace_rscript` (`Rscript` by default). irace's
target runner is a small client, started with the current Python interpreter,
that forwards each experiment over a local socket to the running tuner and
returns the cost; scores are negated when the task maximizes, because irace
minimizes. An error raised by `eval_fn` stops irace and is re-raised as the
cause of a `RuntimeError`.

- **Space:** the `ParamSpace` becomes an irace parameter file. Conditions use
  the subset described in [Conditional parameters](#conditional-parameters);
  conditions with `or` or `not` also require every parent they read to be
  active, as VAMOS does. Categorical choices must have distinct tokens made of
  letters, digits, `_`, `.`, `+` or `-`. irace rounds real values to four
  decimal places by default.
- **Instances and seeds:** by default each instance/seed pair of the task is an
  irace instance of a deterministic race, so `ctx.seed` always comes from
  `task.seeds`. With `irace_instance_seeds="irace"`, irace receives the
  instances and draws the seeds itself.
- **Budget:** `maxExperiments` is `max_trials` times the number of task pairs
  unless `irace_max_experiments` is given; each call uses
  `task.budget_per_run`. There is one fidelity level and no wall-clock limit.
- **Parallelism and options:** `n_jobs` becomes irace's `parallel`, so
  `eval_fn` must be thread-safe when it is greater than one. `seed` becomes
  irace's seed, `irace_initial_configs` its initial configurations, and
  `irace_options` adds other scenario options; the options the backend writes
  itself are rejected there.
- **Result:** the best configuration is irace's first final elite. The history
  has one entry per configuration, scored with `task.aggregator` over its
  evaluations; `details` records the number of evaluations and the final elite
  rank. `irace.Rdata` and `irace.log` stay in `irace_work_dir` when it is set.

### Persistent Optuna studies are scoring-contract versioned

In VAMOS 1.1.0, when `optuna` or `bohb_optuna` uses `--optuna-storage`, the maintained CLI treats
`--optuna-study-name` as a base name and appends
`__vamos_cli_final_population_hv_v1` to the effective persisted study name. The
same versioned namespace is used whether `--optuna-load-if-exists` is enabled or
disabled.

This isolation is intentional: trials persisted by an older VAMOS tuning CLI
may contain archive-derived scores and a different search space, so they must
not compete with trials scored under the current final-population HV contract.
Reusing the same base study name resumes only studies from this contract
version. A future incompatible scoring/search-space change must use a new
namespace rather than mixing historical trial values.

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
- `--backend`: `racing`, `random`, `optuna`, `bohb_optuna`, `smac3`, `bohb`,
  or `irace`;
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
- `--tune-budget`: racing or irace experiments, or model trials;
- `--irace-rscript`, `--irace-instance-seeds`, `--irace-work-dir`,
  `--irace-option`: irace settings; see [The irace backend](#the-irace-backend);
- `--multi-fidelity` / `--no-multi-fidelity`: enable or disable racing
  multi-fidelity execution;
- `--fidelity-levels`: explicit increasing candidate-run budgets for the
  multi-fidelity schedule;
- `--aggregate-mode`: aggregation across instance/seed scores;
- `--n-jobs`: parallel workers (`-1` means CPU cores minus one);
- `--optuna-storage`, `--optuna-study-name`: persistent Optuna storage and the
  base study name used by the contract-versioned namespace;
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

The 1.1.0 CLI intentionally excludes external-archive controls from its
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

What VAMOS calls an ablation is a comparison of predefined variants: every
variant runs on every problem and seed as its own study. It is not the
ablation analysis of Fawcett and Hoos (2016), which walks a path from a source
configuration (the default) to a target configuration (a tuned one), greedily
choosing at each step the parameter change that helps most, and uses racing to
make the steps cheaper. VAMOS neither builds that path nor connects it to the
output of `vamos tune`.

**Development after 1.1.0:** every key of `base_config` and of a variant's
`config_overrides` is either an ablation setting (`population_size`,
`offspring_population_size`, `eval_strategy` and `output_root`) or a field of
the algorithm configuration, such as `pop_size` or
`crossover: [sbx, {prob: 0.5}]`, which reaches the study's `algorithm_configs`.
Any other key fails with a `ConfigurationError` that suggests the closest
field. Operators are accepted as `[name, {params}]` pairs or as mappings with
`method`, in `config_overrides` and in `variations`. Every variant is planned
before the first one runs, so an invalid variant stops the ablation before any
run. With `vamos ablation`, each variant writes its studies to
`<output_root>/<variant>/study-NNNN` unless `--flat-output` or
`per_variant_output_root: false` is given, and `output_root_by_variant` moves
one variant elsewhere. VAMOS 1.1.0 kept only `population_size`,
`offspring_population_size` and `eval_strategy`, so a variant that changed any
other key silently ran the base configuration; it wrote every study to
`<output_root>/study-NNNN`, and the mapping form of an operator in `variations`
failed after the earlier variants had already run.

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

- [`examples/tuning/ablation_runner.py`](https://github.com/vamos-optimization/VAMOS/blob/main/examples/tuning/ablation_runner.py);
- [`notebooks/2_advanced/32_ablation_planning.ipynb`](https://github.com/vamos-optimization/VAMOS/blob/main/notebooks/2_advanced/32_ablation_planning.ipynb);
- [`examples/configs/study_nsgaii.json`](https://github.com/vamos-optimization/VAMOS/blob/main/examples/configs/study_nsgaii.json) for a configuration whose keys map to
  `StudySpec`.

When interpreting variant contributions, define the metric, replicate
aggregation, and statistical procedure before comparing deltas against the
baseline. Tuning, validation, final testing, and ablation answer different
questions; keep their data partitions and provenance explicit.

## A bounded worked experiment

Follow [the tuning tutorial](../tutorials/tuning-experiment.md) for a verified standalone CLI wrapper, two recorded search trials and separate validation/test seeds. For model parameters as decision variables, use [the mixed SVC example](../tutorials/model-hyperparameters.md). For a controlled component removal using stable studies, use [the ablation lesson](../tutorials/controlled-ablation.md).
