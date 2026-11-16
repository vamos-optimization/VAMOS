# Hyperparameter Tuning

VAMOS can tune **algorithm configuration**: operator probabilities, distribution
indices, population-related settings, and other parameters that control a MOEA.
A tuning campaign evaluates candidate configurations over explicit problem and
seed blocks and ranks them by a scalar quality score.

!!! warning "Experimental surface in VAMOS 1.3.0"
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

The install command below selects **VAMOS 1.3.0**, which keeps the corrected
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

VAMOS 1.3.0 does **not** expose a curated public programmatic facade for the
algorithm-configuration tuner. Maintained user workflows should therefore use
`vamos tune` rather than import deep `vamos.engine.*` implementation modules.

The implementation-level tuning and racing classes remain documented for
advanced evaluation and contributors under the
[experimental tuning API reference](../reference/api/experimental/tuning.md),
but those deep imports are not a compatibility commitment or a copy-paste
learning path.

**VAMOS 1.2.0:** the helpers without internal use are removed from
`vamos.engine.tuning`: the `convergence` module (`ConvergenceProfile`,
`ConvergenceDiagnostic`, `TuningCallback`, `diagnose`, `score_with_profile`),
`save_checkpoint` and `load_checkpoint` of the racing I/O (a race cannot be
resumed from them), and `AlgorithmConfigSpace.to_unit_vector` and
`from_unit_vector`. `TwoPhaseTuner` stays.

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

In VAMOS 1.3.0, the maintained `vamos tune` CLI has **no metric selector**. It scores
candidate runs with **hypervolume (HV) and maximizes that score**.

The score source is fixed: VAMOS takes the **final population**, removes
infeasible rows when constraint values are present, Pareto-filters the remaining
objective vectors, and computes HV on that feasible non-dominated population.
It does not score top-level `result.F`, because that field can refer to an
external archive for archive-enabled algorithm configurations.

**VAMOS 1.2.0:** on unconstrained problems, SMPSO is scored on its
leaders archive instead, because its final swarm is not elitist. Constrained
SMPSO runs keep scoring the swarm: the leaders carried no constraint values when
this contract was set, and scoring them now would change the tuning scores. Persistent Optuna studies for SMPSO
take the suffix `__vamos_cli_leaders_hv_v1`, so they do not mix with trials
scored on the swarm.

The maintained CLI also removes `use_external_archive`, `archive_unbounded`, and
`archive_prune_policy` from its built-in search space. This keeps the optimized
parameter space aligned with the set actually used for scoring. Archive design
can still be studied explicitly in a separate controlled experiment, but it is
not mixed into ordinary `vamos tune` comparisons.
**Development after 1.3.0:** the built-in spaces express the archive as one
choice, `external_archive`, which the CLI removes with `archive_prune_policy`.

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
- `--failure-score` (default `0.0`) is the score of a failed evaluation: a run
  of the candidate algorithm that raised, or a score that is not finite (see
  below).
- `--aggregate-mode` then combines the per-block scores using `mean`, `median`,
  `p25`, or `p10`.

**VAMOS 1.2.0:** a failed evaluation has one meaning in every
backend: the run raised, or its score is not finite (NaN or infinite). It
scores `--failure-score` in `vamos tune` (`TuningTask.failure_score` in Python,
-1e30 when maximizing and 1e30 when minimizing if unset), it is counted, and the
model-based backends and irace never receive a NaN. A run that fails in the
`vamos tune` evaluator is marked as failed instead of scoring like a valid
front with no point inside the reference, so the validation's `fail_rate`
counts it, and `tuning_summary.json` records `evaluations` and
`failed_evaluations` for the tuning phase. Racing, random-search and irace
histories record `failed_evaluations` per configuration. When every tuning
evaluation fails, `vamos tune` stops with exit code 1 and the first failure,
and `RacingTuner` and `RandomSearchTuner` raise `TuningFailedError`. VAMOS 1.1.0
reported `fail_rate=0` for runs that could not execute, let a single NaN stop
every elimination of a race, let an exception abort `--backend random`, and
ended a search in which nothing ran with exit code 0 and an arbitrary
configuration.

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
python -m pip install "vamos-optimization[tuning]==1.3.0"
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

`racing` is a special case. In VAMOS 1.1.0, the CLI enables multi-fidelity
racing by default; see the note at the end of this section.
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

**VAMOS 1.2.0:** `--backend racing` runs the statistical race by
default, at `--budget` per run, with `--alpha`, `--use-statistical-tests`,
`--elimination-fraction` and `--min-blocks-before-elimination`; pass
`--multi-fidelity` for the schedule above. The other backends keep
`--multi-fidelity` on by default. Multi-fidelity racing is successive halving:
at each level it keeps the configurations with the best mean score and runs no
statistical test, so it ignores those four options and warns when one of them
is set. Its first level evaluates every initial configuration on every training
instance/seed block; when that exceeds `--tune-budget`, the command now fails
before writing any output and names the options to change. In VAMOS 1.1.0 it
wrote the splits and then stopped with `RuntimeError` without evaluating
anything, as the split-based example below did with `--backend racing`. With
warm starts, each block now resumes only its own run: a block whose run failed
starts again from scratch, where it used to receive another block's checkpoint.

**VAMOS 1.3.0:** the statistical race eliminates with F-race
(Birattari et al., 2002), as irace does. Each instance/seed block ranks the
alive configurations, so problems whose scores lie on different scales weigh
the same. A Friedman test on those ranks comes first; when its p-value is below
`--alpha`, Conover's post-hoc test on the rank sums keeps the best
configuration and every configuration within the critical difference of it.
With two configurations alive, a paired Wilcoxon signed-rank test decides
instead. It drops the blocks where the two tie, and the sign of its statistic
over the other blocks says which configuration it eliminates, whatever their
order. When `Scenario.min_survivors` asks for more configurations than a test
keeps, the race keeps them all and adds the best of the others by aggregate
score. The first test waits for `Scenario.first_test` blocks (5, irace's
`firstTest`) or `--min-blocks-before-elimination`, whichever is larger. VAMOS
1.2.0 compared each configuration with the one of best mean through paired
t-tests on the raw scores, with Holm's correction. When half of the blocks had
scores 100 times larger, those tests eliminated no configuration out of eight
on ten blocks, against about three for F-race. `racing` runs a single race over
its initial configurations: it does not sample new ones around the elites as
irace's iterated racing does. For that, use `--backend irace`; see
[The irace backend](#the-irace-backend).

### Model-based backends run one fidelity unless given levels

**VAMOS 1.2.0:** without `--fidelity-levels` (`budget_levels=None`
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

**VAMOS 1.2.0:** `--smoke` also caps the sampled populations at
half of its 30-evaluation run budget (`--pop-size`, 15), so that every sampled
configuration can run, including the opposition-based initializer, which
evaluates a population and its opposite; for RVEA it caps `n_partitions`, whose
reference directions are the population. VAMOS 1.1.0 kept the [20, 200]
population range, and `optimize` rejected most smoke configurations, which
received `--failure-score`. A `vamos tune` run in which more than half of the
evaluations fail, but not all of them, now ends with a warning.
**Development after 1.3.0:** the spaces no longer sample the population, so
`--smoke` runs every configuration with `--pop-size` capped at 15, or with the
smallest reference lattice of MOEA/D, NSGA-III or RVEA when that is larger, and
records it under `fixed_parameters` in its smoke profile.
Without `--name`, the output folder is named after the `--instances` (the first
three and the number of the others), not after `--problem`.

## Choose a backend deliberately

The current backend families are:

- `random` — built-in random search;
- `racing` — built-in racing that progressively allocates experiments and can
  eliminate weak candidates; since VAMOS 1.2.0, the CLI runs
  its statistical race unless `--multi-fidelity` is given;
- `optuna` — current CLI default; requires the tuning extra;
- `bohb_optuna`, `smac3`, and `bohb` — optional model-based backends from the
  tuning extra;
- `irace` — iterated racing with the R package irace, run through `Rscript`.

A racing run uses the same scientific ingredients as any other tuning run:
problem blocks, algorithm seeds, an explicit fidelity/budget schedule, the CLI
HV score, and an aggregation rule. Its statistical elimination is an
**allocation mechanism during tuning**; it is not a substitute for an
independently designed final comparison on held-out blocks.

**VAMOS 1.2.0:** racing returns the best of the configurations
still alive when the race ends, aggregated over the problem/seed blocks that
all of them were evaluated on; with multi-fidelity racing, over the last
fidelity level they reached. VAMOS 1.1.0 and earlier compared every evaluated
configuration on its own blocks, so a configuration the race had already
eliminated could be returned.

**VAMOS 1.2.0:** with elitist restarts
(`Scenario(use_elitist_restarts=True)`), each configuration spawned to refill
the race is first evaluated on the blocks the race has already played, so it
enters the elimination tests on the same blocks as the elites. A restart
spawns only as many configurations as `max_experiments` leaves room to catch
up. Neighbours of an elite keep the active assignment of the space: a
parameter that the perturbation activates is drawn from its own range. In
VAMOS 1.1.0, restarts failed with a `KeyError` on any conditional space,
including every built-in one, and with statistical tests no configuration was
eliminated after the first refill.

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

**VAMOS 1.2.0:** racing's model-based sampler and
`filter_active_config`, which writes the trial history and
`best_config_active.json`, apply the same rule. In VAMOS 1.1.0 they could still
keep a grandchild such as `x_eta` (active when `x_kind == 'b'`) while its
grandparent `use_x` was false, in spaces with conditions on the direct parent
only. The built-in spaces were not affected.

**VAMOS 1.3.0:** the SMS-EMOA spaces choose the mating selection,
`random` or `tournament`, with `selection_pressure` only for a tournament, and
the offset of the reference point, `reference_point_offset` in [0.1, 2.0].
Both ranges include the defaults of `SMSEMOAConfig`: random selection and an
offset of 1.0. The spaces of VAMOS 1.2.0 always used a tournament and an
offset of 0.1, so no assignment reproduced the default configuration; an
assignment without the new parameters still builds that tournament
configuration.

**VAMOS 1.3.0:** the MOEA/D and SMS-EMOA spaces offer what
Evolver's `MOEADDouble` and `SMSEMOADouble` spaces offer, so that a
configuration tuned or published with Evolver is a point of them:

- every MOEA/D space: `neighbor_size` in [5, 50] and `delta` in [0, 1]
  (before, [5, 40] and [0.5, 0.95]), the `modified_tchebycheff` aggregation
  with `mtch_rho`, which only the permutation space offered, and `pbi_theta`
  up to 200 (before, 10), log-uniform because the usual value, 5, lies at the
  low end;
- the real-coded MOEA/D spaces (`moead`, `moead_constrained`):
  `linked_polynomial` mutation, with `mutation_eta` active for both polynomial
  mutations, `initializer` (`random`, `lhs`, `scatter`) and `repair` (`clip`,
  `reflect`, `random`);
- the real-coded MOEA/D and SMS-EMOA spaces: `mutation_prob_factor` in
  [0.25, 3.0], the probability being the factor divided by the number of
  variables, as in NSGA-II's space, instead of an absolute `mutation_prob` in
  [0.01, 0.5], which depends on the problem. Evolver's factor lies in [0, 2];
  VAMOS rejects a factor of 0.

An assignment saved from the earlier spaces builds the configuration it built:
an absolute `mutation_prob` stays absolute, and without `initializer` or
`repair` MOEA/D keeps its defaults. Seeded MOEA/D and SMS-EMOA tuning runs draw
different trials.

### Parameter names and Evolver's

**VAMOS 1.3.0:** the built-in spaces name their parameters after
Evolver's, in snake_case, so that a configuration tuned with VAMOS can be read
by Evolver and the other way round. The earlier names remain aliases:
`config_from_assignment` accepts both, and an assignment saved from an earlier
space builds the configuration it built. The fields of the configuration
classes (`MOEADConfig.neighbor_size`, ...) keep their names until VAMOS 2.0.
The paragraphs above use the earlier names.
`vamos.engine.tuning.racing.parameter_names` holds the table, with
`to_evolver` and `from_evolver` to exchange assignments:

| Earlier name | Space name | Evolver |
| --- | --- | --- |
| `pop_size` | `pop_size` | `populationSize` |
| `crossover_prob` | `crossover_probability` | `crossoverProbability` |
| `crossover_eta` | `sbx_distribution_index` | `sbxDistributionIndex` |
| `mutation_eta` | `polynomial_mutation_distribution_index` | `polynomialMutationDistributionIndex` |
| `mutation_eta` | `linked_polynomial_mutation_distribution_index` | `linkedPolynomialMutationDistributionIndex` |
| `mutation_prob_factor` | `mutation_probability_factor` | `mutationProbabilityFactor` |
| `neighbor_size` | `neighborhood_size` | `neighborhoodSize` |
| `delta` | `neighborhood_selection_probability` | `neighborhoodSelectionProbability` |
| `replace_limit` | `maximum_number_of_replaced_solutions` | `maximumNumberOfReplacedSolutions` |
| `aggregation` | `aggregation_function` | `aggregationFunction` |
| `pbi_theta` | `pbi_theta` | `pbiTheta` |
| `de_cr` | `de_crossover_rate` | `CR` |
| `de_f` | `de_scale_factor` | `F` |
| `initializer` | `initial_solutions` | `createInitialSolutions` |
| `selection_pressure` | `tournament_size` | `selectionTournamentSize` |
| `selection_size` | `tournament_size` | `selectionTournamentSize` |
| `crossover` | `crossover` | `crossover` |
| `mutation` | `mutation` | `mutation` |
| `selection` | `selection` | `selection` |
| `variation` | `variation` | `variation` |

Values that change: mutation `pm` becomes `polynomial` (the name the other
spaces already used) and crossover `undx` becomes `undc` (Evolver's `UNDC`).
The real-coded `uniform_reset` keeps its name: `random_reset` is the integer
reset mutation, and renaming one to the other broke integer runs. Evolver writes some values
differently, and the exchange functions translate them: `SBX`, `blxAlpha`,
`blxAlphaBeta`, `wholeArithmetic`, `fuzzyRecombination`, `PCX` and `UNDC`;
`linkedPolynomial`, `nonUniform`, `levyFlight` and `powerLaw`;
`tschebyscheff`, `weightedSum`, `penaltyBoundaryIntersection` and
`modifiedTschebyscheff`; `default` (random; **Development after 1.3.0:** `auto`), `latinHypercubeSampling`,
`scatterSearch` and `oppositionBased`; `stochasticUniversalSampling`.

`pop_size` keeps its name: Evolver passes the population size to the algorithm
rather than tuning it, and the CLI and Studio treat it as a run setting
(**Development after 1.3.0:** the spaces no longer offer it). The
external archive parameters (`use_external_archive`, `archive_unbounded`,
`archive_prune_policy`) keep theirs until they become one choice, Evolver's
`algorithmResult` and `archiveType`, in the planned VAMOS-CONFIG grammar.
**Development after 1.3.0:** the archive is one choice, `external_archive`; the
table gains `linked_polynomial_mutation_distribution_index` and CCMO's and
CMOEA-MS's `variation` (`crossoverAndMutationVariation`,
`differentialEvolutionVariation`), and Evolver's
`createInitialSolutions default` reads as `auto`.
The exchange translates names and values only: parameters that VAMOS and
Evolver structure differently, such as Evolver's `variation` or its per-operator
repair strategies, keep their names and are ignored by the other side.

### Parameter spaces as grammars

**Development after 1.3.0:** `vamos.engine.tuning.grammar` reads parameter
spaces written as strict grammars with defaults, under the schema
`vamos.parameter-space` 1.0.0, from YAML, from JSON or from a parsed mapping.
PyYAML, which reads the YAML, is a dependency of VAMOS. A categorical parameter chooses between productions,
one per value, and the production of a value holds the parameters that the
value activates; `global_sub_parameters` are active whatever the value.
Numeric and boolean parameters are terminals, with a `range` (both ends
included, `log` for a log-uniform scale) or nothing more:

```yaml
schema: vamos.parameter-space
schema_version: 1.0.0
name: nsgaii_small
algorithm: nsgaii
encoding: real
parameters:
  crossover:
    type: categorical
    default: sbx
    global_sub_parameters:
      crossover_probability: {type: real, range: [0.6, 1.0], default: 0.9}
    values:
      sbx:
        conditional_parameters:
          sbx_distribution_index: {type: real, range: [5.0, 40.0], default: 20.0}
      blx_alpha:
        conditional_parameters:
          crossover_alpha: {type: real, range: [0.0, 1.0], default: 0.5}
```

Reading a space checks rules G1 to G6 and fails with a `ParameterSpaceError`
that names the rule and the place:

| Rule | Statement |
| --- | --- |
| G1 | Parameter names are unique in the space and match `^[a-z][a-z0-9_]*$`. |
| G2 | Each parameter has one parent; there are no condition expressions. A parameter that applies to some values of a node is duplicated under each of their productions, with its own name. |
| G3 | Every parameter has a `default`, inside its domain. |
| G4 | Ranges are ordered, `log` needs a positive lower end, and integer ranges have integer ends. |
| G5 | `values` is not empty and has no duplicates; a value with a production is a string. |
| G6 | `schema` and `schema_version` are known; `algorithm` and `encoding` exist in VAMOS. |
| G7 | The default configuration runs as the algorithm's `default()` configuration. |

G7 runs the algorithm, so reading a space does not check it:
`check_default_configuration(space, problem)` runs the space's defaults and
the algorithm's `default()` configuration with the same seed and fails unless
both give the same final population. The schema has no forbidden clauses,
because a space offers only configurations that run, and one `range` per
parameter, the same for tuning and for editing a configuration by hand. A
space is identified by its `name`, its `schema_version` and the SHA-256 of its
canonical JSON form without labels and descriptions (`space.identity`).

A configuration is a derivation of the grammar: it holds exactly the active
parameters, each inside its domain. `configuration(space, values)` checks one,
and it has three forms that read back into the same configuration: the
document `vamos.parameter-configuration` 1.0.0 (`to_document()`, with the
space's identity), the flat mapping of its values (`to_values()`), which
`config_from_assignment` reads, and the command line of Evolver's
`VAR_CONF.txt` (`to_command_line()`), with VAMOS's names or, with
`evolver=True`, Evolver's: `--crossover SBX --sbxDistributionIndex 20.0`.

**Development after 1.3.0:** the built-in tuning spaces are such grammars,
packaged with VAMOS as YAML files in `vamos/resources/parameter_spaces`, one
per space (`nsgaii.yaml`, `moead_permutation.yaml`, ...).
`builtin_parameter_space("nsgaii")` reads one, and `build_nsgaii_config_space()`
returns the flat view that the tuners sample. The YAML files are the source of
truth, which VAMOS reads at run time: they were written once from the Python
builders of VAMOS 1.3.0 and are maintained as YAML.
`python tools/update_parameter_spaces.py` checks every file and writes the
snapshot of what the bridges build from them; a test fails when a grammar, or
what a bridge builds from it, changes without its snapshot. Compared with the spaces of VAMOS 1.3.0:

- `pop_size` is not a parameter of any space: it is a run setting, as for
  Evolver and Studio. The caller adds it to the assignment, as it adds `n_obj`
  (see [Populations that depend on the number of objectives](#populations-that-depend-on-the-number-of-objectives));
  `vamos tune` gives every algorithm `--pop-size` (100 by default), which must
  fit in `--budget`. An assignment without it builds a population of 100.
- Every parameter has a default, and the defaults of a space run as its
  algorithm's `default()` configuration for the encoding, which a test checks
  for every space by comparing seeded runs (rule G7). `initial_solutions` and
  `repair` offer `auto`, the configuration's automatic choice, as their
  default, as Evolver's `createInitialSolutions: default`. The nine mixed
  spaces do not pass G7 yet: their segment operators run differently from the
  default mixed operators.
- The segment settings of the mixed spaces of MOEA/D, NSGA-III, SMS-EMOA and
  SPEA2 (`real_mutation`, `int_crossover_prob`, ...) reach the run. Their mixed
  variation gave the operators the problem's segments without the
  configuration's settings, so every point of those spaces ran as the default
  mixed operators, and a mixed configuration of these algorithms that sets them
  runs differently now.
- The external archive is one choice, `external_archive`: `none`,
  `unbounded` or `bounded`, with `archive_prune_policy` under `bounded`, instead
  of `use_external_archive` and `archive_unbounded`.
- Polynomial and linked polynomial mutation have an index each,
  `polynomial_mutation_distribution_index` and
  `linked_polynomial_mutation_distribution_index`, instead of one shared by both.
- The rates hang from their operators: `crossover_probability` from
  `crossover`, the mutation probability from `mutation`. CCMO's and CMOEA-MS's
  spaces choose a `variation`, `crossover_and_mutation` (with `crossover` and its
  probability) or `differential_evolution` (with `de_crossover_rate`,
  `de_scale_factor` and, for CMOEA-MS, `de_variant`), as Evolver's.
- Polynomial mutation is `polynomial` in the integer spaces too (`pm` still
  builds it).
- The mutation probability is `mutation_probability_factor` in [0.25, 3.0],
  divided by the number of variables, in every space but the permutation spaces
  of the algorithms other than NSGA-II and the segments of the mixed spaces,
  whose probability applies to a whole solution or segment and defaults to a
  constant. The real-coded spaces of NSGA-III, SPEA2, IBEA, SMPSO, AGE-MOEA,
  RVEA, CCMO, C-TAEA, ToP and CMOEA-MS, and the binary and integer spaces of
  every algorithm but NSGA-II, sampled an absolute probability in [0.01, 0.5];
  NSGA-II's permutation space defaults to 2.0, the 2/n of
  `NSGAIIConfig.default()`.
- The segments of NSGA-II's mixed space sample their mutation probabilities in
  [0.01, 0.5], as the other mixed spaces do; they took the range of a factor,
  [0.25, 3.0], so above 1 every mutation applied.

The schema has no forbidden clauses: each space offers only what its algorithm
and encoding run, which a test checks by building and running seeded draws of
every space. Every seeded tuning run draws other trials, because every space
loses `pop_size` and gains or changes parameters. An assignment saved from the
spaces of VAMOS 1.3.0, with `pop_size`, `use_external_archive` or an absolute
`mutation_prob`, or with the names of VAMOS 1.2.1, builds the configuration it
built.

### Constrained spaces

**VAMOS 1.2.0:** `nsgaii_constrained`, `nsgaiii_constrained`,
`smsemoa_constrained` and `agemoea_constrained` are the real-coded spaces of
those algorithms with the parameter `constraint_handling`: the six techniques of
[choose a constraint-handling technique](../reference/constraints.md#choose-a-constraint-handling-technique)
and the parameters of three of them, with the ranges of Evolver's constrained
spaces, such as `NSGAIIDoubleConstrained`. The parameters are named after the
technique, such as `epsilon_initial_quantile` or `push_pull_window_fraction`;
`config_from_assignment("nsgaiii_constrained", assignment)` builds the
configuration, and `vamos tune --algorithm nsgaiii_constrained` tunes it.
`rvea_constrained` and `smpso_constrained` offer the four techniques that RVEA
and SMPSO take, with the parameters of `epsilon_constrained` and `push_pull`,
as Evolver's `RVEADoubleConstrained` and `MOPSOConstrained`.
`moead_constrained` adds MOEA/D's `subproblem_update_criterion` and the
parameters of `improved_epsilon` (`improved_epsilon_tao`,
`improved_epsilon_feasibility_ratio_threshold`,
`improved_epsilon_decay_fraction`), with the ranges of Evolver's
`MOEADDoubleConstrained` space.

**VAMOS 1.2.0:** `ccmo` is the real-coded space of
[CCMO](../algorithms/ccmo.md), which handles constraints itself: `pop_size`
(20 to 200, the size of each of its two populations) and the crossover, which
chooses the variation. SBX, BLX-alpha and arithmetic crossover select the GA
variation, with `crossover_prob` and their own parameters; `de` selects the DE
variation, with `de_cr` and `de_f` in [0, 1] and no `crossover_prob`. Mutation,
initializer and repair are those of the other real-coded spaces. An assignment
without `crossover_prob`, `de_cr` or `de_f` gets the published values: 1.0, and
CR = 1 and F = 0.5. `vamos tune --algorithm ccmo` tunes it.

**VAMOS 1.2.0:** `ctaea` is the real-coded space of
[C-TAEA](../algorithms/ctaea.md), which also handles constraints itself:
`pop_size` (20 to 200, the requested number of weight vectors, from which
C-TAEA generates its population: 91 for 100 and three objectives) and a
two-parent crossover, SBX, BLX-alpha or arithmetic, with `crossover_prob` and
its own parameters. Mutation, initializer and repair are those of the other
real-coded spaces. An assignment without `crossover_prob` gets 1.0, the
published value. `vamos tune --algorithm ctaea` tunes it.

**VAMOS 1.2.0:** `top` is the real-coded space of
[ToP](../algorithms/top.md), which also handles constraints itself:
`pop_size` (20 to 200), the crossover rate `de_cr` (0 to 1) and scale factor
`de_f` (0.1 to 1) of its differential evolution, which has no crossover
choice, and the thresholds of its switch between phases, `delta_threshold`
(0.05 to 0.5), `feasible_ratio_threshold` (0.1 to 0.9) and
`phase_one_evaluation_fraction` (0.5 to 1). Mutation, initializer and repair
are those of the other real-coded spaces. An assignment without `de_cr`,
`de_f` or a threshold gets the value of PlatEMO's implementation: CR = 1,
F = 0.5, and 0.2, 1/3 and 0.9. `vamos tune --algorithm top` tunes it.

**VAMOS 1.2.0:** `cmoeams` is the real-coded space of
[CMOEA-MS](../algorithms/cmoeams.md), which also handles constraints itself:
`pop_size` (20 to 200), `feasible_ratio_threshold` (0.1 to 0.9), the paper's
`lambda`, and the crossover, which chooses the variation. SBX, BLX-alpha and
arithmetic crossover select the GA variation, with `crossover_prob` and their
own parameters; `de` selects the DE variation, with `de_cr` (0 to 1), `de_f`
(0.1 to 1), `de_variant` (`rand_1_bin` or `rand_1_exp`) and no
`crossover_prob`. Mutation, initializer and repair are those of the other
real-coded spaces. An assignment without `crossover_prob`, `de_cr`, `de_f`,
`de_variant` or `feasible_ratio_threshold` gets the published values: 1.0,
CR = 1, F = 0.5, the binomial crossover and 0.5. `vamos tune --algorithm
cmoeams` tunes it.

### Parameters that reach the algorithm

**VAMOS 1.3.0:** every parameter of a built-in space changes the
configuration that it builds, and the real-coded spaces no longer offer
`repair="round"` or `blx_repair="round"`: rounding continuous variables to
integers puts the whole population on the vertices of the bounding box, where
mutation has no further effect. The SMPSO spaces drop `repair`, which never
reached the configuration: SMPSO clips each new position to the bounds, and its
polynomial mutation keeps it inside. The SPEA2 spaces drop
`selection_pressure`, because SPEA2 always mates with its own binary
tournament. An assignment saved from an earlier space still builds:
`repair="round"` as before, while SMPSO and SPEA2 ignore the dropped
parameters, as they always did.

### Populations that depend on the number of objectives

The parameter spaces do not know the problem, but two algorithms need its
number of objectives: RVEA derives its population from `n_partitions`, and
MOEA/D needs one weight vector per individual. `vamos tune` adds `n_obj` to the
assignment of both before building the configuration, and code that builds
configurations from assignments with `config_from_assignment` must do the same.

**VAMOS 1.2.0:** with `n_obj` of three or more, MOEA/D runs the
largest population, at most the sampled `pop_size`, that has weight vectors: a
complete Das-Dennis lattice or one of the packaged jMetal weight files. A
sampled `pop_size` of 60 thus runs 55 individuals with three objectives, the
lattice of nine divisions, and a lattice is never truncated. Before, `vamos
tune` did not pass `n_obj` to MOEA/D, and every other sampled population failed
and got the failure score: most MOEA/D candidates on a three-objective problem.

**VAMOS 1.3.0:** `vamos tune` records the population a trial ran
when it differs from the sampled one: `"details": {"effective": {"pop_size": 136}}`
for a sampled 150 with three objectives in `tuning_history.json`, and
`best_config_effective` in `tuning_summary.json` for the returned configuration
(`{}` when it runs as sampled). `assignment_adjustments(algorithm, assignment)`
in `vamos.engine.tuning.racing.bridge_assignments` gives the same values for code
that builds configurations with `config_from_assignment`, and the resolved spec of
a run records its population and where its weights come from (see the
[MOEADConfig reference](../reference/api/algorithms/moead.md)).

**VAMOS 1.3.0:** the RVEA spaces sample `pop_size` in [20, 200],
as the other algorithms do, instead of `n_partitions` in [4, 12]. RVEA runs the
largest complete Das-Dennis lattice with at most `pop_size` directions for the
assignment's `n_obj`: a sampled `pop_size` of 100 runs 100 individuals with two
objectives, 91 with three (12 partitions) and 55 with ten (2 partitions). The
fixed range of partitions gave 5 to 13 individuals with two objectives and up
to 293,930 with ten. An assignment with `n_partitions`, saved from an earlier
space, still builds that lattice, and `--smoke` caps RVEA's `pop_size` as it
does for the other algorithms.

**Development after 1.3.0:** the spaces do not sample `pop_size`, so these
derivations start from the fixed `pop_size` of the run, which the caller adds
to the assignment with `n_obj` and which `vamos tune` takes from `--pop-size`
for every algorithm, RVEA included: with three objectives and `--pop-size 100`,
NSGA-III and RVEA use the lattice of 91 directions (NSGA-III keeps a population
of 100) and MOEA/D the packaged jMetal weights of 100 individuals. The rules do
not change, so a saved assignment with its `pop_size` builds the population it
built. The `effective` record compares with that fixed population: with
`--pop-size 150` and three objectives, every MOEA/D trial records
`{"pop_size": 136}`.

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

In VAMOS 1.2.0, when `optuna` or `bohb_optuna` uses `--optuna-storage`, the maintained CLI treats
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

**VAMOS 1.2.0:** a resumed study runs only the trials that
`--tune-budget` still allows: its finished trials (complete, pruned or failed)
count, so running the same command again after an interruption completes the
budget instead of adding `--tune-budget` trials each time, as VAMOS 1.1.0 did.
Every study also has a name without `--optuna-storage` (the task, backend and
seed), because `bohb_optuna`'s Hyperband pruner assigns each trial to a bracket
from a hash of that name; with one worker, `bohb_optuna` is now reproducible
for a seed, as the other backends were. VAMOS 1.1.0 let Optuna draw a random
name, so the same seed gave different searches.

**VAMOS 1.2.1:** the CLI appends
`__vamos_cli_final_population_hv_v2`. VAMOS 1.2.0 changed the built-in search
spaces and the scores but kept `_v1`, so a `_v1` study can mix trials from two
contracts, and Optuna cannot resume a study in which a categorical parameter
offers other choices: it stops with `ValueError: CategoricalDistribution does
not support dynamic value space`. Since VAMOS 1.2.1 the CLI uses `_v2`, so a
study stored by an earlier version is not resumed: the same command starts a
new study, and the stored one stays in the storage under its `_v1` name.
Persistent SMPSO studies keep `__vamos_cli_leaders_hv_v1`, the namespace that
VAMOS 1.2.0 introduced with its leaders-archive score.

**VAMOS 1.3.0:** VAMOS 1.3.0 changes the built-in search spaces
again, with other categorical choices, so the CLI appends
`__vamos_cli_final_population_hv_v3`, and SMPSO studies, whose spaces also
change, take `__vamos_cli_leaders_hv_v2`. A study stored by VAMOS 1.2.1 or
earlier is not resumed under the new names.

**Development after 1.3.0:** the built-in spaces become grammars without
`pop_size` and with other categorical choices (`auto` for the initializer and
the repair), so the CLI appends `__vamos_cli_final_population_hv_v4`, and SMPSO
studies take `__vamos_cli_leaders_hv_v3`. A study stored by VAMOS 1.3.0 or
earlier is not resumed under the new names.

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
  multi-fidelity execution (off by default for racing since VAMOS 1.2.0);
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

**VAMOS 1.2.0:** `best_score` in `tuning_summary.json` is the
score of the returned configuration (`best_config_*.json`): racing's final
choice, the statistical finisher's winner or irace's first elite, which need
not be the highest score of the history; `best_history_score` is that highest
score. VAMOS 1.1.0 wrote the highest history score as `best_score`. In
`selected_configs_validation.json`, `champion_global` is the validated
candidate with the best tuning score, `champion_robust` has the best 25th
percentile and `champion_fast` the best mean validation score; "fast" does not
refer to runtime. Ties keep the candidate with the better tuning score. In racing, `start_instances`
only sets how many instances the first stages use; the schedule interleaves the
other, already shuffled, instances, so it changes little. Rank elimination
breaks ties by position, so with identical scores (identical failure scores,
for example) the configurations sampled first survive.

Keep these together with the command/configuration and environment used for the
campaign. A best configuration without its search space, seeds, HV reference
point, fixed final-population score source, aggregation rule, budget/fidelity
schedule, and split is not a reproducible tuning result.

## Archive studies are separate from ordinary CLI tuning

The 1.3.0 CLI intentionally excludes external-archive controls from its
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

**VAMOS 1.2.0:** every key of `base_config` and of a variant's
`config_overrides` is either an ablation setting (`population_size`,
`offspring_population_size`, `eval_strategy` and `output_root`) or a field of
the algorithm configuration, such as `pop_size` or
`crossover: [sbx, {prob: 0.5}]`, which reaches the study's `algorithm_configs`.
Any other key fails with a `ConfigurationError` that suggests the closest
field. `pop_size` and `offspring_size` set the same sizes as
`population_size` and `offspring_population_size`: a variant's value wins over
`base_config` whichever name each one uses, and one mapping cannot give the
two names different values. Operators are accepted as `[name, {params}]` pairs or as mappings with
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
