# Experiment blocks: stopping + external archive

!!! warning "Experimental observation and stopping hooks"
    These experiment-spec hooks are Experimental. They record traces inside a
    canonical run artifact, but their observation behavior is not an additional
    stable algorithm API. See [Stability and versioning](../project/stability-and-versioning.md).

Use these blocks within a [CLI experiment configuration](../guide/cli.md#config-files-yamljson).
A [complete NSGA-II/ZDT1 configuration](https://github.com/vamos-optimization/VAMOS/blob/main/experiments/configs/hv_archive_validation_slice.yml)
shows both together. Start with two-objective, unconstrained runs: the hook
observes objective vectors and does not independently enforce feasibility.

**Development after 1.1.0:** a block whose values cannot be built (for example
`window: 1` or `pruning: hypervolume`) stops the command with exit code 2
before any run, and `--validate-config` reports it. VAMOS 1.1.0 and earlier
disabled both hooks with a warning and ran the full evaluation budget without
them.

**Development after 1.1.0:** both blocks merge key by key, nested mappings
included: first `defaults`, then the top-level block, then `problems.<name>`.
A key set to `null` replaces the inherited value, so
`problems.zdt1.stopping.hv_convergence: {patience: 10}` changes only `patience`.
Problem keys are case-insensitive: `ZDT1` is `zdt1`, and two keys that differ
only in case are an error. Each value is checked when the spec is validated,
before any run; the values that conflict only after merging (such as
`min_points` below `window + 2`) are checked when each problem's hooks are
built, also before any run. In VAMOS 1.1.0 a partial problem override replaced
the whole block, so that example turned the stop off and lost `ref_point`; a
top-level block replaced the `defaults` one; `problems.ZDT1` was accepted but
never applied; and misspelled or out-of-range values ran as something else
(`statistic: medain` as the median, `epsilon_mode: relative` as `rel`,
`confidence: 95` as 1).

## stopping.hv_convergence

Enable HV-based convergence stopping driven by a hypervolume trace sampled during the run.

Configuration fragment:

```yaml
stopping:
  hv_convergence:
    enabled: true
    every_k: 200
    window: 10
    patience: 5
    epsilon: 1e-4
    epsilon_mode: rel     # abs|rel
    statistic: median     # mean|median|min
    min_points: 25
    confidence: null      # e.g. 0.95 to enable bootstrap CI
    bootstrap_samples: 300
    ref_point: [2.0, 2.0] # two values (the monitor is two-objective), or "auto"
```

**Development after 1.1.0:** `every_k`, `patience`, `min_points` and
`bootstrap_samples` are positive integers and `window` is an integer of at
least 2; `epsilon` is a finite number `>= 0`; `epsilon_mode` is `abs` or `rel`
and `statistic` is `mean`, `median` or `min`; `confidence` is `null` or
strictly between 0 and 1; `enabled` is `true` or `false`; and `ref_point` is
`auto` or two finite numbers. PyYAML reads a float without a dot, such as
`1e-4`, as a string, so `epsilon` and `confidence` also accept a string that
is a number. In VAMOS 1.1.0 a `ref_point` without two values made every HV
sample fail silently, and the stop never triggered.

Canonical record:

- `manifest.json` outcome metrics under `hooks.stopping`, including the bounded trace

Notes:

- This hook computes HV for **exactly two objectives**. For more objectives it
  records an unavailable-HV reason and cannot apply HV-convergence stopping;
  installing another backend does not change this hook's dimensionality guard.
- `ref_point: "auto"` derives a reference point from the observed set. Use an
  explicit fixed reference point for comparisons across runs.
  **Development after 1.1.0:** the automatic reference is fixed at the first
  sample (its nadir plus 10% of its range in each objective) and recorded as
  `reference_point` in the stopping metadata; later points outside it count as
  zero. VAMOS 1.1.0 and earlier recomputed it from every sample, so the series
  was not comparable and convergence was declared too early.
- The trace uses the hook archive when enabled, otherwise the observed objective
  vectors. Sampling occurs at generation/checkpoint boundaries; `every_k` is a
  sampling interval, not a request for extra objective evaluations.

## archive.external

Enable an **objective-space observation archive** with explicit pruning
policies. This hook archive is separate from an algorithm's result archive;
enabling it does not change which decisions or objectives `result.X` and
`result.F` return.

Configuration fragment:

```yaml
archive:
  external:
    enabled: true
    capacity: 200
    truncate_size: 200
    pruning: crowding          # crowding|hv|mc_hv|knn|maxmin|ref_dirs
    hv_ref_point: null         # optional; hv-based policies use a relative reference without it
    rng_seed: 0
    objective_tolerance: 0.0
    deduplicate_in: objective  # the only value the hook archive accepts
    decision_tolerance: 1.0e-32
```

Canonical record:

- `manifest.json` outcome metrics under `hooks.archive`, including the bounded trace

Notes:

- Set a finite `capacity` explicitly; the hook parser defaults to 200. It does
  not inherit population size from a tuning space.
  **Development after 1.1.0:** `capacity: null` keeps every non-dominated
  objective vector; VAMOS 1.1.0 rejected it.
- The hook stores objective vectors, not a decision-aligned result archive.
  **Development after 1.1.0:** `deduplicate_in` must therefore be `objective`;
  `decision` and `both` are rejected when the spec is validated. VAMOS 1.1.0
  accepted them and the run aborted with an `IndexError` at the first archive
  update.
- **Development after 1.1.0:** `capacity` and `truncate_size` are positive
  integers (not `true` or `10.7`), `rng_seed` is an integer, the tolerances are
  finite numbers `>= 0` and `hv_ref_point` is a list of finite numbers. The same
  rules apply to `ExternalArchiveConfig` in Python, which also warns about a
  field that the chosen pruning does not use: `hv_ref_point` without HV
  pruning, `rng_seed` without `ref_dirs`, or a pruning other than `crowding`
  without a `capacity`. VAMOS 1.1.0 truncated `10.7` to 10, read `true` as 1 and
  failed at the first duplicate with a `NaN` tolerance.
- `objective_tolerance` is an absolute Chebyshev distance: two objective
  vectors within that distance count as duplicates and only the first is kept. Set it
  in objective units, if at all.
- **Development after 1.1.0:** `objective_tolerance` defaults to `0.0`, exact
  equality, as in jMetal; it was `1.0e-10`. The same default applies to the
  SMPSO leader archive and to the algorithm result archives. With objectives on
  a scale of about `1e-9` or smaller, the old tolerance merged distinct
  non-dominated points; at ordinary scales the archive contents are unchanged.
- `pruning: hv` uses exact HV contributions in 2D and, when `moocore` is
  installed, exact higher-dimensional contributions. Archive pruning support does
  not remove the two-objective restriction of the convergence monitor.
- **Development after 1.1.0:** `mc_hv` is an alias of `hv`: both compute the
  exact contributions; VAMOS 1.1.0 documented `mc_hv` as a Monte Carlo estimate,
  but it never was one. `hv_ref_point` is optional. Without it, each truncation
  measures the contributions against `nadir + (nadir - ideal)` of the archived
  points, the reference 2 of the objectives normalized to `[0, 1]`, so the pruning
  does not depend on the objective units and keeps the extreme points. A given
  `hv_ref_point` is used as is, in objective units. VAMOS 1.1.0 and earlier used
  the largest value seen in each objective plus 1, in objective units: with one
  objective in hundreds, the archive dropped extreme points that it keeps with
  the objective in units.
- **Development after 1.1.0:** without `moocore`, HV pruning with two objectives
  uses the exact contributions too, and with three or more objectives the run
  fails with a `DependencyError` naming the `compute` extra. VAMOS 1.1.0 and
  earlier prune by crowding distance instead, with a single warning per process.
  The HV subset selection of `top_k(method="hv_greedy")` is exact with or without
  `moocore`; without it, it is slow with three or more objectives.
- **Development after 1.1.0:** `pruning: knn` applies the SPEA2 truncation
  operator: it repeatedly removes the point whose distances to the others,
  sorted in ascending order, are lexicographically smallest. VAMOS 1.1.0 and
  earlier rank points by the distance to their second-nearest neighbour alone,
  with ties broken by row order, which can remove an extreme point.
- For an **algorithm result archive**, use its public configuration builder and
  inspect the resolved `result_mode`; see [Algorithms and backends](../reference/algorithms.md).
  Since VAMOS 1.0.1, the SMPSO search leaders and optional external result archive are
  separate, and explicit `population` mode returns the final swarm. See the
  [changes since 1.0.0](../reference/algorithms.md#changes-included-in-101).
- Since VAMOS 1.0.1, the tuning CLI excludes result-archive controls from its ordinary search
  and scores the feasible non-dominated final population. See
  [Tuning version differences](../topics/tuning.md#changes-included-in-101).

## Reproducibility

Runs should be launched with fixed seeds and fixed budgets. Early stopping changes executed evaluations,
but the run still reports the original max budget in the resolved spec. Use the
stopping payload and trace in manifest outcome metrics for analysis.
