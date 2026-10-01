# Experiment blocks: stopping + external archive

!!! warning "Experimental observation and stopping hooks"
    These experiment-spec hooks are Experimental. They record traces inside a
    canonical run artifact, but their observation behavior is not an additional
    stable algorithm API. See [Stability and versioning](../project/stability-and-versioning.md).

Use these blocks within a [CLI experiment configuration](../guide/cli.md#config-files-yamljson).
A [complete NSGA-II/ZDT1 configuration](https://github.com/vamos-optimization/VAMOS/blob/main/experiments/configs/hv_archive_validation_slice.yml)
shows both together. Start with two-objective, unconstrained runs: the hook
observes objective vectors and does not independently enforce feasibility.

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
    ref_point: [2.0, 2.0] # must match n_obj (or use "auto")
```

Canonical record:

- `manifest.json` outcome metrics under `hooks.stopping`, including the bounded trace

Notes:

- This hook computes HV for **exactly two objectives**. For more objectives it
  records an unavailable-HV reason and cannot apply HV-convergence stopping;
  installing another backend does not change this hook's dimensionality guard.
- `ref_point: "auto"` derives a reference point from the currently observed set.
  Use an explicit fixed reference point for an interpretable convergence trace
  and comparisons across runs. A changing reference point changes the metric.
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
    hv_ref_point: null         # optional; required for hv-based policies
    rng_seed: 0
    objective_tolerance: 1.0e-10
    deduplicate_in: objective  # objective|decision|both
    decision_tolerance: 1.0e-32
```

Canonical record:

- `manifest.json` outcome metrics under `hooks.archive`, including the bounded trace

Notes:

- Set a finite `capacity` explicitly; the hook parser defaults to 200. It does
  not inherit population size from a tuning space.
- The hook stores objective vectors, not a decision-aligned result archive.
  Use `deduplicate_in: objective` for this observation workflow.
- `pruning: hv` uses exact HV contributions in 2D and, when `moocore` is
  installed, exact higher-dimensional contributions. `mc_hv` uses a Monte Carlo
  estimate. Archive pruning support does not remove the two-objective restriction
  of the convergence monitor.
- For an **algorithm result archive**, use its public configuration builder and
  inspect the resolved `result_mode`; see [Algorithms and backends](../reference/algorithms.md).
  In SMPSO 1.0.1, the search leaders and optional external result archive are
  separate, and explicit `population` mode returns the final swarm. See the
  [changes since 1.0.0](../reference/algorithms.md#changes-included-in-101).
- The 1.0.1 tuning CLI excludes result-archive controls from its ordinary search
  and scores the feasible non-dominated final population. See
  [Tuning version differences](../topics/tuning.md#changes-included-in-101).

## Reproducibility

Runs should be launched with fixed seeds and fixed budgets. Early stopping changes executed evaluations,
but the run still reports the original max budget in the resolved spec. Use the
stopping payload and trace in manifest outcome metrics for analysis.
