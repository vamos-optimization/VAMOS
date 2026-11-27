# Optimization

Run an optimization through `vamos.optimize`. For a first execution rather than a signature lookup, start with the Quickstart.

**Stable public API.** The [1.x compatibility policy](../../project/stability-and-versioning.md) defines the supported surface; import from the public facade shown below.

```python
from vamos import optimize
```

[Quickstart](../../guide/zero_to_hero.md) · [Results](results.md) · [Algorithm configuration](algorithms/index.md)

When `pop_size` or `max_evaluations` is omitted, `optimize` uses quick defaults derived from the problem size, and `algorithm="auto"` selects NSGA-II for two objectives and NSGA-III for more:

- `pop_size` is `min(max(50, 10 * n_var), 200)`, at least 100 with three or more objectives. MOEA/D, NSGA-III and RVEA, and so `algorithm="auto"` with three or more objectives, take the population from their reference directions instead: 100 with two objectives and 91 with three. PPS (development after 1.5.2), which also needs one weight vector per individual, takes the largest population up to that size that has them: 190 for 200 with three objectives.
- `max_evaluations` is `min(max(1000, 100 * n_var), 50000)`, at least 5000 with three or more objectives.

For ZDT1 (30 variables) the quick defaults are 200 individuals and 3000 evaluations, 15 generations: enough for a first run, not for a converged front. Pass both for an experiment you report.

`max_evaluations` is a hard evaluation budget subject to documented algorithm cardinality requirements. An explicit integer seed controls the built-in stochastic path in the same materially relevant environment; cross-backend bitwise equality is not promised. The generated reference below defines the arguments and their defaults.

**VAMOS 1.2.0:** `optimize` raises `ConfigurationError` for arguments it would ignore or apply only in part: `n_var`, `n_obj` or `problem_kwargs` with a problem instance; `problem_kwargs` other than `n_var` and `n_obj` with a registered problem name (build the instance yourself to pass other arguments); `max_evaluations` together with a `termination` that does not stop at the same number of evaluations; the configuration class of another algorithm (`algorithm="moead"` with an `NSGAIIConfig`); `GenericAlgorithmConfig` with a built-in algorithm, whose generic keys, such as SMPSO's coefficient bounds, the typed configuration does not record and a replay cannot rebuild; and an empty seed sequence. `termination` no longer needs an `algorithm_config`. `StudyResult.std` takes `ddof` (default 0, the population standard deviation; pass `ddof=1` for the sample standard deviation). VAMOS 1.1.0 ran these calls and recorded the requested values, raised a `TypeError` naming an internal function or a false "unexpected keyword argument 'termination'", or returned an empty `StudyResult`. Study plans likewise reject, with `INVALID_STUDY_SPEC`, `algorithm_configs` keys that name no algorithm of the study, compared with the exact case; an empty `seeds` list remains a valid zero-task plan.

**VAMOS 1.3.0:** with a registered problem name, `problem_kwargs` also takes the problem's options besides `n_var` and `n_obj`: the keyword-only parameters of its registry factory, such as `capacity_ratio` for `bin_knapsack` and `complicated_pareto_set`, `level`, `bias` and `imbalance` for `zcat1`–`zcat20` (see [Problems](../problems.md)). They are recorded in `resolved_spec.problem.config`, so `verify_run` and `reproduce` rebuild the same problem, and a study's `problem_kwargs` passes them the same way. An option the factory does not declare, or a value other than `None`, a boolean, a string, an integer or a finite number that a float represents exactly (a `Fraction` that `float` would round is rejected), raises `ConfigurationError`. The problem is built with the values as the run records them: another integer type as `int`, another real type as `float`.

::: vamos.optimize
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false
