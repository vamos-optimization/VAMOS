# Optimization

Run an optimization through `vamos.optimize`. For a first execution rather than a signature lookup, start with the Quickstart.

**Stable public API.** The [1.x compatibility policy](../../project/stability-and-versioning.md) defines the supported surface; import from the public facade shown below.

```python
from vamos import optimize
```

[Quickstart](../../guide/zero_to_hero.md) · [Results](results.md) · [Algorithm configuration](algorithms/index.md)

`max_evaluations` is a hard evaluation budget subject to documented algorithm cardinality requirements. An explicit integer seed controls the built-in stochastic path in the same materially relevant environment; cross-backend bitwise equality is not promised. The generated reference below defines the arguments and their defaults.

**Development after 1.1.0:** `optimize` raises `ConfigurationError` for arguments it would ignore or apply only in part: `n_var`, `n_obj` or `problem_kwargs` with a problem instance; `problem_kwargs` other than `n_var` and `n_obj` with a registered problem name (build the instance yourself to pass other arguments); `max_evaluations` together with a `termination` that does not stop at the same number of evaluations; the configuration class of another algorithm (`algorithm="moead"` with an `NSGAIIConfig`); `GenericAlgorithmConfig` with a built-in algorithm, whose generic keys, such as SMPSO's coefficient bounds, the typed configuration does not record and a replay cannot rebuild; and an empty seed sequence. `termination` no longer needs an `algorithm_config`. `StudyResult.std` takes `ddof` (default 0, the population standard deviation; pass `ddof=1` for the sample standard deviation). VAMOS 1.1.0 ran these calls and recorded the requested values, raised a `TypeError` naming an internal function or a false "unexpected keyword argument 'termination'", or returned an empty `StudyResult`. Study plans likewise reject, with `INVALID_STUDY_SPEC`, `algorithm_configs` keys that name no algorithm of the study, compared with the exact case; an empty `seeds` list remains a valid zero-task plan.

::: vamos.optimize
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false
