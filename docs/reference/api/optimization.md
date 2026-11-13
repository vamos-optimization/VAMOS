# Optimization

Run an optimization through `vamos.optimize`. For a first execution rather than a signature lookup, start with the Quickstart.

**Stable public API.** The [1.x compatibility policy](../../project/stability-and-versioning.md) defines the supported surface; import from the public facade shown below.

```python
from vamos import optimize
```

[Quickstart](../../guide/zero_to_hero.md) · [Results](results.md) · [Algorithm configuration](algorithms/index.md)

When `pop_size` or `max_evaluations` is omitted, `optimize` uses quick defaults derived from the problem size, and `algorithm="auto"` selects NSGA-II for two objectives and NSGA-III for more:

- `pop_size` is `min(max(50, 10 * n_var), 200)`, at least 100 with three or more objectives. MOEA/D, NSGA-III and RVEA, and so `algorithm="auto"` with three or more objectives, take the population from their reference directions instead: 100 with two objectives and 91 with three.
- `max_evaluations` is `min(max(1000, 100 * n_var), 50000)`, at least 5000 with three or more objectives.

For ZDT1 (30 variables) the quick defaults are 200 individuals and 3000 evaluations, 15 generations: enough for a first run, not for a converged front. Pass both for an experiment you report.

`max_evaluations` is a hard evaluation budget subject to documented algorithm cardinality requirements. An explicit integer seed controls the built-in stochastic path in the same materially relevant environment; cross-backend bitwise equality is not promised. The generated reference below defines the arguments and their defaults.

::: vamos.optimize
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false
