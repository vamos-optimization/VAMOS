# Optimization

Run an optimization through `vamos.optimize`. For a first execution rather than a signature lookup, start with the Quickstart.

**Stable public API.** The [1.x compatibility policy](../../project/stability-and-versioning.md) defines the supported surface; import from the public facade shown below.

```python
from vamos import optimize
```

[Quickstart](../../guide/zero_to_hero.md) · [Results](results.md) · [Algorithm configuration](algorithms/index.md)

`max_evaluations` is a hard evaluation budget subject to documented algorithm cardinality requirements. An explicit integer seed controls the built-in stochastic path in the same materially relevant environment; cross-backend bitwise equality is not promised. The generated reference below defines the arguments and their defaults.

::: vamos.optimize
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false
