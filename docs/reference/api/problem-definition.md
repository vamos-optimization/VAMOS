# Problem definition

Wrap an objective function with `make_problem`, or implement the `Problem` interface. The guide explains scalar and vectorized evaluation, bounds, encodings, and constraints.

**Stable public API.** The [1.x compatibility policy](../../project/stability-and-versioning.md) defines the supported surface; import from the public facade shown below.

```python
from vamos import Problem, make_problem
```

[Solve your own problem](../../guide/custom-problem.md) · [Built-in problem catalogue](../problems.md) · [Problem discovery](discovery.md)

::: vamos.make_problem
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

::: vamos.Problem
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false
