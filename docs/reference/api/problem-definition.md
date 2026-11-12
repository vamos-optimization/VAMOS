# Problem definition

Wrap an objective function with `make_problem`, or implement the `Problem` interface. The guide explains scalar and vectorized evaluation, bounds, encodings, and constraints.

**Stable public API.** The [1.x compatibility policy](../../project/stability-and-versioning.md) defines the supported surface; import from the public facade shown below.

```python
from vamos import Problem, make_problem
```

[Solve your own problem](../../guide/custom-problem.md) · [Built-in problem catalogue](../problems.md) · [Problem discovery](discovery.md)

**Development after 1.1.0:** `make_problem` accepts any integral type for
`n_var`, `n_obj`, and `n_constraints`, such as `numpy.int64`, and rejects NaN or
infinite bounds with `BoundsError` and a negative or non-integer
`n_constraints` with `ProblemDimensionError`. VAMOS 1.1.0 rejected NumPy
integers and accepted those bounds and counts. Its *Raises* section below names
the errors it raises; the [exception policy](../../project/stability-and-versioning.md#exceptions)
describes their hierarchy. An unknown name passed to `make_problem_selection`
raises `InvalidProblemError`, which is also the `KeyError` of VAMOS 1.1.0.

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
