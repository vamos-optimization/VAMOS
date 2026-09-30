# Constraints

!!! info "Stable problem-definition API"
    `make_problem` and `optimize` are Stable in VAMOS 1.0.0. The symbolic
    constraint DSL and low-level strategy classes are Internal. See
    [Stability and versioning](../project/stability-and-versioning.md).

## Define inequality constraints

VAMOS uses the convention **`g(x) <= 0` means feasible**. Convert a requirement
such as `x[0] + x[1] <= 1` into `x[0] + x[1] - 1`. Bounds belong in `bounds`;
they do not need to be repeated as inequality constraints.

This complete example uses only the core [installation](../guide/installation.md)
and public interfaces:

```python
import numpy as np

from vamos import make_problem, optimize


def objectives(x):
    return [x[0] ** 2 + x[1] ** 2, (x[0] - 1) ** 2 + (x[1] - 1) ** 2]


def constraints(x):
    return [x[0] + x[1] - 1.0]


problem = make_problem(
    objectives,
    n_var=2,
    n_obj=2,
    bounds=[(0.0, 1.0), (0.0, 1.0)],
    encoding="real",
    constraints=constraints,
    n_constraints=1,
)
result = optimize(
    problem,
    algorithm="nsgaii",
    pop_size=40,
    max_evaluations=400,
    seed=7,
    engine="numpy",
)

# Check the returned decisions against the original requirement.
G = np.array([constraints(x) for x in result.X])
feasible = np.all(G <= 0.0, axis=1)
print(f"Feasible returned solutions: {feasible.sum()}/{len(feasible)}")
print("Feasible objective vectors:", result.F[feasible])
```

NSGA-II uses feasibility-aware selection by default. A short run is not a
guarantee that every returned solution is feasible; always inspect feasibility
before interpreting a result or calculating a quality indicator.

## Shapes, vectorization, and equalities

| Evaluation mode | Input | Objectives returned | Constraints returned |
| --- | --- | --- | --- |
| Default, one decision at a time | `(n_var,)` | `(n_obj,)` | `(n_constraints,)` |
| `vectorized=True` | `(n_points, n_var)` | `(n_points, n_obj)` | `(n_points, n_constraints)` |

The objective and constraint functions must follow the same evaluation mode.
Declare `n_constraints` explicitly. In a custom `Problem.evaluate(X, out)`
implementation, put the constraint matrix in `out["G"]` using the same sign
convention.

For an equality `h(x) = 0`, choose a meaningful numerical tolerance and encode
`abs(h(x)) - tolerance <= 0`. State that tolerance and any scaling of constraint
values in the experiment: they affect what counts as feasible and how violations
are compared.

## Choose an algorithm with constraint support

Consult the [algorithm capability matrix](algorithms.md) before selecting an
algorithm. Constraint support is algorithm-specific; the presence of a
constraint function does not make every algorithm feasibility-aware. For
example, NSGA-II's `constraint_mode="none"` explicitly disables its constraint
selection behavior.

Hypervolume and other objective-space metrics do not establish feasibility.
Filter to feasible solutions first and apply the same feasibility rule to every
algorithm in a comparison.

For a fuller modeling example, read
[Solve your own problem](../guide/custom-problem.md). Contributors who need the
implementation-level strategies can consult the
[Internal constraint reference](api/internal/constraints.md); these are not
stable public imports or interchangeable algorithm configuration options.
