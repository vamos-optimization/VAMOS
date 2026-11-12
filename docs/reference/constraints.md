# Constraints

!!! info "Stable problem-definition API"
    `make_problem` and `optimize` are Stable in VAMOS 1.1.0. The symbolic
    constraint DSL and the violation helpers are Internal. See
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

**Development after 1.1.0:** constraint values from a problem that declares
`n_constraints = 0` raise `EvaluationError`. This covers a `G` written by
`evaluate` and an overridden `Problem.constraints()`. VAMOS 1.1.0 ignored them
and optimized the problem as unconstrained. Constraint values, like objectives,
must be finite.

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

## Read the feasibility of a result

**Development after 1.1.0:** every solution array of a result carries the
constraint values of its rows: `result.data["G"]` for `result.X`, and `"G"` in
`result.data["population"]` and in the external archive's entry. The archive's
entry is `result.data["archive"]` for most algorithms, `archive_G` next to
`archive_X` for NSGA-III and IBEA, and `result.data["external_archive"]` for
SPEA2, whose `archive` is its environmental archive. SPEA2 and SMPSO omit
them with `constraint_mode="none"`. VAMOS 1.1.0 dropped them whenever the result
came from an external archive or SMPSO's leaders, and from the population of
most algorithms.

`result.feasibility()` summarizes the top-level rows:

```python
summary = result.feasibility()
# {"n_solutions": 40, "n_feasible": 37, "min_violation": 0.0}
```

The violation of a row is the sum of the positive parts of its constraint
values. `min_violation` is 0 when a row is feasible. A run without a feasible
solution has `n_feasible == 0` and the violation of its least violated row,
which is what feasibility-first comparisons, such as those of constrained
algorithm tuning, rank such runs by.

For a fuller modeling example, read
[Solve your own problem](../guide/custom-problem.md). Contributors who need the
implementation-level strategies can consult the
[Internal constraint reference](api/internal/constraints.md); these are not
stable public imports or interchangeable algorithm configuration options.
