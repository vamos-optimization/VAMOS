# Solve your own problem

`make_problem(...)` turns a plain Python function into a VAMOS-compatible optimization problem. Use this route when you want to optimize your own objective functions without implementing a framework class.

All examples on this page use the public `vamos` facade.

## Start with a scalar objective function

By default, VAMOS calls your function once per candidate solution. The function receives a one-dimensional decision vector and returns one value per objective.

```python
from vamos import make_problem, optimize

problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2,
    n_obj=2,
    bounds=[(0.0, 1.0), (0.0, 1.0)],
    encoding="real",
)

result = optimize(
    problem,
    algorithm="nsgaii",
    max_evaluations=400,
    pop_size=40,
    seed=42,
)

print(result.F.shape)
print(result.X.shape)
```

`result.F` contains objective values and `result.X` contains the corresponding decision variables.

## Bounds

Use `bounds=[(lower, upper), ...]` when variables have different limits. The sequence must contain exactly one pair per decision variable.

```python
problem = make_problem(
    lambda x: [x[0] + x[1], x[0] ** 2 + (x[1] - 1.0) ** 2],
    n_var=2,
    n_obj=2,
    bounds=[(-5.0, 5.0), (0.0, 10.0)],
    encoding="real",
)
```

When every variable shares the same limits, `xl` and `xu` can be scalars:

```python
problem = make_problem(
    lambda x: [x[0] ** 2, (x[1] - 1.0) ** 2],
    n_var=2,
    n_obj=2,
    xl=-2.0,
    xu=2.0,
    encoding="real",
)
```

Do not pass `bounds` together with `xl` or `xu`.

## Vectorized evaluation

Set `vectorized=True` only when your function already accepts an `(N, n_var)` batch and returns an `(N, n_obj)` array.

```python
import numpy as np
from vamos import make_problem


def objectives(X):
    f1 = X[:, 0]
    f2 = 1.0 - np.sqrt(X[:, 0])
    return np.column_stack([f1, f2])


problem = make_problem(
    objectives,
    n_var=2,
    n_obj=2,
    bounds=[(0.0, 1.0), (0.0, 1.0)],
    vectorized=True,
    encoding="real",
)
```

`vectorized=False` is the compatibility path; VAMOS adapts the scalar callable by evaluating one row at a time. `vectorized=True` is the path for actual batch evaluation.

## Constraints

VAMOS uses the convention `g(x) <= 0` for feasibility. Provide a constraint function and declare exactly how many values it returns.

```python
from vamos import make_problem, optimize


def objectives(x):
    return [
        x[0] ** 2 + x[1] ** 2,
        (x[0] - 2.0) ** 2 + (x[1] - 2.0) ** 2,
    ]


def constraints(x):
    # x[0] + x[1] >= 1  ->  1 - x[0] - x[1] <= 0
    return [1.0 - x[0] - x[1]]


problem = make_problem(
    objectives,
    n_var=2,
    n_obj=2,
    bounds=[(-3.0, 3.0), (-3.0, 3.0)],
    encoding="real",
    constraints=constraints,
    n_constraints=1,
)

result = optimize(
    problem,
    algorithm="nsgaii",
    max_evaluations=400,
    pop_size=40,
    seed=42,
)
```

If the objective or constraint function returns an unexpected shape, VAMOS fails with the expected `(N, n_obj)` or `(N, n_constraints)` shape in the error message.

## Encodings

`make_problem(...)` accepts the public encoding names `"real"`, `"binary"`, `"integer"`, `"permutation"`, and `"mixed"`. The selected algorithm, operators, and problem must agree on the encoding. Consult [Algorithms & Backends](../reference/algorithms.md) and [Problems](../reference/problems.md) before assuming that every algorithm supports every encoding.

## When to use a reusable problem class

For a one-off scientific problem, prefer `make_problem(...)`. If you are contributing a reusable problem implementation to VAMOS itself, follow the separate contributor workflow in [Adding a problem](../dev/add_problem.md).

## Next steps

- [Quickstart](zero_to_hero.md) — run, save, verify, replay, and move into a durable study.
- [Constraints](../reference/constraints.md) — constraint-handling reference.
- [API Reference](../reference/api_reference.md) — exact public signatures.
- [Durable Studies](studies.md) — repeat the problem across algorithms and seeds.
