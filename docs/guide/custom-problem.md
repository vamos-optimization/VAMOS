# Solve your own problem

`make_problem(...)` turns your model into a VAMOS-compatible optimization problem. The important step is not the wrapper itself: it is deciding which quantities are **decision variables**, which outputs are **objectives**, which limits are **constraints**, and what bounds make scientific sense.

Start here after you understand how [`X`, `F`, and the returned set](understanding-results.md) relate. For a repository-checkout version that you can run unchanged, use the [Solve my problem executable journey](../examples.md#solve-my-problem).

## Translate your model into VAMOS

Before writing optimization code, make the mapping explicit.

| Domain concept | VAMOS representation | In the example below |
| --- | --- | --- |
| Quantities the optimizer may change | one decision vector `x` | temperature and residence time |
| Allowed range of each quantity | `bounds=[(lower, upper), ...]` | `60–100` and `2–10` |
| Quantities to minimize | objective function return values | energy score and conversion-shortfall score |
| Requirements that must hold | constraint function values with `g(x) <= 0` | process intensity must be at least `450` |
| One evaluated design | one row of `result.X` | `[temperature, residence_time]` |
| Objective values for that design | same row of `result.F` | `[energy_score, shortfall_score]` |

VAMOS minimizes every objective. If your scientific goal is to maximize a quantity, transform it into a minimization objective, for example by minimizing its negative or a meaningful loss/shortfall measure. Record that transformation when reporting results.

## Worked example: a small process-design surrogate

Suppose a model exposes two controllable quantities:

- temperature set point, between `60` and `100`;
- residence time, between `2` and `10`.

We want to minimize two competing scores: an energy proxy that grows with temperature and time, and a conversion-shortfall proxy that falls as temperature and time increase. We also require `temperature * residence_time >= 450`.

The equations below are deliberately simple **teaching surrogates**, not a validated physical process model. Replace them with your equations, simulator, experiment surrogate, or other scientific evaluator.

### 1. Write objectives in domain language

Start with the default scalar mode. VAMOS calls this function once for each candidate solution, so `x` is a one-dimensional decision vector.

```python
from vamos import make_problem, optimize


def objectives(x):
    temperature, residence_time = x

    energy_score = ((temperature - 60.0) / 40.0) ** 2 + 0.25 * (residence_time / 10.0)
    conversion_shortfall = (100.0 - temperature) / 40.0 + 2.0 / residence_time

    return [energy_score, conversion_shortfall]
```

The order is now a contract:

- `x[0]` is temperature and `x[1]` is residence time;
- objective column `F[:, 0]` is the energy score;
- objective column `F[:, 1]` is the conversion-shortfall score.

Keep those meanings stable when you later inspect, plot, save, or publish a result.

### 2. Express requirements as `g(x) <= 0`

The domain requirement is

```text
temperature * residence_time >= 450
```

Move everything to the `<= 0` convention used by VAMOS:

```text
450 - temperature * residence_time <= 0
```

Then write the constraint function:

```python
def constraints(x):
    temperature, residence_time = x
    return [450.0 - temperature * residence_time]
```

For inequalities, a useful translation rule is:

- `a(x) <= b` becomes `a(x) - b <= 0`;
- `a(x) >= b` becomes `b - a(x) <= 0`.

A negative constraint value is feasible, zero is on the boundary, and a positive value violates the requirement.

### 3. Build the problem

```python
problem = make_problem(
    objectives,
    n_var=2,
    n_obj=2,
    bounds=[(60.0, 100.0), (2.0, 10.0)],
    encoding="real",
    constraints=constraints,
    n_constraints=1,
    name="process_design_surrogate",
)
```

`n_var`, `n_obj`, the number of bound pairs, and `n_constraints` must agree with the functions you wrote. Treat the bounds as part of the scientific problem definition, not as arbitrary optimizer settings.

### 4. Optimize it exactly like a built-in problem

```python
result = optimize(
    problem,
    algorithm="nsgaii",
    max_evaluations=400,
    pop_size=40,
    engine="numpy",
    seed=42,
)
```

The custom model changes how candidate solutions are evaluated; it does not require a different optimization entry point.

### 5. Translate the result back to your domain

Rows correspond across `X` and `F`. Extract the non-dominated subset of the returned set while keeping those row indices aligned:

```python
front_F, front_indices = result.front(return_indices=True)
front_X = result.X[front_indices]

for x, f in zip(front_X[:5], front_F[:5]):
    temperature, residence_time = x
    energy_score, conversion_shortfall = f
    print(temperature, residence_time, energy_score, conversion_shortfall)
```

For this simple NSGA-II call, top-level `X` and `F` are the returned population. Do not assume that for every configuration or algorithm: [result modes](understanding-results.md#know-which-set-x-and-f-represent) determine which set is exposed at the top level.

Multi-objective optimization normally gives you several trade-offs. Choosing one operating point from the front is a separate decision step; Pareto dominance alone does not say which trade-off is best for your application.

## Check your evaluator before spending a budget

Optimization can faithfully search the wrong model. Before running thousands of evaluations, test a few hand-picked decision vectors whose behavior you can reason about.

For the example above:

```python
x = [75.0, 6.0]

print(objectives(x))
print(constraints(x))
```

Check at least these properties for your own model:

- objective order and units are what you intend;
- increasing or decreasing a variable changes outputs in a plausible direction;
- feasible and infeasible hand-picked points have the expected constraint sign;
- all values are finite over the allowed bounds;
- stochastic simulators use randomness intentionally and reproducibly.

A seed controls the optimizer. It does not automatically make randomness inside your own evaluator reproducible.

## Vectorize only when your model is already batched

The scalar form is usually the clearest starting point. If your evaluator already works efficiently on a batch, set `vectorized=True`. The function then receives an array with shape `(N, n_var)` and must return `(N, n_obj)`.

```python
import numpy as np
from vamos import make_problem


def objectives_batch(X):
    temperature = X[:, 0]
    residence_time = X[:, 1]

    energy_score = ((temperature - 60.0) / 40.0) ** 2 + 0.25 * (residence_time / 10.0)
    conversion_shortfall = (100.0 - temperature) / 40.0 + 2.0 / residence_time

    return np.column_stack([energy_score, conversion_shortfall])


def constraints_batch(X):
    temperature = X[:, 0]
    residence_time = X[:, 1]
    return (450.0 - temperature * residence_time)[:, None]


problem = make_problem(
    objectives_batch,
    n_var=2,
    n_obj=2,
    bounds=[(60.0, 100.0), (2.0, 10.0)],
    encoding="real",
    vectorized=True,
    constraints=constraints_batch,
    n_constraints=1,
    name="process_design_surrogate",
)
```

Do not set `vectorized=True` just because VAMOS itself works with populations. If your simulator accepts one design at a time, leave the default `vectorized=False`; VAMOS performs the row-by-row adaptation for you.

## Bounds and encodings

Use `bounds=[(lower, upper), ...]` when variables have different limits. When every variable shares the same limits, `xl` and `xu` can be scalars. Do not pass `bounds` together with `xl` or `xu`.

`make_problem(...)` accepts the public encoding names `"real"`, `"binary"`, `"integer"`, `"permutation"`, and `"mixed"`. The encoding describes the decision variables; the selected algorithm and operators must support it. Consult [Algorithms & Backends](../reference/algorithms.md) before assuming that every algorithm supports every encoding.

## When the evaluator is a simulator or external model

Your objective function can call a more substantial model instead of evaluating a closed-form equation. Keep the VAMOS boundary small: receive decision variables, run the model, and return objective values. Put constraints in the separate constraint callable when possible.

For expensive or failure-prone models, define before the experiment what happens when the model fails. Silently returning arbitrary objective values can change the scientific problem. Also avoid hidden global state, uncontrolled random seeds, and mutable files if you need reproducible studies.

## Common mistakes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| VAMOS reports an objective shape mismatch | function returns the wrong number of objectives | make the return length equal `n_obj` |
| Constraint shape mismatch | constraint function and `n_constraints` disagree | return exactly `n_constraints` values |
| Known feasible designs are rejected | inequality sign was translated backwards | rewrite the requirement as `g(x) <= 0` and test hand-picked points |
| Solutions have impossible domain values | bounds do not encode the real admissible range | correct the problem bounds before tuning the algorithm |
| Results change unexpectedly between runs | evaluator has uncontrolled randomness or state | make evaluator randomness explicit and reproducible |
| Vectorized mode fails | batch function does not return `(N, n_obj)` or `(N, n_constraints)` | inspect the array shapes before optimization |

## When to use a reusable problem class

For a one-off scientific problem, prefer `make_problem(...)`. If you are contributing a reusable benchmark or real-world problem implementation to VAMOS itself, follow the contributor workflow in [Adding a problem](../dev/add_problem.md).

## Next steps

- [Understanding optimization results](understanding-results.md) — keep decisions and objective rows aligned and understand result modes.
- [Constraints](../reference/constraints.md) — constraint-handling strategies and reference conventions.
- [Run artifacts & replay](run-artifacts.md) — preserve the run and its resolved configuration.
- [Durable studies](studies.md) — repeat a validated problem across algorithms and seeds.
- [API Reference](../reference/api_reference.md) — exact public signatures.
