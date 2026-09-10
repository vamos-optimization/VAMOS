# Quickstart: run, read, then preserve a result

This guide uses only the stable VAMOS 1.0.0 facades. Install the core package as described in the [installation guide](installation.md).

For a repository-checkout version that you can run unchanged, use the [Try VAMOS executable journey](../examples.md#try-vamos).

## 1. Run one optimization

```python
from vamos import optimize

result = optimize(
    "zdt1",
    algorithm="nsgaii",
    max_evaluations=400,
    pop_size=40,
    engine="numpy",
    seed=42,
)

print(result.F.shape)
print(result.X.shape)
print(result.data["evaluations"])
```

`F` contains objective values and `X` contains the corresponding decision variables. Row `i` in `X` belongs to row `i` in `F`. `max_evaluations` is a hard budget; it is not the number of rows returned.

NumPy is the deterministic reference backend. Reproducibility is a same-environment promise, not a cross-platform or cross-backend bitwise promise.

## 2. Read the result before doing anything else

The simple call above returns the final 40-member NSGA-II population. Extract its non-dominated subset with the result object itself:

```python
front_F, front_indices = result.front(return_indices=True)
front_X = result.X[front_indices]

print(front_F.shape)
print(front_X.shape)
```

![Returned population and non-dominated subset for the Quickstart run.](../assets/results/understanding-results.svg)

*NumPy, seed 42, population 40, 400 evaluations. This short run illustrates the structure of an `OptimizationResult`; it is not a convergence or performance claim.*

The crosses are non-dominated **within the returned population**. They are not automatically the mathematical Pareto front. The dedicated [Understanding optimization results](understanding-results.md) guide explains `X`, `F`, `front()`, result modes, population/archive data, and why selecting one preferred point requires an additional decision rule.

!!! tip "A useful stopping point for a first session"
    If this is your first VAMOS run, you now have the complete minimal loop: execute a bounded optimization and understand what its arrays represent. The remaining sections extend that loop to custom problems, explicit configuration and reproducible evidence.

## 3. Define a problem

```python
from vamos import make_problem, optimize

problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2,
    n_obj=2,
    bounds=[(0, 1), (0, 1)],
    encoding="real",
)

result = optimize(
    problem,
    algorithm="nsgaii",
    max_evaluations=400,
    pop_size=40,
    seed=42,
)
```

The default `vectorized=False` adapter calls this scalar function once per solution. For a function that accepts an `(N, n_var)` batch and returns an `(N, n_obj)` array, pass `vectorized=True` to `make_problem`.

## 4. Use an explicit algorithm configuration

```python
from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import ZDT1

problem = ZDT1(n_var=30)
configuration = NSGAIIConfig.default(pop_size=40, n_var=problem.n_var)

result = optimize(
    problem,
    algorithm="nsgaii",
    algorithm_config=configuration,
    max_evaluations=400,
    seed=42,
)
```

Use a public configuration object when the exact operators and their settings need to be preserved. VAMOS rejects a configuration that does not match the selected algorithm.

Result selection also belongs to the algorithm configuration. In particular, the explicit NSGA-II configuration above uses its configuration-level `non_dominated` result mode, so the top-level row count need not equal 40. See [Understanding optimization results](understanding-results.md#know-which-set-x-and-f-represent) before comparing row counts between configurations.

## 5. Save, verify, and replay

```python
from vamos import load_result, reproduce, save_result, verify_run

stored = save_result(result, "runs/zdt1-seed-42")
verification = verify_run(stored.root, require_level="exact")
loaded = load_result(stored.root)
replay = reproduce(stored.root, output="runs/replays/zdt1-seed-42")

print(verification.environment.level)
print(loaded.F.shape)
print(replay.exact)
```

Loading and verification are data-only. `reproduce` is the separate executable operation and creates a new run directory; it never overwrites the source. Exact replay is limited to reconstructable built-ins in a materially matching environment.

The equivalent stable CLI is:

```bash
vamos results inspect runs/zdt1-seed-42
vamos results verify runs/zdt1-seed-42 --require-level exact
vamos reproduce runs/zdt1-seed-42 --output runs/replays/zdt1-seed-42
```

## 6. Run a durable study

```python
from vamos import StudySpec, create_study

spec = StudySpec(
    problems=["zdt1", "zdt2"],
    algorithms=["nsgaii", "moead"],
    seeds=[0, 1],
    max_evaluations=400,
    pop_size=40,
    on_error="continue",
)

completed = create_study(spec, output="studies/comparison").run()
print(completed.inspect().counts)
print(len(completed.summarize().rows))
```

A durable study is single-owner and sequential in VAMOS 1.0.0. See the [study guide](studies.md) for planning, inspection, resume, and retry.

## Next steps

- [Understanding optimization results](understanding-results.md)
- [Run and understand NSGA-II](../algorithms/nsgaii.md)
- [Solve your own problem](custom-problem.md)
- [Run artifacts and exact replay](run-artifacts.md)
- [Durable studies](studies.md)
- [Stability and versioning](../project/stability-and-versioning.md)
- [Known limitations](../project/known-limitations.md)
