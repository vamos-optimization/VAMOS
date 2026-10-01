# VAMOS Cookbook

Copy-paste recipes for the published **VAMOS 1.0.1** package. Install the core
package using [Installation](installation.md); recipes that need an extra say
so explicitly. Budgets below are demonstrations, not evidence of convergence
or comparative performance.

**Stable** identifies documented core APIs and CLI commands. **Experimental**
identifies custom callbacks, analysis helpers, or optional integrations whose
contracts may change. See [Stability and versioning](../project/stability-and-versioning.md).
Python recipes are self-contained except where an earlier recipe is explicitly
required. Run file-writing examples in a fresh working folder; canonical run
destinations must not already exist.

## Recommended path (optimize)

**Stable.** Start with explicit algorithm, budget, population, backend, and seed:

```python
from vamos import optimize

result = optimize(
    "zdt1", algorithm="nsgaii", max_evaluations=400,
    pop_size=40, engine="numpy", seed=7,
)
front = result.front()
assert front is not None
print(f"Returned: {len(result)} solutions; non-dominated: {len(front)}")
print(front[:3])
```

See [Understanding results](understanding-results.md) before interpreting the
returned population as a Pareto-front approximation.

## 1. Custom problem definition

**Stable.** Use `make_problem` to wrap a batch objective. This formula is finite
at every point in the box, including `x[0] = 0`:

```python
import numpy as np
from vamos import make_problem, optimize


def objectives(X):
    f1 = X[:, 0]
    f2 = (1.0 + X[:, 1]) * (1.0 - np.sqrt(X[:, 0]))
    return np.column_stack([f1, f2])


problem = make_problem(
    objectives, n_var=2, n_obj=2, bounds=[(0.0, 1.0), (0.0, 1.0)],
    encoding="real", vectorized=True, name="two_objective_example",
)
assert np.isfinite(objectives(np.array([[0.0, 0.0], [1.0, 1.0]]))).all()
result = optimize(
    problem, algorithm="nsgaii", max_evaluations=400,
    pop_size=40, engine="numpy", seed=7,
)
print(result.front())
```

For a vectorized class implementation, see
[Solving your own problem](custom-problem.md).

## 2. Handling constraints

**Stable.** Bounds belong in `bounds`. Other constraints use `g(x) <= 0` for
feasibility, with `n_constraints` matching the function output. Here the
requirement is `x[0] + x[1] <= 1.5`:

```python
import numpy as np
from vamos import make_problem, optimize

problem = make_problem(
    lambda x: [x[0], (1.0 + x[1]) * (1.0 - np.sqrt(x[0]))],
    n_var=2, n_obj=2, bounds=[(0.0, 1.0), (0.0, 1.0)], encoding="real",
    constraints=lambda x: [x[0] + x[1] - 1.5], n_constraints=1,
)
result = optimize(
    problem, algorithm="nsgaii", max_evaluations=400,
    pop_size=40, engine="numpy", seed=7,
)
G = result.data["G"]
feasible = np.all(G <= 0.0, axis=1)
print(f"Feasible returned solutions: {feasible.sum()} / {len(feasible)}")
```

`result.front()` filters objective-space dominance; it does not independently
check constraint feasibility. See [Constraints](../reference/constraints.md)
for handling modes and the [algorithm matrix](../reference/algorithms.md)
for support. Internal symbolic-constraint modules are not part of this recipe's
public API.

## 3. Progress callback

**Experimental custom interface.** `live_viz` expects `on_start`,
`on_generation`, and `on_end`; it does not call `__call__(algorithm)`.
A console observer needs no plotting extra:

```python
from vamos import optimize


class Progress:
    def __init__(self):
        self.generations = []
        self.finished = False

    def on_start(self, ctx=None):
        print("Run started")

    def on_generation(self, generation, F=None, X=None, stats=None):
        self.generations.append(generation)
        if F is not None:
            print(f"Generation {generation}: {len(F)} objective rows")

    def on_end(self, final_F=None, final_stats=None):
        self.finished = True
        print("Run finished")


progress = Progress()
result = optimize(
    "zdt1", algorithm="nsgaii", max_evaluations=400,
    pop_size=40, engine="numpy", seed=7, live_viz=progress,
)
assert progress.generations and progress.finished
```

Observe the supplied arrays without modifying them. Custom callbacks are not
covered by the built-in exact-replay contract.

## 4. Saving work and resuming a study

**Stable:** save completed results with `save_result` (recipe 13), and use the
[durable study lifecycle](studies.md) to resume pending/interrupted tasks or
retry eligible failed tasks.

A saved run is a numerical result and its provenance, not a portable snapshot
of an algorithm midway through a generation. Study resume does not promise to
continue an interrupted algorithm at its last in-memory generation. Pickling
algorithm objects is not a supported VAMOS checkpoint format. Internal
checkpoint/state payloads have no public persistence guarantee.

## 5. Using Numba

**Stable backend selection; optional dependency.** Install the `compute` extra
from [Installation](installation.md#optional-extras), then request it explicitly:

```python
from vamos import optimize

result = optimize(
    "zdt1", algorithm="nsgaii", engine="numba",
    max_evaluations=400, pop_size=40, seed=7,
)
print(result.F.shape)
```

The first use can include compilation time. Report warm-up separately in
performance comparisons; cross-backend bitwise equality is not promised.

## 6. Comparing algorithms visually

**Stable optimization; external plotting dependency.** Install the `analysis`
extra. Keep the problem, evaluation budget, population, and seed explicit:

```python
import matplotlib.pyplot as plt
from vamos import optimize

fig, ax = plt.subplots()
for algorithm, label in [("nsgaii", "NSGA-II"), ("moead", "MOEA/D")]:
    result = optimize(
        "zdt1", algorithm=algorithm, max_evaluations=4000,
        pop_size=80, engine="numpy", seed=7,
    )
    front = result.front()
    assert front is not None
    ax.scatter(front[:, 0], front[:, 1], s=12, label=label)
ax.set(xlabel="f1 (minimize)", ylabel="f2 (minimize)")
ax.legend()
fig.savefig("algorithm-comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

One seeded plot is illustrative, not an algorithm ranking. Use multiple seeds
and problems for a comparative study; see [Analysis](../topics/analysis.md).

## 7. Inspect resolved defaults

**Stable.** Keep a bounded budget while asking VAMOS to resolve the algorithm
and population defaults:

```python
from vamos import optimize

result = optimize("zdt1", max_evaluations=400, seed=7)
explanation = result.explain_defaults()
print(explanation["resolved_spec"])
print(explanation["default_sources"])
```

The resolved specification includes the problem, algorithm, operators,
backend, termination, seed, and population. Default sources distinguish
inferred values from explicit settings.

## 8. Discover and configure operators

**Stable.** Use operator identifiers through the public algorithm facade:

```python
from vamos.algorithms import available_crossover_methods, available_mutation_methods

print(available_crossover_methods("real"))
print(available_mutation_methods("real"))
print(available_crossover_methods("permutation"))
```

Pass identifiers to a configuration builder as in recipe 10. Direct operator
classes in internal engine modules are for contributors; see
[Adding an operator](../dev/add_operator.md).

## 9. Multi-seed runs

**Stable.** A list of seeds returns an in-memory `StudyResult`:

```python
from vamos import optimize

study = optimize(
    "zdt1", algorithm="nsgaii", max_evaluations=400,
    pop_size=40, engine="numpy", seed=[0, 1, 2, 3],
)
for seed, result in zip([0, 1, 2, 3], study.runs):
    print(seed, len(result.front()), result.data["evaluations"])
print("Mean evaluations:", study.mean("evaluations"))
```

This convenience does not create a durable study directory. For campaigns
with saved plans, task history, and resume, use [Studies](studies.md).

## 10. Algorithm config objects

**Stable.** Make population and variation settings explicit:

```python
from vamos import optimize
from vamos.algorithms import NSGAIIConfig

cfg = (
    NSGAIIConfig.builder()
    .pop_size(40)
    .offspring_size(40)
    .crossover("sbx", prob=1.0, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .selection("tournament", size=2)
    .build()
)
result = optimize(
    "zdt1", algorithm="nsgaii", algorithm_config=cfg,
    max_evaluations=400, engine="numpy", seed=7,
)
print(result.F.shape)
```

## 11. Multiprocessing evaluation

Use the documented CLI options to select the evaluation strategy and worker
count without importing an internal backend class. Parallel evaluation is for
expensive independent objectives; process overhead can dominate this cheap
demonstration:

```bash
vamos --problem zdt1 --algorithm nsgaii --engine numpy --population-size 40 --max-evaluations 400 --seed 7 --eval-strategy multiprocessing --n-workers 2 --output-root results/cookbook-multiprocessing
```

The single-run CLI is **Stable**; custom evaluators and distributed study
ownership remain **Experimental**. This command distributes evaluations
within one run, not concurrent mutation of a durable study.

## 12. Hypervolume-based early stopping

**Stable single-run CLI.** For built-in ZDT1, stop at a fraction of its reference
hypervolume, with a hard evaluation cap:

```bash
vamos --problem zdt1 --algorithm nsgaii --engine numpy --population-size 40 --max-evaluations 400 --seed 7 --hv-threshold 0.9 --output-root results/cookbook-hv-stop
```

The small budget can expire before the 0.9 target is reached. The target is a
reference-relative stopping criterion, not proof of convergence. See
[CLI options](cli.md#key-flags) for `--hv-reference-front`.

## 13. Save and load a run artifact

**Stable.** Save a complete canonical run and load its result without executing
optimization again:

```python
from vamos import load_result, load_run, optimize, save_result

result = optimize(
    "zdt1", algorithm="nsgaii", max_evaluations=400,
    pop_size=40, engine="numpy", seed=7,
)
stored = save_result(result, "results/cookbook-run")
loaded = load_result(stored.root)
run = load_run(stored.root)
print(loaded.F.shape, run.manifest.run_id)
```

The destination must not exist. See [Run artifacts](run-artifacts.md) for the
layout, integrity checks, resource limits, and non-destructive writes.

## 14. Select one solution

**Stable.** `balanced_sum` minimizes the sum after objective-wise normalization
within the returned non-dominated set:

```python
from vamos import optimize

result = optimize(
    "zdt1", algorithm="nsgaii", max_evaluations=400,
    pop_size=40, engine="numpy", seed=7,
)
choice = result.best("balanced_sum")
print("Decision:", choice["X"])
print("Objectives:", choice["F"])
```

This choice reflects one preference rule, not a uniquely optimal trade-off.
For constrained runs, check feasibility before applying a preference rule.

## 15. Verify and reproduce a stored run

**Stable. Requires recipe 13 first.** Loading reads stored data; verification
checks integrity and compatibility; reproduction executes a new run using the
stored resolved configuration and seed:

```python
from vamos import load_run, reproduce, verify_run

run = load_run("results/cookbook-run", verify="all")
print(run.status, run.result.F.shape)
verification = verify_run("results/cookbook-run", require_level="exact")
print("Environment:", verification.environment.level)
replay = reproduce("results/cookbook-run", output="results/cookbook-replay")
assert replay.exact
print(replay.exact, replay.output_root)
```

Exact replay is available for supported built-ins in a matching material
environment. Custom Python problems, plugins, and cross-backend replay are
outside that contract. The output destination must not already exist.

## 16. Validate a config file

**Stable.** Save this as `experiment.json` in your working folder:

```json
{
  "version": "1",
  "defaults": {
    "problem": "zdt1",
    "algorithm": "nsgaii",
    "engine": "numpy",
    "population_size": 40,
    "max_evaluations": 400,
    "seed": 7,
    "output_root": "results/cookbook-config"
  }
}
```

Validate first, then execute:

```bash
vamos --config experiment.json --validate-config
vamos --config experiment.json
```

See [CLI and config files](cli.md) for overrides.

## 17. Export a DataFrame

**Experimental analysis helper.** Install the `analysis` extra. This exports
the returned result rows, which need not all be non-dominated:

```python
from pathlib import Path
from vamos import optimize
from vamos.ux.api import result_to_dataframe

result = optimize(
    "zdt1", algorithm="nsgaii", max_evaluations=400,
    pop_size=40, engine="numpy", seed=7,
)
output = Path("exports/zdt1-results.csv")
output.parent.mkdir(parents=True, exist_ok=True)
df = result_to_dataframe(result)
df.to_csv(output, index=False)
print(output, df.shape)
```

## 18. Combine fronts from multiple runs

**Stable result API.** Form a pooled objective-space non-dominated set. The
combined object below is an analysis container, not a new optimization run:

```python
import numpy as np
from vamos import OptimizationResult, optimize

study = optimize(
    "zdt1", algorithm="nsgaii", max_evaluations=400,
    pop_size=40, engine="numpy", seed=[0, 1, 2],
)
combined = np.vstack([result.F for result in study.runs])
front = OptimizationResult({"F": combined}).front()
assert front is not None
print(f"Pooled rows: {len(combined)}; non-dominated: {len(front)}")
```

Pool only the same objective definitions, units, and feasibility conventions.
The pooled set is an empirical reference, not the known true Pareto front.

## 19. Compute hypervolume

**Experimental third-party integration.** The `compute` extra installs MooCore.
Use its public function instead of importing VAMOS's internal numerical helpers:

```python
import numpy as np
from moocore import hypervolume
from vamos import optimize

result = optimize(
    "zdt1", algorithm="nsgaii", max_evaluations=400,
    pop_size=40, engine="numpy", seed=7,
)
front = result.front()
assert front is not None
reference_point = np.array([1.1, 11.0])
hv = float(hypervolume(front, ref=reference_point))
print("Hypervolume:", hv)
```

Both objectives are minimized. This fixed point is worse than the full ZDT1
objective bounds on its default box. Use the **same reference point** when
comparing runs; changing it changes the indicator's meaning.

## 20. Reference-relative hypervolume for ZDT1

**Experimental third-party integration; requires `compute`.** Normalize using
a sampled analytical ZDT1 front and the same reference point in numerator and
denominator:

```python
import numpy as np
from moocore import hypervolume
from vamos import optimize

result = optimize(
    "zdt1", algorithm="nsgaii", max_evaluations=400,
    pop_size=40, engine="numpy", seed=7,
)
front = result.front()
assert front is not None
f1 = np.linspace(0.0, 1.0, 1001)
reference_front = np.column_stack([f1, 1.0 - np.sqrt(f1)])
reference_point = np.array([1.1, 11.0])
hv = float(hypervolume(front, ref=reference_point))
reference_hv = float(hypervolume(reference_front, ref=reference_point))
print("Reference-relative HV:", hv / reference_hv)
```

This ratio depends on the reference point and the discretization of the
analytical front. It is not a universal percentage of convergence. Record
both when reporting results; values from different reference conventions are
not directly comparable.
