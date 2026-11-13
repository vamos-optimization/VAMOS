# Constraints

!!! info "Stable problem-definition API"
    `make_problem` and `optimize` are Stable in VAMOS 1.1.0. The violation
    helpers are Internal. See
    [Stability and versioning](../project/stability-and-versioning.md).
    **Development after 1.1.0:** the internal symbolic constraint DSL
    (`vamos.foundation.constraints.dsl`), which no algorithm or guide used, is
    removed; declare constraints with `make_problem` or a `Problem` subclass.

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

## Choose a constraint-handling technique

**Development after 1.1.0:** NSGA-II, NSGA-III, SMS-EMOA, AGE-MOEA, RVEA and
SMPSO take `constraint_handling`, the technique that ranks or compares
constrained solutions. They are the techniques of jMetal's package
`util.constrainthandling`, which Evolver's constrained spaces tune:

| Technique | Parameters | Ranking |
| --- | --- | --- |
| `cdp` (default) | none | Deb's constrained-domination principle: feasible solutions first, by Pareto dominance, then infeasible ones by total violation. |
| `violated_constraints` | none | The same rule over the number of violated constraints. |
| `epsilon_constrained` | `initial_quantile`, `exponent`, `decay_fraction` | Takahama and Sakai's epsilon-constrained method: a solution whose violation is at most epsilon counts as feasible. Epsilon starts at that quantile of the violations of the first ranked population and decays to 0 at that fraction of the run. |
| `push_pull` | `change_threshold`, `window_fraction`, `decay_fraction`, `exponent`, `feasibility_threshold`, `reduction_rate` | Fan et al.'s push and pull search: Pareto dominance alone until the ideal and nadir points settle, then epsilon-constrained. |
| `stochastic_ranking` | `probability` | Runarsson and Yao's stochastic ranking: a bubble sort that compares by objectives with that probability, or always between feasible solutions. |
| `self_adaptive_penalty` | none | Woldesenbet, Yen and Tessema's self-adaptive penalty: Pareto dominance over objectives modified by the violation and the fraction of feasible solutions. |

```python
from vamos.algorithms import NSGAIIConfig

config = (
    NSGAIIConfig.builder()
    .pop_size(100)
    .crossover("sbx", prob=0.9, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .constraint_handling("epsilon_constrained", initial_quantile=0.2, exponent=5.0, decay_fraction=0.8)
    .build()
)
```

The other configurations take the same field and builder method, such as
`NSGAIIIConfig.builder().constraint_handling(...)`. Every parameter of a
technique is required, as in jMetal. The techniques that change during a run,
`epsilon_constrained` and `push_pull`, follow its progress: the evaluations
before the offspring's over the evaluation budget. Each ranks fixed populations
as jMetal does, which `tests/engine/test_constraint_handling_techniques.py`
checks against jMetal's own implementation.

Before each replacement, the technique adapts to the union of parents and
offspring and ranks it, as Evolver's algorithms do, and its fronts take the
place of the non-dominated ones:

| Algorithm | Replacement | Tournament |
| --- | --- | --- |
| NSGA-II | Whole fronts survive; crowding distance cuts the last. | Rank, then crowding distance. |
| NSGA-III | Whole fronts survive; reference-direction niching cuts the last. | Rank, then crowding distance. |
| SMS-EMOA | Once per child, the least hypervolume contributor of the last front leaves. | Rank, then crowding distance. |
| AGE-MOEA | Whole fronts survive; the survival score cuts the last. | Rank, then survival score. |

RVEA and SMPSO take only the four techniques that judge the violation of a
single solution, as Evolver's `EffectiveViolation` does: `cdp` (the total
violation), `violated_constraints` (their number), `epsilon_constrained` (the
violation, or 0 within epsilon) and `push_pull` (0 while it pushes, then as
`epsilon_constrained`). RVEA's feasibility-first survival judges feasibility
and violation by that measure, after the technique adapts to the union of
parents and offspring. In SMPSO, the technique adapts to the swarm once per
iteration and decides, as its jMetal comparator, whether a particle replaces
its personal best: the less violated position wins, and two equally violated
ones compare by Pareto dominance when both count as feasible, and also otherwise
for `cdp` and `violated_constraints`. SMPSO's leaders archive keeps comparing by
the constrained-domination principle.

The tournament compares the technique's ranks of the population, which VAMOS
ranks again, as it does with `cdp`; jMetal reuses the ranks of the replacement.
Stochastic ranking draws its uniforms from the run's generator, so a run stays
a function of its seed, and the engines give the same run. NSGA-II's checkpoints
keep the state of the technique. `cdp` keeps each algorithm's own survival, so
its results are those of VAMOS 1.1.0. A technique other than `cdp` needs
`constraint_mode="feasibility"`.

**Development after 1.1.0:** [CCMO](../algorithms/ccmo.md) is an algorithm
designed for constrained problems rather than a technique: it ranks its main
population with the violation first and evolves a second population that
ignores the constraints, whose offspring the main population can select.

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
