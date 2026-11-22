---
description: 'The evolutionary loop of a genetic algorithm: population, representation (real, binary, integer, permutation, mixed), evaluation, parent selection, crossover and mutation, survival and elitism, generational and steady-state replacement, termination, and the NumPy, Numba and MooCore engines of VAMOS, with an interactive figure and VAMOS code.'
search:
  boost: 0.5
---

# Genetic algorithms: the evolutionary loop

The previous lessons described the goal: a set of solutions close to the
Pareto front and well spread along it, judged with quality indicators. This
lesson explains how evolutionary algorithms build such a set. A **genetic
algorithm** (GA) imitates natural evolution, as proposed by Holland [1] and
popularized by Goldberg [2]: a *population* of candidate solutions reproduces,
the children inherit features of their parents with small changes, and the
better solutions survive. Repeated over many generations, this loop improves
the population without any gradient or model of the problem.

A GA keeps many solutions at once, so one run can return a whole set of
trade-offs. That is why evolutionary algorithms suit multi-objective problems
[3, 4]; Schaffer [5] proposed the first multi-objective genetic algorithm in
1985. Every algorithm of the following lessons runs the loop described here
and differs in a few of its steps.

In this lesson you will learn:

- the steps of the evolutionary loop: population, representation, evaluation,
  parent selection, variation, survival and termination;
- the encodings that VAMOS supports and the operators that each one needs;
- what elitism is, and what (μ + λ), (μ, λ) and (μ + 1) mean;
- how generational and steady-state algorithms differ, and which VAMOS
  algorithms are which;
- why survival is the step that multi-objective algorithms must rethink;
- how VAMOS ends a run, and which engine runs its numerical steps.

## The loop

Some vocabulary first. An *individual* is one candidate solution; the
*population* is the set of N individuals the algorithm keeps; the *parents*
are the individuals chosen to reproduce and the *offspring* (or children) are
the new individuals they produce; one pass through the loop is a
*generation*. The loop itself is short:

<div class="vf-formula">P ← N random individuals;  evaluate P
while the termination criterion is not met:
    parents   ← select(P)                       parent selection
    offspring ← vary(parents)                   crossover and mutation
    evaluate(offspring)
    P ← survive(P, offspring)                   survival, also called replacement
return the best individuals of P</div>

The sections below take the steps one at a time. The interactive figure
further down runs the whole loop on six individuals.

## The population

The population is the memory of the search. Its size N trades breadth for
speed: a large population covers the front densely and keeps more diversity,
but each generation costs more evaluations, so a fixed budget allows fewer
generations. `NSGAIIConfig.default()` uses N = 100 (`pop_size`), and lesson
[NSGA-II at a glance](../intermediate/nsga-ii.md) shows the effect of the
choice on a running algorithm.

The initial population is usually random. Unless told otherwise, VAMOS draws
each individual uniformly within the bounds of the problem, with one branch
per encoding:

```python
# Excerpt from src/vamos/engine/algorithm/components/population.py
    if normalized == "permutation":
        return random_permutation_population(pop_size, n_var, rng)
    if normalized == "binary":
        return random_binary_population(pop_size, n_var, rng)
    if normalized == "integer":
        return random_integer_population(pop_size, n_var, xl.astype(int), xu.astype(int), rng)
    if normalized == "mixed":
        spec = getattr(problem, "mixed_spec", None)
        if spec is None:
            raise ValueError("Mixed-encoding problems must define 'mixed_spec'.")
        return mixed_initialize(pop_size, n_var, spec, rng)
    return rng.uniform(np.asarray(xl, dtype=np.float64), np.asarray(xu, dtype=np.float64), size=(pop_size, n_var))
```

`rng` is the run's random generator, created from its seed, so the same seed
gives the same initial population. For real variables the builders'
`.initializer(...)` also offers Latin hypercube (`"lhs"`), Sobol and Halton
sampling, which cover the bounds more evenly than independent draws.

## Representation

The *representation*, or *encoding*, is how a solution is written down as a
decision vector x. Holland's GAs used bit strings for everything [1]; modern
algorithms use the representation that fits the problem, because the
variation operators must produce valid solutions. VAMOS supports five
encodings, and a problem declares its own in its `encoding` attribute:

| Encoding | A decision vector looks like | Typical problems | VAMOS example |
| --- | --- | --- | --- |
| Real | (0.31, 0.02, 0.77, …), each value within its bounds | Engineering design, continuous benchmarks | `zdt1` |
| Binary | (1, 0, 0, 1, 1, …) | Selecting items or features | `bin_knapsack` |
| Integer | (3, 0, 7, 10, …), each value within its bounds | Allocations, counts | `int_alloc` |
| Permutation | (2, 5, 0, 1, 4, 3), every value exactly once | Tours, schedules | `tsp6` |
| Mixed | real, integer and categorical values side by side | Configurations, designs | `mixed_design` |

Problems, operators and algorithms must agree on the encoding: a real-coded
crossover applied to a tour would repeat some cities and miss others. When a
configuration is built for a problem, VAMOS picks the operators from the
problem's encoding. These are the defaults of NSGA-II for one built-in problem
of each encoding:

```python
from vamos import make_problem_selection
from vamos.algorithms import NSGAIIConfig

for name in ("zdt1", "bin_knapsack", "int_alloc", "tsp6", "mixed_design"):
    problem = make_problem_selection(name).instantiate()
    config = NSGAIIConfig.default(n_var=problem.n_var, encoding=problem.encoding)
    crossover, mutation = config.crossover[0], config.mutation[0]
    print(f"{name:<13} {problem.encoding:<12} {problem.n_var:>2} variables   crossover {crossover:<8} mutation {mutation}")
```

Output:

```text
zdt1          real         30 variables   crossover sbx      mutation pm
bin_knapsack  binary       50 variables   crossover uniform  mutation bitflip
int_alloc     integer      20 variables   crossover sbx      mutation pm
tsp6          permutation   6 variables   crossover ox       mutation swap
mixed_design  mixed         9 variables   crossover mixed    mutation mixed
```

`available_crossover_methods(encoding)` and `available_mutation_methods(encoding)`
in `vamos.algorithms` list the alternatives for each encoding. The
[capability matrix](../../reference/algorithms.md#capability-matrix) gives the
encodings each algorithm accepts; SMPSO, for example, needs real or mixed
variables. The tutorial [Choose a decision representation](../../tutorials/encodings.md)
runs a complete example for each discrete encoding.

## Evaluation

*Evaluation* computes the objectives, and the constraints if there are any, of
a decision vector: it is the only place where the algorithm meets the problem.
In VAMOS a problem's `evaluate(X, out)` receives a whole batch of decision
vectors, one per row of `X`, and writes their objective vectors into
`out["F"]`. For real problems evaluation usually dominates the cost of a run,
because it may involve a simulation or the training of a model. That is why
run lengths are measured in **evaluations** rather than generations or seconds:
the evaluation budget is the fair currency for comparing algorithms.

## Parent selection

Parent selection chooses who reproduces. It should prefer good individuals, so
that their features spread, without always picking the same few, which would
make the population collapse onto them. Two schemes cover most algorithms:

- **Random selection** draws parents uniformly from the population, with no
  preference. The pressure towards better solutions then comes from survival
  alone. SMS-EMOA uses it in VAMOS (`.selection("random")`).
- **Tournament selection** draws k individuals at random and keeps the best of
  them [6]. A larger k means stronger *selection pressure*. Binary tournaments,
  k = 2, are the most common, and the VAMOS default for NSGA-II
  (`.selection("tournament", size=2)`).

A tournament needs to decide which of two individuals is better. With one
objective that is a comparison of two numbers; with several objectives two
individuals can be mutually non-dominated (lesson
[Pareto dominance](pareto-dominance.md)). NSGA-II compares the front each
individual belongs to and, within a front, how isolated it is (lesson
[Crowding and tournaments](../intermediate/crowding-distance.md)).

## Variation

Variation creates the offspring from the parents, with two kinds of operator
[3]:

- **Crossover** (recombination) combines two parents into children that mix
  their features. It is applied to a pair with a crossover probability, 0.9 in
  VAMOS' real-coded default; otherwise the children are copies of the parents.
- **Mutation** perturbs a child at random, variable by variable, with a small
  probability per variable, by default 1/n for n variables. It brings in values
  that no parent has.

Crossover exploits what the population already found; mutation explores
around it. Each encoding has its own operators, as the example above showed.
The section [Real-coded operators](../operators/real-coded.md) explains SBX and
polynomial mutation, the default operators for real variables, and what their
parameters control.

## Survival and elitism

Survival decides which individuals form the next population. The notation
comes from evolution strategies [7], with μ parents and λ offspring:

- **(μ, λ)**: the offspring replace the parents, and the best μ of the λ
  offspring survive. A parent never survives, however good it is.
- **(μ + λ)**: parents and offspring compete together, and the best μ of the
  μ + λ survive. A good parent survives until something better replaces it.

The second scheme is **elitist**: the best solutions found so far are never
lost. Zitzler, Deb and Thiele [8] compared multi-objective algorithms on the
ZDT problems and found elitism to be an important factor of performance, and
the algorithms of VAMOS are elitist, in their population or in an archive.
NSGA-II uses (μ + λ) with λ = μ. SPEA2 and SMPSO replace their population (or
swarm) completely, but keep the best solutions found so far in an archive
(lessons [SPEA2](../intermediate/spea2.md) and
[SMPSO](../intermediate/smpso.md)).

**The multi-objective twist.** "Keep the best μ" assumes that individuals can
be sorted from best to worst. With one objective they can. With several
objectives they cannot: in a well-converged population almost every
individual is non-dominated, and Pareto dominance alone says nothing about
which of them to keep. Every multi-objective algorithm therefore needs a rule
that ranks a set of objective vectors, one that favours both convergence and
spread. The next lesson, [Families of algorithms](algorithm-families.md),
groups the algorithms by that rule.

## Generational and steady-state

When λ = μ, the whole population is renewed in each generation: the algorithm
is **generational**, like NSGA-II. At the other extreme, λ = 1: the algorithm
creates one child, evaluates it, and lets it compete with the population at
once. This is a **steady-state** algorithm, written (μ + 1) when the child
competes with all μ parents [9]:

| | Generational (μ + λ), λ = μ | Steady-state (μ + 1) |
| --- | --- | --- |
| Offspring per cycle | N | 1 |
| Survival steps for a budget of B evaluations | (B − N) / N | B − N |
| When a good child can become a parent | In the next generation | At once |
| Algorithms in VAMOS | NSGA-II, SPEA2, IBEA, NSGA-III, RVEA, AGE-MOEA | SMS-EMOA, MOEA/D, NSGA-II with `.steady_state()` |

A steady-state algorithm uses each new solution as early as possible, which
can speed up convergence for the same number of evaluations; Durillo et al.
[10] studied the effect on NSGA-II and other multi-objective genetic
algorithms. The price is
computational: the survival rule runs after every evaluation instead of once
per N evaluations. When evaluations are expensive that overhead does not
matter; when they are cheap it can dominate the running time.

SMS-EMOA [11] was designed as a steady-state algorithm, and VAMOS runs MOEA/D
steady-state as well (lesson [MOEA/D](../intermediate/moead.md)). NSGA-II [12]
is generational, and `.steady_state()` in its configuration builder switches it
to one offspring per cycle.

## Step through the loop

The figure runs a small GA on a two-variable problem with ZDT1's front:
f₁ = x₁ and f₂ = g·(1 − √(x₁/g)) with g = 1 + x₂, both minimized, so the Pareto
set is the bottom edge x₂ = 0 of the decision space. Six individuals, a budget
of 30 evaluations, NSGA-II's binary tournament and VAMOS' default real-coded
operators (SBX with probability 0.9 and η = 20, polynomial mutation with
probability 1/2 and η = 20). Survival keeps whole fronts while they fit and
breaks ties by crowding distance, the rule of NSGA-II.

Press **Next step**, or focus a plot and use the arrow keys, to go through
select, vary, evaluate and survive. The table states every value the plots
show. Things to try:

1. **Generational, elitist (μ + λ).** Step through the first cycle: six
   tournaments choose three pairs, the pairs produce children 7 to 12, and the
   twelve candidates compete for six places. Parents 1, 3, 4 and 5 and children
   9 and 10 leave.
2. **Generational, non-elitist (μ, λ).** The children replace all six parents.
   In the first cycle, parents 1, 2 and 6 are lost although no child dominates
   them. Run to the end and compare the final population with the elitist one.
3. **Steady-state (μ + 1).** The same 30 evaluations now give 24 cycles of one
   child each. Child 7 replaces parent 5 at once; in cycle 7, child 13 is the
   worst of the seven and is discarded.

<div class="vf-widget" data-vf-widget="ga-loop"><p class="vf-fallback">Interactive figure: a genetic algorithm with six individuals on a two-objective problem, shown in the decision space and in the objective space, step by step through parent selection, variation, evaluation and survival, with generational (μ + λ), generational (μ, λ) and steady-state (μ + 1) replacement. It needs JavaScript.</p></div>

## The loop in VAMOS

VAMOS algorithms run the same loop. In NSGA-II, one cycle is an *ask*, which
selects parents and creates the offspring, followed by the evaluation and a
*tell*, which applies survival. The ask, with its two first steps:

```python
# Excerpt from src/vamos/engine/algorithm/nsgaii/ask_tell.py
    request_size = capped_offspring_size(st.n_eval, st.max_evals, st.offspring_size, "NSGA-II")
    parents_per_group = st.variation.parents_per_group
    children_per_group = st.variation.children_per_group
    parent_count = int(np.ceil(request_size / children_per_group) * parents_per_group)
    mating_pairs = build_mating_pool(
        algo.kernel,
        ranks,
        crowding,
        st.pressure,
        st.rng,
        parent_count,
        parents_per_group,
        st.sel_method,
        candidate_indices=candidate_indices,
    )
    parent_idx = mating_pairs.reshape(-1)
    X_parents = st.variation.gather_parents(st.X, parent_idx)
    X_off = st.variation.produce_offspring(X_parents, st.rng)

    if X_off.shape[0] > request_size:
        X_off = X_off[:request_size]
```

- `request_size` is the number of offspring of this cycle: `pop_size` for the
  generational algorithm, 1 for the steady-state one, and never more than the
  evaluations left in the budget.
- `build_mating_pool` runs the tournaments on the ranks and crowding distances
  of the population, and `produce_offspring` applies crossover and mutation.
- A crossover produces two children per pair, so the steady-state algorithm
  keeps the first one.

The survival of the tell is the subject of lesson
[Elitist survival](../intermediate/elitist-survival.md).

The cycle is visible from the problem's side. The program below wraps ZDT1 so
that it records how many solutions each call to `evaluate` receives, and runs
generational NSGA-II, steady-state NSGA-II and SMS-EMOA with the same seed and
the same budget of 2,000 evaluations:

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import NSGAIIConfig, SMSEMOAConfig
from vamos.problems import ZDT1


class CountedZDT1(ZDT1):
    """ZDT1 that records how many solutions each call to evaluate receives."""

    def __init__(self, n_var):
        super().__init__(n_var=n_var)
        self.batches = []

    def evaluate(self, X, out):
        self.batches.append(len(X))
        super().evaluate(X, out)


steady = (
    NSGAIIConfig.builder()
    .pop_size(40)
    .steady_state()
    .crossover("sbx", prob=0.9, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .selection("tournament", size=2)
    .build()
)
runs = [
    ("NSGA-II generational", "nsgaii", NSGAIIConfig.default(pop_size=40, n_var=10)),
    ("NSGA-II steady-state", "nsgaii", steady),
    ("SMS-EMOA", "smsemoa", SMSEMOAConfig.default(pop_size=40, n_var=10)),
]
for label, algorithm, config in runs:
    problem = CountedZDT1(n_var=10)
    result = optimize(problem, algorithm=algorithm, algorithm_config=config, max_evaluations=2_000, seed=1)
    first, *rest = problem.batches
    front = result.front()
    gap = np.mean(front[:, 1] - (1.0 - np.sqrt(front[:, 0])))  # ZDT1 front: f2 = 1 - sqrt(f1)
    print(f"{label}: {config.selection[0]} selection, {len(problem.batches)} calls to evaluate")
    print(f"  first call {first} solutions, then {sorted(set(rest))} per call; {result.data['evaluations']} evaluations, mean gap {gap:.4f}")
```

Output:

```text
NSGA-II generational: tournament selection, 50 calls to evaluate
  first call 40 solutions, then [40] per call; 2000 evaluations, mean gap 0.0271
NSGA-II steady-state: tournament selection, 1961 calls to evaluate
  first call 40 solutions, then [1] per call; 2000 evaluations, mean gap 0.0075
SMS-EMOA: random selection, 1961 calls to evaluate
  first call 40 solutions, then [1] per call; 2000 evaluations, mean gap 0.0086
```

The first call evaluates the initial population. The generational algorithm
then evaluates 49 batches of 40 offspring, and the two steady-state algorithms
1,960 single children: the same budget, divided into 49 or 1,960 survival
steps. The *mean gap* is the average vertical distance of the returned
non-dominated points to the ZDT1 front. On this run both steady-state
algorithms end closer to the front than the generational one, but a single run
on a single problem proves nothing about algorithms in general (lesson
[Rigorous comparison](../advanced/performance-assessment.md)).

## Termination

A run stops when its termination criterion is met. In VAMOS, `optimize` takes
it as `max_evaluations=n`, or as `termination=` with one of four criteria
([termination criteria](../../reference/algorithms.md#termination-criteria)):

- `("max_evaluations", n)`: the evaluation budget, the usual choice for
  experiments. It counts the initial population, and VAMOS never exceeds it:
  when the budget is not a multiple of the batch size, the last batch is
  shortened (`capped_offspring_size` in the excerpt above).
- `("n_gen", g)`: g generations of `pop_size` evaluations each, the initial
  population included, so g · `pop_size` evaluations.
- `("hv", {...})`: stop once the hypervolume of the result reaches a target, or
  after a maximum number of evaluations.
- `("max_time", seconds)`: a wall-clock limit, available since VAMOS 1.3.0 and
  only through `optimize`. It is checked after each generation (MOEA/D after
  every `pop_size` evaluations, SMS-EMOA after each step), so a run can exceed
  the limit by one generation, and `("max_time", {"seconds": s, "max_evaluations": n})`
  adds an evaluation cap. How many evaluations fit in the time depends on the
  machine and its load, so a run limited by time is **not reproducible**:
  `verify_run` reports it as not exactly replayable and `reproduce` refuses it.

For comparisons, prefer an evaluation budget: it means the same thing on every
machine.

## Engines: NumPy, Numba and MooCore

VAMOS separates what an algorithm does from how its numerical steps are
computed. Non-dominated sorting, crowding distances, survival and the
real-coded operators run in a *kernel*, and three engines provide one:

- **NumPy**, the reference implementation, with vectorized array operations.
- **Numba**, which compiles selected kernels to machine code just in time (JIT).
- **MooCore**, which uses the native non-dominated sorting and hypervolume
  routines of the MooCore library where available.

`optimize` chooses the engine from its `engine` argument
([automatic engine selection](../../reference/algorithms.md#automatic-engine-selection)):

```python
# Excerpt from src/vamos/experiment/runtime/catalog.py
DEFAULT_ENGINE = "numpy"
def resolve_engine_details(engine: str | None, *, algorithm: str | None = None) -> tuple[str, str]:
    if engine is None:
        return DEFAULT_ENGINE, "default"
    engine_name = str(engine).lower()
    if engine_name == "auto":
        if algorithm and algorithm.lower() in get_builtin_algorithm_names() and _has_numba():
            return "numba", "auto"
        return DEFAULT_ENGINE, "auto"
    return engine_name, "explicit"
```

- **Omitting `engine`** means NumPy, the stable default of VAMOS 1.x.
- **`engine="auto"`** uses Numba for the built-in algorithms when Numba is
  installed, loads and has JIT enabled; otherwise it uses NumPy. It never
  chooses MooCore.
- **`"numpy"`, `"numba"` or `"moocore"`** gives exactly that engine; if it is
  not installed, the run fails with an error instead of falling back silently.

The run records the engine it used:

```python
from vamos import optimize

result = optimize("zdt1", algorithm="nsgaii", pop_size=20, max_evaluations=200, seed=1)
print("engine:", result.meta["engine"])
```

Output:

```text
engine: numpy
```

The engines change the speed, not the search. With the same seed, the three
follow the same trajectory for every built-in algorithm, and the repository's
tests compare NumPy with Numba and with MooCore; bitwise equality across
engines and environments is nevertheless outside the stability promise.
Numba and MooCore are optional: install them with the `compute` extra,
`python -m pip install "vamos-optimization[compute]==1.5.2"`.

Numba's speed has a price at the start. Its first call compiles the kernels:
the tutorial [Measure backend performance](../../tutorials/measure-performance.md)
recorded a first run of NSGA-II (ZDT1 with 10 variables, population 40,
400 evaluations) of 3.8 s with Numba against 0.014 s with NumPy, and later runs
of 0.0036 s against 0.0083 s. Numba pays off for long runs, large populations
and many runs in one process, not for the small examples of these lessons,
which therefore use NumPy.

## Key points

- A genetic algorithm repeats select, vary, evaluate and survive on a
  population, and stops when its budget is spent.
- The encoding (real, binary, integer, permutation or mixed) decides which
  operators can be used; VAMOS picks them from the problem's encoding.
- Parent selection can be random or by tournament; variation combines parents
  by crossover and perturbs children by mutation.
- (μ + λ) survival is elitist: parents and offspring compete, and the best
  solutions are never lost. (μ, λ) replaces the parents.
- Generational algorithms create N offspring per cycle, steady-state ones a
  single child, which competes at once.
- With several objectives, survival needs a rule to rank solutions: each family
  of algorithms has its own.
- Use an evaluation budget for comparisons; `max_time` runs are not
  reproducible. Omitting `engine` means NumPy; `engine="auto"` prefers Numba,
  which pays off only for long runs.

**Next:** [Families of multi-objective algorithms](algorithm-families.md).

## References

[1] Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.

[2] Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization and Machine Learning*. Addison-Wesley.

[3] Eiben, A. E., and Smith, J. E. (2015). [*Introduction to Evolutionary Computing*](https://doi.org/10.1007/978-3-662-44874-8) (2nd ed.). Springer.

[4] Deb, K. (2001). *Multi-Objective Optimization Using Evolutionary Algorithms*. Wiley.

[5] Schaffer, J. D. (1985). Multiple objective optimization with vector evaluated genetic algorithms. *Proceedings of the First International Conference on Genetic Algorithms and Their Applications*, 93–100.

[6] Goldberg, D. E., and Deb, K. (1991). [A comparative analysis of selection schemes used in genetic algorithms](https://doi.org/10.1016/B978-0-08-050684-5.50008-2). *Foundations of Genetic Algorithms*, 1, 69–93.

[7] Beyer, H.-G., and Schwefel, H.-P. (2002). [Evolution strategies – A comprehensive introduction](https://doi.org/10.1023/A:1015059928466). *Natural Computing*, 1(1), 3–52.

[8] Zitzler, E., Deb, K., and Thiele, L. (2000). [Comparison of multiobjective evolutionary algorithms: Empirical results](https://doi.org/10.1162/106365600568202). *Evolutionary Computation*, 8(2), 173–195.

[9] Syswerda, G. (1991). A study of reproduction in generational and steady-state genetic algorithms. *Foundations of Genetic Algorithms*, 1, 94–101.

[10] Durillo, J. J., Nebro, A. J., Luna, F., and Alba, E. (2009). On the effect of the steady-state selection scheme in multi-objective genetic algorithms. *Evolutionary Multi-Criterion Optimization – EMO 2009*, Lecture Notes in Computer Science 5467.

[11] Beume, N., Naujoks, B., and Emmerich, M. (2007). [SMS-EMOA: Multiobjective selection based on dominated hypervolume](https://doi.org/10.1016/j.ejor.2006.08.008). *European Journal of Operational Research*, 181(3), 1653–1669.

[12] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://doi.org/10.1109/4235.996017). *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
