---
description: 'NSGA-II at a glance: why it was proposed, its generational loop, a live NSGA-II running in the page, and the same configuration in VAMOS.'
search:
  boost: 0.5
---

# NSGA-II at a glance

NSGA-II, the *Non-dominated Sorting Genetic Algorithm II* of Deb, Pratap,
Agarwal and Meyarivan [1], is the most widely used evolutionary
multi-objective algorithm. It is the default reference point in almost every
comparison, and its two selection criteria, Pareto rank and crowding
distance, reappear in many later algorithms. The Intermediate level takes it
apart: this lesson gives the whole picture, and lessons I2 to I5 explain each
step in detail together with the VAMOS code that implements it.

In this lesson you will learn:

- which problems of its predecessor NSGA-II solved;
- the structure of one NSGA-II generation;
- how a population converges to the front, by watching it happen;
- how to run the same configuration in VAMOS.

## Where NSGA-II comes from

Goldberg [2] suggested ranking a population by Pareto dominance. Srinivas and
Deb [3] turned the idea into NSGA, which sorted the population into
non-dominated fronts and used *fitness sharing* to keep diversity inside each
front. NSGA worked, but had three weaknesses that the NSGA-II paper lists
explicitly [1]:

| Weakness of NSGA | Answer in NSGA-II | Lesson |
| --- | --- | --- |
| Sorting cost O(M N³) for N solutions and M objectives | **Fast non-dominated sorting**, O(M N²) | [Non-dominated sorting](non-dominated-sorting.md) |
| A sharing parameter σ_share that had to be tuned | **Crowding distance**, which has no parameter | [Crowding and tournaments](crowding-distance.md) |
| No elitism: good solutions could be lost | **Elitist survival**: parents and offspring compete together | [Elitist survival](elitist-survival.md) |

## One generation

NSGA-II keeps a population P of N solutions. Each generation creates N
offspring, merges them with the parents, and keeps the best N:

<div class="vf-figure"><svg class="vf-diagram" viewBox="0 0 640 250" role="img" aria-label="NSGA-II generation: the population is selected by tournament, varied by crossover and mutation, the offspring are evaluated and merged with the parents, the merged population is sorted into fronts with crowding distances, and the best N form the next population."><rect class="vf-box" x="10" y="20" width="140" height="62" rx="10"/><text x="80" y="46" text-anchor="middle">Population Pₜ</text><text class="vf-diagram-small" x="80" y="66" text-anchor="middle">N solutions</text><rect class="vf-box" x="170" y="20" width="140" height="62" rx="10"/><text x="240" y="46" text-anchor="middle">Mating selection</text><text class="vf-diagram-small" x="240" y="66" text-anchor="middle">binary tournament</text><rect class="vf-box" x="330" y="20" width="140" height="62" rx="10"/><text x="400" y="46" text-anchor="middle">Variation</text><text class="vf-diagram-small" x="400" y="66" text-anchor="middle">SBX + mutation</text><rect class="vf-box" x="490" y="20" width="140" height="62" rx="10"/><text x="560" y="46" text-anchor="middle">Offspring Qₜ</text><text class="vf-diagram-small" x="560" y="66" text-anchor="middle">N new, evaluated</text><rect class="vf-box" x="490" y="168" width="140" height="62" rx="10"/><text x="560" y="194" text-anchor="middle">Merge Rₜ</text><text class="vf-diagram-small" x="560" y="214" text-anchor="middle">Pₜ ∪ Qₜ, 2N solutions</text><rect class="vf-box" x="330" y="168" width="140" height="62" rx="10"/><text x="400" y="194" text-anchor="middle">Sort and crowd</text><text class="vf-diagram-small" x="400" y="214" text-anchor="middle">fronts, distances</text><rect class="vf-box" x="170" y="168" width="140" height="62" rx="10"/><text x="240" y="194" text-anchor="middle">Survival</text><text class="vf-diagram-small" x="240" y="214" text-anchor="middle">best N → Pₜ₊₁</text><path class="vf-arrow" d="M150 51H164M310 51H324M470 51H484M560 82V162M490 199H476M330 199H316M170 199H80V88"/><path class="vf-arrowhead" d="M170 51l-7-4v8zM330 51l-7-4v8zM490 51l-7-4v8zM560 168l-4-7h8zM470 199l7-4v8zM310 199l7-4v8zM80 82l-4 7h8z"/></svg></div>

In pseudo-code, following the paper [1]:

<div class="vf-formula">P₀ ← N random solutions;  evaluate them;  sort P₀ into fronts and compute crowding distances
for t = 0, 1, 2, … while the evaluation budget lasts:
    Qₜ ← N offspring of Pₜ:  binary tournament on (rank, crowding)  →  SBX crossover  →  polynomial mutation
    evaluate Qₜ
    Rₜ ← Pₜ ∪ Qₜ                                       (2N solutions)
    F₁, F₂, … ← fast non-dominated sort of Rₜ
    Pₜ₊₁ ← whole fronts F₁, F₂, … while they fit in N places;
           fill the remaining places from the next front, largest crowding distance first</div>

Two criteria drive everything: **rank** (the front a solution belongs to)
pushes the population towards the Pareto front, and **crowding distance**
spreads it along the front. Rank always comes first; crowding only breaks ties
between solutions of the same front.

## Watch it converge

The figure runs NSGA-II with VAMOS' default real-coded settings on ZDT1, ZDT2
or ZDT3 with 30 variables: binary tournament, SBX with probability 0.9 and
η_c = 20, polynomial mutation with probability 1/30 and η_m = 20. Press
**Play**, or advance one generation at a time. Blue points are the first front
of the current population; grey points are dominated.

<div class="vf-widget" data-vf-widget="nsga2-live"><p class="vf-fallback">Interactive figure: NSGA-II running on ZDT1, ZDT2 or ZDT3 in the page, with the hypervolume of its first front over the evaluations. It needs JavaScript.</p></div>

What to look for:

1. **Two phases.** For the first generations the population moves down
   towards the front (convergence); the first front holds few points. Later
   almost every solution is non-dominated and the work shifts to spreading
   them evenly (diversity).
2. **Diminishing returns.** The hypervolume curve rises steeply, then slowly.
   Each additional percent costs more evaluations than the previous one.
3. **Population size.** With 20 solutions the front is reached sooner per
   evaluation but covered sparsely; with 100 it takes longer but the coverage
   is dense. Neither is "right": it depends on how many alternatives the
   decision maker needs and on the budget.
4. **Shape.** On ZDT3 the population splits into the five disconnected pieces
   of the front, without any parameter telling it to.

This figure runs a compact JavaScript port of the algorithm, written for
teaching. VAMOS runs the same steps in Python with vectorized NumPy kernels.

## The same run in VAMOS

The configuration of the figure, written with the explicit builder, run for
an increasing number of generations. Every run uses the same seed. The *mean
gap* is the average vertical distance of the non-dominated points to the ZDT1
front f₂ = 1 − √f₁.

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import ZDT1

problem = ZDT1(n_var=30)
config = (
    NSGAIIConfig.builder()
    .pop_size(100)
    .crossover("sbx", prob=0.9, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .selection("tournament", size=2)
    .build()
)
print("generations  evaluations  non-dominated  mean gap")
for generations in (1, 10, 50, 100, 200):
    result = optimize(problem, algorithm="nsgaii", algorithm_config=config, termination=("n_gen", generations), engine="numpy", seed=1)
    front = result.front()
    gap = np.mean(front[:, 1] - (1.0 - np.sqrt(front[:, 0])))
    print(f"{generations:>11}  {result.data['evaluations']:>11}  {len(front):>13}  {gap:>8.4f}")
```

Output:

```text
generations  evaluations  non-dominated  mean gap
          1          100             11    3.1383
         10         1000             13    1.4859
         50         5000             69    0.2758
        100        10000            100    0.0253
        200        20000            100    0.0029
```

One generation means the random initial population only: the budget of a run
always includes it. `NSGAIIConfig.default(pop_size=100, n_var=30)` produces
exactly this configuration. The [NSGA-II lesson](../../algorithms/nsgaii.md)
covers the remaining options, such as the steady-state variant and external
archives, and the [configuration reference](../../reference/api/algorithms/nsgaii.md)
lists every field.

## Where each step lives in VAMOS

| Step | Lesson | VAMOS source |
| --- | --- | --- |
| Generation loop (ask, evaluate, tell) | [Elitist survival](elitist-survival.md) | `src/vamos/engine/algorithm/nsgaii/ask_tell.py` |
| Fast non-dominated sorting | [Non-dominated sorting](non-dominated-sorting.md) | `_fast_non_dominated_sort` in `src/vamos/foundation/kernel/numpy_backend.py` |
| Crowding distance | [Crowding and tournaments](crowding-distance.md) | `_compute_crowding` in the same file |
| Binary tournament | [Crowding and tournaments](crowding-distance.md) | `NumPyKernel.tournament_selection` in the same file |
| SBX and polynomial mutation | [Real-coded operators](operators/real-coded.md) | `src/vamos/foundation/kernel/operator_primitives.py` |
| Elitist survival | [Elitist survival](elitist-survival.md) | `_select_nsga2` and `NumPyKernel.nsga2_survival` in `numpy_backend.py` |

The NumPy kernel is the reference implementation. The Numba and MooCore
engines implement the same kernel interface and are tested against it; see
[algorithms and backends](../../reference/algorithms.md).

## Cost of one generation

Sorting 2N solutions with M objectives costs O(M N²) comparisons, which
dominates everything else; crowding costs O(M N log N) [1]. Jensen [4] showed
that sorting can be done in O(N log^(M−1) N). For two objectives VAMOS uses an
O(N log N) sort (lesson [Non-dominated sorting](non-dominated-sorting.md#in-vamos)). In practice,
for expensive problems the evaluations cost far more than the algorithm.

## Key points

- NSGA-II added fast sorting, a parameter-free diversity measure and elitism
  to NSGA.
- Each generation creates N offspring, merges them with the N parents, and keeps
  the best N by rank, then by crowding distance.
- Rank drives convergence; crowding distance drives spread. Rank always has
  priority.
- The budget counts every evaluation, including the initial population.

**Next:** [Fast non-dominated sorting](non-dominated-sorting.md).

## References

[1] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://doi.org/10.1109/4235.996017). *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

[2] Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization and Machine Learning*. Addison-Wesley.

[3] Srinivas, N., and Deb, K. (1994). [Multiobjective optimization using nondominated sorting in genetic algorithms](https://doi.org/10.1162/evco.1994.2.3.221). *Evolutionary Computation*, 2(3), 221–248.

[4] Jensen, M. T. (2003). [Reducing the run-time complexity of multiobjective EAs: The NSGA-II and other algorithms](https://doi.org/10.1109/TEVC.2003.817234). *IEEE Transactions on Evolutionary Computation*, 7(5), 503–515.
