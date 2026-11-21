---
description: 'Elitist (μ + λ) survival in NSGA-II: merging parents and offspring, filling whole fronts and truncating the last one by crowding distance, step by step and in VAMOS, plus constraints and the steady-state variant.'
search:
  boost: 0.5
---

# Elitist survival and the complete generation

The last step of a generation decides which N of the 2N candidates survive.
NSGA-II makes parents and offspring compete on equal terms, which guarantees
that the best solutions found so far are never lost. This property,
**elitism**, was the third improvement over NSGA [1].

In this lesson you will learn:

- why elitism matters, and how NSGA-II implements it;
- the survival procedure, step by step;
- how VAMOS performs it, and how the whole generation fits together;
- how constraints and the steady-state variant change survival.

## Why elitism

In a non-elitist algorithm the offspring simply replace the parents. A
Pareto-optimal solution found in one generation can disappear in the next if
its offspring are worse. Zitzler, Deb and Thiele [2] compared several
algorithms on the ZDT problems and found elitism to be an important factor of
performance. NSGA-II uses the simplest elitist scheme, a (μ + λ) selection
in the terminology of evolution strategies: the N parents and the N offspring
form one pool, and the best N of the pool survive.

## The procedure

<div class="vf-formula">Rₜ ← Pₜ ∪ Qₜ                               (2N candidates)
F₁, F₂, … ← fast non-dominated sort of Rₜ       (non-dominated sorting)
Pₜ₊₁ ← ∅,   k ← 1
while |Pₜ₊₁| + |F_k| ≤ N:
    Pₜ₊₁ ← Pₜ₊₁ ∪ F_k                          (the whole front fits)
    k ← k + 1
compute the crowding distances of F_k           (crowding distance)
add the N − |Pₜ₊₁| members of F_k with the largest distances</div>

Every member of an earlier front survives before any member of a later
one: convergence first. Only the front that does not fit, the *last front*, is
cut, and crowding decides the cut: diversity second. The fronts after it are
discarded without being examined.

Step through one survival with N = 6. Circles are parents, triangles are
offspring:

<div class="vf-widget" data-vf-widget="survival"><p class="vf-fallback">Interactive figure: six parents and six offspring sorted into fronts, with the last front truncated by crowding distance. It needs JavaScript.</p></div>

In this example three parents and three offspring survive. Nothing in the
procedure prefers either group: a parent survives only if it is still among
the best N.

## In VAMOS

The same twelve candidates, selected with the experimental public helper
`vamos.select_survivors`, which applies NSGA-II's ranking and crowding:

```python
import numpy as np

from vamos import select_survivors

parents = np.array([[0.20, 0.30], [0.25, 0.70], [0.40, 0.48], [0.90, 0.12], [0.50, 0.65], [0.95, 0.35]])
offspring = np.array([[0.45, 0.10], [0.22, 0.90], [0.32, 0.55], [0.60, 0.30], [0.35, 0.90], [0.70, 0.50]])
candidates = np.vstack([parents, offspring])  # rows 0-5: parents, rows 6-11: offspring
labels = [f"P{i}" for i in range(1, 7)] + [f"O{i}" for i in range(1, 7)]

selection = select_survivors(candidates, survivor_count=6)
for label, rank, distance in zip(labels, selection.ranks, selection.crowding_distances):
    print(f"{label}: front {rank + 1}, crowding {distance:.2f}")
print("survivors:", ", ".join(labels[i] for i in selection.selected_indices))
```

Output:

```text
P1: front 1, crowding inf
P2: front 2, crowding 0.60
P3: front 2, crowding 0.73
P4: front 2, crowding inf
P5: front 3, crowding 1.31
P6: front 3, crowding inf
O1: front 1, crowding inf
O2: front 2, crowding inf
O3: front 2, crowding 0.50
O4: front 2, crowding 1.20
O5: front 3, crowding inf
O6: front 3, crowding 1.30
survivors: P1, O1, P4, O2, O4, P3
```

The helper orders candidates by rank, then by decreasing crowding distance,
then by position, and returns the first `survivor_count`. Inside NSGA-II, the
kernel fills the fronts exactly as in the procedure above:

```python
# Excerpt from src/vamos/foundation/kernel/numpy_backend.py
def _select_nsga2(fronts: list[list[int]], crowding: RealArray, pop_size: int) -> IndexArray:
    """
    NSGA-II elitist selection based on fronts + crowding.
    """
    selected: list[int] = []
    for front in fronts:
        if len(front) == 0:
            continue
        front_arr = np.asarray(front, dtype=int)
        if len(selected) + front_arr.size <= pop_size:
            selected.extend(front_arr.tolist())
        else:
            rem = pop_size - len(selected)
            order = np.argsort(crowding[front_arr])[::-1]
            selected.extend(front_arr[order[:rem]].tolist())
            break
    return np.array(selected, dtype=int)
```

and the kernel's survival method chains merge, sort, crowding and selection:

```python
# Excerpt from src/vamos/foundation/kernel/numpy_backend.py
        X_comb = np.vstack([X, X_off])
        F_comb = np.vstack([F, F_off])
        fronts, _ = _fast_non_dominated_sort(F_comb)
        crowding = _compute_crowding(F_comb, fronts)
        sel = _select_nsga2(fronts, crowding, pop_size)
```

## The complete generation

The NSGA-II driver in `src/vamos/engine/algorithm/nsgaii/ask_tell.py`
organizes each generation as *ask* and *tell*:

1. **Ask.** Compute ranks and crowding distances of the current population
   (they are cached from the previous survival when possible), pick parents
   with the binary tournament (lesson [Crowding and tournaments](crowding-distance.md)), and apply
   crossover and mutation (lesson [Real-coded operators](../operators/real-coded.md)). The result is
   the offspring decision matrix.
2. **Evaluate** the offspring, counting every evaluation against the budget.
3. **Tell.** Merge parents and offspring and run the survival above; update any
   external archive; check the termination condition.

Two consequences for experiments:

- **The returned set is not the population size.** `result.F` holds, by
  default, the non-dominated members of the final population, so it can have
  fewer than N rows ([understanding results](../../guide/understanding-results.md)).
- **The budget is exact.** The initial population and every offspring count;
  VAMOS does not overshoot `max_evaluations` silently.

## Survival with constraints

With constraints g_j(x) ≤ 0, VAMOS' default NSGA-II replaces dominance by
**constrained domination** [3]: a feasible solution beats any infeasible one, two
infeasible solutions are compared by their total violation, and two feasible
solutions by Pareto dominance. Sorting, crowding and truncation then proceed as
above. Lesson [Constraints](../advanced/constrained.md) shows its effect and the
alternatives that VAMOS provides.

## Steady-state NSGA-II

The generational scheme creates N offspring before each survival. The
**steady-state** variant creates one offspring, merges it with the N parents,
and removes the worst of the N + 1 immediately, so a good offspring can become
a parent at once. VAMOS enables it with `.steady_state()` in the builder; the
[NSGA-II lesson](../../algorithms/nsgaii.md#steady-state-nsga-ii) explains the
budget accounting and when the two variants differ.

## Key points

- NSGA-II is elitist: parents and offspring compete together and the best N of
  the 2N survive.
- Survival takes whole fronts while they fit and truncates the first front that
  does not fit by crowding distance.
- In VAMOS the kernel's `nsga2_survival` merges, sorts, computes crowding and
  selects; `vamos.select_survivors` exposes the same ranking for any set of
  objective vectors.
- Constraints change the comparison (constrained domination), not the structure
  of survival.

**Next:** [SPEA2](spea2.md).

## References

[1] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://doi.org/10.1109/4235.996017). *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

[2] Zitzler, E., Deb, K., and Thiele, L. (2000). [Comparison of multiobjective evolutionary algorithms: Empirical results](https://doi.org/10.1162/106365600568202). *Evolutionary Computation*, 8(2), 173–195.

[3] Deb, K. (2000). [An efficient constraint handling method for genetic algorithms](https://doi.org/10.1016/S0045-7825(99)00389-8). *Computer Methods in Applied Mechanics and Engineering*, 186(2–4), 311–338.
