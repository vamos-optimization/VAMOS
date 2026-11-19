---
description: 'I2. Fast non-dominated sorting step by step: domination counters and dominated sets, its O(MN²) cost, an interactive stepper, and the vectorized implementation in VAMOS.'
search:
  boost: 0.5
---

# I2. Fast non-dominated sorting

The first criterion of NSGA-II is the **rank** of each solution: the index of
the non-dominated front it belongs to. Computing ranks is the most expensive
step of the algorithm, and NSGA-II owes the "fast" in its title to the way it
does it [1].

In this lesson you will learn:

- what non-domination fronts are;
- the fast non-dominated sorting algorithm, its bookkeeping and its cost;
- how VAMOS implements it with NumPy, and the faster method it uses for two
  objectives.

## Fronts and ranks

Sort a population into layers:

- **Front 1** (F₁) holds the solutions that no other solution dominates;
- **Front 2** (F₂) holds those that no solution dominates once F₁ is removed;
- and so on, until every solution has a front.

The rank of a solution is the number of its front. Rank 1 is best. Every
member of F₂ is dominated by at least one member of F₁, and solutions in the
same front are mutually non-dominated. Goldberg [2] proposed this layering;
Srinivas and Deb's NSGA [3] used it as fitness.

**VAMOS numbers ranks from 0.** In its arrays, front 1 has rank 0.

## The naive method and its cost

The obvious approach finds F₁ by checking every solution against every other
one (O(M N²) comparisons for N solutions and M objectives), removes it, and
repeats on the rest. With up to N fronts, the worst case is O(M N³), the cost
of the original NSGA [1].

## The fast method

NSGA-II compares every pair **once** and remembers the result in two
structures per solution p [1]:

- **n_p**, the *domination counter*: how many solutions dominate p;
- **S_p**, the *dominated set*: the solutions that p dominates.

<div class="vf-formula">for each p:  S_p ← ∅,  n_p ← 0
for each pair (p, q):
    if p ≺ q:  add q to S_p
    if q ≺ p:  n_p ← n_p + 1
F₁ ← { p : n_p = 0 }
k ← 1
while F_k is not empty:
    F_{k+1} ← ∅
    for each p in F_k:
        for each q in S_p:
            n_q ← n_q − 1
            if n_q = 0:  add q to F_{k+1}
    k ← k + 1</div>

When a front is removed, only the counters of the solutions it dominates
change, and a solution joins the next front at the moment its counter reaches
zero: every solution that dominated it has already been placed. Step through
it below. The number after each letter is the current counter n_p; focus or
hover a point to draw its dominated set S_p.

<div class="vf-widget" data-vf-widget="sorting"><p class="vf-fallback">Interactive figure: twelve points sorted into fronts one bookkeeping step at a time, showing each domination counter. It needs JavaScript.</p></div>

**Cost.** The pairwise comparisons cost O(M N²). Peeling the fronts visits each
pair (p, q ∈ S_p) once, O(N²) in total. The memory for the sets S_p is O(N²).
Since NSGA-II sorts the 2N solutions of the merged population every
generation, this step dominates the cost of the algorithm [1].

## In VAMOS

VAMOS does not keep Python lists for S_p. Its NumPy kernel builds the complete
dominance matrix with broadcasting (`dom_matrix[p, q]` is true when p
dominates q) and derives every counter from it at once. Each loop iteration
then extracts a whole front:

```python
# Excerpt from src/vamos/foundation/kernel/numpy_backend.py
def _fast_non_dominated_sort_dense(F: RealArray) -> tuple[list[list[int]], IndexArray]:
    """Dense vectorized fast non-dominated sort for moderate population sizes."""
    N = F.shape[0]
    less_equal = F[:, None, :] <= F[None, :, :]
    strictly_less = F[:, None, :] < F[None, :, :]
    dom_matrix = np.logical_and(
        np.all(less_equal, axis=2),
        np.any(strictly_less, axis=2),
    )

    dominated_count = dom_matrix.sum(axis=0).astype(np.int64)
    rank = np.empty(N, dtype=int)
    fronts: list[list[int]] = []

    current = np.flatnonzero(dominated_count == 0)
    level = 0
    while current.size > 0:
        fronts.append(current.tolist())
        rank[current] = level
        dom_contrib = dom_matrix[current].sum(axis=0)
        dominated_count -= dom_contrib
        dominated_count[current] = -1
        dom_matrix[current] = False
        level += 1
        current = np.flatnonzero(dominated_count == 0)

    return fronts, rank
```

Read it against the pseudo-code:

- the row `dom_matrix[p]` is S_p, and the column sum `dominated_count[q]` is n_q;
- `dom_matrix[current].sum(axis=0)` counts, for every q, how many members of the
  current front dominate it: one subtraction updates all the counters;
- setting the counters of the extracted front to −1 removes it, so it is never
  selected again.

The matrix has N² entries per objective. For very large populations the
kernel switches to `_fast_non_dominated_sort_blocked`, which processes the
comparisons in blocks of rows to bound memory.

**Two objectives.** With M = 2, VAMOS uses a different algorithm in O(N log N):
it visits the points by increasing f₁ and finds the rank of each one with a
binary search over the last point of each front, in the spirit of the
efficient methods of Kung, Luccio and Preparata [4] and Jensen [5]. It returns
the same ranks as the dense sort:

```python
# Excerpt from src/vamos/foundation/kernel/numpy_backend.py
    order = np.lexsort((F[:, 1], F[:, 0]))
    f1 = F[order, 0].tolist()
    f2 = F[order, 1].tolist()
    last_rows: list[tuple[float, float]] = []
    sorted_ranks: list[int] = []
    for key in zip(f2, f1):
        rank = bisect_left(last_rows, key)
        if rank == len(last_rows):
            last_rows.append(key)
        else:
            last_rows[rank] = key
        sorted_ranks.append(rank)
```

You can obtain the ranks of any set of objective vectors through the
experimental public helper `vamos.select_survivors`, which applies the NSGA-II
ranking and crowding of the NumPy kernel. These are the twelve points of the
figure above:

```python
import numpy as np

from vamos import select_survivors

points = np.array(
    [
        [0.08, 0.78], [0.20, 0.50], [0.45, 0.30], [0.80, 0.10],
        [0.18, 0.92], [0.32, 0.62], [0.55, 0.45], [0.90, 0.30],
        [0.45, 0.85], [0.66, 0.62], [0.95, 0.55], [0.75, 0.90],
    ]
)
selection = select_survivors(points, survivor_count=len(points))
labels = "ABCDEFGHIJKL"
for rank in range(max(selection.ranks) + 1):
    members = [labels[i] for i, r in enumerate(selection.ranks) if r == rank]
    print(f"front {rank + 1} (rank {rank}):", ", ".join(members))
```

Output:

```text
front 1 (rank 0): A, B, C, D
front 2 (rank 1): E, F, G, H
front 3 (rank 2): I, J, K
front 4 (rank 3): L
```

`select_survivors` is exported by `vamos` but is **experimental**: it is not
covered by the [1.x stability policy](../../project/stability-and-versioning.md).

## Details that matter

- **Duplicates.** Two identical objective vectors do not dominate each other,
  so they share a front. Crowding distance then has to separate them
  (lesson [I3](crowding-distance.md)).
- **Constraints.** With constraints, VAMOS' default NSGA-II sorts with
  *constrained domination* instead of plain dominance: a feasible solution
  beats an infeasible one. Lesson [A5](../advanced/constrained.md) explains it.
- **Many objectives.** As M grows, almost every solution lands in F₁, and the
  rank stops discriminating. Lesson [A1](../advanced/many-objective.md) shows
  why and what replaces it.

## Key points

- Fronts layer a population by dominance; the rank of a solution is its front.
- Fast non-dominated sorting compares every pair once and stores a counter n_p
  and a set S_p per solution: O(M N²) time and O(N²) memory.
- VAMOS vectorizes it with a dominance matrix and uses an O(N log N) method for
  two objectives.

**Next:** [I3. Crowding distance and tournament selection](crowding-distance.md).

## References

[1] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://doi.org/10.1109/4235.996017). *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

[2] Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization and Machine Learning*. Addison-Wesley.

[3] Srinivas, N., and Deb, K. (1994). [Multiobjective optimization using nondominated sorting in genetic algorithms](https://doi.org/10.1162/evco.1994.2.3.221). *Evolutionary Computation*, 2(3), 221–248.

[4] Kung, H. T., Luccio, F., and Preparata, F. P. (1975). [On finding the maxima of a set of vectors](https://doi.org/10.1145/321906.321910). *Journal of the ACM*, 22(4), 469–476.

[5] Jensen, M. T. (2003). [Reducing the run-time complexity of multiobjective EAs: The NSGA-II and other algorithms](https://doi.org/10.1109/TEVC.2003.817234). *IEEE Transactions on Evolutionary Computation*, 7(5), 503–515.
