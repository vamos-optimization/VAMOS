---
description: 'The crowding distance of NSGA-II, its boundary rule and normalization, the crowded-comparison operator and binary tournament selection, with interactive figures and the VAMOS implementation.'
search:
  boost: 0.5
---

# Crowding distance and tournament selection

Ranks push the population towards the front, but all the members of one front
are equally good by dominance. NSGA-II needs a second criterion to prefer some
of them, and it should favour solutions in sparsely populated regions so that
the population spreads along the front. That criterion is the **crowding
distance** [1].

In this lesson you will learn:

- how the crowding distance is defined and computed;
- why boundary solutions receive an infinite distance;
- the crowded-comparison operator and the binary tournament that uses it;
- the VAMOS implementation, including its rule for objectives without spread.

## From sharing to crowding

NSGA kept diversity with *fitness sharing* [2, 3]: the fitness of a solution
was divided by the number of neighbours closer than a radius σ_share. The
result depended strongly on σ_share, and computing it costs O(N²). The crowding
distance replaces it with a measure that has **no parameter** and adapts to the
scale of each objective [1].

## Definition

The crowding distance of a solution is computed **within its front**. For
each objective, sort the front by that objective; the extreme solutions get an
infinite distance, and every other solution adds the normalized distance
between its two neighbours:

<div class="vf-formula">d_i ← 0 for every solution i of the front
for each objective m:
    sort the front by f_m
    d_first ← ∞,   d_last ← ∞
    for every interior solution i (between its neighbours i − 1 and i + 1 in that order):
        d_i ← d_i + ( f_m(i + 1) − f_m(i − 1) ) / ( f_m^max − f_m^min )</div>

With two objectives, d_i is the sum of the side lengths of the rectangle, the
*cuboid*, whose corners are the two neighbours of i, each side divided by the
range of its objective. A large distance means the neighbours are far away:
the solution sits in an empty region and should be kept.

- **Boundary solutions** get ∞ so that the extremes of each front, which
  define its spread, are always preferred.
- **Normalization** by the range of each objective makes objectives in
  different units comparable within the front.

Focus or hover a point to see its cuboid and the computation. Then reduce the
number of places left for this front and see which points NSGA-II would keep:

<div class="vf-widget" data-vf-widget="crowding"><p class="vf-fallback">Interactive figure: the crowding distance of each point of one front, its cuboid, and the points kept when the front must be truncated. It needs JavaScript.</p></div>

Notice that NSGA-II computes the distances **once** and keeps the largest ones.
It does not recompute them after each removal. When many neighbouring points
are removed together, a gap can open that the one-shot rule did not
anticipate; variants that remove one point at a time and recompute
(Kukkonen and Deb [4]) spread the result more evenly at a higher cost.

## The crowded-comparison operator

Rank and crowding combine into one ordering, the crowded-comparison operator
≺ₙ [1]:

<div class="vf-formula">i ≺ₙ j   ⇔   rank_i &lt; rank_j,   or   ( rank_i = rank_j  and  d_i > d_j )</div>

A lower rank always wins. Within the same front, the less crowded solution
wins. NSGA-II uses this operator in two places: to choose parents (below) and
to truncate the last front during survival (lesson
[Elitist survival](elitist-survival.md)).

## Binary tournament selection

To pick each parent, NSGA-II draws two distinct members of the population at
random and keeps the one that wins under ≺ₙ; if they tie, a coin decides.
Tournament selection [5] needs no scaling of fitness values, only a
comparison, and its *selection pressure* is set by the tournament size: with
two contestants, the worst solution is never selected, and the best is
selected about twice per generation on average.

## In VAMOS

The crowding distance of the NumPy kernel follows the definition line by
line:

```python
# Excerpt from src/vamos/foundation/kernel/numpy_backend.py
    for front in fronts:
        if len(front) == 0:
            continue
        front_arr = np.asarray(front, dtype=int)
        if front_arr.size <= 2:
            crowding[front_arr] = np.inf
            continue

        fvals = F[front_arr]  # shape (k, n_obj)
        n_obj = fvals.shape[1]
        d = np.zeros(front_arr.size, dtype=float)

        for m in range(n_obj):
            order = np.argsort(fvals[:, m], kind="mergesort")
            sorted_vals = fvals[order, m]

            span = sorted_vals[-1] - sorted_vals[0]
            if span <= 0.0:
                continue
            d[order[0]] = np.inf
            d[order[-1]] = np.inf

            contrib = np.zeros_like(sorted_vals)
            contrib[1:-1] = (sorted_vals[2:] - sorted_vals[:-2]) / span
            d[order[1:-1]] += contrib[1:-1]

        crowding[front_arr] = d
```

Two details go beyond the paper and follow jMetal and pymoo:

- a front of one or two solutions gives every member ∞;
- an objective whose values are all equal in a front (`span <= 0`) is skipped,
  instead of marking two arbitrary members as boundaries. The
  [NSGA-II lesson](../../algorithms/nsgaii.md#how-selection-works) records when
  VAMOS adopted this rule.

The tournament of two (`pressure == 2`) draws the second contestant so that it
always differs from the first, then compares rank, then crowding, then flips
a coin:

```python
# Excerpt from src/vamos/foundation/kernel/numpy_backend.py
            first = rng.integers(0, N, size=n_parents, dtype=int)
            second = rng.integers(0, N - 1, size=n_parents, dtype=int)
            second = second + (second >= first)

            rank_first = ranks[first]
            rank_second = ranks[second]
            crowd_first = np.nan_to_num(crowding[first], nan=-np.inf)
            crowd_second = np.nan_to_num(crowding[second], nan=-np.inf)

            pick_first = rank_first < rank_second
            pick_second = rank_second < rank_first
            unresolved = ~(pick_first | pick_second)

            crowd_first_better = crowd_first > crowd_second
            crowd_second_better = crowd_second > crowd_first
            pick_first = pick_first | (unresolved & crowd_first_better)
            pick_second = pick_second | (unresolved & crowd_second_better)
```

All the tournaments of a generation are resolved at once with array
operations; there is no Python loop over parents.

The public helper `vamos.select_survivors` returns the crowding distances of
any set of objective vectors. These are the nine points of the figure, with
five places to fill:

```python
import numpy as np

from vamos import select_survivors

f1 = np.array([0.0, 0.04, 0.09, 0.16, 0.38, 0.47, 0.55, 0.80, 1.0])
front = np.column_stack([f1, 1.0 - np.sqrt(f1)])  # nine points of the ZDT1 front
selection = select_survivors(front, survivor_count=5)

labels = "ABCDEFGHI"
for label, distance in zip(labels, selection.crowding_distances):
    print(label, "inf" if np.isinf(distance) else f"{distance:.3f}")
print("kept:", ", ".join(labels[i] for i in sorted(selection.selected_indices)))
```

Output:

```text
A inf
B 0.390
C 0.320
D 0.606
E 0.596
F 0.295
G 0.539
H 0.708
I inf
kept: A, D, E, H, I
```

## Known limitations

- **Duplicates.** Identical solutions are neighbours at distance zero; with
  many duplicates the ranking within a front becomes arbitrary. Fortin and
  Parizeau [6] discuss corrections.
- **More than two or three objectives.** The sum of per-objective gaps no
  longer reflects how crowded a region is, and nearly every solution is a
  boundary in some objective. NSGA-III replaces crowding with reference
  directions (lesson [Many objectives](../advanced/many-objective.md)).
- **Not a distance to the front.** A large crowding distance says nothing
  about convergence; rank takes care of that.

## Key points

- The crowding distance sums, over the objectives, the normalized gap between
  the two neighbours of a solution in its front; boundary solutions get ∞.
- NSGA-II compares solutions by rank first and crowding distance second, the
  crowded-comparison operator ≺ₙ.
- Parents are chosen by binary tournaments with ≺ₙ; the last front is truncated
  by keeping the largest distances, computed once.

**Next:** [Elitist survival and the complete generation](elitist-survival.md).

## References

[1] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://doi.org/10.1109/4235.996017). *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

[2] Goldberg, D. E., and Richardson, J. (1987). Genetic algorithms with sharing for multimodal function optimization. *Proceedings of the Second International Conference on Genetic Algorithms and Their Application*, 41–49.

[3] Srinivas, N., and Deb, K. (1994). [Multiobjective optimization using nondominated sorting in genetic algorithms](https://doi.org/10.1162/evco.1994.2.3.221). *Evolutionary Computation*, 2(3), 221–248.

[4] Kukkonen, S., and Deb, K. (2006). Improved pruning of non-dominated solutions based on crowding distance for bi-objective optimization problems. *Proceedings of the 2006 IEEE Congress on Evolutionary Computation*, 1179–1186.

[5] Goldberg, D. E., and Deb, K. (1991). [A comparative analysis of selection schemes used in genetic algorithms](https://doi.org/10.1016/B978-0-08-050684-5.50008-2). *Foundations of Genetic Algorithms*, 1, 69–93.

[6] Fortin, F.-A., and Parizeau, M. (2013). [Revisiting the NSGA-II crowding-distance computation](https://doi.org/10.1145/2463372.2463456). *Proceedings of the 15th Annual Conference on Genetic and Evolutionary Computation (GECCO '13)*, 623–630.
