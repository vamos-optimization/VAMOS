---
description: 'I7. SPEA2 in depth: strength and raw fitness, k-th nearest-neighbour density, environmental selection into a fixed-size archive, the truncation operator and mating selection, with an interactive figure and the VAMOS implementation.'
search:
  boost: 0.5
---

# I7. SPEA2: strength, density and archive truncation

NSGA-II sorts the population into fronts and breaks ties within a front by
crowding distance. SPEA2, the second version of the Strength Pareto
Evolutionary Algorithm by Zitzler, Laumanns and Thiele [1], answers the same two
questions, convergence and spread, in a different way: every solution receives
one number, its **fitness**, that counts the dominance relations it takes part
in and adds a small density term, and the best solutions are kept in an
**archive** of fixed size.

When the archive overflows with non-dominated solutions, SPEA2 does not cut
it in one shot as NSGA-II does with crowding distances (lesson
[I3](crowding-distance.md)). It removes one solution at a time, always the one
whose neighbours are closest, and measures the distances again after every
removal. It is the second classic of the Pareto-based family of lesson
[I6](beyond-nsga-ii.md#pareto-based-algorithms).

In this lesson you will learn:

- how strength and raw fitness turn dominance into a number;
- how the distance to the k-th nearest neighbour estimates density, and the k
  that VAMOS uses;
- how the archive is filled, and how the truncation operator chooses which
  solutions to remove;
- how parents are chosen, and how VAMOS implements each step.

## From SPEA to SPEA2

The first SPEA [2] already kept an external archive of non-dominated solutions.
An archive member's *strength* was the share of the population it dominated, the
fitness of a population member was computed from the strengths of the archive
members that dominated it, and clustering reduced the archive when it grew too
large. Zitzler, Laumanns and Thiele [1] identified three weaknesses:

- **Coarse fitness.** Solutions dominated by the same archive members had the
  same fitness. With a single archive member, the whole population had one
  fitness value, whether its members dominated each other or not.
- **No density information** when many solutions are mutually non-dominated, as
  the dominance relation alone cannot order them.
- **Lost extremes.** Clustering could remove the outer solutions of the front.

SPEA2 fixes them with a finer fitness, a nearest-neighbour density estimate and
a new truncation operator, and keeps the archive at a constant size.

## Strength and raw fitness

SPEA2 assigns fitness to the union of the archive Aₜ and the current population
Pₜ, M = N_A + N solutions in total. The **strength** of a solution is the number
of solutions it dominates, and its **raw fitness** sums the strengths of the
solutions that dominate it (≺ is Pareto dominance, lesson
[B2](../basic/pareto-dominance.md)):

<div class="vf-formula">S(i) = | { j ∈ Aₜ ∪ Pₜ :  i ≺ j } |
R(i) = Σ_{j ∈ Aₜ ∪ Pₜ,  j ≺ i}  S(j)</div>

Raw fitness is minimized. R(i) = 0 means that no solution dominates i. A
dominated solution is penalized more when it has many dominators and when they
are strong, that is, when they dominate many other solutions as well. Unlike
SPEA, dominators are counted in the whole union, archive and population alike,
so even a dominated solution contributes its strength.

## Density from the k-th nearest neighbour

Raw fitness cannot separate mutually non-dominated solutions, which all have
R = 0. SPEA2 adds a density term based on σᵢᵏ, the Euclidean distance in
objective space from i to its k-th nearest neighbour among all M solutions, a
classical density estimator [3]:

<div class="vf-formula">D(i) = 1 / ( σᵢᵏ + 2 )
F(i) = R(i) + D(i)</div>

The constant 2 keeps D(i) in (0, 0.5], so D(i) &lt; 1. Because R is an integer,
density only orders solutions of equal raw fitness, and every non-dominated
solution has F(i) &lt; 1. A solution with a close k-th neighbour sits in a
crowded region and has a larger D, a worse fitness.

The paper uses k = √M, the square root of the sample size [1]. VAMOS follows
jMetal instead: `k_neighbors` defaults to **1**, the nearest neighbour, and
`SPEA2Config.builder().k_neighbors(k)` sets another value. The distances use the
objective values as they are, without normalization, so an objective with a much
larger range dominates them; scale such objectives in the problem definition.

## Environmental selection

Each generation, the new archive is built from the union, in two cases:

<div class="vf-formula">Q ← Aₜ ∪ Pₜ                                (M solutions)
compute S, R, D and F for every member of Q
Aₜ₊₁ ← { i ∈ Q : R(i) = 0 }                 (the non-dominated, F(i) &lt; 1)
if |Aₜ₊₁| > N_A:   remove members with the truncation operator until |Aₜ₊₁| = N_A
if |Aₜ₊₁| &lt; N_A:   add the N_A − |Aₜ₊₁| dominated members of Q with the lowest F</div>

Early in a run, few solutions are non-dominated and the archive is completed
with the best dominated ones. Later, more than N_A solutions are usually
non-dominated, and the truncation operator decides.

## The truncation operator

Let σᵢˡ now denote the distance from i to its l-th nearest neighbour **among the
members still in the archive**. The truncation removes, one at a time, a member
i that satisfies i ≤_d j for every other member j, where

<div class="vf-formula">i ≤_d j   ⇔   σᵢˡ = σⱼˡ for every l,
             or there is a k with σᵢˡ = σⱼˡ for every l &lt; k, and σᵢᵏ &lt; σⱼᵏ</div>

In words: remove the member with the closest nearest neighbour; on a tie,
compare the second-closest neighbours, then the third, and so on. This is a
**lexicographic** comparison of the sorted neighbour distances, and it is
recomputed after every removal. Two consequences:

- **The closest pair always ties.** Its two members share the same nearest
  distance, so the second neighbour decides which of them goes: the one whose
  other neighbours are also close.
- **Boundary solutions are protected.** An extreme solution has neighbours on one
  side only, so its second-nearest distance is usually larger than its
  neighbour's, and its neighbour is removed instead. Zitzler, Laumanns and
  Thiele designed the operator for this purpose [1].

The cost is higher than one crowding computation: each removal examines the
distances among all remaining members.

## Mating selection

Only archive members become parents. SPEA2 fills the mating pool with binary
tournaments on the archive, comparing fitness F [1, 4]. The population of the
next generation consists of the N offspring created from that pool, so SPEA2 is
elitist through its archive: a solution survives only while it stays in Aₜ.

## Try it: fitness and truncation

The ten points below play the union Aₜ ∪ Pₜ. The table gives each point's
strength, raw fitness, k-th neighbour distance, density and fitness, and the
archive-size control applies environmental selection. Things to try:

1. Keep archive size 5. Seven points are non-dominated, so two must go: P5
   leaves first, then P2. P5 and P6 form the closest pair; P5 goes because its
   second neighbour, P4, is closer than P6's.
2. Set the archive size to 8 and 9. P8 and P10 are dominated with the same
   R = 2; P10 sits next to P6, so its density is higher and P8 enters first.
3. Drag a point onto another one and watch S, R and the truncation order change.
   Switch k from 1 to 3 (√10 ≈ 3, the paper's rule for ten points).

<div class="vf-widget" data-vf-widget="spea2-fitness"><p class="vf-fallback">Interactive figure: strength, raw fitness, density and fitness of ten draggable points, and the points that SPEA2's environmental selection keeps or removes, in order, for a chosen archive size. It needs JavaScript.</p></div>

## In VAMOS

The definitions above fit in a few lines of NumPy. These are the ten points of
the figure, with VAMOS' default k = 1 and an archive of five:

```python
import numpy as np

F = np.array([[0.05, 0.92], [0.12, 0.70], [0.16, 0.64], [0.30, 0.45], [0.45, 0.33],
              [0.50, 0.30], [0.85, 0.08], [0.40, 0.62], [0.70, 0.55], [0.55, 0.32]])  # P1 ... P10
dominates = np.all(F[:, None] <= F[None, :], axis=2) & np.any(F[:, None] < F[None, :], axis=2)
S = dominates.sum(axis=1)  # strength: how many points each point dominates
R = dominates.T.astype(int) @ S  # raw fitness: the strengths of its dominators
distances = np.linalg.norm(F[:, None] - F[None, :], axis=2)
k = 1  # VAMOS' default k_neighbors
sigma_k = np.sort(distances, axis=1)[:, k]  # column 0 is the distance to itself
D = 1.0 / (sigma_k + 2.0)
for i in range(len(F)):
    print(f"P{i + 1}: S={S[i]} R={R[i]} D={D[i]:.4f} F={R[i] + D[i]:.4f}")

archive = [int(i) for i in np.flatnonzero(R == 0)]  # the non-dominated points
while len(archive) > 5:  # archive size 5: truncate
    sub = distances[np.ix_(archive, archive)]
    np.fill_diagonal(sub, np.inf)
    neighbours = np.sort(sub, axis=1)  # each row: distances to the others, nearest first
    worst = min(range(len(archive)), key=lambda i: tuple(neighbours[i]))
    print(f"truncation removes P{archive.pop(worst) + 1}")
print("archive:", ", ".join(f"P{i + 1}" for i in archive))
```

Output:

```text
P1: S=0 R=0 D=0.4483 F=0.4483
P2: S=0 R=0 D=0.4826 F=0.4826
P3: S=0 R=0 D=0.4826 F=0.4826
P4: S=2 R=0 D=0.4562 F=0.4562
P5: S=1 R=0 D=0.4858 F=0.4858
P6: S=2 R=0 D=0.4869 F=0.4869
P7: S=0 R=0 D=0.4194 F=0.4194
P8: S=0 R=2 D=0.4551 F=2.4551
P9: S=0 R=6 D=0.4396 F=6.4396
P10: S=1 R=2 D=0.4869 F=2.4869
truncation removes P5
truncation removes P2
archive: P1, P3, P4, P6, P7
```

P9 is dominated by P4, P5, P6 and P10, so R(P9) = 2 + 1 + 2 + 1 = 6. P6's
nearest neighbour is the dominated P10: the density uses all M solutions,
dominated ones included, while the truncation only measures distances among
the non-dominated.

VAMOS computes strength and raw fitness from a boolean dominance matrix
(`dom[i, j]` is true when i dominates j), and the k-th neighbour distance with
a partial sort:

```python
# Excerpt from src/vamos/engine/algorithm/spea2/helpers.py
def strength_raw_fitness(dom: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Compute SPEA2 raw fitness (strength ranking) from dominance matrix."""
    n = dom.shape[0]
    if n == 0:
        return np.empty(0, dtype=float)
    strength = dom.sum(axis=1, dtype=np.int64)
    raw = (dom.T @ strength).astype(float, copy=False)
    return np.asarray(raw, dtype=float)

def _kth_neighbour_distance(dist: np.ndarray[Any, Any], k: int) -> np.ndarray[Any, Any]:
    n = dist.shape[0]
    if n == 0:
        return np.empty(0, dtype=float)
    if n == 1:
        return np.full(1, np.inf, dtype=float)
    k = min(max(int(k), 1), n - 1)
    # Each row's zero self-distance sorts first, so position k is the k-th neighbour.
    return np.asarray(np.partition(dist, kth=k, axis=1)[:, k], dtype=float)
```

`dom.sum(axis=1)` counts the solutions each one dominates, S; `dom.T @ strength`
adds, for each solution, the strengths of the rows that dominate it, R. With
constraints (the default `constraint_mode="feasibility"`), `dom` holds
constrained domination instead (lesson [A5](../advanced/constrained.md)), so
feasible solutions dominate infeasible ones.

Environmental selection follows the two cases of the procedure:

```python
# Excerpt from src/vamos/engine/algorithm/spea2/helpers.py
    n = F.shape[0]
    dist = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
    kth_distance = _kth_neighbour_distance(dist, int(k_neighbors) if k_neighbors is not None else 1)
    dom, _, _ = dominance_matrix(F, G, constraint_mode)
    raw_fitness = strength_raw_fitness(dom)

    nondominated = np.flatnonzero(raw_fitness == 0.0)
    if n <= archive_size:
        selected: NDArray[np.int_] = np.arange(n, dtype=int)
    elif nondominated.size > archive_size:
        keep = truncate_lexicographic(dist[np.ix_(nondominated, nondominated)], archive_size, selection_kernels)
        selected = nondominated[keep]
    else:
        dominated = np.flatnonzero(raw_fitness > 0.0)
        fitness = raw_fitness[dominated] + 1.0 / (kth_distance[dominated] + 2.0)
        fill = dominated[np.argsort(fitness, kind="stable")[: archive_size - nondominated.size]]
        selected = np.concatenate([nondominated, fill])

    return selected, raw_fitness[selected], kth_distance[selected]
```

- `n <= archive_size` happens only at the start, when the initial population
  fits in the archive.
- The fill sorts the dominated solutions by F = R + 1 / (σᵏ + 2) with a stable
  sort, so equal fitness keeps the input order.
- The function also returns R and σᵏ of each survivor, measured in the union;
  the mating tournament reuses them.

The truncation operator removes one row per iteration:

```python
# Excerpt from src/vamos/engine/algorithm/spea2/helpers.py
    active: NDArray[np.int_] = np.arange(n, dtype=int)
    while active.size > keep:
        sub = dist[np.ix_(active, active)]
        np.fill_diagonal(sub, np.inf)
        nearest = sub.min(axis=1)
        candidates = np.flatnonzero(nearest == nearest.min())
        if candidates.size > 1:
            # The self-distance sorts last in every row, so it never decides the order.
            rows = np.sort(sub[candidates], axis=1)
            candidates = candidates[np.lexsort(rows.T[::-1])]
        active = np.delete(active, int(candidates[0]))

    return active
```

The candidates are the members with the smallest nearest distance. When there
are several, their rows of sorted distances are ordered lexicographically with
`np.lexsort`, which sorts by the last key given, hence the reversed `rows.T`.
`np.lexsort` is stable, so identical rows remove the first member. With
`engine="numba"`, a compiled loop gives the same result for finite distances.

Mating uses a binary tournament between two distinct archive members. It
compares raw fitness first and, on a tie, the k-th neighbour distance, where
the larger distance wins:

```python
# Excerpt from src/vamos/engine/algorithm/spea2/spea2.py
    fa = raw_fitness[first]
    fb = raw_fitness[second]
    da = density[first]
    db = density[second]

    winners = np.empty(n_parents, dtype=int)
    better_first = fa < fb
    better_second = fb < fa
    winners[better_first] = first[better_first]
    winners[better_second] = second[better_second]

    tied = ~(better_first | better_second)
    if tied.any():
        tied_first = first[tied]
        tied_second = second[tied]
        tied_density_first = da[tied]
        tied_density_second = db[tied]

        density_first = tied_density_first > tied_density_second
        density_second = tied_density_second > tied_density_first
```

Since D = 1 / (σᵏ + 2) is below 1 and decreases with σᵏ, comparing R and then
σᵏ is the same as comparing F. A remaining tie is decided by a coin. The
contestants are drawn as in NSGA-II's tournament (lesson
[I3](crowding-distance.md#binary-tournament-selection)). Each generation forms
N pairs of winners, and each pair produces one offspring: VAMOS keeps the first
child of the crossover and mutates it.

A complete run on ZDT1, with an archive smaller than the population and the
default k written out:

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import SPEA2Config
from vamos.problems import ZDT1

problem = ZDT1(n_var=30)
config = (
    SPEA2Config.builder()
    .pop_size(100)
    .archive_size(60)
    .k_neighbors(1)
    .crossover("sbx", prob=0.9, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .selection("tournament")
    .build()
)
result = optimize(problem, algorithm="spea2", algorithm_config=config, max_evaluations=20_000, engine="numpy", seed=1)

gap = np.abs(result.F[:, 1] - (1.0 - np.sqrt(result.F[:, 0])))  # ZDT1 front: f2 = 1 - sqrt(f1)
print("evaluations:", result.data["evaluations"])
print("returned:", result.F.shape)
print("largest vertical gap to the front:", round(float(gap.max()), 3))
```

Output:

```text
evaluations: 20000
returned: (60, 2)
largest vertical gap to the front: 0.013
```

Each generation creates 100 offspring and keeps 60 solutions; the result holds
the non-dominated members of the final archive, here all 60. With the paper's
rule, k would be √160 ≈ 13. The [SPEA2 lesson](../../algorithms/spea2.md)
records a run with its figure, and the
[configuration reference](../../reference/api/algorithms/spea2.md) lists every
builder method.

## SPEA2, NSGA-II and when to use it

- **Same family, different estimators.** Both are elitist and Pareto-based.
  NSGA-II ranks fronts and uses one crowding computation per front; SPEA2 uses a
  scalar fitness, a k-nearest-neighbour density and an iterative truncation that
  spreads the archive more evenly, at a higher cost: M² distances per
  generation, and a truncation that re-examines them after every removal.
- **Archive and population are decoupled.** `archive_size` sets how many
  solutions survive and are returned; `pop_size` sets how many offspring each
  generation evaluates.
- **Many objectives.** When nearly every solution is non-dominated, R = 0 for
  all of them and selection is decided by density alone, which rewards spread,
  not convergence. Ishibuchi, Tsukamoto and Nojima [5] showed this loss of
  selection pressure for NSGA-II and SPEA2 (lesson
  [A1](../advanced/many-objective.md)).
- **When to use it.** Two or three objectives, when an evenly spread set of a
  given size matters, and the population is small enough for M² distances.

## Key points

- Strength S(i) counts the solutions that i dominates; raw fitness R(i) sums the
  strengths of the solutions that dominate i, and is 0 for non-dominated ones.
- Density D(i) = 1 / (σᵢᵏ + 2) is below 1, so F = R + D orders by dominance
  first and density second. VAMOS uses k = 1 by default; the paper uses √M.
- The archive keeps every non-dominated solution if they fit, completes it with
  the best dominated ones if they do not fill it, and otherwise truncates.
- The truncation removes, one at a time, the member whose sorted neighbour
  distances are lexicographically smallest, which tends to keep the extremes.
- Parents come from the archive through binary tournaments on fitness.

**Next:** [I8. MOEA/D](moead.md).

## References

[1] Zitzler, E., Laumanns, M., and Thiele, L. (2001). [SPEA2: Improving the strength Pareto evolutionary algorithm](https://tik-db.ee.ethz.ch/db/public/tik/?db=publications&form=report_single_publication&publication_id=1287). TIK Report 103, Computer Engineering and Networks Laboratory, ETH Zurich.

[2] Zitzler, E., and Thiele, L. (1999). [Multiobjective evolutionary algorithms: A comparative case study and the strength Pareto approach](https://doi.org/10.1109/4235.797969). *IEEE Transactions on Evolutionary Computation*, 3(4), 257–271.

[3] Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman and Hall.

[4] Goldberg, D. E., and Deb, K. (1991). [A comparative analysis of selection schemes used in genetic algorithms](https://doi.org/10.1016/B978-0-08-050684-5.50008-2). *Foundations of Genetic Algorithms*, 1, 69–93.

[5] Ishibuchi, H., Tsukamoto, N., and Nojima, Y. (2008). [Evolutionary many-objective optimization: A short review](https://doi.org/10.1109/CEC.2008.4631121). *Proceedings of the 2008 IEEE Congress on Evolutionary Computation*, 2419–2426.
