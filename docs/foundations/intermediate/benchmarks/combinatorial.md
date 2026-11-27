---
description: 'Binary, integer, permutation and mixed benchmark problems in VAMOS: the deceptive ZDT5, the synthetic knapsack, feature-selection, QUBO, allocation, job and mixed-design instances, the TSP instances, what their objectives mean, and how to build a reference front by enumeration.'
search:
  boost: 0.5
---

# Binary, integer, permutation and mixed problems

The previous lessons used real-valued variables. Many problems have discrete
ones: a feature is selected or not, a number of units is an integer, a tour is
an order of cities, a design mixes all three. Their operators must produce
valid values of each type, and their benchmarks are much less developed than
the continuous suites. VAMOS has one classic discrete benchmark, ZDT5, a set of
small synthetic instances, and five TSPLIB instances. This lesson says what they
test, what their objectives mean, and what they cannot tell you.

In this lesson you will learn:

- which operators VAMOS uses for each encoding;
- why ZDT5 is deceptive, and what NSGA-II finds on it;
- what the synthetic binary, integer and mixed instances compute;
- how to build an exact reference front of a small instance by enumeration;
- what the TSP instances measure, and why the six-city one has a single
  optimal point.

## Encodings and operators

A problem declares its encoding: `binary`, `integer`, `permutation` or `mixed`
(real-valued problems use `real`). `optimize(problem, algorithm=...)` passes it
to the algorithm's default configuration, which picks the operators of a shared
table ([algorithm reference](../../../reference/algorithms.md)): uniform crossover
and bit-flip mutation for binary variables, order crossover (OX) and swap
mutation for permutations, SBX with probability 0.9 and polynomial mutation for
integers, and dedicated mixed operators for mixed
variables. The [decision representations tutorial](../../../tutorials/encodings.md)
runs one problem of each encoding. The catalog's
[discrete examples table](../../../reference/problems.md#binary-integer-mixed-and-permutation-examples)
gives the keys and dimensions used below.

## ZDT5: deception in bits

ZDT5 [1] is the binary member of ZDT. Its 80 bits form a first block of 30 and
ten blocks of 5. With u₁ the number of ones in the first block and u_i that of
block i, f₁ = 1 + u₁ and f₂ = g / f₁, where g sums a value v(u_i) over the ten
small blocks: v(u) = 2 + u for u < 5, and v(5) = 1.

A block with no ones scores 2, and each added one makes it worse, until the
fifth makes it best. Filling the first small block one bit at a time, with
every other bit at 0:

```python
import numpy as np

from vamos import make_problem_selection

problem = make_problem_selection("zdt5").instantiate()
X = np.zeros((6, problem.n_var))  # the first block is empty, so f1 = 1 and f2 = g
for ones in range(6):
    X[ones, 30 : 30 + ones] = 1  # 0 to 5 ones in the first small block
out = {}
problem.evaluate(X, out)
print("g for 0 to 5 ones:", out["F"][:, 1].round().astype(int).tolist())
```

Output:

```text
g for 0 to 5 ones: [20, 21, 22, 23, 24, 19]
```

Every small step leads towards empty blocks and g = 20, while the optimum,
g = 10, needs every block full. That is deception: the local information
points away from the global optimum [1]. The Pareto front is f₂ = 10 / f₁ for
f₁ = 1, …, 31, and VAMOS packages its 31 points:

```python
from importlib.resources import files

import numpy as np

from vamos import optimize

fronts = {path.name.lower(): path for path in (files("vamos.resources") / "reference_fronts").iterdir()}
R = np.loadtxt(fronts["zdt5.csv"].read_text().splitlines(), delimiter=",")
print(f"packaged front: {len(R)} points, g = f1 f2 from {(R[:, 0] * R[:, 1]).min():.0f} to {(R[:, 0] * R[:, 1]).max():.0f}")

result = optimize("zdt5", algorithm="nsgaii", pop_size=100, max_evaluations=20_000, seed=1)
F = np.unique(result.front(), axis=0)
g = F[:, 0] * F[:, 1]
on_front = sum(np.any(np.all(np.isclose(R, f), axis=1)) for f in F)
print(f"NSGA-II: {len(F)} distinct points, g from {g.min():.0f} to {g.max():.0f}, {on_front} on the Pareto front")
```

Output:

```text
packaged front: 31 points, g = f1 f2 from 10 to 10
NSGA-II: 27 distinct points, g from 19 to 20, 0 on the Pareto front
```

After 20,000 evaluations with uniform crossover and bit-flip mutation, NSGA-II
sits on the deceptive front: g = 20 means that every small block is empty, and
g = 19 that one of them is full. None of its points is Pareto-optimal. ZDT5 needs a fixed number of
variables, 35 or more with n − 30 divisible by 5, and its packaged front holds
only for the default 80 bits.

## The synthetic instances

The six synthetic problems of VAMOS draw their data from a fixed seed, so every
run sees the same instance. Each is a trade-off between two simple terms; the
catalog's [objective table](../../../reference/problems.md#what-the-objectives-actually-mean)
states them:

| Key | Encoding | Objectives (both minimized) |
| --- | --- | --- |
| `bin_feat` | Binary, 50 | Minus the summed utility of the selected features; their summed cost |
| `bin_knapsack` | Binary, 50 | The distance of the selected weight from a target capacity; minus the selected value |
| `bin_qubo` | Binary, 30 | A quadratic energy xᵀQx + bᵀx; minus the number of selected bits |
| `int_alloc` | Integer, 20 in 0..10 | The allocation cost; minus a sum of square-root returns |
| `int_jobs` | Integer, 30 in 0..4 | Weighted mismatches with preferred job labels; the largest share of one label |
| `mixed_design` | 3 real, 3 integer, 3 categorical | Squared errors to targets plus category mismatches; integer and category costs |

They are illustrations, not models: `bin_feat` trains no classifier,
`bin_knapsack` allows overweight selections and treats the capacity as an
objective, `int_alloc` has no budget and `int_jobs` no machines. Evaluating the
extreme designs shows what each objective rewards:

```python
import numpy as np

from vamos import make_problem_selection

for key in ["bin_feat", "bin_knapsack", "bin_qubo", "int_alloc", "int_jobs"]:
    problem = make_problem_selection(key).instantiate()
    X = np.vstack([np.broadcast_to(problem.xl, (problem.n_var,)), np.broadcast_to(problem.xu, (problem.n_var,))])
    out = {}
    problem.evaluate(X, out)
    low, high = out["F"] + 0.0  # + 0.0 turns -0.0 into 0.0
    print(f"{key:12s} {problem.encoding:8s} all lower bounds: ({low[0]:8.2f}, {low[1]:8.2f})   all upper bounds: ({high[0]:8.2f}, {high[1]:8.2f})")
```

Output:

```text
bin_feat     binary   all lower bounds: (    0.00,     0.00)   all upper bounds: (  -54.01,    36.58)
bin_knapsack binary   all lower bounds: (  120.52,     0.00)   all upper bounds: (  180.78,  -150.29)
bin_qubo     binary   all lower bounds: (    0.00,     0.00)   all upper bounds: (  -38.10,   -30.00)
int_alloc    integer  all lower bounds: (    0.00,     0.00)   all upper bounds: (  295.12,  -122.98)
int_jobs     integer  all lower bounds: (   42.76,     1.00)   all upper bounds: (   34.03,     1.00)
```

Selecting nothing costs nothing and gains nothing; selecting everything gains
the most at the highest cost. The fronts lie between these extremes, and VAMOS
packages none of them. Changing `n_var` draws a different instance from the
same seed: `bin_knapsack` with 16 items is a new problem, not a smaller view of
the 50-item one.

## A reference front by enumeration

Without a packaged front, indicators need a reference you build yourself. For a
small enough instance, enumeration gives the exact front: 16 binary variables
have 2¹⁶ = 65,536 designs, which VAMOS evaluates in one call.

```python
from itertools import product

import numpy as np

from vamos import make_problem_selection, optimize

problem = make_problem_selection("bin_knapsack", n_var=16).instantiate()
X = np.array(list(product((0, 1), repeat=problem.n_var)))
out = {}
problem.evaluate(X, out)
F = out["F"]
front, best = [], np.inf
for i in np.lexsort((F[:, 1], F[:, 0])):  # by f1, then f2: keep each strict improvement in f2
    if F[i, 1] < best:
        front.append(F[i])
        best = F[i, 1]
front = np.array(front)

result = optimize("bin_knapsack", algorithm="nsgaii", n_var=16, pop_size=40, max_evaluations=2_000, seed=1)
found = np.unique(result.front(), axis=0)
on_front = sum(np.any(np.all(np.isclose(front, f), axis=1)) for f in found)
print(f"{len(X)} designs, {len(front)} on the exact front")
print(f"NSGA-II with 2,000 evaluations: {len(found)} distinct points, {on_front} of them on the exact front")
```

Output:

```text
65536 designs, 60 on the exact front
NSGA-II with 2,000 evaluations: 35 distinct points, 30 of them on the exact front
```

The exact front makes every indicator exact for this instance, and the share of
front points found is a meaningful measure in its own right. The approach stops
at a few dozen variables; beyond that, a reference front is the non-dominated
union of long runs of several algorithms, as for the real-world suites
(lesson [Real-world problems](real-world.md#where-the-reference-fronts-come-from)).

## The TSP instances

All the TSP keys have the same two objectives for a closed tour: its total
length and its longest edge, with unrounded Euclidean distances. `tsp6` places
six cities on a circle, and `kroa100` to `kroe100` use the coordinates of the
100-city instances of TSPLIB [2]. A permutation of six cities can be
enumerated:

```python
from itertools import permutations

import numpy as np

from vamos import make_problem_selection

problem = make_problem_selection("tsp6").instantiate()
routes = np.array(list(permutations(range(problem.n_var))))
out = {}
problem.evaluate(routes, out)
F = np.round(out["F"], 9)
ideal = F.min(axis=0)
print(f"{len(routes)} routes, {len(np.unique(F, axis=0))} distinct objective vectors")
print(f"ideal point ({ideal[0]:.3f}, {ideal[1]:.3f}), reached by {np.all(F == ideal, axis=1).sum()} routes")
```

Output:

```text
720 routes, 10 distinct objective vectors
ideal point (6.000, 1.000), reached by 12 routes
```

The hexagon tour is the shortest and has the shortest longest edge, so it
dominates every other tour: on `tsp6` the objectives do not conflict, and the
front is a single point. That makes it a test of the permutation operators, not
of trade-offs. On `kroa100` the objectives do conflict. A greedy tour from a
simple rule is a useful yardstick for a run:

```python
import numpy as np

from vamos import make_problem_selection, optimize

problem = make_problem_selection("kroa100").instantiate()
xy = problem.coordinates
tour, unvisited = [0], set(range(1, problem.n_var))
while unvisited:  # nearest neighbour: always go to the closest unvisited city
    here = xy[tour[-1]]
    tour.append(min(unvisited, key=lambda c: np.hypot(*(xy[c] - here))))
    unvisited.remove(tour[-1])
out = {}
problem.evaluate(np.array([tour]), out)
print(f"nearest-neighbour tour: length {out['F'][0, 0]:.0f}, longest edge {out['F'][0, 1]:.0f}")

result = optimize("kroa100", algorithm="nsgaii", pop_size=100, max_evaluations=20_000, seed=1)
F = result.front()
print(f"NSGA-II, 20,000 evaluations: {len(F)} tours, lengths {F[:, 0].min():.0f} to {F[:, 0].max():.0f}, longest edges {F[:, 1].min():.0f} to {F[:, 1].max():.0f}")
```

Output:

```text
nearest-neighbour tour: length 26856, longest edge 2524
NSGA-II, 20,000 evaluations: 5 tours, lengths 60746 to 68706, longest edges 1552 to 2032
```

The greedy tour is less than half as long as any tour NSGA-II found with OX,
swap mutation and 20,000 evaluations, although NSGA-II finds tours with a
shorter longest edge. Generic permutation operators need far larger budgets,
or problem-specific operators and local search, on a 100-city tour. Published
TSPLIB tour lengths use integer-rounded distances and a single objective, so
they do not compare directly with these values.

## Using them in a study

- **Most of these instances are synthetic.** They check that operators produce
  valid values and that an algorithm handles an encoding; they are not
  evidence that it solves real discrete problems.
- **Only ZDT5 has a packaged reference front**, and only for 80 bits. Build one
  by enumeration for a small instance, or as the union of long runs, and say
  how.
- **Record `n_var`**: it changes the instance, not just its size.
- **For realistic mixed variables**, the real-world suites have them: RE22,
  RE23, RE25 and RE35 are mixed and RE36 integer (lesson
  [Real-world problems](real-world.md)), and `welded_beam` mixes real, integer
  and categorical variables.

## Key points

- VAMOS picks operators by encoding: uniform crossover and bit flips, OX and
  swaps, integer SBX and polynomial mutation, and mixed operators.
- ZDT5 is deceptive: every step towards fewer ones in a block looks better,
  although only a full block is optimal; NSGA-II ends on the deceptive front.
- The synthetic binary, integer and mixed instances are fixed by seeds,
  illustrate simple trade-offs, and have no packaged fronts.
- Small instances can be enumerated, which gives an exact reference front.
- `tsp6`'s objectives do not conflict; on `kroa100` a greedy tour beats a short
  generic NSGA-II run in length, a reminder to compare with simple baselines.

**Next section:** [Many-objective optimization](../../advanced/many-objective.md).

## References

[1] Zitzler, E., Deb, K., and Thiele, L. (2000). [Comparison of multiobjective evolutionary algorithms: Empirical results](https://doi.org/10.1162/106365600568202). *Evolutionary Computation*, 8(2), 173–195.

[2] Reinelt, G. (1991). [TSPLIB—A traveling salesman problem library](https://doi.org/10.1287/ijoc.3.4.376). *ORSA Journal on Computing*, 3(4), 376–384.
