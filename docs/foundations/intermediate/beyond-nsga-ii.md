---
description: 'I6. The three families of evolutionary multi-objective algorithms after NSGA-II: Pareto-based (SPEA2), decomposition-based (MOEA/D, Tchebycheff) and indicator-based (IBEA, SMS-EMOA), with interactive figures and the VAMOS algorithms of each family.'
search:
  boost: 0.5
---

# I6. Beyond NSGA-II

NSGA-II answers the question "which solutions survive?" with dominance ranks
and crowding distances. Other algorithms answer it differently, and the
answers fall into three families. Knowing them is enough to understand almost
every algorithm in VAMOS.

In this lesson you will learn:

- the selection principle of Pareto-based, decomposition-based and
  indicator-based algorithms;
- how a Tchebycheff scalarization reaches fronts that a weighted sum cannot;
- how SMS-EMOA uses hypervolume contributions;
- which VAMOS algorithms belong to each family.

## Pareto-based algorithms

Pareto-based algorithms rank solutions by dominance, then break ties with a
density estimate. NSGA-II is one; **SPEA2** [1] is the other classic:

- every solution receives a *strength*, the number of solutions it dominates;
  its raw fitness is the sum of the strengths of the solutions that dominate it
  (0 for non-dominated solutions);
- density is estimated from the distance to the k-th nearest neighbour, a finer
  measure than crowding distance;
- an external archive of fixed size keeps the best solutions; when it
  overflows, the solution with the closest neighbours is removed one at a time.

Pareto-based selection works well with two or three objectives and makes no
assumption about the shape of the front. Its weakness appears with more
objectives, when nearly every solution is non-dominated (lesson
[A1](../advanced/many-objective.md)).

## Decomposition-based algorithms

**MOEA/D** [2] decomposes the problem into N single-objective subproblems, one
per weight vector w¹, …, wᴺ, and optimizes them simultaneously. Each subproblem
keeps one solution. New solutions are created from *neighbouring* subproblems,
those with the closest weight vectors, and a child replaces a neighbour's
solution if it is better **for that neighbour's subproblem**. Neighbours share
information because their optima are close on the front.

The subproblems are scalarizations. The weighted sum fails on non-convex
fronts (lesson [B1](../basic/multi-objective-problems.md#turning-many-objectives-into-one-scalarization)).
The **Tchebycheff** scalarization [3, 4] uses the ideal point z* instead:

<div class="vf-formula">g^te(x | w, z*) = max_i  w_i · | f_i(x) − z*_i |</div>

Its contour lines are corners, not straight lines, and every Pareto-optimal
point minimizes it for some weight vector, whatever the shape of the front.
Compare the two methods on the non-convex front: sweep the weights with each
method and count the distinct solutions.

<div class="vf-widget" data-vf-widget="scalarization" data-modes="ws,tch" data-shape="concave"><p class="vf-fallback">Interactive figure: weighted-sum and Tchebycheff scalarizations on a convex and on a non-convex front. It needs JavaScript.</p></div>

Evenly spaced weights do not give evenly spaced solutions: their spacing
depends on the shape of the front, and Ishibuchi et al. [5] showed that the
performance of decomposition-based algorithms depends strongly on how well the
shape of the front matches the distribution of the weights. NSGA-III and RVEA
use the same idea of a fixed set of directions with Pareto-based or
angle-based selection; lesson [A1](../advanced/many-objective.md) explains
NSGA-III.

## Indicator-based algorithms

If the goal is a set with a high hypervolume (lesson
[B4](../basic/quality-indicators.md)), why not select for it directly?
Indicator-based algorithms do exactly that.

- **IBEA** [6] defines the fitness of a solution from a binary indicator,
  usually the additive ε, computed against every other solution, and removes
  the worst solution one at a time.
- **SMS-EMOA** [7] is steady-state: each step creates one offspring, sorts the
  N + 1 solutions into fronts, and removes the solution of the last front with
  the **smallest exclusive hypervolume contribution**, the one whose removal
  loses the least hypervolume.

Drag the points below and watch which one SMS-EMOA would discard:

<div class="vf-widget" data-vf-widget="hypervolume" data-smsemoa="true"><p class="vf-fallback">Interactive figure: the exclusive hypervolume contribution of each point and the point that SMS-EMOA would discard. It needs JavaScript.</p></div>

Indicator-based selection inherits the Pareto compliance of the hypervolume,
and it adapts the distribution of points to the front's shape. It also
inherits its cost: computing contributions becomes expensive with many
objectives [8].

## The families in VAMOS

| Family | Selection principle | VAMOS algorithms | In-depth lessons |
| --- | --- | --- | --- |
| Pareto-based | Dominance rank, then density | [NSGA-II](../../algorithms/nsgaii.md), [SPEA2](../../algorithms/spea2.md) | [I1 NSGA-II](nsga-ii.md), [I7 SPEA2](spea2.md) |
| Decomposition and reference directions | Scalarized subproblems or directions in objective space | [MOEA/D](../../algorithms/moead.md), [NSGA-III](../../algorithms/nsgaiii.md), [RVEA](../../algorithms/rvea.md) | [I8 MOEA/D](moead.md), [A2 NSGA-III](../advanced/nsga-iii.md), [A3 RVEA](../advanced/rvea.md) |
| Indicator-based | Contribution to a quality indicator | [SMS-EMOA](../../algorithms/smsemoa.md), [IBEA](../../algorithms/ibea.md) | [I9 SMS-EMOA](smsemoa.md), [I10 IBEA](ibea.md) |
| Other survival rules | Geometry of the front, particle swarms | [AGE-MOEA](../../algorithms/agemoea.md), [SMPSO](../../algorithms/smpso.md) | [A4 AGE-MOEA](../advanced/agemoea.md), [I11 SMPSO](smpso.md) |
| Constrained problems | Several populations, archives or stages | [CCMO](../../algorithms/ccmo.md), [C-TAEA](../../algorithms/ctaea.md), [ToP](../../algorithms/top.md), [CMOEA-MS](../../algorithms/cmoeams.md) (lesson [A5](../advanced/constrained.md)) | [A6 CCMO](../advanced/ccmo.md), [A7 C-TAEA](../advanced/ctaea.md), [A8 ToP](../advanced/top.md), [A9 CMOEA-MS](../advanced/cmoeams.md) |

Because every algorithm receives the same problem and budget through
`optimize`, switching family is a one-word change:

```python
from vamos import optimize

for algorithm in ("nsgaii", "moead", "smsemoa"):
    result = optimize("zdt1", algorithm=algorithm, pop_size=100, max_evaluations=10_000, engine="numpy", seed=1)
    print(f"{algorithm:<8} {result.data['evaluations']} evaluations, {result.F.shape[0]} returned solutions")
```

Output:

```text
nsgaii   10000 evaluations, 100 returned solutions
moead    10000 evaluations, 100 returned solutions
smsemoa  10000 evaluations, 100 returned solutions
```

This loop deliberately prints no quality values. One run per algorithm on one
problem with default settings says little about which algorithm is better:
lesson [A10](../advanced/performance-assessment.md) explains what a comparison
needs. Each algorithm's lesson describes its configuration; the
[capability matrix](../../reference/algorithms.md) lists the encodings,
constraint handling and engines each one supports.

## Key points

- Pareto-based algorithms select by dominance, then density (NSGA-II, SPEA2).
- Decomposition-based algorithms solve many scalarized subproblems at once
  (MOEA/D); the Tchebycheff scalarization reaches non-convex fronts.
- Indicator-based algorithms select for a quality indicator directly (IBEA,
  SMS-EMOA), at the price of computing it.
- In VAMOS all of them share one interface: changing `algorithm=` changes the
  family.

**Next:** [I7. SPEA2](spea2.md).

## References

[1] Zitzler, E., Laumanns, M., and Thiele, L. (2001). [SPEA2: Improving the strength Pareto evolutionary algorithm](https://tik-db.ee.ethz.ch/db/public/tik/?db=publications&form=report_single_publication&publication_id=1287). TIK Report 103, Computer Engineering and Networks Laboratory, ETH Zurich.

[2] Zhang, Q., and Li, H. (2007). [MOEA/D: A multiobjective evolutionary algorithm based on decomposition](https://doi.org/10.1109/TEVC.2007.892759). *IEEE Transactions on Evolutionary Computation*, 11(6), 712–731.

[3] Steuer, R. E., and Choo, E.-U. (1983). [An interactive weighted Tchebycheff procedure for multiple objective programming](https://doi.org/10.1007/BF02591870). *Mathematical Programming*, 26(3), 326–344.

[4] Miettinen, K. (1999). [*Nonlinear Multiobjective Optimization*](https://doi.org/10.1007/978-1-4615-5563-6). Kluwer Academic Publishers.

[5] Ishibuchi, H., Setoguchi, Y., Masuda, H., and Nojima, Y. (2017). [Performance of decomposition-based many-objective algorithms strongly depends on Pareto front shapes](https://doi.org/10.1109/TEVC.2016.2587749). *IEEE Transactions on Evolutionary Computation*, 21(2), 169–190.

[6] Zitzler, E., and Künzli, S. (2004). [Indicator-based selection in multiobjective search](https://doi.org/10.1007/978-3-540-30217-9_84). *Parallel Problem Solving from Nature – PPSN VIII*, Lecture Notes in Computer Science 3242, 832–842.

[7] Beume, N., Naujoks, B., and Emmerich, M. (2007). [SMS-EMOA: Multiobjective selection based on dominated hypervolume](https://doi.org/10.1016/j.ejor.2006.08.008). *European Journal of Operational Research*, 181(3), 1653–1669.

[8] Bader, J., and Zitzler, E. (2011). [HypE: An algorithm for fast hypervolume-based many-objective optimization](https://doi.org/10.1162/EVCO_a_00009). *Evolutionary Computation*, 19(1), 45–76.
