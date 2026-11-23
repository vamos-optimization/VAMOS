---
description: 'AGE-MOEA: normalizing the first front, estimating its curvature p from a central point, proximity and diversity in the Lp norm, the greedy survival score and the truncation of the last front, with an interactive figure and VAMOS code.'
search:
  boost: 0.5
---

# AGE-MOEA: adapting to the geometry of the front

NSGA-III and RVEA (lessons [NSGA-III](nsga-iii.md) and [RVEA](rvea.md)) spread their
solutions along directions fixed before the run. That works when the front
covers the simplex the directions were made for, and less well when it does
not [6]. AGE-MOEA, the adaptive geometry estimation MOEA of Panichella [1],
takes the opposite route: it has no reference directions. At every generation
it estimates the shape of the current front and measures distances in the
geometry that shape implies.

The rest is NSGA-II [2]: merge parents and offspring, sort them into fronts,
keep whole fronts while they fit (lesson
[Elitist survival](../intermediate/elitist-survival.md)). Only the measure that cuts the
last front changes. The crowding distance is replaced by a **survival score**
that combines the proximity of a solution to the ideal point with its
distance to the solutions already kept, both measured in the estimated
geometry.

In this lesson you will learn:

- how AGE-MOEA normalizes the first front with its corner points;
- how a single point estimates the curvature p of a front
  f₁ᵖ + … + f_mᵖ = 1;
- how proximity and diversity are measured in the Lp norm and combined into
  the survival score;
- how the last front is cut and parents are chosen, and where VAMOS follows
  pymoo or the author's code.

## Why the geometry matters

Many fronts are well described, after normalization, by one family of
surfaces:

<div class="vf-formula">f̃₁ᵖ + f̃₂ᵖ + … + f̃_mᵖ = 1                           (the front model)
‖y‖_p = ( |y₁|ᵖ + |y₂|ᵖ + … + |y_m|ᵖ )^(1/p)         (Lp norm; Minkowski distance ‖y − y′‖_p)</div>

p = 1 gives a flat front (DTLZ1 [4]), p = 2 a sphere (DTLZ2), p below 1 a
convex front bulging towards the ideal point (ZDT1 [3] is close to p = 0.72)
and p above 2 a front that bulges even further away from it than the sphere.

The model has a useful property: every point of the front lies at distance 1
from the ideal point **in the Lp norm with the right p**. Measured with the
Euclidean norm instead, the points of a convex front near its knee look closer
to the ideal point, that is better converged, than the points near its
corners, although all of them are Pareto-optimal. Distances between
neighbours along the front are distorted in the same way. Crowding distance
and Euclidean distances implicitly assume one geometry; AGE-MOEA estimates p
and uses it [1].

## Normalizing the first front

The geometry is estimated on the first front F₁ of the merged population,
after three steps:

1. **Translate** by the ideal point of F₁, the smallest value of each objective
   in F₁, so that it becomes the origin (VAMOS rounds the translated values to
   12 decimals to settle near-ties).
2. **Corner points.** For each axis i, take the point of F₁ closest to the line
   through the origin and the unit vector eᵢ (VAMOS adds 10⁻⁶ to every
   component of eᵢ), excluding points already chosen.
3. **Intercepts.** Solve for the hyperplane through the m corner points; its
   intercepts a₁, …, a_m divide each objective: f̃ᵢ = f′ᵢ / aᵢ. When the corners
   are not distinct, the system is singular, or an intercept is not finite or
   not positive, VAMOS divides by the largest value of each objective in F₁
   instead.

This is the hyperplane of NSGA-III [5], with two differences: the corners are
the points nearest to the axes rather than the minimizers of an achievement
scalarizing function, and nothing is remembered between generations. If F₁
has fewer points than objectives, no hyperplane exists: VAMOS sets p = 1 and
gives every point of F₁ the score 0.

## Estimating the curvature

AGE-MOEA reads p from one point. If a point of the front had all its
coordinates equal to c̄, the model would give m · c̄ᵖ = 1, so

<div class="vf-formula">central point c = the non-corner point of F₁ closest to the diagonal through (1, 1, …, 1)
c̄ = mean of the coordinates of c
p̂ = ln m / ln( 1 / c̄ )</div>

On DTLZ2, c̄ = 1/√2 with two objectives and p̂ = ln 2 / ln √2 = 2. On a flat
front c̄ = 1/m and p̂ = 1. VAMOS guards the estimate: if c̄ ≤ 0, or p̂ is
undefined or at most 0.1, it uses p̂ = 1, and it caps p̂ at 20. One point makes
the estimate cheap and noisy, but it is repeated every generation, and the
first front changes slowly.

## Proximity, diversity and the survival score

With p̂ fixed, every point s of F₁ gets a proximity, and every pair of points a
distance, both in the Lp norm:

<div class="vf-formula">P(s) = ‖ f̃(s) ‖_p̂                                   (proximity: distance to the ideal point)
d̂(s, q) = ‖ f̃(s) − f̃(q) ‖_p̂ / P(s)                   (distance from s to q, relative to P(s))
score(s) = ∞ for the m corner points, which form the selected set S
repeat until every point of F₁ is in S:
    for each s outside S:  D(s) = d̂(s, q₁) + d̂(s, q₂),   q₁, q₂ the two points of S nearest to s
    move the s with the largest D(s) into S;  score(s) ← D(s)</div>

The score is diversity divided by proximity: a point scores high when it is far
from the points already kept and close to the ideal point. The selection is
greedy, so each score depends on the points kept before it, and the scores
never increase along the order of selection. With only one point in S, D(s) is the
distance to that point. VAMOS, like pymoo and the author's code, uses the two
nearest selected points.

## Survival and mating

AGE-MOEA keeps whole fronts while they fit and cuts the last one by
decreasing score:

- If the last front is F₁, its points are kept in the order of the greedy
  selection: the corners first, then the most diverse and best converged
  points.
- If it is a later front, VAMOS follows the author's code in PlatEMO: each of
  its points scores 1/P(s), the inverse of its Lp distance to the ideal point,
  measured with F₁'s p̂ in F₁'s translated and normalized coordinates, so the
  points closest to the ideal point survive. pymoo cuts a later front by index
  instead. VAMOS translates the later fronts by F₁'s ideal point before
  dividing, which PlatEMO and pymoo do not, so the survivors do not depend on
  where the objectives' origin lies.

Parents are chosen by binary tournament on rank, then survival score, as in
the author's code: the lower rank wins, and on equal rank the higher score.
With constraints VAMOS puts feasible solutions first: when more are feasible
than survive, the survival runs on the feasible ones; otherwise every feasible
solution survives and the least violated infeasible ones fill the population.
The tournament compares infeasible solutions by violation, and
`constraint_handling` can replace the rule (lesson [Constraints](constrained.md)).

### Try it: one generation step by step

Step through one generation on six solutions of a problem with two variables
whose Pareto front is ZDT1's, f₂ = 1 − √f₁, the problem of the
[NSGA-II lesson](../intermediate/nsga-ii.md)'s figure. Parents are circles,
children triangles, and darker points lie on better fronts. In steps 6 to 8
the left panel shows the first front in normalized coordinates instead of
the decision space. The operators are AGE-MOEA's defaults in VAMOS.

<div class="vf-widget" data-vf-widget="agemoea-steps"><p class="vf-fallback">Interactive figure: one AGE-MOEA generation in ten steps on six solutions: the population with its fronts and survival scores, the binary tournaments, SBX crossover and polynomial mutation, the merge of parents and children, non-dominated sorting, the normalization of the first front by its ideal point and corners, the estimate of p from the central point, the survival scores, the whole fronts and the cut of the last front, and the survivors. It needs JavaScript.</p></div>

Things to try:

1. Step 2: in the tournament 5 vs 4, both are on front 1, and 4 wins because
   it is a corner, with score ∞ against 1.33.
2. Steps 6 and 7: front 1 has nine members. Its ideal point is
   z = (0.03, 0.09), its corners are 8 and 4, and the intercepts are
   a₁ = 0.84 and a₂ = 1.64. The central point 7 lies at f̃ = (0.39, 0.39),
   so p̂ = 0.74: front 1 is still far from the Pareto front, but its shape is
   already close to ZDT1's, p = 0.72.
3. Step 8: after the corners, 10 joins S first with D = 2.92; it is the
   member closest to the ideal point, P = 0.86. Member 1, near the corner 8,
   comes last with 0.22.
4. Step 9: front 1 is also the last front, with nine members for six places.
   The six first in the greedy order survive, and 12 (0.77), 2 (0.46) and
   1 (0.22) are removed. Press **New population** until a later front is
   cut: its members are then ranked by 1/P.

## Explore the geometry

The figure samples ten points at fixed angles on the front f₁ᵖ + f₂ᵖ = 1. All
ten are non-dominated, so they form F₁, and with fewer than ten survivors F₁ is
also the last front. Move the true p from 0.5 to 3 and watch the estimate p̂
and the fitted curve follow the true front; they differ slightly because no
point lies exactly on the diagonal. With six survivors, compare p = 0.5 and
p = 2: the same angles, different picks. Then, at p = 0.5, tick the Euclidean
comparison: the knee points look best converged and their scores jump.

<div class="vf-widget" data-vf-widget="agemoea-geometry"><p class="vf-fallback">Interactive figure: ten points on a two-objective front f₁ᵖ + f₂ᵖ = 1 with a slider for the true p, AGE-MOEA's estimated p and fitted curve, and the survival scores that decide which points of the last front survive. It needs JavaScript.</p></div>

## In VAMOS

AGE-MOEA does not report p̂, but the estimate is easy to repeat on a returned
front. The example runs AGE-MOEA on three two-objective problems, ZDT1
(convex), ZDT2 (concave) and DTLZ2 (a quarter circle), and applies the
estimate to each result. The value in parentheses is the curvature of the
true front, read from its point on the diagonal:

```python
import numpy as np

from vamos import optimize
from vamos.problems import DTLZ2, ZDT1, ZDT2


def estimate_p(F):
    """AGE-MOEA's curvature estimate for a two-objective front."""
    Z = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))  # ideal → (0, 0), corners → (1, 0) and (0, 1)
    central = Z[np.argmin(np.abs(Z[:, 0] - Z[:, 1]))]  # the point closest to the diagonal
    return np.log(2) / np.log(1 / central.mean())


problems = [("ZDT1, convex", ZDT1(n_var=30), 0.72), ("ZDT2, concave", ZDT2(n_var=30), 1.44), ("DTLZ2, circle", DTLZ2(n_var=11, n_obj=2), 2.0)]
for name, problem, true_p in problems:
    result = optimize(problem, algorithm="agemoea", pop_size=100, max_evaluations=10_000, engine="numpy", seed=1)
    print(f"{name}: {result.F.shape[0]} solutions, estimated p = {estimate_p(result.F):.2f} (front: {true_p:.2f})")
```

Output:

```text
ZDT1, convex: 100 solutions, estimated p = 0.72 (front: 0.72)
ZDT2, concave: 100 solutions, estimated p = 1.41 (front: 1.44)
DTLZ2, circle: 100 solutions, estimated p = 2.00 (front: 2.00)
```

With two objectives the corner points are the extremes of each objective, so
dividing by the range is the hyperplane normalization, and the point closest
to the diagonal is the one with the smallest |f̃₁ − f̃₂|. ZDT2's front,
f₂ = 1 − f₁², is not exactly an Lp curve, and its estimate depends on which
point happens to be nearest the diagonal.

**VAMOS 1.5.0:** `variant("agemoea_ii")` selects AGE-MOEA-II [7] instead of
AGE-MOEA, the default. AGE-MOEA-II estimates p from the non-corner point
nearest to the ideal point by Newton-Raphson, so that its Lp norm is 1, and
replaces the Lp distances of F₁ by approximate geodesic distances on the
estimated front: each pair of points is projected onto the Lp surface and
measured through their projected midpoint. The rest of the survival is
AGE-MOEA's. The builder sets the variant; here both variants run on
three-objective DTLZ2, whose front is the part of the unit sphere in the
positive orthant:

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import AGEMOEAConfig
from vamos.problems import DTLZ2

problem = DTLZ2(n_var=12, n_obj=3)
for variant in ("agemoea", "agemoea_ii"):
    config = (
        AGEMOEAConfig.builder()
        .pop_size(100)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .variant(variant)
        .build()
    )
    result = optimize(problem, algorithm="agemoea", algorithm_config=config, max_evaluations=10_000, engine="numpy", seed=1)
    radius = np.linalg.norm(result.F, axis=1)  # 1 on DTLZ2's front
    print(f"{variant}: {len(result.F)} solutions, median distance to the origin {np.median(radius):.3f}")
```

Output:

```text
agemoea: 100 solutions, median distance to the origin 1.003
agemoea_ii: 100 solutions, median distance to the origin 1.002
```

Both variants bring the whole population close to the front. Which one spreads
it better depends on the problem and the seed; lesson
[Comparing algorithms rigorously](performance-assessment.md) explains how to
compare them.

`AGEMOEAConfig.default` uses a population of 100, SBX crossover (probability
0.9, distribution index 15) and polynomial mutation (probability 1/n, index
20). Unlike NSGA-III and RVEA, the population size has no lattice to match.
The [AGE-MOEA lesson](../../algorithms/agemoea.md) records a run on
three-objective DTLZ2, and the
[configuration reference](../../reference/api/algorithms/agemoea.md) lists the
builder methods.

## AGE-MOEA and the reference-direction algorithms

| | NSGA-II | NSGA-III, RVEA | AGE-MOEA |
| --- | --- | --- | --- |
| Diversity measure | Crowding distance | Fixed reference directions | Lp distances with the estimated p |
| Convergence measure | Rank | Rank (NSGA-III), distance to the ideal point (RVEA) | Rank, then proximity in the Lp norm |
| Population size | Free | Tied to the lattice | Free |
| Cost of the last cut | Sorting per objective | Association with H directions | Pairwise distances in F₁ |

**Strengths.** AGE-MOEA needs no directions, so the population can have any
size and nothing is wasted on parts of the simplex that the front does not
cover. Its distances adapt to convex, flat and concave fronts, and its
proximity does not favour the knee of a convex front.

**Limits.** The model has one exponent for all objectives and assumes a
connected front that touches every axis: disconnected, degenerate or
irregular fronts break it, and the estimate rests on a single point. The
pairwise distances cost O(m·|F₁|²) per generation, which grows with large
populations. And like every algorithm that normalizes by the corners of the
current front, it can be misled early in the run, when those corners are still
poor.

## Key points

- AGE-MOEA keeps NSGA-II's structure and replaces the crowding distance by a
  survival score computed in the geometry of the first front.
- The first front is translated by its ideal point and divided by the
  intercepts of the hyperplane through its corner points.
- The curvature is read from the point closest to the diagonal:
  p̂ = ln m / ln(1/c̄); VAMOS replaces an estimate of at most 0.1 by 1 and
  caps it at 20.
- Proximity is the Lp distance to the ideal point; the greedy score adds the Lp
  distances to the two nearest kept points and divides by the proximity.
- VAMOS scores later fronts by inverse proximity, as the author's code does,
  and measures them in the first front's translated, normalized coordinates.

**Next:** [Visualizing many objectives](visualizing-many-objectives.md).

## References

[1] Panichella, A. (2019). [An adaptive evolutionary algorithm based on non-Euclidean geometry for many-objective optimization](https://doi.org/10.1145/3321707.3321839). *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '19)*, 595–603.

[2] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://doi.org/10.1109/4235.996017). *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

[3] Zitzler, E., Deb, K., and Thiele, L. (2000). [Comparison of multiobjective evolutionary algorithms: Empirical results](https://doi.org/10.1162/106365600568202). *Evolutionary Computation*, 8(2), 173–195.

[4] Deb, K., Thiele, L., Laumanns, M., and Zitzler, E. (2005). [Scalable test problems for evolutionary multiobjective optimization](https://doi.org/10.1007/1-84628-137-7_6). In A. Abraham, L. Jain and R. Goldberg (Eds.), *Evolutionary Multiobjective Optimization*, 105–145. Springer.

[5] Deb, K., and Jain, H. (2014). [An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, Part I: Solving problems with box constraints](https://doi.org/10.1109/TEVC.2013.2281535). *IEEE Transactions on Evolutionary Computation*, 18(4), 577–601.

[6] Ishibuchi, H., Setoguchi, Y., Masuda, H., and Nojima, Y. (2017). [Performance of decomposition-based many-objective algorithms strongly depends on Pareto front shapes](https://doi.org/10.1109/TEVC.2016.2587749). *IEEE Transactions on Evolutionary Computation*, 21(2), 169–190.

[7] Panichella, A. (2022). [An improved Pareto front modeling algorithm for large-scale many-objective optimization](https://doi.org/10.1145/3512290.3528732). *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '22)*, 565–573.
