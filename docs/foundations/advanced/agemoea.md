---
description: 'AGE-MOEA: normalizing the first front, estimating its curvature p from a central point, proximity and diversity in the Lp norm, the greedy survival score and the truncation of the last front, as VAMOS implements them, with an interactive figure and VAMOS code.'
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

Inside VAMOS, the estimate and the distances follow the formulas. In
`_compute_geometry`, `d` is the squared distance of each point to the
diagonal, the corners are excluded with `np.inf`, and `p` is the formula of
p̂ with its guards. In `_survival_score`, `nn` holds the proximities and
`distances` the Lp distances, each row divided by the proximity of its point:

```python
# Excerpt from src/vamos/engine/algorithm/agemoea/geometry.py
def _compute_geometry(front: np.ndarray[Any, Any], extreme: np.ndarray[Any, Any], n_obj: int) -> float:
    d = _point_to_line_distance(front, np.zeros(n_obj), np.ones(n_obj))
    d[extreme] = np.inf
    index = int(np.argmin(d))
    mean_val = np.mean(front[index, :])
    if mean_val <= 0.0:
        return 1.0
    p = np.log(n_obj) / np.log(1.0 / mean_val)
    if np.isnan(p) or p <= 0.1:
        p = 1.0
    elif p > 20.0:
        p = 20.0
    return float(p)
    front = np.round(front - ideal_point, 12)
    extreme = _find_corner_solutions(front)
    front, normalization = _normalize_front(front, extreme)

    crowd_dist[extreme] = np.inf
    selected = np.full(m, False)
    selected[extreme] = True

    ii = variant == "agemoea_ii"
    p = _compute_geometry_ii(front, extreme) if ii else _compute_geometry(front, extreme, n)
    nn = np.linalg.norm(front, ord=p, axis=1)
    nn[nn < 1e-8] = 1.0

    distances = _geodesic_distances(front, p) if ii else _pairwise_distances(front, p)
    distances[distances < 1e-8] = 1e-8
    distances = distances / nn[:, None]
```

**VAMOS 1.5.0:** the `ii` lines choose between AGE-MOEA, the
default, and AGE-MOEA-II [7], selected with `variant("agemoea_ii")`. AGE-MOEA-II
estimates p from the non-corner point nearest to the ideal point by
Newton-Raphson, so that its Lp norm is 1, and replaces the Lp distances of F₁
by approximate geodesic distances on the estimated front: each pair of points
is projected onto the Lp surface and measured through their projected
midpoint. The rest of the survival is AGE-MOEA's.

The greedy loop keeps, for every unselected point, its two smallest distances
to the selected set (`best1`, `best2`) and their sum (`scores`). Each pass
selects the largest sum, records it as the survival score in `crowd_dist`, and
updates the two nearest distances with the newly selected point. The
`selected_count == 2` branch only matters when S started from a single
corner: it sorts the first two distances once both exist:

```python
# Excerpt from src/vamos/engine/algorithm/agemoea/geometry.py
    while remaining.size > 0:
        index = int(np.argmax(scores))
        best = int(remaining[index])
        d = float(scores[index])
        selected[best] = True
        crowd_dist[best] = d

        remaining = np.delete(remaining, index)
        best1 = np.delete(best1, index)
        best2 = np.delete(best2, index)
        scores = np.delete(scores, index)
        selected_count += 1
        if remaining.size == 0:
            break

        new_dist = distances[remaining, best]
        if selected_count == 2:
            lo = np.minimum(best1, new_dist)
            hi = np.maximum(best1, new_dist)
            best1 = lo
            best2 = hi
            scores = best1 + best2
            continue

        better_first = new_dist < best1
        best2 = np.where(better_first, best1, best2)
        best1 = np.where(better_first, new_dist, best1)
        better_second = (~better_first) & (new_dist < best2)
        best2 = np.where(better_second, new_dist, best2)
        scores = best1 + best2
```

`age_survival` then assembles the generation: whole fronts below `last_rank`
survive, F₁ is scored by the greedy procedure, the later fronts up to the last
one by the inverse Lp distance to the ideal point, and the last front is cut by
decreasing score:

```python
# Excerpt from src/vamos/engine/algorithm/agemoea/geometry.py
    selected = ranks < last_rank
    crowd_dist = np.zeros(F.shape[0], dtype=float)

    front0 = F[ranks == 0, :]
    ideal_point = np.min(front0, axis=0)
    crowd_dist[ranks == 0], p, normalization = _survival_score(front0, ideal_point, kernel.selection_kernels(), variant)

    for r in range(1, last_rank + 1):
        front_idx = fronts[r]
        if front_idx.size == 0:
            continue
        # Same ideal-centred, normalized coordinates as the first-front score.
        front = (F[front_idx] - ideal_point) / normalization
        dist = _minkowski_distances(front, np.zeros((1, F.shape[1])), p).squeeze()
        dist = np.where(dist < 1e-12, 1e-12, dist)
        crowd_dist[front_idx] = 1.0 / dist

    last = fronts[last_rank]
    if last.size > 0:
        order = np.argsort(crowd_dist[last])[::-1]
        remaining = n_survive - int(np.sum(selected))
        selected[last[order[:remaining]]] = True

    survivors = np.flatnonzero(selected)
    return survivors, crowd_dist[survivors]
```

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

**Next:** [Constrained multi-objective optimization](constrained.md).

## References

[1] Panichella, A. (2019). [An adaptive evolutionary algorithm based on non-Euclidean geometry for many-objective optimization](https://doi.org/10.1145/3321707.3321839). *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '19)*, 595–603.

[2] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://doi.org/10.1109/4235.996017). *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

[3] Zitzler, E., Deb, K., and Thiele, L. (2000). [Comparison of multiobjective evolutionary algorithms: Empirical results](https://doi.org/10.1162/106365600568202). *Evolutionary Computation*, 8(2), 173–195.

[4] Deb, K., Thiele, L., Laumanns, M., and Zitzler, E. (2005). [Scalable test problems for evolutionary multiobjective optimization](https://doi.org/10.1007/1-84628-137-7_6). In A. Abraham, L. Jain and R. Goldberg (Eds.), *Evolutionary Multiobjective Optimization*, 105–145. Springer.

[5] Deb, K., and Jain, H. (2014). [An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, Part I: Solving problems with box constraints](https://doi.org/10.1109/TEVC.2013.2281535). *IEEE Transactions on Evolutionary Computation*, 18(4), 577–601.

[6] Ishibuchi, H., Setoguchi, Y., Masuda, H., and Nojima, Y. (2017). [Performance of decomposition-based many-objective algorithms strongly depends on Pareto front shapes](https://doi.org/10.1109/TEVC.2016.2587749). *IEEE Transactions on Evolutionary Computation*, 21(2), 169–190.

[7] Panichella, A. (2022). [An improved Pareto front modeling algorithm for large-scale many-objective optimization](https://doi.org/10.1145/3512290.3528732). *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '22)*, 565–573.
