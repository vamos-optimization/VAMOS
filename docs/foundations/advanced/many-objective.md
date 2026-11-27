---
description: 'Many-objective optimization: why Pareto dominance and crowding lose power with four or more objectives, the remedies, and NSGA-III reference directions in detail, with interactive figures and VAMOS code.'
search:
  boost: 0.5
---

# Many-objective optimization

The algorithms of the Intermediate level were designed, and are mostly tested,
with two or three objectives. Real problems often have more: a car design is
judged on cost, mass, emissions, safety and comfort at once. With four or more
objectives, the problem is called *many-objective*, and several mechanisms that
NSGA-II relies on stop working [1, 2, 3].

In this lesson you will learn:

- why Pareto dominance loses its selective power as objectives are added;
- the other difficulties: diversity, visualization, indicators, population size;
- the families of remedies;
- how NSGA-III replaces crowding with reference directions, and how VAMOS
  configures it.

## Dominance resistance

Two random vectors are comparable only if one is better in **every**
objective. With m objectives that chance shrinks quickly, so most pairs become
incomparable, and most solutions end up in the first front. The figure samples
random points uniformly in the unit hypercube and measures the share that is
non-dominated:

<div class="vf-widget" data-vf-widget="dominance-resistance"><p class="vf-fallback">Interactive figure: the share of mutually non-dominated points among uniform random points for 2 to 15 objectives. It needs JavaScript.</p></div>

When the whole population is in front 1, rank no longer pushes it towards the
Pareto front; selection is decided by the secondary criterion, crowding
distance, which rewards spread, not convergence. Purshouse and Fleming [2]
analysed this loss of selection pressure, and Ishibuchi, Tsukamoto and Nojima
[1] showed its effect on NSGA-II and SPEA2.

## Other difficulties

- **Diversity estimates degrade.** The crowding distance sums per-objective
  gaps, and in high dimension nearly every solution is extreme in some
  objective. Distances also concentrate: all points look equally far apart.
- **The front grows.** The front of an m-objective problem is an
  (m − 1)-dimensional surface. Covering it with the same density needs a
  population that grows exponentially with m.
- **Recombination becomes less effective.** Parents that are both
  non-dominated can be far apart in decision space, and their children land
  far from either.
- **Indicators get expensive or unreliable.** Exact hypervolume grows
  exponentially with m (lesson [Quality indicators](../basic/quality-indicators.md#hypervolume)),
  and reference sets for IGD+ need many points to represent the front.
- **People cannot see the front.** Beyond three objectives, scatter plots give
  way to parallel coordinates, heat maps or projections, each of which hides
  something.

## Families of remedies

Li et al. [3] survey the many-objective algorithms. The main ideas are:

- **Relaxed dominance**, which lets more pairs be compared, for example
  ε-dominance [4].
- **Decomposition and reference directions**: spread a set of directions over
  the objective space and maintain one solution per direction, as MOEA/D
  [5], NSGA-III [6] and RVEA [7] do. Diversity then comes from the directions,
  not from distances between solutions.
- **Indicator-based selection** with fast or approximate hypervolume, such as
  HypE [8].
- **Adaptive geometry**: AGE-MOEA [9] estimates the shape of the front from the
  population and measures diversity accordingly.

## NSGA-III in detail

NSGA-III [6] keeps NSGA-II's structure (merge, non-dominated sort, fill whole
fronts) and changes only the way the last front is truncated. Crowding distance
is replaced by **niching around reference directions**, in four steps.
Lesson [NSGA-III](nsga-iii.md) explains each of them in detail, with the options VAMOS offers.

**1. Reference points.** Before the run, place H points on the unit simplex
(w₁ + … + w_m = 1, w_i ≥ 0) with Das and Dennis' systematic method [10]:
every point whose coordinates are multiples of 1/p, for p *divisions*. Their
number is

<div class="vf-formula">H = C(p + m − 1, m − 1)</div>

Each point defines a reference direction from the origin. Change m and p
below:

<div class="vf-widget" data-vf-widget="reference-directions"><p class="vf-fallback">Interactive figure: Das and Dennis reference points on the simplex and the number of reference directions for m objectives and p divisions. It needs JavaScript.</p></div>

H grows fast: with m = 10 and p = 3, H = 220 directions, but they all lie on
the boundary of the simplex. For eight or more objectives Deb and Jain combine
two layers with few divisions, an outer one and an inner one scaled towards
the centre [6]. Lesson [Reference directions and weight vectors](reference-directions.md)
explains both designs, the weight files that hold other sets, and how MOEA/D,
NSGA-III and RVEA use the same vectors.

**2. Adaptive normalization.** The population's objectives are translated so
that the ideal point is the origin, and scaled by the intercepts of the
hyperplane through the extreme points of the population. Fronts with very
different scales per objective then map onto the simplex.

**3. Association.** Each normalized solution is associated with the reference
direction to which it has the smallest perpendicular distance.

**4. Niche preservation.** Count, for each direction, the solutions already
selected from the earlier fronts. Repeatedly pick the direction with the fewest
associated solutions and add a member of the last front associated with it
(the closest one if the niche is empty, a random one otherwise), until the
population is full.

The population size must be at least H, so that every direction can keep a
solution. VAMOS checks this before the run: a population smaller than the
number of directions stops with an error that asks for a larger population or
fewer divisions, instead of silently dropping directions.

## In VAMOS

NSGA-III on DTLZ2 [11] with three objectives and p = 12, so H = 91 and a
population of 91. DTLZ2's front is the part of the unit sphere in the positive
orthant, so the distance of each point to the origin should approach 1:

```python
from math import comb

import numpy as np

from vamos import optimize
from vamos.algorithms import NSGAIIIConfig
from vamos.problems import DTLZ2

problem = DTLZ2(n_var=12, n_obj=3)
divisions = 12
pop_size = comb(divisions + problem.n_obj - 1, problem.n_obj - 1)
config = (
    NSGAIIIConfig.builder()
    .pop_size(pop_size)
    .crossover("sbx", prob=1.0, eta=30.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .selection("random")
    .reference_directions(divisions=divisions)
    .result_mode("population")
    .build()
)
result = optimize(problem, algorithm="nsgaiii", algorithm_config=config, max_evaluations=10_000, engine="numpy", seed=42)

radius = np.linalg.norm(result.F, axis=1)
print("reference directions:", pop_size)
print("returned:", result.F.shape)
print("distance to the origin: median", round(float(np.median(radius)), 3), "max", round(float(radius.max()), 3))
```

Output:

```text
reference directions: 91
returned: (91, 3)
distance to the origin: median 1.003 max 1.029
```

The [NSGA-III lesson](../../algorithms/nsgaiii.md) draws this result and
discusses the configuration. The [problem reference](../../reference/problems.md)
lists the scalable families (DTLZ and others) for experiments with more
objectives.

## Key points

- With many objectives most solutions are mutually non-dominated, so rank stops
  driving convergence and distance-based diversity degrades.
- Covering a higher-dimensional front needs larger populations, and indicators
  become harder to compute and interpret.
- Reference-direction methods restore diversity by keeping one solution per
  direction; NSGA-III does this inside NSGA-II's structure.
- For H = C(p + m − 1, m − 1) directions the population must hold at least H
  solutions.

**Next:** [Reference directions and weight vectors](reference-directions.md).

## References

[1] Ishibuchi, H., Tsukamoto, N., and Nojima, Y. (2008). [Evolutionary many-objective optimization: A short review](https://doi.org/10.1109/CEC.2008.4631121). *Proceedings of the 2008 IEEE Congress on Evolutionary Computation*, 2419–2426.

[2] Purshouse, R. C., and Fleming, P. J. (2007). [On the evolutionary optimization of many conflicting objectives](https://doi.org/10.1109/TEVC.2007.910138). *IEEE Transactions on Evolutionary Computation*, 11(6), 770–784.

[3] Li, B., Li, J., Tang, K., and Yao, X. (2015). [Many-objective evolutionary algorithms: A survey](https://doi.org/10.1145/2792984). *ACM Computing Surveys*, 48(1), Article 13.

[4] Laumanns, M., Thiele, L., Deb, K., and Zitzler, E. (2002). [Combining convergence and diversity in evolutionary multiobjective optimization](https://doi.org/10.1162/106365602760234108). *Evolutionary Computation*, 10(3), 263–282.

[5] Zhang, Q., and Li, H. (2007). [MOEA/D: A multiobjective evolutionary algorithm based on decomposition](https://doi.org/10.1109/TEVC.2007.892759). *IEEE Transactions on Evolutionary Computation*, 11(6), 712–731.

[6] Deb, K., and Jain, H. (2014). [An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, Part I: Solving problems with box constraints](https://doi.org/10.1109/TEVC.2013.2281535). *IEEE Transactions on Evolutionary Computation*, 18(4), 577–601.

[7] Cheng, R., Jin, Y., Olhofer, M., and Sendhoff, B. (2016). [A reference vector guided evolutionary algorithm for many-objective optimization](https://doi.org/10.1109/TEVC.2016.2519378). *IEEE Transactions on Evolutionary Computation*, 20(5), 773–791.

[8] Bader, J., and Zitzler, E. (2011). [HypE: An algorithm for fast hypervolume-based many-objective optimization](https://doi.org/10.1162/EVCO_a_00009). *Evolutionary Computation*, 19(1), 45–76.

[9] Panichella, A. (2019). [An adaptive evolutionary algorithm based on non-Euclidean geometry for many-objective optimization](https://doi.org/10.1145/3321707.3321839). *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '19)*, 595–603.

[10] Das, I., and Dennis, J. E. (1998). [Normal-boundary intersection: A new method for generating the Pareto surface in nonlinear multicriteria optimization problems](https://doi.org/10.1137/S1052623496307510). *SIAM Journal on Optimization*, 8(3), 631–657.

[11] Deb, K., Thiele, L., Laumanns, M., and Zitzler, E. (2005). [Scalable test problems for evolutionary multiobjective optimization](https://doi.org/10.1007/1-84628-137-7_6). In A. Abraham, L. Jain and R. Goldberg (Eds.), *Evolutionary Multiobjective Optimization*, 105–145. Springer.
