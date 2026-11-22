---
description: 'A map of the families of multi-objective evolutionary algorithms, told by how they choose survivors: dominance-based (NSGA-II, SPEA2), decomposition-based (MOEA/D), indicator-based (SMS-EMOA, IBEA) and swarm-based (SMPSO), with one population seen through each rule and the lessons that teach each algorithm.'
search:
  boost: 0.5
---

# Families of multi-objective algorithms

Lesson [Genetic algorithms](genetic-algorithms.md) ended with an open
question. Survival keeps the best individuals, but with several objectives
there is no single "best": in a converged population almost every individual
is non-dominated, and something else must decide who stays. Each **family** of
multi-objective algorithms is one answer to that question. The rest of the
loop (population, parent selection, variation, evaluation, termination) is
largely shared, so knowing the families is enough to place almost every
algorithm in VAMOS.

This lesson is a map. For each family it gives the survival principle, the
VAMOS algorithms that follow it, and the lessons where they are taught step by
step. The Intermediate level is organized by the same families.

In this lesson you will learn:

- the survival principle of the dominance-based, decomposition-based,
  indicator-based and swarm-based families;
- how one population is filtered differently by each principle;
- which VAMOS algorithms belong to each family, and where each is taught;
- how the many-objective and constrained algorithms of the Advanced level
  build on these families.

## The map

<div class="vf-figure"><svg class="vf-diagram" viewBox="0 0 640 286" role="img" aria-label="Map of the families. Survival must decide which solutions stay when there are several objectives. Four families answer: dominance-based, Pareto rank first and then density, with NSGA-II and SPEA2; decomposition-based, the best solution for each weight vector, with MOEA/D; indicator-based, the solution whose loss costs the least indicator value leaves, with SMS-EMOA and IBEA; swarm-based, every particle moves on and an archive keeps the leaders, with SMPSO. AGE-MOEA, CCMO, ToP and CMOEA-MS of the Advanced level build on the dominance-based family; NSGA-III, RVEA and C-TAEA build on the decomposition-based family."><rect class="vf-box" x="170" y="8" width="300" height="44" rx="10"/><text x="320" y="27" text-anchor="middle">Survival: which solutions stay?</text><text class="vf-diagram-small" x="320" y="44" text-anchor="middle">several objectives, no single ranking</text><path class="vf-arrow" d="M320 52V66M80 66H560M80 66V78M240 66V78M400 66V78M560 66V78"/><path class="vf-arrowhead" d="M80 84l-4-7h8zM240 84l-4-7h8zM400 84l-4-7h8zM560 84l-4-7h8z"/><rect class="vf-box" x="10" y="84" width="140" height="96" rx="10"/><text x="80" y="106" text-anchor="middle">Dominance-based</text><text class="vf-diagram-small" x="80" y="127" text-anchor="middle">Pareto rank first,</text><text class="vf-diagram-small" x="80" y="143" text-anchor="middle">then density</text><text x="80" y="167" text-anchor="middle">NSGA-II · SPEA2</text><rect class="vf-box" x="170" y="84" width="140" height="96" rx="10"/><text x="240" y="106" text-anchor="middle">Decomposition-based</text><text class="vf-diagram-small" x="240" y="127" text-anchor="middle">the best solution</text><text class="vf-diagram-small" x="240" y="143" text-anchor="middle">for each weight vector</text><text x="240" y="167" text-anchor="middle">MOEA/D</text><rect class="vf-box" x="330" y="84" width="140" height="96" rx="10"/><text x="400" y="106" text-anchor="middle">Indicator-based</text><text class="vf-diagram-small" x="400" y="127" text-anchor="middle">the least loss of</text><text class="vf-diagram-small" x="400" y="143" text-anchor="middle">indicator value leaves</text><text x="400" y="167" text-anchor="middle">SMS-EMOA · IBEA</text><rect class="vf-box" x="490" y="84" width="140" height="96" rx="10"/><text x="560" y="106" text-anchor="middle">Swarm-based</text><text class="vf-diagram-small" x="560" y="127" text-anchor="middle">every particle moves;</text><text class="vf-diagram-small" x="560" y="143" text-anchor="middle">an archive keeps leaders</text><text x="560" y="167" text-anchor="middle">SMPSO</text><path class="vf-arrow" d="M80 180V208M240 180V208"/><path class="vf-arrowhead" d="M80 214l-4-7h8zM240 214l-4-7h8z"/><rect class="vf-box" x="10" y="214" width="140" height="62" rx="10"/><text class="vf-diagram-small" x="80" y="240" text-anchor="middle">AGE-MOEA · CCMO</text><text class="vf-diagram-small" x="80" y="258" text-anchor="middle">ToP · CMOEA-MS</text><rect class="vf-box" x="170" y="214" width="140" height="62" rx="10"/><text class="vf-diagram-small" x="240" y="240" text-anchor="middle">NSGA-III · RVEA</text><text class="vf-diagram-small" x="240" y="258" text-anchor="middle">C-TAEA</text><text class="vf-diagram-small" x="330" y="240">Advanced algorithms that build on</text><text class="vf-diagram-small" x="330" y="258">a family: many objectives, constraints</text></svg></div>

The first three families share the generational or steady-state loop of a
genetic algorithm and differ only in survival. The swarm-based family also
changes how new solutions are made.

## One population, four survival rules

The ten solutions below compete for five places. Each family keeps a
different five, and the swarm keeps all ten. Choose a family to see its rule,
drag the points (or focus one with <kbd>Tab</kbd> and move it with the arrow
keys), and change the number of places. The table compares the four families
at once. Things to try:

1. **Dominance-based.** Seven points are non-dominated, so front 1 does not
   fit in five places. Crowding distance keeps the extremes P1 and P7 and the
   most isolated points P2, P5 and P6, and removes P3 and P4.
2. **Decomposition-based.** Five weight vectors, five subproblems. Each keeps
   the point closest to the ideal point z* along its direction, in the
   Tchebycheff sense: P1, P3, P4, P6 and P7. Set the places to 7: P6 is the
   best point for two weight vectors, so only six distinct points are kept.
3. **Indicator-based.** The dominated P9, P10 and P8 leave first, then P5 and
   P3, the points whose loss costs the least hypervolume. The shaded region is
   the hypervolume of the five that stay.
4. **Swarm-based.** No particle is removed. The seven non-dominated ones are
   the leaders that guide the swarm.

<div class="vf-widget" data-vf-widget="family-survivors"><p class="vf-fallback">Interactive figure: ten draggable solutions and the ones that each family keeps for five places. With the default points, the dominance-based rule (NSGA-II) keeps P1, P2, P5, P6 and P7; the decomposition-based rule (MOEA/D with five weight vectors) keeps P1, P3, P4, P6 and P7; the indicator-based rule (SMS-EMOA) keeps P1, P2, P4, P6 and P7; the swarm-based family (SMPSO) keeps every particle, and the seven non-dominated ones are its leaders. It needs JavaScript.</p></div>

With the default points and five places, the three rules that remove points
all keep the extremes P1 and P7 and remove the three dominated points. They disagree in the middle of the front, where each
rule's idea of a good spread decides. None of the three answers is wrong;
they are different definitions of a good approximation set, as lesson
[Quality aspects](quality-aspects.md) anticipated.

## Dominance-based algorithms

<a id="pareto-based-algorithms"></a>

Dominance-based algorithms, also called **Pareto-based**, rank the population
by Pareto dominance first: the non-dominated solutions form front 1, the
solutions dominated only by them form front 2, and so on. Convergence comes
first, because a solution of an earlier front always beats one of a later
front. Within a front, a **density estimate** breaks the tie, so that isolated
solutions are preferred to crowded ones and the population spreads along the
front.

- **NSGA-II** [1] sorts the merged parents and offspring into fronts, keeps
  whole fronts while they fit, and cuts the last one by *crowding distance*,
  the size of the gap around each solution.
- **SPEA2** [2] gives each solution one fitness number from the dominance
  relations it takes part in, plus a density term from the distance to its
  k-th nearest neighbour, and keeps the best in an archive of fixed size,
  removing the most crowded one at a time.

Dominance-based survival makes no assumption about the shape of the front and
works well with two or three objectives. With many objectives nearly every
solution becomes non-dominated, front 1 holds the whole population, and
selection is left to the density estimate alone (lesson
[Many-objective optimization](../advanced/many-objective.md)).

**In VAMOS:** `nsgaii` and `spea2`. **Taught in:**
[NSGA-II at a glance](../intermediate/nsga-ii.md) and the three lessons that
follow it, then [SPEA2](../intermediate/spea2.md).

## Decomposition-based algorithms

Decomposition-based algorithms turn the multi-objective problem into N
single-objective **subproblems**, one per weight vector w¹, …, wᴺ, and solve
them together. **MOEA/D** [3] keeps one solution per subproblem. A new
solution is bred from *neighbouring* subproblems, those with the closest
weight vectors, and replaces a neighbour's solution if it is better **for that
neighbour's subproblem**. Diversity comes from the spread of the weight
vectors, fixed before the run, not from a density estimate.

Each subproblem is a scalarization. Lesson
[Multi-objective problems](multi-objective-problems.md#turning-many-objectives-into-one-scalarization)
showed that the weighted sum cannot reach the non-convex parts of a front. The
**Tchebycheff** scalarization [4, 5] measures the distance to the ideal point
z*, the vector of the best value of each objective, with a weighted maximum:

<div class="vf-formula">g^te(x | w, z*) = max_i  w_i · | f_i(x) − z*_i |</div>

Its contour lines are corners, not straight lines, and every Pareto-optimal
point minimizes it for some weight vector, whatever the shape of the front.
VAMOS computes it for many solutions at once, and counts a zero weight as a
small positive one:

```python
# Excerpt from src/vamos/engine/algorithm/moead/aggregation.py
def tchebycheff(fvals: Array, weights: Array, ideal: Array) -> Array:
    """Tchebycheff aggregation: max(w * |f - z*|)."""
    diff = np.abs(fvals - ideal)
    weighted = np.where(weights == 0, ZERO_WEIGHT_EPS * diff, weights * diff)
    return np.asarray(np.max(weighted, axis=-1), dtype=float)
```

Compare the two scalarizations on a non-convex front: sweep the weights with
each one and count the distinct solutions it finds.

<div class="vf-widget" data-vf-widget="scalarization" data-modes="ws,tch" data-shape="concave"><p class="vf-fallback">Interactive figure: weighted-sum and Tchebycheff scalarizations on a convex and on a non-convex front. It needs JavaScript.</p></div>

Evenly spaced weights do not give evenly spaced solutions: their spacing
depends on the shape of the front, and Ishibuchi et al. [6] showed that the
performance of decomposition-based algorithms depends strongly on how well
the shape of the front matches the distribution of the weights.

**Reference vectors for many objectives.** The idea of a fixed set of
directions in objective space carries over to problems with four or more
objectives, where dominance-based selection weakens. **NSGA-III** keeps
non-dominated sorting and uses reference directions to count how many
solutions each region already has; **RVEA** selects, along each reference
vector, the solution with the best angle-penalized distance.

**In VAMOS:** `moead`, and `nsgaiii` and `rvea` for many objectives.
**Taught in:** [MOEA/D](../intermediate/moead.md), then
[NSGA-III](../advanced/nsga-iii.md) and [RVEA](../advanced/rvea.md).

## Indicator-based algorithms

If the goal is a set with a high hypervolume (lesson
[Quality indicators](quality-indicators.md)), why not select for it directly?
Indicator-based algorithms do exactly that: the solution that leaves is the
one whose loss costs the least indicator value.

- **SMS-EMOA** [7] is steady-state: each step creates one child, sorts the
  N + 1 solutions into fronts, and removes the solution of the last front with
  the **smallest exclusive hypervolume contribution**, the hypervolume that
  only it dominates.
- **IBEA** [8] defines the fitness of a solution from a binary indicator,
  usually the additive ε, computed against every other solution, and removes
  the worst solution one at a time.

Indicator-based selection inherits the Pareto compliance of the hypervolume,
and the distribution of its points adapts to the shape of the front. It also
inherits its cost: computing hypervolume contributions becomes expensive as
the number of objectives grows [9].

**In VAMOS:** `smsemoa` and `ibea`. **Taught in:**
[SMS-EMOA](../intermediate/smsemoa.md) and [IBEA](../intermediate/ibea.md).

## Swarm-based algorithms

Particle swarm optimization replaces crossover and mutation with movement.
Each particle has a position and a velocity, remembers the best position it
has visited, and is pulled towards a *leader*, a good position found by the
swarm. **SMPSO** [10] adapts this to several objectives with an archive of
non-dominated **leaders**, bounded with crowding distance, from which each
particle picks its guide, and bounds each particle's speed.

The swarm itself has no survival step: every particle moves to its new
position at every iteration, whatever its quality. Elitism lives in the
leaders archive, which keeps the best positions found so far and is what VAMOS
returns by default. Particles need a velocity, so SMPSO accepts real and mixed
variables only.

**In VAMOS:** `smpso`. **Taught in:** [SMPSO](../intermediate/smpso.md).

## The families in VAMOS

| Family | Survival principle | VAMOS algorithms | Foundations lessons | Advanced algorithms that build on it |
| --- | --- | --- | --- | --- |
| Dominance-based (Pareto-based) | Pareto rank first, then a density estimate | [NSGA-II](../../algorithms/nsgaii.md), [SPEA2](../../algorithms/spea2.md) | [NSGA-II](../intermediate/nsga-ii.md), [SPEA2](../intermediate/spea2.md) | [AGE-MOEA](../advanced/agemoea.md); [CCMO](../advanced/ccmo.md), [ToP](../advanced/top.md), [CMOEA-MS](../advanced/cmoeams.md) |
| Decomposition-based | The best solution for each weight vector | [MOEA/D](../../algorithms/moead.md) | [MOEA/D](../intermediate/moead.md) | [NSGA-III](../advanced/nsga-iii.md), [RVEA](../advanced/rvea.md); [C-TAEA](../advanced/ctaea.md) |
| Indicator-based | The smallest loss of indicator value leaves | [SMS-EMOA](../../algorithms/smsemoa.md), [IBEA](../../algorithms/ibea.md) | [SMS-EMOA](../intermediate/smsemoa.md), [IBEA](../intermediate/ibea.md) | |
| Swarm-based | Every particle moves; an archive keeps the leaders | [SMPSO](../../algorithms/smpso.md) | [SMPSO](../intermediate/smpso.md) | |

The Advanced level shows how these principles are extended:

- **Many objectives.** NSGA-III and RVEA bring decomposition's reference
  directions to four or more objectives. AGE-MOEA keeps NSGA-II's fronts and
  cuts the last one with a score computed from the estimated geometry of the
  front (lesson [Many-objective optimization](../advanced/many-objective.md)).
- **Constraints.** The constrained algorithms reuse a family's survival and
  add a second population, archive, phase or stage (lesson
  [Constrained optimization](../advanced/constrained.md)). CCMO and CMOEA-MS
  use SPEA2's fitness and environmental selection, ToP ends with NSGA-II and
  constrained domination, and C-TAEA organizes its two archives with weight
  vectors, as decomposition does.

Every algorithm receives the same problem and budget through `optimize`, so
switching family is a one-word change:

```python
from vamos import optimize

families = {
    "nsgaii": "dominance-based",
    "spea2": "dominance-based",
    "moead": "decomposition-based",
    "smsemoa": "indicator-based",
    "ibea": "indicator-based",
    "smpso": "swarm-based",
}
for algorithm, family in families.items():
    result = optimize("zdt1", algorithm=algorithm, pop_size=100, max_evaluations=5_000, seed=1)
    print(f"{algorithm:<8} {family:<20} {result.data['evaluations']} evaluations, {result.F.shape[0]} returned solutions")
```

Output:

```text
nsgaii   dominance-based      5000 evaluations, 100 returned solutions
spea2    dominance-based      5000 evaluations, 100 returned solutions
moead    decomposition-based  5000 evaluations, 100 returned solutions
smsemoa  indicator-based      5000 evaluations, 100 returned solutions
ibea     indicator-based      5000 evaluations, 100 returned solutions
smpso    swarm-based          5000 evaluations, 32 returned solutions
```

Every run spends exactly the budget. SMPSO returns its leaders archive, which
here holds 32 non-dominated positions; the others return the non-dominated
solutions of their final population (for SPEA2, of its archive). The loop deliberately prints no quality
values: one run per algorithm on one problem with default settings says little
about which algorithm is better, and lesson
[Rigorous comparison](../advanced/performance-assessment.md) explains what a
comparison needs. The [capability matrix](../../reference/algorithms.md#capability-matrix)
lists the encodings, constraint handling and engines of each algorithm.

## Key points

- Families differ in how survival ranks solutions with several objectives;
  the rest of the evolutionary loop is largely shared.
- Dominance-based algorithms rank by Pareto dominance, then by density
  (NSGA-II, SPEA2).
- Decomposition-based algorithms solve one scalarized subproblem per weight
  vector (MOEA/D); the Tchebycheff scalarization reaches non-convex fronts.
  NSGA-III and RVEA carry the idea to many objectives.
- Indicator-based algorithms remove the solution whose loss costs the least
  indicator value (SMS-EMOA, IBEA), at the price of computing the indicator.
- Swarm-based algorithms move every particle and keep the elite in an archive
  of leaders (SMPSO).
- The constrained algorithms of VAMOS build on these families. In VAMOS all of
  them share one interface: changing `algorithm=` changes the family.

**Next level:** [NSGA-II at a glance](../intermediate/nsga-ii.md).

## References

[1] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://doi.org/10.1109/4235.996017). *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

[2] Zitzler, E., Laumanns, M., and Thiele, L. (2001). [SPEA2: Improving the strength Pareto evolutionary algorithm](https://tik-db.ee.ethz.ch/db/public/tik/?db=publications&form=report_single_publication&publication_id=1287). TIK Report 103, Computer Engineering and Networks Laboratory, ETH Zurich.

[3] Zhang, Q., and Li, H. (2007). [MOEA/D: A multiobjective evolutionary algorithm based on decomposition](https://doi.org/10.1109/TEVC.2007.892759). *IEEE Transactions on Evolutionary Computation*, 11(6), 712–731.

[4] Steuer, R. E., and Choo, E.-U. (1983). [An interactive weighted Tchebycheff procedure for multiple objective programming](https://doi.org/10.1007/BF02591870). *Mathematical Programming*, 26(3), 326–344.

[5] Miettinen, K. (1999). [*Nonlinear Multiobjective Optimization*](https://doi.org/10.1007/978-1-4615-5563-6). Kluwer Academic Publishers.

[6] Ishibuchi, H., Setoguchi, Y., Masuda, H., and Nojima, Y. (2017). [Performance of decomposition-based many-objective algorithms strongly depends on Pareto front shapes](https://doi.org/10.1109/TEVC.2016.2587749). *IEEE Transactions on Evolutionary Computation*, 21(2), 169–190.

[7] Beume, N., Naujoks, B., and Emmerich, M. (2007). [SMS-EMOA: Multiobjective selection based on dominated hypervolume](https://doi.org/10.1016/j.ejor.2006.08.008). *European Journal of Operational Research*, 181(3), 1653–1669.

[8] Zitzler, E., and Künzli, S. (2004). [Indicator-based selection in multiobjective search](https://doi.org/10.1007/978-3-540-30217-9_84). *Parallel Problem Solving from Nature – PPSN VIII*, Lecture Notes in Computer Science 3242, 832–842.

[9] Bader, J., and Zitzler, E. (2011). [HypE: An algorithm for fast hypervolume-based many-objective optimization](https://doi.org/10.1162/EVCO_a_00009). *Evolutionary Computation*, 19(1), 45–76.

[10] Nebro, A. J., Durillo, J. J., García-Nieto, J., Coello Coello, C. A., Luna, F., and Alba, E. (2009). [SMPSO: A new PSO-based metaheuristic for multi-objective optimization](https://doi.org/10.1109/MCDM.2009.4938830). *IEEE Symposium on Computational Intelligence in Multi-Criteria Decision-Making*, 66–73.
