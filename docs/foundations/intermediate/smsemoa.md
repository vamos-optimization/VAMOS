---
description: 'SMS-EMOA in depth: the steady-state (μ + 1) loop, non-dominated sorting of the N + 1 solutions, removal of the least hypervolume contributor of the last front, the reference point, special cases and cost, with interactive figures and VAMOS code.'
search:
  boost: 0.5
---

# SMS-EMOA: selection by hypervolume contribution

Lesson [Quality indicators](../basic/quality-indicators.md#hypervolume) showed that the
hypervolume rewards convergence, spread and uniformity at once, and lesson
[Families of algorithms](../basic/algorithm-families.md#indicator-based-algorithms) asked why an algorithm
should not select for it directly. SMS-EMOA, the *S-metric selection*
evolutionary multi-objective algorithm of Emmerich, Beume and Naujoks [1, 2],
does exactly that, in the simplest possible way: create one offspring, and
remove the solution whose loss costs the least hypervolume.

SMS-EMOA keeps NSGA-II's non-dominated sorting for convergence and replaces the
crowding distance with the **exclusive hypervolume contribution** for
diversity. Because it removes a single solution per step, its contributions
always describe the current set; NSGA-II, which cuts a whole front with
distances computed once (lesson [Crowding and tournaments](crowding-distance.md)), has no such
guarantee.

In this lesson you will learn:

- the steady-state (μ + 1) loop and what it changes;
- how the N + 1 solutions are sorted, and which front loses a member;
- how the exclusive hypervolume contribution is computed, and how the reference
  point changes it;
- what VAMOS does when the last front has one point, with more than two
  objectives and with constraints;
- what the selection costs, and when to use SMS-EMOA.

## From an indicator to a selection rule

The hypervolume, or S metric, was introduced to compare the results of
multi-objective algorithms [3]. A set of a given size that maximizes it
contains only Pareto-optimal points [4], so an algorithm whose population never
loses hypervolume moves towards the front. Emmerich, Beume and Naujoks turned
that observation into a selection operator [2], and Beume, Naujoks and Emmerich
evaluated the resulting algorithm on two- and three-objective benchmarks and on
aeronautical design problems [1].

## The steady-state loop

SMS-EMOA is a (μ + 1) algorithm, in the terminology of evolution strategies:
μ = N parents create one offspring, and the worst of the N + 1 solutions
leaves.

<div class="vf-formula">P ← N random solutions, evaluated
repeat until the evaluation budget is spent:
    choose two parents in P
    q ← mutation( first child of crossover(parents) ),   evaluate q
    Q ← P ∪ {q}                                     (N + 1 solutions)
    F₁, …, F_v ← non-dominated sort of Q             (non-dominated sorting)
    r ← the member s of F_v with the smallest ΔS(s, F_v)
    P ← Q \ {r}</div>

Each evaluation changes the population, so a good child can become a parent
at once, as in steady-state NSGA-II (lesson
[Elitist survival](elitist-survival.md#steady-state-nsga-ii)). The child competes on equal
terms: if it is the worst of the N + 1, it is the one removed, and the
population does not change.

Parent selection is not part of the method. VAMOS draws both parents uniformly
at random by default, `selection("random")`, and `selection("tournament")`
uses NSGA-II's tournament on rank and crowding distance instead.

## Sorting the N + 1 solutions

The N + 1 solutions are sorted into fronts exactly as in NSGA-II (lesson
[Non-dominated sorting](non-dominated-sorting.md)). Only the **last front** F_v can lose a member;
every earlier front is safe. Early in a run, F_v often holds dominated
solutions or the child alone. Later, all N + 1 solutions are often mutually
non-dominated, v = 1, and the hypervolume alone decides.

VAMOS does not sort from scratch at every step: it inserts the child into the
fronts of the population, which gives the same fronts as a full sort at a lower
cost.

Beume et al. [1] also describe a cheaper rule for a population with more than
one front: remove the member of the last front that the most solutions
dominate. VAMOS does not use it; it always measures hypervolume contributions.

## The exclusive hypervolume contribution

With S(A) the hypervolume of a set A with respect to a reference point r
(lesson [Quality indicators](../basic/quality-indicators.md#hypervolume)), the exclusive
contribution of s to the last front is the hypervolume lost if s leaves:

<div class="vf-formula">ΔS(s, F_v) = S(F_v) − S(F_v \ {s})</div>

With two objectives it is a rectangle. Sort F_v by f₁, so that s₁, …, s_n have
increasing f₁ and decreasing f₂; each point's rectangle reaches to the next
point along f₁ and to the previous point along f₂, and the two extremes reach
the reference point:

<div class="vf-formula">ΔS(sᵢ) = ( f₁(sᵢ₊₁) − f₁(sᵢ) ) · ( f₂(sᵢ₋₁) − f₂(sᵢ) ),      with  f₂(s₀) = r₂  and  f₁(sₙ₊₁) = r₁</div>

A point squeezed between close neighbours has a thin rectangle and leaves
first, so the removals thin out crowded regions. Drag the points of the figure
and watch which one SMS-EMOA would discard:

<div class="vf-widget" data-vf-widget="hypervolume" data-smsemoa="true"><p class="vf-fallback">Interactive figure: the exclusive hypervolume contribution of each point and the point that SMS-EMOA would discard. It needs JavaScript.</p></div>

## The reference point

The two extremes' contributions extend to r, so r decides how much they are
worth. Too close, and the extremes contribute almost nothing and are removed
first; far away, and they are always kept. VAMOS offers three rules through
`SMSEMOAConfig.builder().reference_point(...)`:

<div class="vf-formula">"relative" (default):   r = nadir + offset · (nadir − ideal)     offset = 1.0, recomputed at every step
"jmetal":               rᵢ = 1.1 · max fᵢ,  or  max fᵢ / 1.1 when max fᵢ &lt; 0,    recomputed at every step
vector=[…]:             a fixed r given by the user</div>

Ideal and nadir are the best and worst value of each objective over **all
N + 1 solutions**, not only the last front (lesson
[Pareto dominance](../basic/pareto-dominance.md#ideal-and-nadir-points)). The default places r
one objective range beyond the worst values, so rescaling an objective rescales
r with it and the selection does not depend on the units. jMetal's rule lies
close to the worst values. A fixed vector suits an objective space whose
bounds are known; points outside its box contribute 0 and are removed before any
point inside it with a positive contribution. The
[SMS-EMOA lesson](../../algorithms/smsemoa.md#understand-the-configuration)
compares the rules and records when VAMOS introduced them. Auger et al. [5]
showed how r shapes the distribution that maximizes the hypervolume, and
Ishibuchi et al. [6] how to choose it.

## Special cases in VAMOS

- **The last front has one solution.** It is removed at once, and no
  hypervolume is computed. If that solution is the child, the population does
  not change.
- **Ties.** The first of the equal smallest contributions is removed, in the
  order of the population with the child last. Duplicated points contribute 0.
- **More than two objectives.** The contributions are exact for any number of
  objectives; VAMOS uses no approximation. When MooCore is installed, VAMOS
  computes them with it.
- **Constraints.** With the default `constraint_mode="feasibility"`, while any of
  the N + 1 solutions is infeasible, the one with the largest total violation is
  removed (the last one on ties), and no hypervolume is computed. A
  `constraint_handling(...)` technique ranks the N + 1 solutions instead, and the
  least contributor of its last front leaves; see
  [choose a constraint-handling technique](../../reference/constraints.md#choose-a-constraint-handling-technique).
- **Duplicates.** With `eliminate_duplicates(True)`, a child that repeats a
  decision vector or an objective vector of the population is discarded.

## Step through one selection

Six parents, P1 to P6, and one child C. Press the step buttons: sort into
fronts, identify the last front, compute its contributions, remove the worst.
Things to try:

1. In the default scenario, the child is dominated by P3 and P4. It is alone in
   the last front, so VAMOS removes it without computing any hypervolume.
2. Switch to the second scenario, or drag C below the front, and step again. All
   seven points form one front; P4, squeezed between P3 and C, has the smallest
   contribution and leaves.
3. Lower the reference offset to 0. The extremes P1 and P6 now contribute
   nothing, and P1, the first of them, is removed: the population starts losing
   its spread.

<div class="vf-widget" data-vf-widget="smsemoa-step"><p class="vf-fallback">Interactive figure: one SMS-EMOA selection step on six parents and one child, sorted into fronts, with the hypervolume contributions of the last front and the solution removed. It needs JavaScript.</p></div>

### Try it: one step, step by step

The figure above starts from a given population and child. This one runs a
whole step as VAMOS does by default, on six solutions of a problem with two
variables whose Pareto front is ZDT1's, f₂ = 1 − √f₁: two random parents, the
first child of SBX, mutated, the reference point of the seven solutions, the
fronts and the contributions of the last front. Parents are circles, the child
is a triangle, and darker points lie on better fronts.

<div class="vf-widget" data-vf-widget="smsemoa-steps"><p class="vf-fallback">Interactive figure: one SMS-EMOA step on six solutions, step by step: the population, two random parents, the child, the reference point of the seven solutions, non-dominated sorting, the last front, its hypervolume contributions, and the next population. It needs JavaScript.</p></div>

Things to try:

1. Step 2: the parents, 4 and 6, are drawn at random. 4 lies on front 2, and
   its front plays no part.
2. Step 5: child 7 joins front 2, the last front, with 2, 3 and 4.
3. Step 7: 2, squeezed between 4 and 3, has the smallest contribution, 0.0193.
   Child 7, an extreme of front 2, contributes 0.1600: its rectangle reaches
   r = (1.67, 1.98).
4. Step 8: child 7 replaces 2. Press **New population** for another step whose
   last front has at least three members.

## In VAMOS

The contributions of the second scenario, with MooCore and VAMOS' default
reference point:

```python
import moocore
import numpy as np

parents = np.array([[0.08, 0.90], [0.20, 0.62], [0.35, 0.45], [0.40, 0.40], [0.62, 0.22], [0.90, 0.06]])
child = np.array([[0.50, 0.30]])
candidates = np.vstack([parents, child])  # the N + 1 solutions, here a single front
nadir, ideal = candidates.max(axis=0), candidates.min(axis=0)
reference = nadir + 1.0 * (nadir - ideal)  # VAMOS' default reference point, offset 1.0
contributions = moocore.hv_contributions(candidates, ref=reference)

labels = ["P1", "P2", "P3", "P4", "P5", "P6", "C"]
print("reference point:", reference.round(3))
for label, (f1, f2), value in zip(labels, candidates, contributions):
    print(f"{label} ({f1:.2f}, {f2:.2f}): contribution {value:.4f}")
print("SMS-EMOA removes", labels[int(np.argmin(contributions))])
```

Output:

```text
reference point: [1.72 1.74]
P1 (0.08, 0.90): contribution 0.1008
P2 (0.20, 0.62): contribution 0.0420
P3 (0.35, 0.45): contribution 0.0085
P4 (0.40, 0.40): contribution 0.0050
P5 (0.62, 0.22): contribution 0.0224
P6 (0.90, 0.06): contribution 0.1312
C (0.50, 0.30): contribution 0.0120
SMS-EMOA removes P4
```

The large margin of the default reference point gives the extremes the two
largest contributions. In a run, each step draws the two parents (with
replacement under random selection) and keeps only the first child of the
crossover, mutated. In the default reference-point rule, an objective without
spread counts as having a range of 1.

A complete run on ZDT1 with the default reference point written out:

```python
from vamos import optimize
from vamos.algorithms import SMSEMOAConfig
from vamos.problems import ZDT1

problem = ZDT1(n_var=30)
config = (
    SMSEMOAConfig.builder()
    .pop_size(50)
    .crossover("sbx", prob=0.9, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .selection("random")
    .reference_point(offset=1.0, adaptive=True)
    .build()
)
result = optimize(problem, algorithm="smsemoa", algorithm_config=config, max_evaluations=5_000, engine="numpy", seed=1)

F = result.F
print("evaluations:", result.data["evaluations"], "| generations of 50 children:", result.data["generation"])
print("returned:", F.shape, "| f1 from", round(float(F[:, 0].min()), 3), "to", round(float(F[:, 0].max()), 3))
print("reference point of the last step:", result.data["reference_point"].round(3))
```

Output:

```text
evaluations: 5000 | generations of 50 children: 99
returned: (50, 2) | f1 from 0.0 to 0.991
reference point of the last step: [1.982 2.118]
```

After the 50 initial evaluations, 4,950 children were created, one per step;
VAMOS counts a generation every 50 children. The returned set spans nearly the
whole range of f₁, because the extremes carry large contributions. The
[SMS-EMOA lesson](../../algorithms/smsemoa.md) records a run with a fixed
reference point, and the
[configuration reference](../../reference/api/algorithms/smsemoa.md) lists every
builder method.

## Cost

Each evaluation is followed by one contributions computation on the last front.
With two objectives, a sort of that front suffices, O(n log n) for n points. The
cost of the exact hypervolume, and of exact contributions, grows exponentially
with the number of objectives [7, 8]; Guerreiro, Fonseca and Paquete [9] survey
the exact and approximate algorithms. A run of E evaluations computes E sets of
contributions, where NSGA-II computes E / N sets of crowding distances.

With `engine="numba"` and a supported configuration (real encoding with SBX
and polynomial mutation, among others), VAMOS compiles the whole step, including
the two-objective contributions, and reproduces the NumPy path. For many objectives,
HypE [10] estimates the contributions by Monte Carlo sampling, and
reference-direction algorithms avoid them altogether (lesson
[Many objectives](../advanced/many-objective.md)).

## SMS-EMOA compared

- **With NSGA-II.** Same sorting; the hypervolume contribution replaces the
  crowding distance, and one child per step replaces N. The contributions
  account for the shape of the front and are recomputed after every removal.
- **With SPEA2** (lesson [SPEA2](spea2.md)). Both remove one solution at a time;
  SPEA2 measures distances to neighbours, SMS-EMOA measures the volume a point
  adds, so its distribution depends on the shape of the front and on the
  reference point [5].
- **With IBEA** (lesson [IBEA](ibea.md)). IBEA also selects by an indicator but
  needs only pairwise indicator values, which stay cheap with many objectives.
- **When to use it.** Two or three objectives, when the hypervolume is the
  measure that matters, and especially when evaluations are expensive enough
  that the per-step selection cost does not matter.

## Key points

- SMS-EMOA is steady-state: each step creates one child and removes one of the
  N + 1 solutions.
- Only the last front loses a member: the one with the smallest exclusive
  hypervolume contribution ΔS(s, F_v) = S(F_v) − S(F_v \ {s}).
- The reference point sets the contributions of the extremes; VAMOS' default
  is nadir + 1.0 · (nadir − ideal) of the N + 1 solutions, recomputed at every
  step.
- VAMOS removes a single-member last front directly, computes exact
  contributions for any number of objectives, and handles constraints by
  violation first.
- The cost of exact contributions limits SMS-EMOA to few objectives.

**Next:** [IBEA: indicator-based fitness](ibea.md).

## References

[1] Beume, N., Naujoks, B., and Emmerich, M. (2007). [SMS-EMOA: Multiobjective selection based on dominated hypervolume](https://doi.org/10.1016/j.ejor.2006.08.008). *European Journal of Operational Research*, 181(3), 1653–1669.

[2] Emmerich, M., Beume, N., and Naujoks, B. (2005). An EMO algorithm using the hypervolume measure as selection criterion. *Evolutionary Multi-Criterion Optimization – EMO 2005*, Lecture Notes in Computer Science 3410, 62–76.

[3] Zitzler, E., and Thiele, L. (1998). [Multiobjective optimization using evolutionary algorithms – A comparative case study](https://doi.org/10.1007/BFb0056872). *Parallel Problem Solving from Nature – PPSN V*, Lecture Notes in Computer Science 1498, 292–301.

[4] Fleischer, M. (2003). [The measure of Pareto optima: Applications to multi-objective metaheuristics](https://doi.org/10.1007/3-540-36970-8_37). *Evolutionary Multi-Criterion Optimization – EMO 2003*, Lecture Notes in Computer Science 2632, 519–533.

[5] Auger, A., Bader, J., Brockhoff, D., and Zitzler, E. (2009). [Theory of the hypervolume indicator: Optimal μ-distributions and the choice of the reference point](https://doi.org/10.1145/1527125.1527138). *Proceedings of the Tenth ACM SIGEVO Workshop on Foundations of Genetic Algorithms (FOGA '09)*, 87–102.

[6] Ishibuchi, H., Imada, R., Setoguchi, Y., and Nojima, Y. (2018). [How to specify a reference point in hypervolume calculation for fair performance comparison](https://doi.org/10.1162/evco_a_00226). *Evolutionary Computation*, 26(3), 411–440.

[7] Beume, N., Fonseca, C. M., López-Ibáñez, M., Paquete, L., and Vahrenhold, J. (2009). [On the complexity of computing the hypervolume indicator](https://doi.org/10.1109/TEVC.2009.2015575). *IEEE Transactions on Evolutionary Computation*, 13(5), 1075–1082.

[8] Bringmann, K., and Friedrich, T. (2010). [Approximating the volume of unions and intersections of high-dimensional geometric objects](https://doi.org/10.1016/j.comgeo.2010.03.004). *Computational Geometry*, 43(6–7), 601–610.

[9] Guerreiro, A. P., Fonseca, C. M., and Paquete, L. (2021). [The hypervolume indicator: Computational problems and algorithms](https://doi.org/10.1145/3453474). *ACM Computing Surveys*, 54(6), Article 119.

[10] Bader, J., and Zitzler, E. (2011). [HypE: An algorithm for fast hypervolume-based many-objective optimization](https://doi.org/10.1162/EVCO_a_00009). *Evolutionary Computation*, 19(1), 45–76.
