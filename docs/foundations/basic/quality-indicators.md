---
description: 'B4. Quality indicators for approximation sets: hypervolume, GD, IGD, IGD+, additive epsilon, spacing and spread; Pareto compliance, reference points and reference sets, with interactive figures and MooCore code on a VAMOS result.'
search:
  boost: 0.5
---

# B4. Quality indicators

A quality indicator turns an approximation set into a number, so that runs
and algorithms can be compared. Lesson [B3](quality-aspects.md) showed that
any single number weights the quality aspects in its own way. This lesson
defines the indicators in common use, shows what each one sees, and explains
the property that matters most when choosing one: Pareto compliance.

In this lesson you will learn:

- the definitions of hypervolume, GD, IGD, IGD+ and the additive ε-indicator;
- what Pareto compliance is, and which indicators have it;
- how reference points and reference sets change the results;
- how to compute the indicators for a VAMOS result.

## What an indicator needs

Every indicator below takes an approximation set A of *m*-dimensional
objective vectors, all minimized, plus reference data fixed **in advance**:

| Indicator | Reference data | Better | Pareto compliant | Introduced by |
| --- | --- | --- | --- | --- |
| Hypervolume (HV) | a reference point r | higher | yes, strictly | Zitzler and Thiele [1, 2] |
| Generational distance (GD) | a reference set R | lower | no | Van Veldhuizen and Lamont [3] |
| Inverted generational distance (IGD) | a reference set R | lower | no | Coello Coello and Reyes Sierra [4, 5] |
| IGD+ | a reference set R | lower | weakly | Ishibuchi et al. [6] |
| Additive ε-indicator (ε+) | a reference set R | lower | weakly | Zitzler et al. [7] |
| Spacing | none | lower | no | Schott [8] |
| Spread Δ | the extremes of the front | lower | no | Deb et al. [9] |

The reference set R is usually a dense sample of the true Pareto front, which
benchmark problems provide. For a real problem, the non-dominated union of all
the sets being compared is a common substitute [10].

## Pareto compliance

An indicator I (where higher is better) is

- **strictly Pareto compliant** if A ◁ B implies I(A) > I(B): a better set always
  scores strictly better;
- **weakly Pareto compliant** if A ⪯ B implies I(A) ≥ I(B): a better set never
  scores worse.

(For indicators where lower is better, reverse the inequalities.) An indicator
that is not compliant can rank a set that dominates another one **below** it.
That is not a theoretical curiosity; the example further down shows it with
two points.

## Hypervolume

The hypervolume of A is the size of the region of objective space that A
dominates and that is bounded by the reference point r [1, 2]:

<div class="vf-formula">HV(A; r) = λ( ⋃_{a ∈ A, a ≺ r}  [a₁, r₁] × [a₂, r₂] × … × [a_m, r_m] )</div>

where λ is the Lebesgue measure: area with two objectives, volume with three.
Each point contributes the box between itself and r; overlapping parts count
once. Drag the five points or the reference point r:

<div class="vf-widget" data-vf-widget="hypervolume"><p class="vf-fallback">Interactive figure: the hypervolume of five draggable points and the exclusive contribution of each point. It needs JavaScript.</p></div>

The hypervolume has unique strengths:

- It is the only commonly used unary indicator known to be **strictly Pareto
  compliant** [7]. A set of a given size that maximizes it contains only
  Pareto-optimal points [11].
- It needs no reference set, only a reference point, so it applies to problems
  whose front is unknown.
- It rewards all four aspects at once: convergence, spread, uniformity and
  cardinality.

And real costs:

- **Computation** is fast for two and three objectives (O(n log n) for n points)
  but grows exponentially with the number of objectives; the exact problem is
  #P-hard [12, 13]. Guerreiro, Fonseca and Paquete survey the algorithms [14].
- **The reference point changes the ranking.** Points beyond r contribute
  nothing, and r controls how much the extremes are worth. After normalizing
  each objective to [0, 1], r = (1.1, …, 1.1) is a common choice; Auger et al.
  [15] and Ishibuchi et al. [16] analyse the bias that the choice introduces.
- **Scale matters.** Normalize the objectives with fixed bounds first.

The area that only one point dominates is its **exclusive contribution**:
the hypervolume lost if that point is removed. SMS-EMOA builds its selection on
it (lesson [I6](../intermediate/beyond-nsga-ii.md)).

## Distance-based indicators

The other indicators measure distances between A and a reference set R. With
‖·‖ the Euclidean distance:

<div class="vf-formula">GD(A, R)   = (1 / |A|) · Σ_{a ∈ A}  min_{r ∈ R} ‖a − r‖
IGD(A, R)  = (1 / |R|) · Σ_{r ∈ R}  min_{a ∈ A} ‖a − r‖
IGD+(A, R) = (1 / |R|) · Σ_{r ∈ R}  min_{a ∈ A} d⁺(a, r),    d⁺(a, r) = √( Σ_i max(a_i − r_i, 0)² )
ε+(A, R)   = max_{r ∈ R}  min_{a ∈ A}  max_i ( a_i − r_i )</div>

- **GD** looks from A to the front: it only measures convergence. A single
  well-placed point scores perfectly.
- **IGD** looks from the front to A: a reference point far from every point of
  A raises it, so it also measures spread and uniformity.
- **IGD+** is IGD with a modified distance d⁺ that only counts the objectives in
  which a is **worse** than r. That change makes it weakly Pareto compliant [6].
- **ε+** is the smallest amount by which A must be translated towards the
  origin so that every reference point is weakly dominated. It is a
  worst-case measure: one uncovered region decides its value [7].

GD and IGD appear in the literature with an exponent p, as
(Σ dᵖ)^(1/p) / n. Van Veldhuizen and Lamont used p = 2 [3]; the mean distance
shown here (p = 1) is the version of MooCore and of
[VAMOS' indicator guide](../../dev/add_metric.md#indicator-contract). Always
state which one you report. The averaged Hausdorff distance,
Δ_p = max(GD_p, IGD_p), combines both directions [17].

Select an indicator and drag the points of A to see which distances it uses:

<div class="vf-widget" data-vf-widget="distance-indicators"><p class="vf-fallback">Interactive figure: GD, IGD, IGD+ and additive epsilon drawn as distances between a draggable approximation set and a reference set. It needs JavaScript.</p></div>

## Pareto compliance in action

Take a reference set with the two extremes of a linear front, and two
single-point sets. A = {(0, 0.6)} dominates B = {(0.5, 0.6)}: it is better in f₁
and equal in f₂. A compliant indicator must not prefer B.

```python
import moocore
import numpy as np

reference_set = np.array([[0.0, 1.0], [1.0, 0.0]])
A = np.array([[0.0, 0.6]])
B = np.array([[0.5, 0.6]])  # A dominates B: better f1, equal f2

for name, approximation in (("A", A), ("B", B)):
    igd = moocore.igd(approximation, ref=reference_set)
    igd_plus = moocore.igd_plus(approximation, ref=reference_set)
    epsilon = moocore.epsilon_additive(approximation, ref=reference_set)
    hv = moocore.hypervolume(approximation, ref=[1.1, 1.1])
    print(f"{name}: IGD {igd:.3f}  IGD+ {igd_plus:.3f}  eps+ {epsilon:.3f}  HV {hv:.3f}")
```

Output:

```text
A: IGD 0.783  IGD+ 0.300  eps+ 0.600  HV 0.550
B: IGD 0.711  IGD+ 0.550  eps+ 0.600  HV 0.300
```

IGD prefers B, the dominated set, because B happens to lie closer to the
reference point (1, 0). IGD+ and the hypervolume prefer A. The additive ε gives
both the same value: weak compliance allows a tie, never a reversal.

## Diversity-only indicators

Two older indicators measure the distribution of A and nothing else:

<div class="vf-formula">Spacing(A) = √( 1/(|A| − 1) · Σ_i ( d̄ − d_i )² ),   d_i = min_{j ≠ i} Σ_k | a_{i,k} − a_{j,k} |
Δ(A)       = ( d_f + d_l + Σ_{i=1}^{N−1} | d_i − d̄ | ) / ( d_f + d_l + (N − 1) · d̄ )</div>

Schott's **spacing** [8] is the standard deviation of the nearest-neighbour
distances: 0 for a perfectly even set. Deb's **spread** Δ [9], for two
objectives, uses the N − 1 distances d_i between consecutive points and the
distances d_f and d_l from the extremes of the true front to the extremes of
A: it penalizes both uneven gaps and missing ends. Neither looks at
convergence, so both must be paired with another indicator.

## All the indicators side by side

The playground of B3 with every indicator of this lesson. Watch which values
move when you change a single aspect.

<div class="vf-widget" data-vf-widget="quality"><p class="vf-fallback">Interactive figure: an approximation set whose convergence, extent, clustering and size are controlled by sliders, with the values of eight indicators. It needs JavaScript.</p></div>

## Choosing and using indicators

1. **Fix the reference data before running anything**: the reference point,
   the reference set, and the normalization bounds. Choosing them after
   seeing the results invites bias.
2. **Normalize** every objective with the same fixed bounds for all the sets
   compared, for example the ideal and nadir points of the reference front.
3. **Prefer compliant indicators**: the hypervolume, IGD+ or ε+. Use GD,
   spacing or spread only as complements that explain *why* a set is good or bad.
4. **Report more than one** indicator, its direction, and its exact variant
   (for example IGD with p = 1).
5. **Measure only feasible, non-dominated points** for a constrained problem,
   and report the share of feasible runs separately (lesson
   [A5](../advanced/constrained.md)).

## In VAMOS

Compute the indicators of a VAMOS result with MooCore, using the analytical
ZDT1 front as reference set, as the
[quality and budget tutorial](../../tutorials/t07-quality-and-budget.md) does
for saved studies. ZDT1 is already scaled to [0, 1] in both objectives, so no
further normalization is needed here.

```python
import moocore
import numpy as np

from vamos import optimize

result = optimize("zdt1", algorithm="nsgaii", pop_size=100, max_evaluations=10_000, engine="numpy", seed=7)
front = result.front()

f1 = np.linspace(0.0, 1.0, 1001)
reference_set = np.column_stack([f1, 1.0 - np.sqrt(f1)])  # the ZDT1 Pareto front
reference_point = [1.1, 1.1]

print("HV  ", round(moocore.hypervolume(front, ref=reference_point), 4))
print("HV of the reference front", round(moocore.hypervolume(reference_set, ref=reference_point), 4))
print("IGD ", round(moocore.igd(front, ref=reference_set), 4))
print("IGD+", round(moocore.igd_plus(front, ref=reference_set), 4))
print("eps+", round(moocore.epsilon_additive(front, ref=reference_set), 4))
```

Output:

```text
HV   0.8459
HV of the reference front 0.8762
IGD  0.0177
IGD+ 0.0176
eps+ 0.0241
```

The exact front has a hypervolume of 1.21 − 1/3 ≈ 0.8767 for this reference
point; the 1,001-point sample of it reaches 0.8762. This run therefore attains
about 96.5% of the best possible value after 10,000 evaluations. Durable VAMOS studies record the
same indicators for every run: see the
[study lifecycle guide](../../guide/studies.md) and
[T07](../../tutorials/t07-quality-and-budget.md).

## Key points

- The hypervolume is strictly Pareto compliant and needs only a reference
  point, but its value depends on that point and on the scaling.
- IGD+ and ε+ are weakly compliant; GD and IGD are not, and can prefer a
  dominated set.
- Fix reference points, reference sets and normalization bounds before
  running, and report the exact variant of each indicator.
- Spacing and spread describe the distribution only; pair them with a
  convergence-aware indicator.

**Next level:** [I1. NSGA-II at a glance](../intermediate/nsga-ii.md).

## References

[1] Zitzler, E., and Thiele, L. (1998). [Multiobjective optimization using evolutionary algorithms – A comparative case study](https://doi.org/10.1007/BFb0056872). *Parallel Problem Solving from Nature – PPSN V*, Lecture Notes in Computer Science 1498, 292–301.

[2] Zitzler, E., and Thiele, L. (1999). [Multiobjective evolutionary algorithms: A comparative case study and the strength Pareto approach](https://doi.org/10.1109/4235.797969). *IEEE Transactions on Evolutionary Computation*, 3(4), 257–271.

[3] Van Veldhuizen, D. A., and Lamont, G. B. (2000). [On measuring multiobjective evolutionary algorithm performance](https://doi.org/10.1109/CEC.2000.870296). *Proceedings of the 2000 Congress on Evolutionary Computation*, 204–211.

[4] Coello Coello, C. A., and Reyes Sierra, M. (2004). [A study of the parallelization of a coevolutionary multi-objective evolutionary algorithm](https://doi.org/10.1007/978-3-540-24694-7_71). *MICAI 2004: Advances in Artificial Intelligence*, Lecture Notes in Computer Science 2972, 688–697.

[5] Coello Coello, C. A., and Cruz Cortés, N. (2005). [Solving multiobjective optimization problems using an artificial immune system](https://doi.org/10.1007/s10710-005-6164-x). *Genetic Programming and Evolvable Machines*, 6(2), 163–190.

[6] Ishibuchi, H., Masuda, H., Tanigaki, Y., and Nojima, Y. (2015). [Modified distance calculation in generational distance and inverted generational distance](https://doi.org/10.1007/978-3-319-15892-1_8). *Evolutionary Multi-Criterion Optimization – EMO 2015*, Lecture Notes in Computer Science 9019, 110–125.

[7] Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C. M., and Grunert da Fonseca, V. (2003). [Performance assessment of multiobjective optimizers: An analysis and review](https://doi.org/10.1109/TEVC.2003.810758). *IEEE Transactions on Evolutionary Computation*, 7(2), 117–132.

[8] Schott, J. R. (1995). *Fault tolerant design using single and multicriteria genetic algorithm optimization*. Master's thesis, Massachusetts Institute of Technology.

[9] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://doi.org/10.1109/4235.996017). *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

[10] Knowles, J., Thiele, L., and Zitzler, E. (2006). *A tutorial on the performance assessment of stochastic multiobjective optimizers*. TIK Report 214, Computer Engineering and Networks Laboratory, ETH Zurich.

[11] Fleischer, M. (2003). [The measure of Pareto optima: Applications to multi-objective metaheuristics](https://doi.org/10.1007/3-540-36970-8_37). *Evolutionary Multi-Criterion Optimization – EMO 2003*, Lecture Notes in Computer Science 2632, 519–533.

[12] Beume, N., Fonseca, C. M., López-Ibáñez, M., Paquete, L., and Vahrenhold, J. (2009). [On the complexity of computing the hypervolume indicator](https://doi.org/10.1109/TEVC.2009.2015575). *IEEE Transactions on Evolutionary Computation*, 13(5), 1075–1082.

[13] Bringmann, K., and Friedrich, T. (2010). [Approximating the volume of unions and intersections of high-dimensional geometric objects](https://doi.org/10.1016/j.comgeo.2010.03.004). *Computational Geometry*, 43(6–7), 601–610.

[14] Guerreiro, A. P., Fonseca, C. M., and Paquete, L. (2021). [The hypervolume indicator: Computational problems and algorithms](https://doi.org/10.1145/3453474). *ACM Computing Surveys*, 54(6), Article 119.

[15] Auger, A., Bader, J., Brockhoff, D., and Zitzler, E. (2009). [Theory of the hypervolume indicator: Optimal μ-distributions and the choice of the reference point](https://doi.org/10.1145/1527125.1527138). *Proceedings of the Tenth ACM SIGEVO Workshop on Foundations of Genetic Algorithms (FOGA '09)*, 87–102.

[16] Ishibuchi, H., Imada, R., Setoguchi, Y., and Nojima, Y. (2018). [How to specify a reference point in hypervolume calculation for fair performance comparison](https://doi.org/10.1162/evco_a_00226). *Evolutionary Computation*, 26(3), 411–440.

[17] Schütze, O., Esquivel, X., Lara, A., and Coello Coello, C. A. (2012). [Using the averaged Hausdorff distance as a performance measure in evolutionary multiobjective optimization](https://doi.org/10.1109/TEVC.2011.2161872). *IEEE Transactions on Evolutionary Computation*, 16(4), 504–522.
