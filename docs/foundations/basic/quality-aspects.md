---
description: 'B3. The quality aspects of an approximation set (convergence, spread, uniformity and cardinality), how sets are compared by dominance, and why no single number can say everything.'
search:
  boost: 0.5
---

# B3. What makes a good approximation?

An algorithm returns an approximation set, not the Pareto front. Two runs, or
two algorithms, return different sets. Before measuring them with numbers
(lesson [B4](quality-indicators.md)), we need to agree on **what** a good set
looks like.

In this lesson you will learn:

- the four quality aspects of an approximation set;
- how two sets are compared using dominance alone, and why that is often
  inconclusive;
- why one number cannot capture every aspect.

## The four quality aspects

Zitzler, Deb and Thiele [1] stated the goals of an a posteriori algorithm as
three: a small distance to the Pareto front, a good (in most cases uniform)
distribution of the solutions, and a large extent of the front covered. The
survey of Li and Yao [2], which reviews about one hundred indicators, groups
the properties of a set into four aspects:

| Aspect | The question it answers | A set that fails it |
| --- | --- | --- |
| **Convergence** | How close are the points to the Pareto front? | A perfectly spread set far from the front. |
| **Spread** (extent, coverage) | How much of the front's range does the set cover? | A set clustered around one end of the front. |
| **Uniformity** (evenness) | Are the points evenly distributed along what they cover? | Points bunched together with large gaps between groups. |
| **Cardinality** | How many non-dominated points does the set offer? | Two excellent points where the decision maker needed choices. |

Spread and uniformity together are often called **diversity**. Convergence
and diversity pull in different directions: an algorithm that pushes hard
towards the front tends to lose diversity, and one that preserves diversity
spends evaluations away from the front. NSGA-II balances them with two
criteria, ranks and crowding distances (lessons
[I2](../intermediate/non-dominated-sorting.md) and
[I3](../intermediate/crowding-distance.md)).

## See the aspects one at a time

The figure builds an approximation set of the ZDT1 front with four sliders,
one for each aspect: the distance added to every point, the share of the
front's range covered, how strongly the points cluster towards one end, and
how many points there are. The table shows how a few indicators react; B4
defines them.

<div class="vf-widget" data-vf-widget="quality" data-indicators="hv,gd,igd,spacing,n"><p class="vf-fallback">Interactive figure: an approximation set whose convergence, extent, clustering and size are controlled by sliders, with indicator values. It needs JavaScript.</p></div>

Things to try:

1. Set the **convergence gap** to 0 and the **extent** to 20%. GD stays near
   zero, because every point lies on the front, yet four fifths of the front
   are missing. IGD and the hypervolume notice it.
2. Set the extent back to 100% and increase **clustering**. GD barely moves,
   spacing and IGD grow, and the hypervolume drops slightly.
3. Reduce the **points** to 2. The hypervolume collapses and IGD grows
   fivefold, GD hardly changes, and spacing reports a perfect 0, because two
   points are always evenly spaced. Indicators that each see one aspect can
   disagree completely.

## Comparing two sets by dominance

Before choosing an indicator, note what dominance alone can say about two
approximation sets A and B [3]:

<div class="vf-formula">A ≺≺ B   every b ∈ B is strictly dominated by some a ∈ A
A ≺ B    every b ∈ B is dominated by some a ∈ A
A ⪯ B    A weakly dominates B: every b ∈ B is weakly dominated by some a ∈ A
A ◁ B    A is better than B: A ⪯ B, but not B ⪯ A
A ∥ B    incomparable: neither A ⪯ B nor B ⪯ A</div>

When A ◁ B, A is better than B under **every** preference a decision maker
could hold. That is the only conclusion free of assumptions, but it is rare:
two good algorithms usually produce incomparable sets, each better in some
region of the front. To go further we must add preferences, for example "I
care about the whole front equally", and that is what a quality indicator
encodes.

## Why one number cannot say everything

A *unary* quality indicator maps one approximation set to one real number.
Zitzler et al. [3] proved that no unary indicator, and no finite combination
of them, can in general detect whether A is better than B; the best an
indicator can guarantee is that a better value means "not worse". Any single
number is therefore a summary that weights the aspects in its own way. Three
practical consequences:

- **Choose indicators for the question.** Convergence-only measures suit a
  question about convergence; a decision maker who needs the whole front needs
  an indicator that rewards spread.
- **Use more than one indicator**, and check that their conclusions agree.
- **Look at the sets.** For two objectives, a plot of the sets, or of their
  attainment surfaces (lesson [A10](../advanced/performance-assessment.md)),
  often shows what the numbers hide.

## Scale matters

Every aspect depends on the units of the objectives. If f₁ is in metres and f₂
in millimetres, distances are dominated by f₂ and "uniform" means uniform in
millimetres. Normalize the objectives with fixed bounds, for example the ideal
and nadir points of a reference front (lesson
[B2](pareto-dominance.md#ideal-and-nadir-points)), **before** computing
distances, and use the same bounds for every algorithm you compare.

## In VAMOS

The returned set has a cardinality you can read directly, and its quality
aspects can be inspected against a known front. For ZDT1 the front is
f₂ = 1 − √f₁, so the vertical gap of each point is a direct convergence check,
and the covered range of f₁ measures the spread:

```python
import numpy as np

from vamos import optimize

result = optimize("zdt1", algorithm="nsgaii", pop_size=100, max_evaluations=10_000, engine="numpy", seed=7)
front = result.front()
f1, f2 = front[:, 0], front[:, 1]
gap = f2 - (1.0 - np.sqrt(f1))
spacing = np.diff(np.sort(f1))

print("cardinality:", len(front))
print("convergence, mean vertical gap:", round(float(gap.mean()), 4))
print("spread, f1 covered:", round(float(f1.min()), 3), "to", round(float(f1.max()), 3))
print("uniformity, largest / median gap in f1:", round(float(spacing.max() / np.median(spacing)), 1))
```

Output:

```text
cardinality: 100
convergence, mean vertical gap: 0.0261
spread, f1 covered: 0.0 to 0.995
uniformity, largest / median gap in f1: 4.2
```

These ad hoc measures are useful for intuition on a problem whose front is
known. The next lesson introduces the standard indicators, which work on any
problem given a reference point or reference set.

## Key points

- Judge an approximation set on convergence, spread, uniformity and
  cardinality; diversity is spread plus uniformity.
- Dominance between sets (A ◁ B) is preference-free but often inconclusive.
- No single unary indicator can establish that one set is better than another;
  use several, chosen for the question, and look at the sets.
- Normalize objectives with fixed bounds before measuring distances.

**Next:** [B4. Quality indicators](quality-indicators.md).

## References

[1] Zitzler, E., Deb, K., and Thiele, L. (2000). [Comparison of multiobjective evolutionary algorithms: Empirical results](https://doi.org/10.1162/106365600568202). *Evolutionary Computation*, 8(2), 173–195.

[2] Li, M., and Yao, X. (2019). [Quality evaluation of solution sets in multiobjective optimisation: A survey](https://doi.org/10.1145/3300148). *ACM Computing Surveys*, 52(2), Article 26.

[3] Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C. M., and Grunert da Fonseca, V. (2003). [Performance assessment of multiobjective optimizers: An analysis and review](https://doi.org/10.1109/TEVC.2003.810758). *IEEE Transactions on Evolutionary Computation*, 7(2), 117–132.
