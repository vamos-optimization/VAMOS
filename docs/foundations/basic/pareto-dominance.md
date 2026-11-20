---
description: 'B2. Pareto dominance, weak dominance and incomparability; the Pareto set and front; non-dominated sets, ideal and nadir points, front shapes and knees, with an interactive dominance explorer.'
search:
  boost: 0.5
---

# B2. Pareto dominance and the Pareto front

Lesson [B1](multi-objective-problems.md) showed that two candidates can be
incomparable. This lesson makes "better" precise for vectors, defines the
Pareto set and front, and introduces the reference points used to normalize
objectives.

In this lesson you will learn:

- the definitions of Pareto dominance, weak dominance and incomparability;
- what Pareto optimality, the Pareto set and the Pareto front are;
- the difference between a non-dominated set and the Pareto front;
- what the ideal and nadir points are and how front shapes differ.

## Comparing objective vectors

Numbers have a total order: for any two of them, one is smaller or they are
equal. Vectors do not. Is (2, 3) better than (3, 2)? It depends on which
objective matters more, and that is a preference, not a fact. Pareto
dominance is the strongest statement that can be made **without** any
preference, and it is named after Vilfredo Pareto, who used the idea in
economics [1].

For two objective vectors **a** and **b** of a minimization problem:

<div class="vf-formula">a ≺ b   (a dominates b)          ⇔   a_i ≤ b_i for every i,  and  a_j &lt; b_j for at least one j
a ⪯ b   (a weakly dominates b)   ⇔   a_i ≤ b_i for every i
a ≺≺ b  (a strictly dominates b) ⇔   a_i &lt; b_i for every i
a ∥ b   (incomparable)           ⇔   neither a ⪯ b nor b ⪯ a</div>

In words: **a** dominates **b** when it is at least as good in every objective
and strictly better in at least one. Dominance is transitive (if a ≺ b and
b ≺ c then a ≺ c) and never holds in both directions, but two vectors may not
be related at all: it is a *partial* order [2, 3].

Geometrically, a point dominates everything in the box above and to its right,
and is dominated by everything in the box below and to its left. The other two
quadrants contain the incomparable points. Explore it below: select a point,
drag it, or focus it and move it with the arrow keys.

<div class="vf-widget" data-vf-widget="dominance"><p class="vf-fallback">Interactive figure: ten points; the selected point shades the region it dominates and the region that dominates it. It needs JavaScript.</p></div>

## Pareto optimality

A feasible decision vector x* is **Pareto optimal** when no feasible x has
F(x) ≺ F(x*): nothing improves an objective without worsening another [2, 4].

- The **Pareto set** PS is the set of all Pareto-optimal decision vectors.
- The **Pareto front** PF = { F(x) : x ∈ PS } is their image in objective space.

Two refinements matter in practice:

- **Weak Pareto optimality.** x* is *weakly* Pareto optimal when no x
  strictly dominates it. Such a point may still be improved in one objective
  without loss in the others. A flat stretch at the end of a front contains
  weakly Pareto-optimal points.
- **Local Pareto optimality.** x* is *locally* Pareto optimal when no
  dominating vector exists *in a neighbourhood* of x*. Many benchmark problems
  have local fronts that trap algorithms: in ZDT4, for example, 21⁹ of them [5].

## Non-dominated sets are not the Pareto front

An algorithm only ever sees the finite set of candidates it has evaluated. The
members of that set that no other member dominates form its **non-dominated
set**. A set whose members are mutually non-dominated is called an
**approximation set** [6]. It is the best that run found, but it is the Pareto
front only if the run found truly Pareto-optimal vectors. Keep the two apart:
"non-dominated" is relative to a set, "Pareto optimal" is relative to the
whole problem.

## Ideal and nadir points

Two reference vectors summarize the range of the front:

- the **ideal point** z* has, in each objective, the best value over the Pareto
  front: z*_i = min { f_i(x) : x ∈ PS };
- the **nadir point** z^nad has, in each objective, the worst value over the
  Pareto front: z^nad_i = max { f_i(x) : x ∈ PS }.

The ideal point is usually not attainable (otherwise the objectives would not
conflict). Together they define the box that contains the front, and they are
used to **normalize** objectives measured in different units:

<div class="vf-formula">f̃_i(x) = ( f_i(x) − z*_i ) / ( z^nad_i − z*_i )</div>

With two objectives, the nadir point is given by the two extremes of the front.
With three or more, it is hard to estimate because extreme points of the
non-dominated set can come from dominated regions of the true front [7].
Enable the option in the figure above to see the ideal and nadir points of the
current non-dominated set.

## Shapes of Pareto fronts

The geometry of the front affects which methods work:

- **Convex** fronts bulge towards the ideal point (ZDT1). Every point can be
  found with a weighted sum.
- **Non-convex** (concave) fronts bulge away from it (ZDT2). Weighted sums only
  find their extremes, as you saw in [B1](multi-objective-problems.md#turning-many-objectives-into-one-scalarization).
- **Disconnected** fronts have gaps (ZDT3). Algorithms must keep several
  separate groups of solutions.
- **Degenerate** fronts have a lower dimension than m − 1, for example a curve
  in a three-objective space.

The ZDT problems were designed by Zitzler, Deb and Thiele to isolate these
features with two objectives [5]; the DTLZ problems scale them to any number of
objectives [8]. The [problem reference](../../reference/problems.md) lists
every benchmark that VAMOS ships.

## Choosing one solution: knees

When the decision maker has no strong preference, a common choice is a
**knee**: a point where a small gain in one objective costs a large loss in
another, so moving away from it in any direction is a poor bargain [9].
[T03](../../tutorials/t03-pareto-decision.md) selects a solution from a real
VAMOS result with explicit preferences.

## In VAMOS

Pareto dominance takes two lines of NumPy. The non-dominated subset of a
finite set follows directly:

```python
import numpy as np


def dominates(a, b):
    """True when a Pareto-dominates b; every objective is minimized."""
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


F = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 4.0], [4.0, 1.0], [2.5, 2.5]])
print(dominates(F[1], F[2]))  # (2, 3) is better than (3, 4) in both objectives
print(dominates(F[0], F[3]))  # (1, 5) and (4, 1) are incomparable
non_dominated = [i for i in range(len(F)) if not any(dominates(F[j], F[i]) for j in range(len(F)))]
print(non_dominated)
```

Output:

```text
True
False
[0, 1, 3, 4]
```

A VAMOS result does this for you. `OptimizationResult.front()` returns the
non-dominated rows of `result.F`, and their indices when you also need the
decision vectors:

```python
from vamos import optimize

result = optimize("zdt1", algorithm="nsgaii", pop_size=40, max_evaluations=400, engine="numpy", seed=42)
front_F, front_indices = result.front(return_indices=True)
print("returned:", result.F.shape, "non-dominated:", front_F.shape)
print("decisions of the non-dominated rows:", result.X[front_indices].shape)
```

Output:

```text
returned: (40, 2) non-dominated: (12, 2)
decisions of the non-dominated rows: (12, 30)
```

After only 400 evaluations this non-dominated set is still far from the ZDT1
front. [Understanding results](../../guide/understanding-results.md#separate-the-returned-set-from-its-non-dominated-subset)
draws it and explains which set `result.F` holds.

## Key points

- a ≺ b when a is no worse in every objective and better in at least one.
  Many pairs are incomparable: dominance is a partial order.
- The Pareto set holds the decisions that nothing dominates; the Pareto front
  holds their objective vectors.
- A non-dominated set is relative to the points evaluated; it approximates the
  front but is not the front.
- The ideal and nadir points bound the front and are used to normalize the
  objectives.

**Next:** [B3. What makes a good approximation?](quality-aspects.md)

## References

[1] Pareto, V. (1896). *Cours d'économie politique*. F. Rouge, Lausanne.

[2] Miettinen, K. (1999). [*Nonlinear Multiobjective Optimization*](https://doi.org/10.1007/978-1-4615-5563-6). Kluwer Academic Publishers.

[3] Ehrgott, M. (2005). [*Multicriteria Optimization*](https://doi.org/10.1007/3-540-27659-9) (2nd ed.). Springer.

[4] Deb, K. (2001). *Multi-Objective Optimization Using Evolutionary Algorithms*. Wiley.

[5] Zitzler, E., Deb, K., and Thiele, L. (2000). [Comparison of multiobjective evolutionary algorithms: Empirical results](https://doi.org/10.1162/106365600568202). *Evolutionary Computation*, 8(2), 173–195.

[6] Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C. M., and Grunert da Fonseca, V. (2003). [Performance assessment of multiobjective optimizers: An analysis and review](https://doi.org/10.1109/TEVC.2003.810758). *IEEE Transactions on Evolutionary Computation*, 7(2), 117–132.

[7] Deb, K., Miettinen, K., and Chaudhuri, S. (2010). [Toward an estimation of nadir objective vector using a hybrid of evolutionary and local search approaches](https://doi.org/10.1109/TEVC.2010.2041667). *IEEE Transactions on Evolutionary Computation*, 14(6), 821–841.

[8] Deb, K., Thiele, L., Laumanns, M., and Zitzler, E. (2005). [Scalable test problems for evolutionary multiobjective optimization](https://doi.org/10.1007/1-84628-137-7_6). In A. Abraham, L. Jain and R. Goldberg (Eds.), *Evolutionary Multiobjective Optimization*, 105–145. Springer.

[9] Branke, J., Deb, K., Dierolf, H., and Osswald, M. (2004). [Finding knees in multi-objective optimization](https://doi.org/10.1007/978-3-540-30217-9_73). *Parallel Problem Solving from Nature – PPSN VIII*, Lecture Notes in Computer Science 3242, 722–731.
