---
description: 'Visualizing many objectives: why scatter plots fail beyond three objectives, scatter-plot matrices, parallel coordinates with axis order, normalization and brushing, heatmaps, radar charts and projections, what each one hides, and the plotting helpers of VAMOS, with an interactive figure.'
search:
  boost: 0.5
---

# Visualizing many objectives

A run with m objectives returns a matrix with one row per solution and m
columns. With two objectives a scatter plot shows all of it; with three, a
rotating 3-D plot nearly does. With four or more no picture shows everything,
and every method keeps some information and hides the rest. Lesson
[Many objectives](many-objective.md#other-difficulties) listed visualization
among the difficulties of many objectives; this lesson goes through the
methods and says what each one hides.

In this lesson you will learn:

- why scatter plots of two objectives mislead beyond three objectives;
- how to read scatter-plot matrices and parallel coordinates, and why the axis
  order, the normalization and brushing matter;
- what heatmaps and radar charts add;
- what projections and objective reduction hide;
- which of these VAMOS draws for you.

## Why scatter plots stop at three

The examples of this lesson use points of the front of DTLZ2 [9] with five
objectives, which lies on the unit sphere. None of them dominates another.
Plot two of the five objectives, though, and almost every point appears to be
dominated: it is better in some objective that the plot does not show.

```python
from itertools import combinations

import numpy as np

rng = np.random.default_rng(1)
n, m = 200, 5
x = rng.random((n, m - 1))
F = np.ones((n, m))  # 200 points of DTLZ2's front with five objectives
for i in range(m):
    F[:, i] = np.prod(np.cos(x[:, : m - 1 - i] * np.pi / 2), axis=1)
    if i > 0:
        F[:, i] *= np.sin(x[:, m - 1 - i] * np.pi / 2)


def dominated(P):
    """Whether another row of P dominates each row (minimization)."""
    no_worse = np.all(P[:, None] <= P[None, :], axis=2)
    better = np.any(P[:, None] < P[None, :], axis=2)
    return np.any(no_worse & better, axis=0)


print(f"dominated in all five objectives: {dominated(F).sum()} of {n}")
for a, b in combinations(range(m), 2):
    print(f"dominated in the plot of f{a + 1} and f{b + 1}: {dominated(F[:, [a, b]]).sum()} of {n}")
```

Output:

```text
dominated in all five objectives: 0 of 200
dominated in the plot of f1 and f2: 197 of 200
dominated in the plot of f1 and f3: 199 of 200
dominated in the plot of f1 and f4: 197 of 200
dominated in the plot of f1 and f5: 195 of 200
dominated in the plot of f2 and f3: 197 of 200
dominated in the plot of f2 and f4: 198 of 200
dominated in the plot of f2 and f5: 196 of 200
dominated in the plot of f3 and f4: 198 of 200
dominated in the plot of f3 and f5: 189 of 200
dominated in the plot of f4 and f5: 191 of 200
```

A plot of two objectives is a projection: it keeps their values exactly and
drops the others, and with them the reason why a point that looks dominated is
not. VAMOS draws fronts of two or three objectives with
`plot_pareto_front_2d`, `plot_pareto_front_3d` and `plot_result_front` from
`vamos.ux.api`; `plot_result_front` refuses more than three objectives.

## Scatter-plot matrices

A scatter-plot matrix draws every pair of objectives in a grid, m(m − 1)/2
distinct panels: 10 for five objectives, 45 for ten. Each panel shows the
trade-off between two objectives: points along a falling line are in conflict,
points along a rising line move together. The matrix shows every pair, but
each panel is still the projection of the example above, and no panel shows how
three objectives interact. Beyond three objectives, VAMOS Studio overlays the
fronts of several algorithms as a scatter-plot matrix on its Results and
Compare pages.

Selecting points in one panel and highlighting the same points in the others,
**brushing**, links the panels; Becker and Cleveland [3] introduced it for
scatter-plot matrices.

## Parallel coordinates

Inselberg's parallel coordinates [1] draw m vertical axes side by side and each
solution as a polyline that crosses axis i at the height of f_i. Any number of
objectives fits, and a few rules make the picture readable [4]:

- **Dominance is visible.** If every axis increases with its objective, a
  solution dominates another when its line is nowhere above the other's and
  somewhere below. The lines of two mutually non-dominated solutions therefore
  cross, or touch, somewhere.
- **Neighbours tell the trade-off.** Between two neighbouring axes, conflicting
  objectives produce many crossing lines, objectives that move together
  produce nearly parallel lines.
- **Only neighbours are compared.** An order of m axes shows m − 1 of the
  m(m − 1)/2 pairs, so seeing every pair takes at least ⌈m/2⌉ orders. Wegman
  [2] gave a systematic construction of such orders.

### Axis order

The number of crossings between neighbouring axes measures how much the
neighbours conflict. The example counts them for every pair of the five DTLZ2
objectives and compares three orders: the one with the fewest crossings, the
natural order and the one with the most.

```python
from itertools import combinations, permutations

import numpy as np

rng = np.random.default_rng(1)
n, m = 200, 5
x = rng.random((n, m - 1))
F = np.ones((n, m))  # the same 200 points of DTLZ2's front
for i in range(m):
    F[:, i] = np.prod(np.cos(x[:, : m - 1 - i] * np.pi / 2), axis=1)
    if i > 0:
        F[:, i] *= np.sin(x[:, m - 1 - i] * np.pi / 2)


def crossings(a, b):
    """Pairs of solutions whose lines cross between the axes of f_a and f_b."""
    da = F[:, None, a] - F[None, :, a]
    db = F[:, None, b] - F[None, :, b]
    return int(np.sum(da * db < 0)) // 2


pairs = {(a, b): crossings(a, b) for a, b in combinations(range(m), 2)}


def total(order):
    return sum(pairs[tuple(sorted(pair))] for pair in zip(order, order[1:]))


orders = sorted({min(o, o[::-1]) for o in permutations(range(m))}, key=total)  # an order and its reverse look alike
print("pairs of solutions:", n * (n - 1) // 2)
for label, order in (("fewest", orders[0]), ("natural", tuple(range(m))), ("most", orders[-1])):
    print(f"{label:<8} {', '.join(f'f{i + 1}' for i in order)}: {total(order)} crossings")
```

Output:

```text
pairs of solutions: 19900
fewest   f4, f3, f2, f1, f5: 35778 crossings
natural  f1, f2, f3, f4, f5: 37778 crossings
most     f1, f5, f3, f4, f2: 46762 crossings
```

The front is the same in the three orders; only what the eye compares changes.
Orders that put objectives in conflict next to each other show the trade-offs;
orders that put similar objectives together show groups of objectives.

### Normalization

Each axis needs a scale. Drawing every axis on its own range, from the
smallest to the largest value in the set, uses the full height for each
objective; drawing all axes on one scale keeps their units comparable but
squeezes objectives with small ranges. VAMOS's `plot_parallel_coordinates`
takes the first choice: it rescales each column from its minimum to its
maximum in the rows it draws, so every axis runs from 0 to 1, and an objective
without spread gets a range of 1.

The scale depends on the set. Two sets drawn in two calls get two different
scales, even when they were normalized with the same bounds beforehand, because
the helper rescales every matrix to its own minima and maxima again. To compare
sets on one scale, stack them into one matrix and draw it in one call, or draw
the lines yourself on common bounds, such as the ideal and nadir points of a
reference front.

### Brushing

A brush selects an interval on one axis; the lines that pass through it stay
highlighted and the others fade. Brushes on several axes combine, and linked to
a scatter-plot matrix they show where the selected solutions lie in every pair
of objectives. This is how a decision maker narrows a front down: "f₅ below
this value", and then sees what that costs in the other objectives.

### Try it

The figure shows 60 solutions of DTLZ2's front with five objectives. Drag an
axis label sideways to move the axis, or drag along an axis to brush it. With
the keyboard, focus an axis with <kbd>Tab</kbd>: <kbd>←</kbd> and <kbd>→</kbd>
move it, <kbd>↑</kbd> and <kbd>↓</kbd> create and move its brush,
<kbd>Shift</kbd> with <kbd>↑</kbd> and <kbd>↓</kbd> resizes the brush, and
<kbd>Esc</kbd> clears it.

1. In the natural order, f₄ and f₅ cross most: 1,229 of the 1,770 pairs of
   lines, 3,258 crossings in all. Move f₅ to the far left: f₅ and f₁ cross
   less, and the total drops to 3,011. The scatter-plot matrix outlines the
   pairs that are neighbours in the current order.
2. Focus the f₅ axis and press <kbd>↓</kbd>: the brush keeps the bottom quarter
   of f₅, 10 solutions. Their f₃ and f₄ are larger than those of the others: a
   small f₅ is paid for elsewhere.
3. Choose the scaled objectives (f_i × 2i, as in WFG) and one scale for every
   axis: f₁, whose range is a fifth of f₅'s, is squeezed at the bottom.

<div class="vf-widget" data-vf-widget="parallel-coordinates"><p class="vf-fallback">Interactive figure: parallel coordinates of 60 solutions of a five-objective DTLZ2 front, with axes that can be reordered and brushed, a scatter-plot matrix of the same solutions with the brushed ones highlighted, and the number of line crossings between neighbouring axes. It needs JavaScript.</p></div>

Parallel coordinates scale to many objectives, but not to many solutions:
hundreds of lines merge into a band, and the eye then reads the envelope rather
than single solutions. VAMOS Studio draws the front of a running experiment as
parallel coordinates when it has more than three objectives, and
`plot_parallel_coordinates(F, labels=[...], show=False)` from `vamos.ux.api`
draws them with Matplotlib, from the `analysis` extra.

## Heatmaps and radar charts

A **heatmap** draws the solutions as rows and the objectives as columns, with
each cell coloured by the normalized value [6]. It fits hundreds of solutions
and dozens of objectives in one picture, and sorting the rows, by one objective
or by clustering similar solutions, shows groups of solutions with the same
profile. Colour is read less precisely than position, so exact trade-offs are
hard to read, and the picture changes with the order of the rows.

A **radar chart**, or star plot, draws m spokes from a centre and each solution
as a polygon through its normalized values. It suits a handful of solutions,
for example the final candidates of a decision maker, compared side by side.
The shape and area of a polygon depend on the order of the spokes, and dozens
of overlapping polygons are unreadable. Tušar and Filipič [5] review these and
other methods for fronts of several objectives.

## Projections and what they hide

A projection maps the m objectives to two or three coordinates. Every
projection loses something:

| Method | Keeps | Hides |
| --- | --- | --- |
| Two objectives (a scatter plot, a panel of the matrix) | the exact values of two objectives | why points that look dominated are not (first example) |
| Objective reduction | the objectives it keeps | the trade-offs with the objectives it drops |
| Linear projection, such as principal components | the directions in which the set varies most | dominance; distances along the dropped directions |
| RadViz [7] | the proportions between the objectives | their magnitude: proportional vectors land on the same point |

RadViz places each solution at the average of m anchors on a circle, weighted
by its normalized objective values. Two solutions with equal values in every
objective, such as (0.1, …, 0.1) and (0.9, …, 0.9), both land in the centre,
although one is far better than the other. **Objective reduction** keeps a subset
of the objectives, and Brockhoff and Zitzler [8] studied when that preserves
dominance. VAMOS's `reduce_objectives` drops an objective that is almost
perfectly correlated with another (correlation 0.99 or more by default), and
with `target_dim` keeps the objectives with the largest variance relative to
their range. Below, f₆ is f₁ in other units, with noise:

```python
import numpy as np

from vamos.ux.api import reduce_objectives

rng = np.random.default_rng(1)
n, m = 200, 5
x = rng.random((n, m - 1))
F = np.ones((n, m))  # the same 200 points of DTLZ2's front
for i in range(m):
    F[:, i] = np.prod(np.cos(x[:, : m - 1 - i] * np.pi / 2), axis=1)
    if i > 0:
        F[:, i] *= np.sin(x[:, m - 1 - i] * np.pi / 2)


def dominated(P):
    """Whether another row of P dominates each row (minimization)."""
    no_worse = np.all(P[:, None] <= P[None, :], axis=2)
    better = np.any(P[:, None] < P[None, :], axis=2)
    return np.any(no_worse & better, axis=0)


cost = 1000 * F[:, 0] + rng.normal(0.0, 1.0, n)  # f1 again, in other units and with noise
_, kept = reduce_objectives(np.column_stack([F, cost]))
print("kept of f1 to f6:", ", ".join(f"f{i + 1}" for i in kept))

reduced, kept = reduce_objectives(F, target_dim=3)
print("kept of f1 to f5 with target_dim=3:", ", ".join(f"f{i + 1}" for i in kept))
print(f"dominated in those three objectives: {dominated(reduced).sum()} of {n}")
```

Output:

```text
kept of f1 to f6: f2, f3, f4, f5, f6
kept of f1 to f5 with target_dim=3: f3, f4, f5
dominated in those three objectives: 152 of 200
```

The redundant pair loses one member without loss: f₁ and f₆ order the
solutions almost identically. Cutting five conflicting objectives to three is
another matter: most solutions then look dominated, as in the first example.

## Key points

- Beyond three objectives every picture is partial: a plot of two objectives
  makes almost every non-dominated point look dominated.
- A scatter-plot matrix shows every pair of objectives, m(m − 1)/2 panels,
  but no interaction of three or more.
- Parallel coordinates show every objective at once; the axis order decides
  which pairs are compared, the normalization decides what heights mean, and
  brushing filters solutions across linked views.
- Heatmaps fit many solutions, radar charts a few; projections and objective
  reduction hide magnitude, dominance or the dropped objectives.
- VAMOS draws fronts of two or three objectives, parallel coordinates for any
  number, and reduces objectives, through the experimental `vamos.ux.api`;
  Studio uses parallel coordinates and scatter-plot matrices.

**Next:** [Constrained multi-objective optimization](constrained.md).

## References

[1] Inselberg, A. (1985). [The plane with parallel coordinates](https://doi.org/10.1007/BF01898350). *The Visual Computer*, 1(2), 69–91.

[2] Wegman, E. J. (1990). [Hyperdimensional data analysis using parallel coordinates](https://doi.org/10.1080/01621459.1990.10474926). *Journal of the American Statistical Association*, 85(411), 664–675.

[3] Becker, R. A., and Cleveland, W. S. (1987). [Brushing scatterplots](https://doi.org/10.1080/00401706.1987.10488204). *Technometrics*, 29(2), 127–142.

[4] Li, M., Zhen, L., and Yao, X. (2017). [How to read many-objective solution sets in parallel coordinates](https://doi.org/10.1109/MCI.2017.2742869). *IEEE Computational Intelligence Magazine*, 12(4), 88–100.

[5] Tušar, T., and Filipič, B. (2015). [Visualization of Pareto front approximations in evolutionary multiobjective optimization: A critical review and the prosection method](https://doi.org/10.1109/TEVC.2014.2313407). *IEEE Transactions on Evolutionary Computation*, 19(2), 225–245.

[6] Pryke, A., Mostaghim, S., and Nazemi, A. (2007). [Heatmap visualization of population based multi objective algorithms](https://doi.org/10.1007/978-3-540-70928-2_29). *Evolutionary Multi-Criterion Optimization – EMO 2007*, Lecture Notes in Computer Science 4403, 361–375.

[7] Hoffman, P., Grinstein, G., Marx, K., Grosse, I., and Stanley, E. (1997). [DNA visual and analytic data mining](https://doi.org/10.1109/VISUAL.1997.663916). *Proceedings of Visualization '97*, 437–441.

[8] Brockhoff, D., and Zitzler, E. (2009). [Objective reduction in evolutionary multiobjective optimization: Theory and applications](https://doi.org/10.1162/evco.2009.17.2.135). *Evolutionary Computation*, 17(2), 135–166.

[9] Deb, K., Thiele, L., Laumanns, M., and Zitzler, E. (2005). [Scalable test problems for evolutionary multiobjective optimization](https://doi.org/10.1007/1-84628-137-7_6). In A. Abraham, L. Jain and R. Goldberg (Eds.), *Evolutionary Multiobjective Optimization*, 105–145. Springer.
