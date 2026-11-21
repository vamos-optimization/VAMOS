---
description: 'What a multi-objective optimization problem is: decision and objective spaces, conflicting objectives, why weighted sums are not enough, and why population-based algorithms are used.'
search:
  boost: 0.5
---

# What is a multi-objective problem?

Most real decisions have more than one goal. A bridge should be light **and**
stiff; a machine-learning model should be accurate **and** small; a production
plan should be cheap **and** fast **and** clean. When improving one goal makes
another worse, there is no single best answer, only a set of best
compromises. Multi-objective optimization finds that set.

In this lesson you will learn:

- how a multi-objective problem is written, and what its two spaces are;
- why conflicting objectives produce a set of solutions instead of one;
- why adding the objectives into one weighted sum is not enough;
- why evolutionary algorithms, which work with a population, fit this task.

## The formal problem

A multi-objective optimization problem with *m* objectives is written as

<div class="vf-formula">minimize     F(x) = ( f₁(x), f₂(x), …, f_m(x) )
subject to   g_j(x) ≤ 0,   j = 1, …, J
             x ∈ X,  for example  x_iᴸ ≤ x_i ≤ x_iᵁ,   i = 1, …, n</div>

- **x** = (x₁, …, x_n) is a *decision vector*: one candidate design, plan or
  model. The set X of allowed vectors is the **decision space** (or search
  space). Each x_i is a *decision variable*; here they are real numbers
  between bounds, but they can also be integers, bits or permutations.
- **F(x)** is the *objective vector* of that candidate. The set of all
  objective vectors is the **objective space**. VAMOS minimizes every
  objective: an objective to maximize is passed as its negative, −f(x).
- **g_j(x) ≤ 0** are inequality *constraints*. A candidate that satisfies all
  of them is *feasible*. VAMOS uses exactly this convention, `G <= 0`; lesson
  [Constraints](../advanced/constrained.md) is devoted to constraints.

Problems with two or three objectives are usually called *multi-objective*.
With four or more, they are called *many-objective*, because, as lesson
[Many objectives](../advanced/many-objective.md) shows, several mechanisms that work well
with two objectives stop working.

## Two spaces

The optimizer changes the decision vector, but we judge the result by its
objective vector. The figure below uses a two-variable relative of the ZDT1
benchmark [1]:

<div class="vf-formula">f₁(x) = x₁
f₂(x) = g · ( 1 − √(x₁ / g) ),   where  g = 1 + x₂,   0 ≤ x₁, x₂ ≤ 1</div>

The grey dots are random designs (left) and their objective vectors (right).
The orange point is a design you control: drag it, or focus it and use the
arrow keys.

<div class="vf-widget" data-vf-widget="decision-map"><p class="vf-fallback">Interactive figure: a design in decision space and its image in objective space. It needs JavaScript; the text below describes what it shows.</p></div>

Three things to notice:

1. **The mapping is not one-to-one.** Moving x₂ upwards moves F(x) upwards
   without changing f₁: the variable x₂ only adds "distance" from the best
   trade-offs.
2. **The best designs form a set.** Every design with x₂ = 0 is a best
   compromise: you cannot lower f₁ without raising f₂. Those designs are the
   *Pareto set* (blue segment on the left); their objective vectors form the
   *Pareto front* (blue curve on the right). Lesson
   [Pareto dominance](pareto-dominance.md) defines both precisely.
3. **The front, not the set, is what a decision maker looks at.** It answers
   "how much f₂ must I accept to gain this much f₁?". The Pareto set answers
   "which design gives me that trade-off?".

## Why one optimum is not enough

With one objective, any two candidates can be compared: one is better, or they
are equal. With two conflicting objectives, a candidate can be better in f₁
and worse in f₂. Neither is better overall; they are **incomparable**. Along
the Pareto front every move that improves one objective costs something in
another, and that cost is called the *trade-off*. The front shows the
complete menu of trade-offs before anyone has to decide which one is preferred.

## Turning many objectives into one: scalarization

A tempting shortcut is to add the objectives with weights and solve an
ordinary single-objective problem:

<div class="vf-formula">minimize   w₁ f₁(x) + w₂ f₂(x) + … + w_m f_m(x),    w_i ≥ 0,   Σ w_i = 1</div>

With strictly positive weights, the minimizer of a weighted sum is always
Pareto optimal, and changing the weights moves along the front [2]. But the
method has two well-known limits:

- **It cannot reach non-convex parts of the front.** The contour lines of a
  weighted sum are straight. On a front that bulges outwards (non-convex),
  the straight line always touches the front first at an extreme point. Das
  and Dennis [3] showed that no choice of weights reaches the interior of
  such a front, and that evenly spaced weights do not give evenly spaced
  solutions even on convex fronts.
- **Weights mix units.** Adding kilograms to millimetres only makes sense once
  both are scaled, and the result depends on that scaling.

Try both fronts below. Choose the non-convex front, press **Sweep 21 weights**
and count how many different solutions the weighted sum finds.

<div class="vf-widget" data-vf-widget="scalarization" data-modes="ws" data-shape="concave"><p class="vf-fallback">Interactive figure: the solution selected by a weighted sum on a convex and on a non-convex front. It needs JavaScript.</p></div>

Other scalarizations avoid the first limit. The *ε-constraint* method
optimizes one objective and turns the others into constraints [4]; the
*Tchebycheff* scalarization uses a weighted maximum instead of a weighted sum
and reaches every Pareto-optimal point. Lesson
[Families of algorithms](algorithm-families.md) shows how MOEA/D builds a complete
algorithm on that idea.

## When do preferences enter?

Somebody eventually picks one solution. Methods are classified by when that
person, the *decision maker*, states preferences [2, 5]:

- **A priori:** preferences come first, for example as weights, and one
  solution is computed. Cheap, but the decision maker must know the trade-offs
  before seeing them.
- **A posteriori:** a representative approximation of the whole front is
  computed first, and the decision maker chooses afterwards with the
  trade-offs in view.
- **Interactive:** computation and preference statements alternate, focusing
  the search on the region of interest.

The algorithms in VAMOS, like most evolutionary multi-objective algorithms,
are *a posteriori* methods. [T03. From a Pareto set to a
decision](../../tutorials/t03-pareto-decision.md) shows the decision step
on a real result.

## Why population-based algorithms?

An a posteriori method must return many solutions, spread along the front.
Evolutionary algorithms keep a *population* of candidates and improve it
generation after generation, so a single run naturally produces a set. Schaffer's
VEGA [6] was the first multi-objective genetic algorithm; Goldberg [7]
proposed ranking the population by Pareto dominance, the idea that NSGA-II,
the subject of the [Intermediate level](../intermediate/nsga-ii.md), still uses.
Evolutionary algorithms also need no gradients, so they handle objectives that
are noisy, discontinuous or computed by a simulator [8, 9].

## In VAMOS

The toy problem of the first figure can be written as a plain Python function
and solved with NSGA-II. The returned non-dominated solutions should all have
x₂ close to 0, the Pareto set.

```python
import numpy as np

from vamos import make_problem, optimize


def toy(x):
    """Two decision variables and two objectives, both minimized."""
    g = 1.0 + x[1]
    f1 = x[0]
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return [f1, f2]


problem = make_problem(toy, n_var=2, n_obj=2, bounds=[(0.0, 1.0), (0.0, 1.0)], encoding="real", name="toy")
result = optimize(problem, algorithm="nsgaii", pop_size=40, max_evaluations=4_000, engine="numpy", seed=1)

front_F, front_indices = result.front(return_indices=True)
front_X = result.X[front_indices]
print(len(front_F), "non-dominated solutions")
print("largest x2 among them:", round(float(front_X[:, 1].max()), 4))
print("f1 from", round(float(front_F[:, 0].min()), 3), "to", round(float(front_F[:, 0].max()), 3))
```

Output:

```text
40 non-dominated solutions
largest x2 among them: 0.0166
f1 from 0.0 to 1.0
```

`result.X` holds decision vectors and `result.F` the matching objective
vectors, row by row. The [custom problem guide](../../guide/custom-problem.md)
explains `make_problem` in full, including constraints and vectorized
functions.

## Key points

- A multi-objective problem minimizes a vector F(x) = (f₁, …, f_m) over a
  decision space, usually subject to constraints g_j(x) ≤ 0.
- Conflicting objectives have no single optimum; the best compromises form the
  Pareto set (decisions) and the Pareto front (objective values).
- Weighted sums are simple but miss non-convex parts of the front and depend
  on scaling.
- Evolutionary algorithms keep a population and therefore approximate the
  whole front in one run, leaving the final choice to the decision maker.

**Next:** [Pareto dominance and the Pareto front](pareto-dominance.md).

## References

[1] Zitzler, E., Deb, K., and Thiele, L. (2000). [Comparison of multiobjective evolutionary algorithms: Empirical results](https://doi.org/10.1162/106365600568202). *Evolutionary Computation*, 8(2), 173–195.

[2] Miettinen, K. (1999). [*Nonlinear Multiobjective Optimization*](https://doi.org/10.1007/978-1-4615-5563-6). Kluwer Academic Publishers.

[3] Das, I., and Dennis, J. E. (1997). [A closer look at drawbacks of minimizing weighted sums of objectives for Pareto set generation in multicriteria optimization problems](https://doi.org/10.1007/BF01197559). *Structural Optimization*, 14(1), 63–69.

[4] Haimes, Y. Y., Lasdon, L. S., and Wismer, D. A. (1971). [On a bicriterion formulation of the problems of integrated system identification and system optimization](https://doi.org/10.1109/TSMC.1971.4308298). *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-1(3), 296–297.

[5] Branke, J., Deb, K., Miettinen, K., and Słowiński, R. (Eds.) (2008). [*Multiobjective Optimization: Interactive and Evolutionary Approaches*](https://doi.org/10.1007/978-3-540-88908-3). Lecture Notes in Computer Science 5252, Springer.

[6] Schaffer, J. D. (1985). Multiple objective optimization with vector evaluated genetic algorithms. *Proceedings of the First International Conference on Genetic Algorithms and Their Applications*, 93–100.

[7] Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization and Machine Learning*. Addison-Wesley.

[8] Deb, K. (2001). *Multi-Objective Optimization Using Evolutionary Algorithms*. Wiley.

[9] Coello Coello, C. A., Lamont, G. B., and Van Veldhuizen, D. A. (2007). [*Evolutionary Algorithms for Solving Multi-Objective Problems*](https://doi.org/10.1007/978-0-387-36797-2) (2nd ed.). Springer.
