---
description: 'A5. Constrained multi-objective optimization: feasibility and constraint violation, constrained domination, epsilon-constrained and stochastic ranking, why feasibility-first selection can fail, and the constrained algorithms in VAMOS.'
search:
  boost: 0.5
---

# A5. Constrained multi-objective optimization

Real problems come with limits: a stress below the yield strength, a budget,
a delivery time. Constraints split the decision space into feasible and
infeasible regions, and the best feasible trade-offs can look very different
from the unconstrained ones. This lesson explains how evolutionary algorithms
compare solutions when some of them break the rules.

In this lesson you will learn:

- how constraints and constraint violation are defined;
- constrained domination, the default rule of NSGA-II in VAMOS;
- the alternatives that relax feasibility first;
- why difficult feasible regions call for dedicated algorithms, and which ones
  VAMOS provides.

## Feasibility and violation

With inequality constraints written as g_j(x) ≤ 0, the convention of VAMOS
(lesson [B1](../basic/multi-objective-problems.md#the-formal-problem)), a
solution is **feasible** when every g_j(x) ≤ 0. Its **constraint violation**
aggregates how much it breaks them:

<div class="vf-formula">CV(x) = Σ_j  max( 0, g_j(x) )          CV(x) = 0  ⇔  x is feasible</div>

An equality h(x) = 0 is usually relaxed to |h(x)| − δ ≤ 0 for a small tolerance
δ. Constraints of different magnitudes should be normalized before they are
added, or the largest one dominates CV. The [constraint
reference](../../reference/constraints.md#define-inequality-constraints) shows
how to declare them in VAMOS.

The **constrained Pareto front** is the Pareto front of the feasible solutions.
It can coincide with the unconstrained front, lie partly on it, or lie on the
boundary of the feasible region, possibly in disconnected pieces.

## Constrained domination

Deb's feasibility rules [1], used by NSGA-II [2], define a solution a to
*constrained-dominate* b when

<div class="vf-formula">a is feasible and b is not,                      or
both are infeasible and CV(a) &lt; CV(b),            or
both are feasible and a ≺ b  (Pareto dominance)</div>

Fast non-dominated sorting runs unchanged with this relation in place of
dominance (lesson [I2](../intermediate/non-dominated-sorting.md)). The rule has
no parameter: feasibility comes first, then the objectives.

The figure uses CTP2 of Deb, Pratap and Meyarivan [3], a problem whose
constraint carves the objective space into feasible "tunnels". The dashed line
is the unconstrained front; it is infeasible everywhere except at its end. The
constrained front, in blue, is a set of short disconnected segments on the
boundary of the feasible region. Switch between the two ranking rules and
compare the first front:

<div class="vf-widget" data-vf-widget="constraints"><p class="vf-fallback">Interactive figure: CTP2 objective space with its infeasible regions and a population ranked with and without constrained domination. It needs JavaScript.</p></div>

Ignoring the constraints, the infeasible solutions closest to the unconstrained
front take the best ranks, and selection would push the population into a
region it can never use. Constrained domination ranks every feasible solution
first and orders the infeasible ones by how close they are to feasibility.

## Beyond feasibility first

Feasibility first is simple and effective when the feasible region is large
and connected. It fails when it is not: once a few feasible solutions are found,
they dominate the population, and feasible regions separated from them by
infeasible barriers are never reached. Several techniques relax the rule:

- **Penalty functions** add a multiple of CV(x) to each objective. They are
  simple, but the result depends on the penalty factor.
- **ε-constrained methods** treat a solution as feasible when its violation is
  at most ε, and shrink ε to zero during the run, so that slightly infeasible
  solutions can cross barriers early on [4].
- **Stochastic ranking** compares two solutions by their objectives with a fixed
  probability, even when they are infeasible, and by their violation otherwise
  [5].

Modern constrained algorithms go further and keep **two sources of
information**: one that ignores or relaxes the constraints and explores, and one
that enforces them and converges to the constrained front.

| Algorithm | Idea | Lessons |
| --- | --- | --- |
| CCMO [6] | Two populations evolve in parallel, one solving the constrained problem and one a helper problem without constraints, sharing their offspring. | [A6](ccmo.md) · [CCMO](../../algorithms/ccmo.md) |
| C-TAEA [7] | Two archives, one that pushes towards feasibility and convergence and one that keeps diversity, including infeasible solutions. | [A7](ctaea.md) · [C-TAEA](../../algorithms/ctaea.md) |
| ToP [8] | Two phases: first find promising regions by treating the problem as a single-objective constrained problem, then optimize all objectives there. | [A8](top.md) · [ToP](../../algorithms/top.md) |
| CMOEA-MS [9] | Two stages: one gives the objectives the same priority as the constraint violation, the other puts the constraints first; the share of feasible solutions decides which stage runs. | [A9](cmoeams.md) · [CMOEA-MS](../../algorithms/cmoeams.md) |

The [constraint reference](../../reference/constraints.md#choose-a-constraint-handling-technique)
lists the constraint-handling techniques that VAMOS' algorithms accept, such as
`epsilon_constrained`, `stochastic_ranking` and `push_pull`, and the
[problem reference](../../reference/problems.md) lists the constrained benchmark
families.

## In VAMOS

A CTP2-style problem, written with `make_problem`. Both functions are
vectorized: they receive a whole population, one row per solution. The
constraint function returns G = −c, because the paper writes the constraint as
c(x) ≥ 0 and VAMOS treats G ≤ 0 as feasible. Here g = 1 + Σ x_i² is a simpler
distance function than the paper's; VAMOS releases after 1.4.0 also ship the
original CTP1 to CTP7 as built-in problems.

```python
import numpy as np

from vamos import make_problem, optimize

THETA, A, B, C, D, E = -0.2 * np.pi, 0.2, 10.0, 1.0, 6.0, 1.0  # CTP2 parameters


def objectives(X):
    g = 1.0 + np.sum(X[:, 1:] ** 2, axis=1)
    f1 = X[:, 0]
    return np.column_stack([f1, g - f1])  # f2 = g (1 - f1 / g)


def constraints(X):
    F = objectives(X)
    f1, f2 = F[:, 0], F[:, 1]
    across = np.cos(THETA) * (f2 - E) - np.sin(THETA) * f1
    along = np.sin(THETA) * (f2 - E) + np.cos(THETA) * f1
    c = across - A * np.abs(np.sin(B * np.pi * along**C)) ** D  # c(x) >= 0 in the paper
    return -c[:, None]  # VAMOS: G <= 0 is feasible


problem = make_problem(
    objectives,
    n_var=5,
    n_obj=2,
    bounds=[(0.0, 1.0)] + [(-1.0, 1.0)] * 4,
    vectorized=True,
    encoding="real",
    constraints=constraints,
    n_constraints=1,
    name="ctp2_style",
)
result = optimize(problem, algorithm="nsgaii", pop_size=100, max_evaluations=20_000, engine="numpy", seed=3)

G = result.data["G"]
feasible = np.all(G <= 0.0, axis=1)
front = result.front()
print("feasible returned solutions:", int(feasible.sum()), "of", len(G))
print("constrained front:", len(front), "points, f1 from", round(float(front[:, 0].min()), 3), "to", round(float(front[:, 0].max()), 3))
```

Output:

```text
feasible returned solutions: 100 of 100
constrained front: 100 points, f1 from 0.0 to 0.985
```

`result.front()` keeps feasible rows only (`feasible_only=True` by default) and
falls back to the least violated rows when none is feasible. To change the
rule, pass a technique to the builder, for example
`.constraint_handling("epsilon_constrained")`; to change the algorithm, pass
`algorithm="ccmo"` or one of the others in the table.

## Measuring constrained results

- Compute quality indicators on the **feasible** non-dominated points only, with
  a reference set from the **constrained** front.
- Report, per algorithm, how many runs found at least one feasible solution,
  and how quickly. An indicator averaged over the successful runs alone hides
  the failures.

## Key points

- A solution is feasible when every g_j(x) ≤ 0; the violation CV(x) sums the
  positive parts.
- Constrained domination (feasible first, then less violation, then Pareto
  dominance) is the default rule of NSGA-II in VAMOS.
- Feasibility-first selection can be trapped by disconnected or narrow feasible
  regions; ε-constrained methods, stochastic ranking and two-population or
  two-stage algorithms address that.
- Evaluate constrained results on feasible solutions and report the feasibility
  rate separately.

**Next:** [A6. CCMO](ccmo.md).

## References

[1] Deb, K. (2000). [An efficient constraint handling method for genetic algorithms](https://doi.org/10.1016/S0045-7825(99)00389-8). *Computer Methods in Applied Mechanics and Engineering*, 186(2–4), 311–338.

[2] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://doi.org/10.1109/4235.996017). *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

[3] Deb, K., Pratap, A., and Meyarivan, T. (2001). [Constrained test problems for multi-objective evolutionary optimization](https://doi.org/10.1007/3-540-44719-9_20). *Evolutionary Multi-Criterion Optimization – EMO 2001*, Lecture Notes in Computer Science 1993, 284–298.

[4] Takahama, T., and Sakai, S. (2006). Constrained optimization by the ε constrained differential evolution with gradient-based mutation and feasible elites. *Proceedings of the 2006 IEEE Congress on Evolutionary Computation*, 1–8.

[5] Runarsson, T. P., and Yao, X. (2000). [Stochastic ranking for constrained evolutionary optimization](https://doi.org/10.1109/4235.873238). *IEEE Transactions on Evolutionary Computation*, 4(3), 284–294.

[6] Tian, Y., Zhang, T., Xiao, J., Zhang, X., and Jin, Y. (2021). [A coevolutionary framework for constrained multiobjective optimization problems](https://doi.org/10.1109/TEVC.2020.3004012). *IEEE Transactions on Evolutionary Computation*, 25(1), 102–116.

[7] Li, K., Chen, R., Fu, G., and Yao, X. (2019). [Two-archive evolutionary algorithm for constrained multiobjective optimization](https://doi.org/10.1109/TEVC.2018.2855411). *IEEE Transactions on Evolutionary Computation*, 23(2), 303–315.

[8] Liu, Z.-Z., and Wang, Y. (2019). [Handling constrained multiobjective optimization problems with constraints in both the decision and objective spaces](https://doi.org/10.1109/TEVC.2019.2894743). *IEEE Transactions on Evolutionary Computation*, 23(5), 870–884.

[9] Tian, Y., Zhang, Y., Su, Y., Zhang, X., Tan, K. C., and Jin, Y. (2022). [Balancing objective optimization and constraint satisfaction in constrained evolutionary multiobjective optimization](https://doi.org/10.1109/TCYB.2020.3021138). *IEEE Transactions on Cybernetics*, 52(9), 9559–9572.
