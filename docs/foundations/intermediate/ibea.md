---
description: 'IBEA in depth: binary quality indicators (additive epsilon and hypervolume difference), the exponential fitness with its scaling c and parameter kappa, objective normalization, iterative removal with fitness updates and binary tournament mating, with an interactive figure and VAMOS code.'
search:
  boost: 0.5
---

# IBEA: indicator-based fitness

SMS-EMOA (lesson [SMS-EMOA](smsemoa.md)) selects with one unary indicator, the
hypervolume, and pays for exact contributions at every step. The
Indicator-Based Evolutionary Algorithm of Zitzler and Künzli, IBEA [1], takes a
more general route. It asks the user for a **binary indicator**, a number that
says how much better one solution is than another, turns the indicator values
of all pairs into a fitness, and removes the worst solution until the
population fits.

IBEA needs no non-dominated sorting, no crowding distance and no density
estimate: the indicator alone provides both convergence and diversity. With the
additive ε indicator, the default, the fitness of a population costs O(N² m)
operations for N solutions and m objectives, whatever m is.

In this lesson you will learn:

- what a binary indicator is, and the two that VAMOS supports: additive ε and
  hypervolume difference;
- how IBEA turns indicator values into a fitness, and what the scaling c and
  the parameter κ do;
- why VAMOS normalizes the objectives first;
- how environmental selection removes one solution at a time and updates the
  others' fitness, and how parents are chosen;
- how to configure IBEA in VAMOS.

## Binary indicators

The indicators of lesson [Quality indicators](../basic/quality-indicators.md) are *unary*: they
give one number to one set. A **binary** indicator I(A, B) compares two sets
[2]. IBEA applies it to single solutions, I(x₁, x₂), and Zitzler and Künzli
require it to be **dominance preserving** [1]:

<div class="vf-formula">x₁ ≺ x₂   ⇒   I(x₁, x₂) &lt; I(x₂, x₁)
x₁ ≺ x₂   ⇒   I(x₃, x₁) ≥ I(x₃, x₂)   for every x₃</div>

The first condition says that a dominating solution scores better against the
solution it dominates than the reverse; the second, that the better solution is
harder for any third solution to cover. VAMOS offers two such indicators.

**Additive ε.** The smallest amount ε by which x₁ must be improved in every
objective to weakly dominate x₂ (lesson
[Quality indicators](../basic/quality-indicators.md#distance-based-indicators)):

<div class="vf-formula">I_ε+(x₁, x₂) = max_i ( f_i(x₁) − f_i(x₂) )</div>

It is ≤ 0 exactly when x₁ weakly dominates x₂, and small and positive when x₁
almost covers x₂.

**Hypervolume difference.** With H(A) the hypervolume of A
(lesson [Quality indicators](../basic/quality-indicators.md#hypervolume)) [3]:

<div class="vf-formula">I_HD(x₁, x₂) = H(x₂) − H(x₁)            if x₁ dominates x₂
I_HD(x₁, x₂) = H({x₁, x₂}) − H(x₁)      otherwise</div>

In the second case it is the volume that x₂ dominates and x₁ does not. Select it
with `.indicator("hypervolume")`; `.indicator("eps")` is the default.

## Fitness from indicator values

The fitness of a solution x of the population P sums one negative term per other
solution y:

<div class="vf-formula">F(x) = Σ_{y ∈ P, y ≠ x}  −exp( −I(y, x) / (c · κ) )</div>

The term for y measures how well y covers x:

- If y dominates x, I(y, x) &lt; 0 and the term is below −1; the more y
  dominates x, and the smaller κ, the larger its magnitude.
- If y nearly covers x, I(y, x) is small and positive and the term is close
  to −1. Two close non-dominated solutions penalize each other this way, which is
  how IBEA rewards spread.
- If y is far from covering x, I(y, x) is large and the term is close to 0.

F(x) ≤ 0, and a value closer to 0 is better: the solution that the others cover
the least. Zitzler and Künzli read F(x) as the loss of quality caused by
removing x [1].

## Normalization, the scaling c and κ

Indicator values depend on the objective units. Zitzler and Künzli's adaptive
IBEA [1] therefore maps each objective to [0, 1] with the bounds of the
population, and divides every indicator value by

<div class="vf-formula">c = max_{x₁, x₂ ∈ P}  | I(x₁, x₂) |</div>

so that I / c lies in [−1, 1] and each exponent in [−1/κ, 1/κ]. VAMOS always
applies both steps, with either indicator, to the union of population and
offspring. A positive affine change of the units of any objective then leaves
the selection unchanged. A solution far from the others widens the
normalization range, which lowers the resolution among the rest.

κ sets the selection pressure. VAMOS' default is κ = 0.05, the value of
Zitzler and Künzli [1].

- **Small κ.** The terms of dominating solutions dwarf all others: dominated
  solutions get a very negative fitness and leave first, and among the
  non-dominated ones the nearest neighbours decide.
- **Large κ.** Every term approaches −1 + I(y, x) / (c κ), so the fitness
  approaches a sum of indicator values. The penalty for being dominated is no
  longer special, and dominated solutions survive more often.

## Environmental selection

IBEA merges the N parents with the N offspring and removes solutions one at a
time:

<div class="vf-formula">Q ← P ∪ P′                                    (2N solutions)
normalize Q, compute I and c, compute F(x) for every x in Q
while |Q| > N:
    x* ← the member of Q with the smallest F
    Q ← Q \ {x*}
    for every x in Q:   F(x) ← F(x) + exp( −I(x*, x) / (c · κ) )</div>

The update removes x\*'s term from the sums of the others, so the fitness of the
remaining solutions is exactly the one they would have in Q without x\*; nothing
is recomputed. The normalization and c stay those of the merged set. Removing
one solution at a time matters for the same reason as in SPEA2 (lesson
[SPEA2](spea2.md)): when two solutions cover each other, the first removal lifts
the fitness of the second, which then survives.

## Mating selection

Parents are chosen by binary tournaments on the survivors, comparing their
updated fitness: the higher fitness wins [1, 4]. Crossover and mutation then
create the N offspring of the next generation.

## Try it: fitness and removal order

The eight points below are the merged set Q. The table gives each point's
fitness and the order in which IBEA removes points for the chosen population
size, with the updated fitness of the survivors. Things to try:

1. With κ = 0.05 and population size 5, P8 and P7, the dominated points, go
   first, then P3, one of the close pair P2 and P3.
2. Reduce the population to 4: the next to go is P1, an extreme. IBEA with
   ε has no rule that protects extremes.
3. Raise κ to 1 with population size 5: the third removal becomes P4, the only
   point in the middle of the front, and both P2 and P3 survive.
4. Switch to the hypervolume-difference indicator and compare the orders.

<div class="vf-widget" data-vf-widget="ibea-fitness"><p class="vf-fallback">Interactive figure: the IBEA fitness of eight draggable points for a chosen indicator and kappa, and the order in which environmental selection removes them for a chosen population size. It needs JavaScript.</p></div>

## In VAMOS

The figure's default case in NumPy, with the additive ε indicator, κ = 0.05
and a population of five:

```python
import numpy as np

F = np.array([[0.05, 0.95], [0.15, 0.60], [0.20, 0.55], [0.40, 0.38], [0.60, 0.20], [0.95, 0.05], [0.50, 0.60], [0.75, 0.45]])
kappa = 0.05
Z = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))  # each objective mapped to [0, 1]
eps = np.max(Z[None, :, :] - Z[:, None, :], axis=2)  # eps[x, y] = I_eps+(y, x) = max_i (z_i(y) - z_i(x))
c = np.abs(eps).max()  # adaptive scaling: the largest indicator magnitude
terms = np.exp(-eps / (c * kappa))
np.fill_diagonal(terms, 0.0)  # no term for y = x
fitness = -terms.sum(axis=1)
for i, value in enumerate(fitness):
    print(f"P{i + 1}: F = {value:.3f}")

alive = list(range(len(F)))
while len(alive) > 5:  # population size 5
    worst = min(alive, key=lambda i: fitness[i])  # the first lowest fitness
    alive.remove(worst)
    fitness[alive] += terms[alive, worst]  # its terms leave the others' sums
    print(f"remove P{worst + 1}")
print("survivors:", ", ".join(f"P{i + 1} ({fitness[i]:.3f})" for i in alive))
```

Output:

```text
P1: F = -0.145
P2: F = -0.334
P3: F = -0.342
P4: F = -0.050
P5: F = -0.023
P6: F = -0.036
P7: F = -13.378
P8: F = -32.961
remove P8
remove P7
remove P3
survivors: P1 (-0.109), P2 (-0.004), P4 (-0.019), P5 (-0.019), P6 (-0.036)
```

P7 is dominated by P2, P3 and P4, and P8 by P4 and P5: their fitness is two
orders of magnitude below the others. P2 and P3 cover each other closely, so both have
a poor fitness; once P3 is removed, P2's fitness rises from −0.334 to −0.004.

VAMOS' IBEA follows the same steps, with a few details worth knowing:

- Each objective is mapped to [0, 1] with its minimum and maximum over the
  finite values; an objective without spread maps to 0.
- The hypervolume-difference indicator uses the reference point max + 1 of the
  normalized objectives, (2, …, 2) when every objective varies.
- On a tie, the first of the solutions with the lowest fitness is removed.
- With constraints, every infeasible solution gets a lower fitness than all
  feasible ones, so infeasible solutions leave first, the most violated first.
- A κ so small that the exponentials would overflow still gives the correct
  removals, and `engine="numba"` gives the same removals as NumPy.

Mating uses NSGA-II's binary tournament (lesson
[Crowding and tournaments](crowding-distance.md#binary-tournament-selection))
on the fitness that the survivors have at the end of the removal loop: the
solution with the higher fitness always wins. SBX creates two children per
pair, so N / 2 pairs give the N offspring, and the last batch is shortened so
that the budget is never exceeded.

Two runs on ZDT1 that differ only in κ:

```python
from vamos import optimize
from vamos.algorithms import IBEAConfig
from vamos.problems import ZDT1

problem = ZDT1(n_var=30)
for kappa in (0.05, 1.0):
    config = (
        IBEAConfig.builder()
        .pop_size(100)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament")
        .indicator("eps")
        .kappa(kappa)
        .build()
    )
    result = optimize(problem, algorithm="ibea", algorithm_config=config, max_evaluations=20_000, engine="numpy", seed=1)
    print(f"kappa {kappa}: {result.data['evaluations']} evaluations, {result.F.shape[0]} non-dominated of 100 in the final population")
```

Output:

```text
kappa 0.05: 20000 evaluations, 97 non-dominated of 100 in the final population
kappa 1.0: 20000 evaluations, 78 non-dominated of 100 in the final population
```

By default the result holds the non-dominated members of the final population.
With κ = 1, dominated solutions are penalized less and more of them survive, as
the previous section predicts; one seed shows the mechanism, not which value is
better. The [IBEA lesson](../../algorithms/ibea.md) records a run with its
figure, and the [configuration reference](../../reference/api/algorithms/ibea.md)
lists every builder method.

## IBEA compared

- **With NSGA-II and SPEA2.** No fronts and no separate density estimate: one
  fitness combines convergence and spread. Because the fitness of a dominated
  solution depends on how far it is dominated, IBEA keeps pushing when nearly
  every solution is non-dominated. Wagner, Beume and Naujoks [5] found that
  indicator-based algorithms, IBEA among them, kept converging on many-objective
  problems where NSGA-II and SPEA2 lost selection pressure (lesson
  [Many objectives](../advanced/many-objective.md)).
- **With SMS-EMOA.** Both remove one solution at a time by an indicator.
  SMS-EMOA measures exact hypervolume contributions, whose cost grows
  exponentially with m; IBEA with ε needs only pairwise differences.
- **Distribution.** The figure shows that IBEA with ε does not protect the
  extremes of the front; the distribution it reaches depends on the indicator, on
  κ and on the normalization.
- **When to use it.** Problems with three or more objectives, where Pareto
  ranking stalls and the hypervolume is too expensive; or whenever a single,
  parameter-light fitness is preferable to sorting and density estimation.

## Key points

- IBEA turns a binary indicator into a fitness:
  F(x) = Σ_{y ≠ x} −exp(−I(y, x) / (c κ)); values closer to 0 are better.
- VAMOS supports the additive ε indicator (default) and the hypervolume
  difference, always on objectives normalized to [0, 1] and divided by
  c = max |I|.
- κ (default 0.05) sets the pressure: small κ punishes dominated solutions
  strongly, large κ flattens the fitness.
- Environmental selection removes the lowest fitness one at a time and adds the
  removed solution's term back to every survivor.
- Parents come from binary tournaments on the updated fitness.

**Next:** [SMPSO](smpso.md).

## References

[1] Zitzler, E., and Künzli, S. (2004). [Indicator-based selection in multiobjective search](https://doi.org/10.1007/978-3-540-30217-9_84). *Parallel Problem Solving from Nature – PPSN VIII*, Lecture Notes in Computer Science 3242, 832–842.

[2] Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C. M., and Grunert da Fonseca, V. (2003). [Performance assessment of multiobjective optimizers: An analysis and review](https://doi.org/10.1109/TEVC.2003.810758). *IEEE Transactions on Evolutionary Computation*, 7(2), 117–132.

[3] Zitzler, E., and Thiele, L. (1998). [Multiobjective optimization using evolutionary algorithms – A comparative case study](https://doi.org/10.1007/BFb0056872). *Parallel Problem Solving from Nature – PPSN V*, Lecture Notes in Computer Science 1498, 292–301.

[4] Goldberg, D. E., and Deb, K. (1991). [A comparative analysis of selection schemes used in genetic algorithms](https://doi.org/10.1016/B978-0-08-050684-5.50008-2). *Foundations of Genetic Algorithms*, 1, 69–93.

[5] Wagner, T., Beume, N., and Naujoks, B. (2007). Pareto-, aggregation-, and indicator-based methods in many-objective optimization. *Evolutionary Multi-Criterion Optimization – EMO 2007*, Lecture Notes in Computer Science 4403, 742–756.
