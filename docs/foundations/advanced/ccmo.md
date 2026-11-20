---
description: 'A6. CCMO in depth: a constrained population and a helper population that ignores the constraints, cooperation through shared offspring, SPEA2 fitness with the violation first, environmental selection, budget, and where VAMOS follows the authors'' code, with an interactive figure and VAMOS code.'
search:
  boost: 0.5
---

# A6. CCMO: two cooperating populations

Lesson [A5](constrained.md#beyond-feasibility-first) ended with a warning:
selection that puts feasibility first can be trapped. Once a few feasible
solutions are found they take over the population, and the parts of the
constrained front that lie behind an infeasible barrier are never reached.
CCMO, the **coevolutionary framework for constrained multi-objective
optimization** of Tian, Zhang, Xiao, Zhang and Jin [1], avoids the trap by
keeping a second population that is not bound by the constraints at all.

The two populations have the same size. Population 1 solves the problem as
stated; population 2 solves a *helper problem*, the same objectives without the
constraints. They never mix their members: in every generation each population
breeds its own offspring, and then each one selects its survivors from itself
together with the offspring of **both**. Population 2 moves freely across
infeasible regions towards the unconstrained front, and its offspring give
population 1 candidates in places it could not reach on its own.

NSGA-II with constrained domination (lesson [A5](constrained.md#constrained-domination))
has one population and one rule. CCMO has two populations, two rules and one
pool of offspring. In VAMOS, as in the authors' code, each population is
evolved with the fitness and the environmental selection of SPEA2 [6] (lesson
[I7](../intermediate/spea2.md)), with the constraint violation put first in
population 1.

In this lesson you will learn:

- why a population that ignores the constraints helps the one that respects
  them;
- the helper problem and the weak cooperation through shared offspring;
- the fitness, environmental selection and mating selection of each population
  in VAMOS;
- how a generation spends the budget, and where VAMOS follows the authors' code
  instead of the paper.

## Why a second population

Deb's feasibility rules [2], used by NSGA-II [3], compare two solutions by
feasibility first and by their objectives second. The rules have no parameter
and work well when the feasible region is large and connected. On problems
such as CTP2 [4], whose feasible region near the front is a row of narrow
"tunnels" (lesson [A5](constrained.md#constrained-domination)), they fail in a
characteristic way: a population that has entered one tunnel cannot cross the
infeasible band beside it, because every step across is an infeasible
solution, and feasibility first discards it.

Two remedies were common before CCMO: relax the constraints (penalties,
ε-constrained methods, stochastic ranking; lesson
[A5](constrained.md#beyond-feasibility-first)) or keep infeasible solutions in
an archive. Tian et al. [1] proposed a framework instead. A *helper problem*,
easier than the original one, is solved by a second population, and the two
populations cooperate weakly: they share their offspring and nothing else. In
CCMO the helper problem is the original problem without its constraints. Its
population converges to the unconstrained front, crossing infeasible regions
on the way, and wherever the constrained and the unconstrained fronts are close,
its offspring land near the constrained front.

## The helper problem and the shared offspring

One generation of CCMO, with N solutions in each population:

<div class="vf-formula">population 1:  minimize f(x)  subject to  g_j(x) ≤ 0       (the original problem)
population 2:  minimize f(x)                                     (the helper problem)
Q₁ ← offspring of population 1,   Q₂ ← offspring of population 2   (mating within each population)
P₁ ← the best N of  P₁ ∪ Q₁ ∪ Q₂,   ranked with the constraint violation
P₂ ← the best N of  P₂ ∪ Q₁ ∪ Q₂,   ranked without it</div>

Each population chooses its parents among its own members, so neither one is
pulled towards the other's region. Every offspring is evaluated once and
offered to both. The cooperation is *weak* in that sense: no migration, no
shared archive, no change to either selection rule.

With the default GA variation each population creates N/2 offspring, so both
selections choose N survivors from 2N candidates, as NSGA-II does. The result
of the run is population 1; population 2 is a means, not an answer.

## Fitness with the violation first

Each population ranks its 2N candidates with SPEA2's fitness (lesson
[I7](../intermediate/spea2.md#strength-and-raw-fitness)), computed with a
dominance relation in which the constraint violation CV of lesson
[A5](constrained.md#feasibility-and-violation) comes first:

<div class="vf-formula">a ≺_c b   ⇔   CV(a) &lt; CV(b),   or   CV(a) = CV(b)  and  a ≺ b  (Pareto dominance)
S(i) = |{ j : i ≺_c j }|                      (strength: how many candidates i dominates)
R(j) = Σ_{i ≺_c j} S(i)                       (raw fitness: the strengths of j's dominators)
D(i) = 1 / ( σ_i^k + 2 ),     k = ⌊√n⌋          (σ_i^k: distance to the k-th nearest other candidate)
fitness(i) = R(i) + D(i)                      (to be minimized)</div>

- Every feasible candidate dominates every infeasible one, and of two
  infeasible candidates the less violated one dominates. Two candidates with
  the same violation, both feasible or both infeasible, are compared by Pareto
  dominance.
- Population 2 passes a violation of zero for every candidate, so ≺_c becomes
  plain Pareto dominance: population 2 sees the helper problem.
- σ_i^k is the Euclidean distance in the objective space, without
  normalization, and n is the number of candidates (k = 14 for 2N = 200).
- R is 0 exactly for the candidates that no other one dominates, and
  D ≤ 1/2, so those candidates are the ones with a fitness below 1.

## Environmental selection and mating

Each population keeps N of its candidates by SPEA2's environmental selection:

<div class="vf-formula">keep every candidate with fitness &lt; 1           (the non-dominated ones)
if fewer than N:  keep the N lowest fitness values instead
if more than N:   repeatedly remove the candidate whose sorted distances
                  to the remaining ones are lexicographically smallest</div>

The removal is SPEA2's truncation operator (lesson
[I7](../intermediate/spea2.md#the-truncation-operator)): the candidate with the
closest neighbour goes first, a tie is broken by the second closest, and so on.
The survivors are stored in increasing order of fitness, and that fitness,
computed on the 2N candidates, is reused for mating.

Parents are chosen by binary tournaments on the fitness: two members are drawn
at random (possibly the same one twice), the lower fitness wins, and a tie goes
to the one that comes first. The crossover chooses the variation:

- **GA** (SBX by default, or any two-parent crossover): N tournaments; the i-th
  winner of the first half mates with the i-th of the second half, and only the
  first child of each pair is kept and mutated. Each population creates N/2
  offspring.
- **DE** (crossover `de`, real variables): the i-th child is
  x_i + F (r1_i − r2_i), with r1 and r2 from 2N tournaments, binomial crossover
  of rate CR, then mutation, after Storn and Price [5]. Each population creates
  N offspring.

The two initial populations cost 2N evaluations, so the budget must be at least
2N. A GA generation then costs 2·⌊N/2⌋ evaluations and a DE generation 2N.

## Try it: one set of candidates, two rules

The figure shows the CTP2 objective space of lesson
[A5](constrained.md#constrained-domination) with 16 candidates: eight members
of a population (P1 to P8) and eight offspring (O1 to O8). Circles are
feasible, squares infeasible. Both selection rules of CCMO are applied to the
same 16 candidates and keep N = 8. Start with population 1, then switch to
population 2 and compare the regions each one keeps. Hover a point to see its
violation, its dominators and its fitness.

<div class="vf-widget" data-vf-widget="constrained-selection" data-rule="ccmo"><p class="vf-fallback">Interactive figure: sixteen candidate solutions of CTP2 and the eight that survive in CCMO's constrained population and in its helper population, with the fitness of each candidate. It needs JavaScript.</p></div>

Nine candidates are feasible. Population 1 keeps eight of them: the six that no
feasible candidate dominates, then P2 and O2, each dominated only by O1. It
drops P6, dominated by three feasible candidates, and every infeasible one.
Population 2 keeps the seven candidates that are non-dominated when the
constraints are ignored, six of them infeasible and close to the dashed
unconstrained front, plus P8, dominated only by O7. The two populations cover
different parts of the plane, which is the point: the next offspring of
population 2 will be bred near the unconstrained front, and population 1 will
keep those that turn out to be feasible.

## In VAMOS

CCMO on CTP2 [4], with two populations of 100 and 20,000 evaluations. On CTP2
the sum f₁ + f₂ equals the distance function g(x), whose minimum 1 is reached
on the unconstrained front; on the constrained front f₁ + f₂ grows from 1 at
f₁ = 0 to about 1.27 at f₁ = 1.

```python
import numpy as np

from vamos import optimize
from vamos.problems import CTP

result = optimize(CTP(2), algorithm="ccmo", pop_size=100, max_evaluations=20_000, engine="numpy", seed=1)
print("evaluations:", result.data["evaluations"])
for name in ("population", "helper_population"):
    F, G = result.data[name]["F"], result.data[name]["G"]
    feasible = int(np.all(G <= 0.0, axis=1).sum())
    print(f"{name}: {feasible} of {len(F)} feasible, mean f1 + f2 = {F.sum(axis=1).mean():.3f}")
```

Output:

```text
evaluations: 20000
population: 100 of 100 feasible, mean f1 + f2 = 1.088
helper_population: 1 of 100 feasible, mean f1 + f2 = 1.001
```

Population 1 is entirely feasible. Population 2 sits on the unconstrained front
(f₁ + f₂ ≈ 1), and all but one of its members are infeasible, as expected from
a population that never looks at the constraints. `result.data["population"]`
is population 1, the result of the authors' code; `result.X` and `result.F`
hold its rows, or its feasible non-dominated rows when the configuration asks
for `result_mode="non_dominated"` (the default of `CCMOConfig.default()`).

The dominance relation and the fitness follow the formulas above line by line.
`strength_raw_fitness` is SPEA2's: the strength of each row, then the sum of the
strengths of its dominators.

```python
# Excerpt from src/vamos/engine/algorithm/ccmo/selection.py
def constraint_first_dominance(F: NDArray[Any], violation: NDArray[Any]) -> NDArray[np.bool_]:
    objectives = np.asarray(F, dtype=float)
    cv = np.asarray(violation, dtype=float)
    smaller = cv[:, None] < cv[None, :]
    equal = cv[:, None] == cv[None, :]
    no_worse = np.all(objectives[:, None, :] <= objectives[None, :, :], axis=2)
    better = np.any(objectives[:, None, :] < objectives[None, :, :], axis=2)
    return np.asarray(smaller | (equal & no_worse & better), dtype=bool)


def ccmo_fitness(F: NDArray[Any], violation: NDArray[Any]) -> tuple[FloatArray, FloatArray]:
    objectives = np.asarray(F, dtype=float)
    n = objectives.shape[0]
    distances = objective_distances(objectives)
    if n == 0:
        return np.empty(0, dtype=float), distances
    raw = strength_raw_fitness(constraint_first_dominance(objectives, violation))
    others = distances.copy()
    np.fill_diagonal(others, np.inf)
    k = math.isqrt(n)
    # Every row has n - 1 finite distances, and k <= n - 1 for n >= 2; with n = 1 the distance is infinite.
    kth = np.partition(others, kth=k - 1, axis=1)[:, k - 1]
    return np.asarray(raw + 1.0 / (kth + 2.0), dtype=float), distances
```

`smaller` and `equal` compare every pair of violations at once; a row
dominates another if its violation is smaller or, with an equal violation, if
it is no worse in every objective and better in one. In the fitness, the
diagonal of the distance matrix is set to infinity so that a candidate is not
its own neighbour, and `np.partition` finds the k-th smallest distance of each
row without sorting it.

The environmental selection keeps the rows with a fitness below 1 and falls
back on the lowest fitness or on the truncation:

```python
# Excerpt from src/vamos/engine/algorithm/ccmo/selection.py
    fitness, distances = ccmo_fitness(F, violation)
    n = fitness.shape[0]
    size = min(int(n_select), n)
    nondominated = np.flatnonzero(fitness < 1.0)
    if nondominated.size < size:
        selected = np.argsort(fitness, kind="stable")[:size]
    elif nondominated.size > size:
        keep = truncate_lexicographic(distances[np.ix_(nondominated, nondominated)], size, selection_kernels)
        selected = nondominated[keep]
    else:
        selected = nondominated
    selected = np.sort(selected)
    order = np.asarray(selected[np.argsort(fitness[selected], kind="stable")], dtype=int)
    return order, np.asarray(fitness[order], dtype=float)
```

The stable sorts give every tie to the row that comes first, as the authors'
code does. The truncation receives only the distances among the non-dominated
rows. The last two lines return the survivors in increasing order of fitness.

The algorithm class puts the pieces together. `ask` breeds the offspring of
both populations, population 1's first, and cuts them to the remaining budget;
`tell` runs the two selections on the same offspring, once with the violation
and once with zeros:

```python
# Excerpt from src/vamos/engine/algorithm/ccmo/ccmo.py
        request = capped_offspring_size(st.n_eval, st.max_evals, st.offspring_size, "CCMO")
        main = self._offspring(st.X, st.fitness)
        helper = self._offspring(st.helper_X, st.helper_fitness)
        X_off = np.concatenate([main, helper])[:request]

        # Population 1 selects from [itself, offspring] with the constraint violation.
        X_union = np.vstack([st.X, X_off])
        F_union = np.vstack([st.F, F_off])
        G_union = combine_constraints(st.G, G_off, st.constraint_mode, algorithm="CCMO")
        selected, st.fitness = ccmo_environmental_selection(F_union, _violation(G_union, F_union.shape[0]), st.pop_size, kernels)
        st.X, st.F = X_union[selected], F_union[selected]
        st.G = G_union[selected] if G_union is not None else None

        # Population 2 selects from [itself, offspring] without it.
        helper_X_union = np.vstack([st.helper_X, X_off])
        helper_F_union = np.vstack([st.helper_F, F_off])
        helper_G_union = combine_constraints(st.helper_G, G_off, st.constraint_mode, algorithm="CCMO")
        selected, st.helper_fitness = ccmo_environmental_selection(helper_F_union, np.zeros(helper_F_union.shape[0]), st.pop_size, kernels)
```

`[:request]` is the exact budget: when fewer evaluations are left than a full
generation creates, the offspring of population 2, which come last, are the
first to be cut, and both populations then select from the offspring that were
evaluated. Population 2 still stores the constraint values of its members
(`helper_G_union`), so that a result can report their feasibility, but its
selection never reads them.

## Where VAMOS differs from the paper

VAMOS follows the authors' reference code through jMetal's port, whose fitness
and selection the [CCMO lesson](../../algorithms/ccmo.md#where-vamos-departs-from-the-reference-code)
checks against jMetal's own classes. The code differs from the paper, and VAMOS
follows the code:

- the paper embeds NSGA-II with SPEA2's truncation; the code evolves each
  population with SPEA2's fitness and environmental selection;
- the paper treats two infeasible solutions with the same violation as
  non-dominated; the code compares them by Pareto dominance;
- the GA variation selects N parents for N/2 offspring per population; the
  paper selects N/2. The DE variation is not described in the paper.

VAMOS departs from the code itself only where the framework requires it:

- **Budget:** exact; the reference code checks the budget before a generation
  and can exceed it in the last one.
- **Initialization:** both populations come from the configured initializer
  (uniform by default), population 1 first, from the run's generator.
- **Result:** the reference code returns population 1, which is
  `result.data["population"]`; `result.X` and `result.F` follow the configured
  `result_mode`.

The [CCMO lesson](../../algorithms/ccmo.md#agreement-with-the-published-results)
records the agreement with jMetal and with the paper on MW1, MW3 and C1-DTLZ1.

## CCMO among the constrained algorithms

| | NSGA-II with constrained domination | CCMO |
| --- | --- | --- |
| Populations | One | Two of the same size |
| Constraints | Feasible first, then violation, then dominance | Population 1 like NSGA-II; population 2 ignores them |
| Diversity | Crowding distance | SPEA2 density and truncation |
| Extra parameters | None | None (the helper problem is fixed) |
| Evaluations per generation | N | N with the GA variation, shared by both populations |

Strengths:

- **No new parameter.** There is no penalty factor, no ε schedule and no
  switching threshold; the configuration is that of a GA.
- **Exploration across barriers.** Population 2 is never stopped by an
  infeasible region, so population 1 receives offspring from places it could
  not reach.
- **Population 1 is unaffected when the help is useless.** It selects with the
  violation first, so infeasible offspring of population 2 cannot displace its
  feasible members.

Limits:

- **Half of the offspring come from the helper.** When the unconstrained front
  is far from the constrained one, population 2 converges where its offspring
  are of little use, and population 1 effectively evolves with half its
  offspring.
- **Population 1 can still be trapped.** It keeps feasibility first; the
  [CCMO lesson](../../algorithms/ccmo.md#agreement-with-the-published-results)
  reports two of 31 runs on MW1 that ended trapped; jMetal's own CCMO was
  trapped in one of 100.

The next three lessons answer the same question differently: C-TAEA keeps two
archives with one population of offspring, ToP splits the run into two phases,
and CMOEA-MS switches between two ranking rules.

## Key points

- CCMO evolves population 1 on the constrained problem and population 2 on a
  helper problem, the same objectives without constraints.
- The populations cooperate only through their offspring: each one selects N
  survivors from itself and the offspring of both.
- In VAMOS, as in the authors' code, each population uses SPEA2's fitness,
  R + 1/(σ^k + 2), with a dominance relation that puts the violation first in
  population 1 and ignores it in population 2.
- Survivors are the candidates with fitness below 1, completed by the lowest
  fitness or reduced by SPEA2's truncation.
- The result is population 1; the budget is exact, and the offspring of
  population 2 are cut first.

**Next:** [A7. C-TAEA: a convergence archive and a diversity archive](ctaea.md).

## References

[1] Tian, Y., Zhang, T., Xiao, J., Zhang, X., and Jin, Y. (2021). [A coevolutionary framework for constrained multiobjective optimization problems](https://doi.org/10.1109/TEVC.2020.3004012). *IEEE Transactions on Evolutionary Computation*, 25(1), 102–116.

[2] Deb, K. (2000). [An efficient constraint handling method for genetic algorithms](https://doi.org/10.1016/S0045-7825(99)00389-8). *Computer Methods in Applied Mechanics and Engineering*, 186(2–4), 311–338.

[3] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://doi.org/10.1109/4235.996017). *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

[4] Deb, K., Pratap, A., and Meyarivan, T. (2001). [Constrained test problems for multi-objective evolutionary optimization](https://doi.org/10.1007/3-540-44719-9_20). *Evolutionary Multi-Criterion Optimization – EMO 2001*, Lecture Notes in Computer Science 1993, 284–298.

[5] Storn, R., and Price, K. (1997). [Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces](https://doi.org/10.1023/A:1008202821328). *Journal of Global Optimization*, 11(4), 341–359.

[6] Zitzler, E., Laumanns, M., and Thiele, L. (2001). [SPEA2: Improving the strength Pareto evolutionary algorithm](https://tik-db.ee.ethz.ch/db/public/tik/?db=publications&form=report_single_publication&publication_id=1287). TIK Report 103, Computer Engineering and Networks Laboratory, ETH Zurich.
