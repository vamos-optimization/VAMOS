---
description: 'A8. ToP in depth: a first phase that solves a constrained single-objective problem with differential evolution and the feasibility rule, the switch on the feasible share and on delta, a second phase with NSGA-II and constrained domination, and where VAMOS follows PlatEMO and jMetal, with an interactive figure and VAMOS code.'
search:
  boost: 0.5
---

# A8. ToP: two phases

CCMO and C-TAEA (lessons [A6](ccmo.md) and [A7](ctaea.md)) run their
constrained and their exploring parts side by side for the whole run. The
**two-phase framework**, ToP, of Liu and Wang [1] runs them one after the
other. Phase 1 looks for the feasible regions that hold the front by solving a
simpler problem: a *single-objective* constrained problem whose objective is
the mean of the objectives. When the population has enough feasible solutions
and its best ones agree on a region, or when 90% of the budget is spent, phase
2 starts: NSGA-II with constrained domination spreads the population along the
front until the end of the run.

Liu and Wang address problems whose constraints act in the decision space as
well as in the objective space, where the feasible regions can be small and
scattered. A multi-objective search spreads its effort along the whole front
from the start; a single-objective search concentrates it. ToP spends the
first part of the run on that concentration and the rest on the spread.

Compared with NSGA-II, ToP adds a whole phase before it, with its own
variation (differential evolution), its own selection (one-to-one replacement
by the feasibility rule) and a switching rule with three thresholds.

In this lesson you will learn:

- the single-objective problem of phase 1 and how its population is replaced;
- the two quantities that end phase 1, the feasible share Pf and the spread δ,
  and their thresholds;
- phase 2, NSGA-II with constrained domination as ToP implements it;
- where VAMOS follows PlatEMO's implementation through jMetal's port instead of
  the paper.

## Why two phases

Deb's feasibility rules [2] rank feasible solutions before infeasible ones,
and NSGA-II [3] uses them to compare solutions inside its non-dominated
sorting. Both work well once the population sits in the feasible regions that
matter. The hard part, on problems with small or scattered feasible regions,
is to get there. The idea of ToP [1] is that a constrained *single-objective*
problem is easier to search: the feasibility rule then compares two feasible
solutions on one number, and the population contracts around the best feasible
region it finds instead of spreading along a front.

ToP therefore splits the run. Phase 1 replaces the m objectives by their
weighted sum with equal weights, which VAMOS computes as their mean, and solves
the resulting constrained single-objective problem. Its population does not
try to cover the front, only to find feasible solutions where the objectives
are good. Phase 2 starts from that population and solves the original problem.

## Phase 1: one objective and the feasibility rule

Each solution x_i of the population creates one trial vector u_i by
differential evolution [4], and u_i competes only with x_i:

<div class="vf-formula">f̄(x) = ( f₁(x) + … + f_m(x) ) / m                     (the objective of phase 1)
u_i replaces x_i   ⇔   CV(u_i) &lt; CV(x_i),   or   CV(u_i) = CV(x_i)  and  f̄(u_i) &lt; f̄(x_i)</div>

The rule is Deb's feasibility rule applied to one pair: a feasible trial beats
an infeasible parent, the less violated of two infeasible ones wins, and two
solutions with the same violation, feasible ones included, are compared by f̄.
The population never shrinks or grows, and a solution is only ever replaced by
a better one.

The trial vector comes from one of two DE strategies, chosen with probability
0.5 each:

<div class="vf-formula">DE/current-to-rand/1:     v_i = x_i + F (x_r1 − x_i) + F (x_r2 − x_r3)           (base vector x_i)
DE/rand-to-best/1/bin:    v_i = x_r1 + F (x_best − x_r1) + F (x_r2 − x_r3)        (base vector x_r1)
u_i:  each variable from v_i with probability CR, otherwise from the base vector</div>

x_best is the solution with the lowest f̄, feasible or not, and r1, r2, r3 are
random members of the population. In VAMOS both mutants go through the
binomial crossover with their base vector; the trial vector is then clamped to
the bounds and mutated with polynomial mutation. The defaults are CR = 1.0,
with which the trial vector is the whole mutant, and F = 0.5.

On CTP2 [5] phase 1 has a simple meaning: f₁ + f₂ is the distance function
g(x), so f̄ = g/2, and phase 1 minimizes g subject to the constraint. Its lines
of equal f̄ are parallel to the unconstrained front.

## The switch: Pf and δ

After each generation of phase 1 two quantities are updated from the
population:

<div class="vf-formula">Pf = (number of feasible solutions) / N
for the n feasible solutions:  f̂ = ( f − min ) / ( max − min )   (per objective, min and max over them)
s₁ ≤ s₂ ≤ … ≤ s_n  = the sorted sums  f̂₁ + … + f̂_m
δ = s_⌊n/3⌋ − s₁                                (unchanged while fewer than 3 are feasible)
the next generation is in phase 1  ⇔  ( δ ≥ 0.2  or  Pf ≤ 1/3 )  and  evaluations ≤ 0.9 · budget</div>

Pf asks whether enough of the population is feasible. δ asks whether the best
third of the feasible solutions, by the sum of their normalized objectives,
are close to the best one: a small δ means that the feasible solutions with the
best objective values agree on a region. Phase 1 ends when both answers are
yes, or when 90% of the budget is spent. δ and Pf change only in phase 1, so
once phase 2 starts it lasts until the end. The thresholds are
`delta_threshold` (0.2), `feasible_ratio_threshold` (1/3) and
`phase_one_evaluation_fraction` (0.9) in `ToPConfig`.

The normalization uses the feasible solutions of the current population only,
so δ is scale-free: a population that has contracted to a tiny region still
has normalized sums between 0 and m, and δ is small only if the best third are
close *relative to the spread of all feasible solutions*.

## Phase 2: NSGA-II with constrained domination

Phase 2 is NSGA-II (lesson [I5](../intermediate/elitist-survival.md#survival-with-constraints))
with ToP's constrained dominance:

<div class="vf-formula">a ≺ b   ⇔   both feasible and a dominates b in the objectives,   or   CV(a) &lt; CV(b)
fronts:  the feasible solutions by non-dominated sorting, then one front per distinct violation
survival:  whole fronts of P ∪ Q while they fit, the last front by crowding distance</div>

Every infeasible solution is behind every feasible one, and infeasible
solutions with the same violation share a front whatever their objectives.
Each offspring is x₁ + F (x₂ − x₃) with binomial crossover (DE/rand/1/bin),
where x₁ wins a binary tournament on front and crowding distance and x₂, x₃ are
random; it is then clamped and mutated, as in phase 1.

## Try it: the two phases on the same candidates

The figure shows the 16 CTP2 candidates of lesson
[A6](ccmo.md#try-it-one-set-of-candidates-two-rules). In phase 1, O1 to O8 are
the trial vectors of P1 to P8, and each pair is joined by a segment. In phase 2,
all 16 are the union of population and offspring. Follow the eight pairs of
phase 1, read Pf and δ under the plot, then switch to phase 2.

<div class="vf-widget" data-vf-widget="constrained-selection" data-rule="top"><p class="vf-fallback">Interactive figure: sixteen candidate solutions of CTP2 and the eight that survive in ToP's phase 1, trial vector against parent by the feasibility rule on the mean objective, and in phase 2, NSGA-II with constrained domination, with the switch quantities Pf and delta. It needs JavaScript.</p></div>

In phase 1 the violation decides six of the eight pairs. O3 has a smaller mean
objective than P3 (0.505 against 0.548) but is infeasible, so P3 stays; O8 has
a larger mean than P8 (0.635 against 0.575) but is feasible, so it replaces P8.
Only O2 and O6 win on the mean objective, against feasible parents. Seven of
the eight survivors are feasible, so Pf = 0.875, and the best third of them lie
close together: δ = 0.051. Both tests pass, and the next generation would be in
phase 2. In phase 2, the six non-dominated feasible candidates fill the first
front; the second front, P2, O2 and P6, must lose one, and crowding removes O2,
the middle one. No infeasible candidate survives either phase 2 or, apart from
P4, phase 1.

## In VAMOS

ToP on CTP2 and on MW3 (15 variables, two constraints), with a population of
100 and 20,000 evaluations, and on CTP2 a second time with phase 1 limited to
50% of the budget. The [problem reference](../../reference/problems.md) lists
both problems.

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import ToPConfig

for name, fraction in (("ctp2", 0.9), ("ctp2", 0.5), ("mw3", 0.9)):
    config = (
        ToPConfig.builder()
        .pop_size(100)
        .crossover("de", cr=1.0, f=0.5)
        .mutation("pm", prob="1/n", eta=20.0)
        .phase_one_evaluation_fraction(fraction)
        .result_mode("population")
        .build()
    )
    result = optimize(name, algorithm="top", algorithm_config=config, max_evaluations=20_000, engine="numpy", seed=1)
    G = result.data["population"]["G"]
    feasible = int(np.all(G <= 0.0, axis=1).sum())
    print(f"{name}, phase 1 for at most {fraction:.0%} of the budget: phase 2 from evaluation {result.data['phase_two_start']}, {feasible} of {len(G)} feasible")
```

Output:

```text
ctp2, phase 1 for at most 90% of the budget: phase 2 from evaluation 18100, 100 of 100 feasible
ctp2, phase 1 for at most 50% of the budget: phase 2 from evaluation 10100, 100 of 100 feasible
mw3, phase 1 for at most 90% of the budget: phase 2 from evaluation 3400, 100 of 100 feasible
```

`phase_two_start` is the number of evaluations done when phase 2 started. On
MW3, Pf and δ pass their thresholds early, and phase 2 starts after 3,400
evaluations. On CTP2 they never do: a
random solution of CTP2 has a large g and lies far above the front, where the
whole objective space is feasible, so the population is feasible from the first
generation and Pf = 1 throughout (the feasibility rule never replaces a
feasible solution by an infeasible one). What keeps phase 1 going is δ: while
DE descends the many local minima of g, the best third of the population stays
spread relative to the rest, δ stays above 0.2, and phase 1 runs until its
evaluation limit. The generation that starts at 18,000 evaluations, exactly
90%, still belongs to phase 1, so phase 2 starts at 18,100; with a limit of
50% it starts at 10,100. The switch thresholds are therefore problem
parameters, not constants: `phase_one_evaluation_fraction` bounds what phase 1
may cost when δ does not fall.

The replacement rule, δ and the switch:

```python
# Excerpt from src/vamos/engine/algorithm/top/phases.py
def offspring_replaces_parent(
    offspring_violation: NDArray[Any],
    offspring_mean: NDArray[Any],
    parent_violation: NDArray[Any],
    parent_mean: NDArray[Any],
) -> NDArray[np.bool_]:
    offspring_cv = np.asarray(offspring_violation, dtype=float)
    parent_cv = np.asarray(parent_violation, dtype=float)
    smaller_mean = np.asarray(offspring_mean, dtype=float) < np.asarray(parent_mean, dtype=float)
    return np.asarray((offspring_cv < parent_cv) | ((offspring_cv == parent_cv) & smaller_mean), dtype=bool)


def updated_delta(F: NDArray[Any], violation: NDArray[Any], delta: float) -> float:
    objectives = np.asarray(F, dtype=float)[np.asarray(violation, dtype=float) == 0.0]
    n = objectives.shape[0]
    if n < 3:
        return float(delta)
    minimum = np.min(objectives, axis=0)
    maximum = np.max(objectives, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = (objectives - minimum) / (maximum - minimum)
    sums = np.zeros(n, dtype=float)
    for j in range(normalized.shape[1]):
        sums = sums + normalized[:, j]
    sums = np.sort(sums)
    return float(sums[n // 3 - 1] - sums[0])


def phase_one_continues(
    delta: float,
    ratio: float,
    evaluations: int,
    max_evaluations: int,
    *,
    delta_threshold: float = DELTA_THRESHOLD,
    feasible_ratio_threshold: float = FEASIBLE_RATIO_THRESHOLD,
    phase_one_evaluation_fraction: float = PHASE_ONE_EVALUATION_FRACTION,
) -> bool:
    searching = delta >= delta_threshold or ratio <= feasible_ratio_threshold
    return bool(searching and evaluations <= phase_one_evaluation_fraction * max_evaluations)
```

The comparison is strict, so a trial vector with the same violation and the
same mean objective does not replace its parent. `sums[n // 3 - 1]` is the
⌊n/3⌋-th smallest sum. The sums are added objective by objective from zero, as
jMetal's loops do, so that the floating-point values, and the ties they
create, are jMetal's. In the switch, a NaN δ, which an objective with a single
value among the feasible solutions produces, fails `delta >= delta_threshold`;
the algorithm calls it with the evaluations done when a generation starts.

Phase 2's fronts and survival:

```python
# Excerpt from src/vamos/engine/algorithm/top/selection.py
def constrained_front_numbers(F: NDArray[Any], violation: NDArray[Any]) -> IndexArray:
    objectives = np.asarray(F, dtype=float)
    cv = np.asarray(violation, dtype=float)
    fronts = np.zeros(cv.shape[0], dtype=int)
    feasible = cv == 0.0
    feasible_fronts = 0
    if feasible.any():
        ranks = front_ranks(objectives[feasible])
        fronts[feasible] = ranks
        feasible_fronts = int(ranks.max()) + 1
    if not feasible.all():
        _, levels = np.unique(cv[~feasible], return_inverse=True)
        fronts[~feasible] = feasible_fronts + np.asarray(levels, dtype=int).reshape(-1)
    return fronts
```

Instead of running a sort with the constrained dominance, the function builds
its result directly: Pareto fronts for the feasible rows, then `np.unique`
numbers the distinct violations of the infeasible rows in increasing order,
one front each. `environmental_selection` then keeps whole fronts while they
fit and takes the rest of the last front by decreasing crowding distance.

Phase 1 in the algorithm class. The trial vectors:

```python
# Excerpt from src/vamos/engine/algorithm/top/top.py
    def _phase_one_offspring(self) -> np.ndarray[Any, Any]:
        st = self._st
        X = np.asarray(st.X, dtype=float)
        n = int(X.shape[0])
        best = X[index_of_lowest_mean_objective(st.F)]
        current_to_rand = st.rng.random(n) < CURRENT_TO_RAND_PROBABILITY
        drawn = st.rng.integers(0, n, size=(n, 3))
        r1, r2, r3 = X[drawn[:, 0]], X[drawn[:, 1]], X[drawn[:, 2]]
        choice = current_to_rand[:, None]
        base = np.where(choice, X, r1)
        mutant = np.where(choice, current_to_rand_1(X, r1, r2, r3, st.f), rand_to_best_1(best, r1, r2, r3, st.f))
        return self._trial_vectors(base, mutant)
```

The random numbers are drawn in blocks: the choice of strategy for every
solution, then the indices r1, r2 and r3 (with replacement), then, inside
`_trial_vectors`, the crossover and the mutation. `base` is the vector that
the crossover starts from: x_i for DE/current-to-rand/1, x_r1 for
DE/rand-to-best/1/bin. When the evaluated trial vectors come back, `tell`
replaces the parents and updates Pf and δ:

```python
# Excerpt from src/vamos/engine/algorithm/top/top.py
        if st.pending_phase == 1:
            # Each evaluated trial vector faces the solution at its position; the others are kept.
            violation = constraint_violation(st.G, st.F.shape[0])
            replaces = offspring_replaces_parent(
                constraint_violation(G_off, k), mean_objectives(F_off), violation[:k], mean_objectives(st.F[:k])
            )
            rows = np.flatnonzero(replaces)
            X, F = st.X.copy(), st.F.copy()
            X[rows], F[rows] = X_off[rows], F_off[rows]
            st.X, st.F, st.G = X, F, G
            violation = constraint_violation(st.G, st.F.shape[0])
            st.feasible_ratio = feasible_ratio(violation)
            st.delta = updated_delta(st.F, violation, st.delta)
```

`k` is the number of trial vectors that the budget allowed: a cut last
generation compares only the first k pairs and keeps the other solutions. The
violation is computed again after the replacement, so Pf and δ describe the new
population.

## Where VAMOS differs from the paper

VAMOS follows jMetal's port, which follows PlatEMO's implementation where it
differs from the paper; the [ToP lesson](../../algorithms/top.md#where-vamos-departs-from-jmetals-port)
lists the checks against jMetal's classes. Relative to the paper:

- phase 1 draws r1, r2 and r3 with replacement, instead of mutually different
  and different from i, and uses a fixed F and CR instead of the paper's pools
  of values;
- the binomial crossover has no j_rand (no variable is forced from the mutant)
  and, in DE/rand-to-best/1/bin, the variables that are not crossed come from
  x_r1 instead of x_i; neither difference matters with CR = 1;
- the trial vectors of phase 1 are clamped and mutated;
- δ normalizes with the feasible solutions of the current population instead
  of all those found, keeps its value with fewer than three feasible solutions,
  and phase 1 also ends at 90% of the budget;
- the mean of the objectives replaces their sum, which gives the same
  comparisons;
- phase 2 creates its offspring with DE/rand/1/bin and polynomial mutation
  instead of SBX.

Relative to jMetal's port:

- **Budget:** exact; a cut last generation evaluates the first offspring only,
  while jMetal can exceed the budget by up to N − 1 evaluations. The phase of a
  generation is decided when it starts, so the decisions are jMetal's.
- **Random draws:** in blocks, as described above, while jMetal draws them
  solution by solution; each draw has jMetal's distribution, but runs with the
  same seed differ.
- **Result:** jMetal returns the whole final population, which is
  `result.data["population"]`; `result.X` and `result.F` follow the configured
  `result_mode`.

The [ToP lesson](../../algorithms/top.md#agreement-with-jmetals-port-and-the-published-results)
records the agreement with jMetal and with the published results.

## ToP among the constrained algorithms

| | CCMO, C-TAEA (lessons A6, A7) | ToP |
| --- | --- | --- |
| Exploring part | Runs alongside the constrained part all the time | Phase 1, before the constrained part |
| What it explores with | The problem without constraints | A constrained single-objective problem |
| Infeasible solutions | Kept by the helper population or the DA | Replaced by a less violated trial vector in phase 1, ranked last in phase 2 |
| Switching rule | None | δ, Pf and an evaluation limit |
| Variation | GA (SBX) or DE | DE in both phases |

Strengths:

- **Fast entry into small feasible regions.** A single objective and one-to-one
  replacement concentrate the population quickly.
- **Plain NSGA-II afterwards.** Phase 2 is cheap and well understood.

Limits:

- **The switch is a guess about the problem.** On CTP2 above, δ stays high and
  phase 1 uses 90% of the budget, leaving 2,000 evaluations to spread the
  population along the front.
- **One region at a time.** The mean objective rewards one part of the front;
  the [ToP lesson](../../algorithms/top.md#agreement-with-jmetals-port-and-the-published-results)
  records MW3 runs whose population converges to one end of the front, and MW1
  runs that end without a feasible solution.
- **No exploration after the switch.** Phase 2 ranks every feasible solution
  first, so infeasible barriers met later are not crossed.

## Key points

- ToP solves a constrained single-objective problem first, the mean of the
  objectives with the feasibility rule, then the original problem with NSGA-II
  and constrained dominance.
- In phase 1 each trial vector faces only its own parent: a smaller violation
  wins, then a smaller mean objective.
- Phase 1 lasts while δ ≥ 0.2 or Pf ≤ 1/3, and at most until 90% of the budget;
  δ measures how close the best third of the feasible solutions are.
- Phase 2 sorts the feasible solutions into Pareto fronts and puts one front per
  violation value after them.
- VAMOS follows PlatEMO's implementation through jMetal's port, with DE in both
  phases and an exact budget.

**Next:** [A9. CMOEA-MS: two stages](cmoeams.md).

## References

[1] Liu, Z.-Z., and Wang, Y. (2019). [Handling constrained multiobjective optimization problems with constraints in both the decision and objective spaces](https://doi.org/10.1109/TEVC.2019.2894743). *IEEE Transactions on Evolutionary Computation*, 23(5), 870–884.

[2] Deb, K. (2000). [An efficient constraint handling method for genetic algorithms](https://doi.org/10.1016/S0045-7825(99)00389-8). *Computer Methods in Applied Mechanics and Engineering*, 186(2–4), 311–338.

[3] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://doi.org/10.1109/4235.996017). *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

[4] Storn, R., and Price, K. (1997). [Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces](https://doi.org/10.1023/A:1008202821328). *Journal of Global Optimization*, 11(4), 341–359.

[5] Deb, K., Pratap, A., and Meyarivan, T. (2001). [Constrained test problems for multi-objective evolutionary optimization](https://doi.org/10.1007/3-540-44719-9_20). *Evolutionary Multi-Criterion Optimization – EMO 2001*, Lecture Notes in Computer Science 1993, 284–298.
