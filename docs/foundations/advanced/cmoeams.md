---
description: 'A9. CMOEA-MS in depth: a normalized constraint violation, stage A that ranks density and violation as two objectives, stage B that puts the violation first, the switch on the feasible share and lambda, SPEA2 fitness with cosine distances, and where VAMOS follows PlatEMO and jMetal, with an interactive figure and VAMOS code.'
search:
  boost: 0.5
---

# A9. CMOEA-MS: two stages

ToP (lesson [A8](top.md)) explores first and converges afterwards, and once its
second phase starts it never goes back. The **constrained multi-objective
evolutionary algorithm with multiple stages**, CMOEA-MS, of Tian, Zhang, Su,
Zhang, Tan and Jin [1] also has two ways of ranking solutions, but it chooses
between them at every generation, from the share of feasible solutions, and it
can return to the first one whenever feasibility becomes scarce again.

In **stage A** the objectives and the constraints have the same priority: each
solution is described by two numbers, its density in the objective space and
its constraint violation, and these two numbers are ranked by Pareto
dominance. A sparse infeasible solution can then beat a crowded feasible one.
In **stage B** the constraints come first: a smaller violation is better, and
equal violations are compared by the objectives. The algorithm is in stage B
while more than half of the candidates are feasible, once 10% of the budget is
spent.

CMOEA-MS keeps a single population, like NSGA-II, and both stages use the
fitness and the environmental selection of SPEA2 [3] (lesson
[I7](../intermediate/spea2.md)). What changes between the stages is the
dominance relation behind the fitness.

In this lesson you will learn:

- the normalized constraint violation of CMOEA-MS;
- the shift-based density estimation, and how stage A ranks density and
  violation as two objectives;
- stage B, the fitness and the environmental selection with cosine distances;
- the rule that chooses the stage, its parameter λ, and where VAMOS follows
  PlatEMO's implementation through jMetal's port.

## Balancing objectives and constraints

A feasibility-first rule (lesson [A5](constrained.md#constrained-domination))
gives constraint satisfaction absolute priority, and can be trapped behind
infeasible barriers. Ignoring the constraints gives the objectives absolute
priority, and converges where the solutions cannot be used. CMOEA-MS [1]
makes the balance depend on the state of the search: while few solutions are
feasible, objective optimization and constraint satisfaction weigh the same,
and the population can keep well-spread solutions on both sides of the
feasibility boundary; once most solutions are feasible, constraint
satisfaction comes first, and the population converges inside the feasible
region.

The two stages implement these two priorities, and the share of feasible
solutions decides between them. Unlike ToP's phases, the stages alternate as
often as the share crosses its threshold.

## The normalized violation

The violation of each constraint is divided by its largest value among the
candidates, and the violation of a solution is the mean over the constraints:

<div class="vf-formula">CV(x) = (1/q) Σ_j  max( 0, g_j(x) ) / max_{y ∈ Q} max( 0, g_j(y) )         (a term is 0 if nobody violates g_j)</div>

Q is the union of the population and the offspring, and q the number of
constraints. Every constraint contributes on the same scale, and CV is in
[0, 1]. A solution is feasible when CV(x) = 0.

## Stage A: density and violation as two objectives

Stage A replaces the m objectives by one measure of how good and how isolated
a solution is in the objective space, the **shift-based density estimation**
(SDE) of Li, Yang and Liu [2]. To estimate the density around x, every other
solution y is first *shifted*: in each objective where y is better than x, it
takes x's value. Then distances are measured as usual:

<div class="vf-formula">f̂ = objectives normalized to [0, 1] over Q
y′_i = max( f̂_i(y), f̂_i(x) )                           (y shifted towards x)
d(x) = the (⌊√|Q|⌋ + 1)-th smallest of ‖ f̂(x) − y′ ‖ over all y ∈ Q   (x itself included, at distance 0)
SDE(x) = 1 / ( d(x) + 2 )                               (lower = less crowded)
stage A ranks the points  [ SDE(x), CV(x) ]  by Pareto dominance</div>

Shifting moves the solutions that are better than x in some objective onto x
in that objective. A solution that is worse than many others therefore sees
them close by and gets a high density; a solution that is good in the
objectives and far from its neighbours gets a low one. SDE was designed for
many-objective Pareto-based algorithms, where it penalizes poor convergence and
crowding in one number [2]. In stage A that number becomes one objective and
the violation the other: a solution survives if no other one is both less
crowded and less violated.

## Stage B: the violation first

Stage B ranks the objectives themselves, with the violation first:

<div class="vf-formula">a ≺_B b   ⇔   CV(a) &lt; CV(b),   or   CV(a) = CV(b)  and  a ≺ b  (Pareto dominance on the objectives)</div>

This is the relation of CCMO's population 1 (lesson
[A6](ccmo.md#fitness-with-the-violation-first)): every feasible solution
dominates every infeasible one, and infeasible solutions are ordered by their
violation.

## Fitness and environmental selection

Both stages compute SPEA2's fitness of the 2N solutions of Q with their own
dominance relation, on their own points ([SDE, CV] in stage A, the objectives in
stage B):

<div class="vf-formula">S(i) = number of solutions that i dominates        R(j) = Σ_{i dominates j} S(i)
fitness(i) = R(i) + 1 / ( d_cos(i) + 2 ),    d_cos = cosine distance to the ⌊√|Q|⌋-th nearest other point
keep every solution with fitness &lt; 1
if fewer than N:  keep the N lowest fitness values
if more than N:   remove one at a time the solution whose sorted cosine distances
                  between objective vectors are lexicographically smallest</div>

The cosine distance, 1 − cos of the angle between two vectors, compares
directions from the origin rather than positions. The truncation always uses
the objective vectors, in stage A too. The survivors keep their order in Q, and
their fitness is reused by the binary tournaments of the next generation.

## The stage rule

<div class="vf-formula">Q is in stage B   ⇔   |{ x ∈ Q : CV(x) = 0 }| / |Q|  >  λ    and    evaluations ≥ 10% of the budget
otherwise, and for the initial population, Q is in stage A                        (λ = 0.5 by default)</div>

The rule is checked at every generation, after the offspring are evaluated, so
the algorithm moves back to stage A whenever half or more of Q is infeasible.
λ is `feasible_ratio_threshold` in `CMOEAMSConfig`. Stage A tends to keep
infeasible solutions, which lowers the feasible share, so a high λ can hold the
algorithm in stage A for good, as the example below shows.

## Try it: the two stages on the same candidates

The figure shows the 16 CTP2 candidates of lesson
[A6](ccmo.md#try-it-one-set-of-candidates-two-rules): P1 to P8 are the
population and O1 to O8 the offspring, so the 16 together are Q. Start with
stage A, whose ranking happens in the plane [SDE, CV] (hover a point to see both
values), then switch to stage B. The line under the plot applies the stage rule
to Q.

<div class="vf-widget" data-vf-widget="constrained-selection" data-rule="cmoeams"><p class="vf-fallback">Interactive figure: sixteen candidate solutions of CTP2 and the eight that survive in CMOEA-MS's stage A, density and violation ranked as two objectives, and in its stage B, violation first, with the stage rule. It needs JavaScript.</p></div>

Nine of the 16 candidates are feasible: 9/16 = 0.5625 > λ = 0.5, so after the
first 10% of the budget VAMOS would use stage B. In stage A, O1 has both the
lowest density and no violation, so it dominates every other point [SDE, CV]
and is the only one with a fitness below 1; the other seven survivors are those
with the lowest fitness: P1, P3, O7, O5, O3, P8 and P2. Five of the eight are
infeasible, and the feasible P5, P7, O6 and O8 on the constrained front are
dropped because their region of the front is crowded. Stage B keeps eight of
the nine feasible candidates, as a feasibility-first rule would, and drops P6,
which three of them dominate.

## In VAMOS

CMOEA-MS on CTP2 with λ = 0.5 and on MW3 with λ = 0.5 and 0.9, with a
population of 100 and 20,000 evaluations:

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import CMOEAMSConfig

for name, threshold in (("ctp2", 0.5), ("mw3", 0.5), ("mw3", 0.9)):
    config = (
        CMOEAMSConfig.builder()
        .pop_size(100)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .feasible_ratio_threshold(threshold)
        .result_mode("population")
        .build()
    )
    result = optimize(name, algorithm="cmoeams", algorithm_config=config, max_evaluations=20_000, engine="numpy", seed=1)
    G = result.data["population"]["G"]
    feasible = int(np.all(G <= 0.0, axis=1).sum())
    stage_b = result.data["stage_b_generations"]
    print(f"{name}, lambda {threshold}: stage B in {stage_b} of {result.data['generation']} generations, final stage {result.data['stage']}, {feasible} of {len(G)} feasible")
```

Output:

```text
ctp2, lambda 0.5: stage B in 181 of 199 generations, final stage B, 100 of 100 feasible
mw3, lambda 0.5: stage B in 177 of 199 generations, final stage B, 100 of 100 feasible
mw3, lambda 0.9: stage B in 0 of 199 generations, final stage A, 64 of 100 feasible
```

Each generation evaluates 100 offspring, so the first 18 generations end before
2,000 evaluations, 10% of the budget, and are in stage A whatever the share of
feasible solutions. With λ = 0.5, every later generation on CTP2 is in stage B
(181 = 199 − 18): its random solutions lie far above the front, where the whole
objective space is feasible. On MW3 four later generations fall back to stage
A. With λ = 0.9 the picture changes: the feasible share of Q never exceeds 0.9
and the run stays in stage A, which keeps infeasible solutions: 64 of the final
100 are feasible.

A λ close to the feasible share that a run settles at makes the stages fragile.
On CTP2 with λ = 0.9 the share stays near 0.9, so the last-digit differences
between platforms decide whether a generation is in stage A or B, and the same
seed gives different runs: on the Windows machine that recorded these outputs
the share is 0.9 or less in 22 of the generations after the first 10% and the
run ends in stage A with 26 feasible solutions, while on Linux it stays above
0.9 and the run ends in stage B with all 100 feasible. The example leaves that
case out because its output depends on the platform.

The shift-based density, for all the solutions of Q at once:

```python
# Excerpt from src/vamos/engine/algorithm/cmoeams/fitness.py
def shift_based_density(F: NDArray[Any]) -> FloatArray:
    normalized = _normalized_objectives(F)
    n, n_obj = normalized.shape
    if n < 2:
        raise ValueError("CMOEA-MS's shift-based density needs at least two solutions.")
    squares = np.zeros((n, n), dtype=float)
    for j in range(n_obj):
        own = normalized[:, j][:, None]
        other = normalized[:, j][None, :]
        # Row i, column k: solution k shifted to solution i.
        shifted = np.where(other < own, own, other)
        difference = own - shifted
        squares = squares + difference * difference
    distances = np.sort(np.sqrt(squares), axis=1)
    neighbour = int(math.floor(math.sqrt(n)))
    return np.asarray(1.0 / (distances[:, neighbour] + 2.0), dtype=float)
```

`np.where(other < own, own, other)` is the shift: where solution k is better
than solution i in objective j, it takes i's value, and the difference is zero.
Each row of `distances` starts with the zero distance of i to itself, so index
⌊√n⌋ is the (⌊√n⌋ + 1)-th smallest distance, as in PlatEMO.

The fitness of both stages and the stage rule:

```python
# Excerpt from src/vamos/engine/algorithm/cmoeams/fitness.py
def spea2_fitness(points: NDArray[Any], violation: NDArray[Any]) -> FloatArray:
    values = np.asarray(points, dtype=float)
    n = values.shape[0]
    if n == 0:
        raise ValueError("CMOEA-MS's fitness needs at least one solution.")
    dominates = dominance_matrix(values, violation)
    strength = dominates.sum(axis=1, dtype=np.int64)
    raw = (dominates.T.astype(np.int64) @ strength).astype(float)
    distances = cosine_distances(values)
    np.fill_diagonal(distances, np.inf)
    neighbour = int(math.floor(math.sqrt(n))) - 1
    kth = np.sort(distances, axis=1)[:, neighbour]
    return np.asarray(raw + 1.0 / (kth + 2.0), dtype=float)


def stage_a_fitness(F: NDArray[Any], violation: NDArray[Any]) -> FloatArray:
    cv = np.asarray(violation, dtype=float)
    points = np.column_stack([shift_based_density(F), cv])
    return spea2_fitness(points, np.zeros(cv.shape[0], dtype=float))


def stage_b_fitness(F: NDArray[Any], violation: NDArray[Any]) -> FloatArray:
    return spea2_fitness(F, violation)


def is_stage_b(violation: NDArray[Any], threshold: float, evaluations: int, max_evaluations: int) -> bool:
    cv = np.asarray(violation, dtype=float)
    ratio = int(np.count_nonzero(cv <= 0.0)) / cv.shape[0]
    return bool(ratio > threshold and evaluations >= STAGE_B_EVALUATION_FRACTION * max_evaluations)
```

One function serves both stages. Stage A passes the points [SDE, CV] and zero
violations, so `dominance_matrix` reduces to Pareto dominance on those two
numbers; stage B passes the objectives and the normalized violations. In stage
A every feasible point is [SDE, 0], so the feasible points all have the same
direction and a cosine distance of 0 between them; when Q holds enough
feasible solutions, their density term is exactly 1/2.

The environmental selection:

```python
# Excerpt from src/vamos/engine/algorithm/cmoeams/selection.py
def environmental_selection(
    fitness: NDArray[Any], F: NDArray[Any], n_select: int, selection_kernels: SelectionKernels | None = None
) -> IndexArray:
    values = np.asarray(fitness, dtype=float)
    objectives = np.asarray(F, dtype=float)
    n = values.shape[0]
    selected = values < 1.0
    count = int(np.count_nonzero(selected))
    if count < n_select:
        selected[sorted_indices(values)[: int(n_select)]] = True
    elif count > n_select:
        candidates = np.flatnonzero(selected)
        selected[candidates[truncation(objectives[candidates], count - int(n_select), selection_kernels)]] = False
    return np.asarray(np.flatnonzero(selected), dtype=int)
```

`sorted_indices` is a stable sort that puts NaN last, as MATLAB's `sort`, so
ties go to the earlier solution. `truncation` is SPEA2's operator on the cosine
distances between the objective vectors of the candidates.

Finally, `tell` builds Q, chooses the stage and selects:

```python
# Excerpt from src/vamos/engine/algorithm/cmoeams/cmoeams.py
        # The union of the population and the offspring, in that order.
        X_union = np.vstack([st.X, X_off])
        F_union = np.vstack([st.F, F_off])
        G_union = combine_constraints(st.G, G_off, st.constraint_mode, algorithm="CMOEA-MS")
        violation = normalized_violation(G_union, F_union.shape[0])
        st.stage_b = is_stage_b(violation, st.feasible_ratio_threshold, st.n_eval, st.max_evals)
        fitness = stage_b_fitness(F_union, violation) if st.stage_b else stage_a_fitness(F_union, violation)
        selected = environmental_selection(fitness, F_union, st.pop_size, self.kernel.selection_kernels())
        st.X, st.F = X_union[selected], F_union[selected]
        st.G = G_union[selected] if G_union is not None else None
        st.fitness = fitness[selected]
        st.stage_b_generations += int(st.stage_b)
```

`st.n_eval` already includes the offspring of this generation when the stage is
decided, as in jMetal. The violation is normalized over Q, so the same
solution can have different values of CV in two generations.

## Where VAMOS differs from the paper

VAMOS follows jMetal's port, which follows the authors' implementation in
PlatEMO where it differs from the paper; the
[CMOEA-MS lesson](../../algorithms/cmoeams.md#where-vamos-departs-from-jmetals-port)
lists the checks against jMetal's classes. Relative to the paper:

- stage B needs a feasible share greater than λ **and** 10% of the
  evaluations, where the paper's Algorithm 1 has no condition on the
  evaluations;
- the violation is normalized per constraint and averaged, instead of the sum
  of the violations;
- SDE is computed on normalized objectives with the (⌊√|Q|⌋ + 1)-th shifted
  neighbour, instead of the nearest one;
- the density term of the fitness and the truncation use the cosine distance
  instead of the Euclidean distance, and the truncation uses the objectives in
  stage A too;
- the tournaments use the fitness computed on the union of the previous
  generation.

Relative to jMetal's port:

- **Budget:** exact; a cut last generation evaluates the first offspring only,
  while jMetal can exceed the budget.
- **Random draws:** in blocks (tournaments, then crossovers, then mutations),
  so runs with the same seed differ from jMetal's, with the same
  distributions.
- **Parent order:** the GA variation passes the second parent of each pair
  first, so that each child of VAMOS's SBX has the distribution of jMetal's
  child at its position.
- **Result:** jMetal returns the whole final population, which is
  `result.data["population"]`; `result.X` and `result.F` follow the configured
  `result_mode`.

The [CMOEA-MS lesson](../../algorithms/cmoeams.md#agreement-with-jmetals-port-and-the-published-results)
records the agreement with jMetal's classes, bit for bit, and with the
published results.

## The four constrained algorithms

| | CCMO (A6) | C-TAEA (A7) | ToP (A8) | CMOEA-MS (A9) |
| --- | --- | --- | --- | --- |
| Exploring part | Population without constraints | Diversity archive | Phase 1, single objective | Stage A, density and violation |
| Converging part | Population with violation first | Convergence archive | Phase 2, NSGA-II | Stage B, violation first |
| When each acts | Always, in parallel | Always, mating by non-dominated shares | Phase 1, then phase 2 | Each generation, by the feasible share |
| Switch parameters | None | None | δ, Pf, 90% limit | λ, 10% limit |
| Selection | SPEA2 | Subregions of weight vectors | Feasibility rule, then NSGA-II | SPEA2 with cosine distances |

Strengths of CMOEA-MS:

- **One population, one parameter.** No helper population, no archives; λ is
  the only switching parameter.
- **Reversible.** The algorithm returns to stage A whenever feasibility becomes
  scarce, for example after the population reaches a new infeasible barrier.
- **A balanced stage A.** Sparse infeasible solutions survive next to feasible
  ones, without a penalty factor to tune.

Limits:

- **λ matters.** In the example, λ = 0.9 keeps the run on MW3 in stage A from
  start to end, and 64 of the final 100 solutions are feasible instead of all
  of them. A λ near the feasible share that a run settles at makes the stage
  sequence, and the result, depend on rounding.
- **Stage A ranks two scalars.** SDE condenses the objectives into one number,
  so a feasible solution with the lowest density can dominate every other
  point, as O1 does in the figure, and the rest is ranked by raw fitness.
- **Gaps on some problems.** The
  [CMOEA-MS lesson](../../algorithms/cmoeams.md#agreement-with-jmetals-port-and-the-published-results)
  records LIR-CMOP10 runs, in VAMOS and in jMetal, that miss a part of the
  front.

Lesson [A10](performance-assessment.md) explains how to compare such
algorithms fairly: on several problems, with several runs, and with the
feasibility rate reported separately (lesson
[A5](constrained.md#measuring-constrained-results)).

## Key points

- CMOEA-MS ranks one population with SPEA2's fitness under two dominance
  relations, one per stage.
- Stage A ranks the points [SDE, CV] by Pareto dominance: density in the
  objective space and normalized violation weigh the same.
- Stage B puts the violation first and compares equal violations by the
  objectives.
- The union Q is in stage B when more than λ = 0.5 of it is feasible and 10% of
  the budget is spent; otherwise in stage A, and the stages can alternate.
- VAMOS follows PlatEMO's implementation through jMetal's port: normalized and
  averaged violations, SDE on normalized objectives, cosine distances, and an
  exact budget.

**Next:** [A10. Comparing algorithms rigorously](performance-assessment.md).

## References

[1] Tian, Y., Zhang, Y., Su, Y., Zhang, X., Tan, K. C., and Jin, Y. (2022). [Balancing objective optimization and constraint satisfaction in constrained evolutionary multiobjective optimization](https://doi.org/10.1109/TCYB.2020.3021138). *IEEE Transactions on Cybernetics*, 52(9), 9559–9572.

[2] Li, M., Yang, S., and Liu, X. (2014). [Shift-based density estimation for Pareto-based algorithms in many-objective optimization](https://doi.org/10.1109/TEVC.2013.2262178). *IEEE Transactions on Evolutionary Computation*, 18(3), 348–365.

[3] Zitzler, E., Laumanns, M., and Thiele, L. (2001). [SPEA2: Improving the strength Pareto evolutionary algorithm](https://tik-db.ee.ethz.ch/db/public/tik/?db=publications&form=report_single_publication&publication_id=1287). TIK Report 103, Computer Engineering and Networks Laboratory, ETH Zurich.
