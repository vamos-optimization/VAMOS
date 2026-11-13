---
description: 'ToP: a two-phase framework for constrained multi-objective problems, its switch between phases, its exact budget rule and its agreement with jMetal''s port and the published results.'
---

# ToP

The **two-phase framework** for constrained multi-objective optimization [1] first looks for the feasible regions that hold the front and then searches them. Phase 1 solves the problem as a constrained single-objective one, whose objective is the mean of the objectives, with differential evolution and the feasibility rule. Once the population has enough feasible solutions and the best of them are close in the objective space, or once 90% of the budget is spent, phase 2 runs NSGA-II with the constrained-domination principle until the end. The result is the final population.

**Development after 1.1.0.** ToP is new in the development series after VAMOS 1.1.0. It follows jMetal's port (pull request [jMetal/jMetal#525](https://github.com/jMetal/jMetal/pull/525)), which follows PlatEMO's implementation where it differs from the paper, and its sorting, crowding distance, truncation, tournament, differential evolution, mutation and phase-1 rules are checked against that port's own classes.

[All algorithms](../reference/algorithms.md) · [Configuration reference](../reference/api/algorithms/top.md) · [Constraints](../reference/constraints.md)

## Run an example

Use **MW3** with 15 variables (two constraints), the configuration of PlatEMO's implementation (a population of **100**, differential evolution with CR = 1.0 and F = 0.5, polynomial mutation with probability `1/n` and distribution index 20), **25,000 evaluations**, **NumPy** and **seed 42**.

```python
import numpy as np

from vamos import make_problem_selection, optimize
from vamos.algorithms import ToPConfig

problem = make_problem_selection("mw3").instantiate()
config = ToPConfig.default(n_var=problem.n_var)
result = optimize(
    problem,
    algorithm="top",
    algorithm_config=config,
    max_evaluations=25_000,
    engine="numpy",
    seed=42,
)
population = result.data["population"]
print(result.data["evaluations"])  # 25000
print(result.data["phase_two_start"])  # 1800
print(population["F"].shape)  # (100, 2)
print(bool(np.all(population["G"] <= 0)))  # True
```

`result.data["population"]` is the final population, which is the result of PlatEMO's implementation and of jMetal and can contain infeasible solutions; `result.X` and `result.F` hold its feasible non-dominated solutions. `result.data["phase_two_start"]` is the number of evaluations done when phase 2 started: here, 17 generations of phase 1 found a feasible population whose best third is close in the objective space, and phase 2 ran for the remaining 23,200 evaluations. It is `None` when the run did not reach phase 2.

## How it works

**Phase 1.** For each solution `x_i` of the population, with probability 0.5 a trial vector is created with DE/current-to-rand/1, `x_i + F (x_r1 - x_i) + F (x_r2 - x_r3)`, and otherwise with DE/rand-to-best/1/bin, `x_r1 + F (x_best - x_r1) + F (x_r2 - x_r3)`, where `x_best` is the first solution with the lowest mean objective, feasible or not, and `r1`, `r2` and `r3` are drawn with replacement. The trial vector replaces `x_i` if its overall violation is smaller, or if both violations are equal and its mean objective is smaller. Then `Pf`, the ratio of feasible solutions, and `delta` are updated: the objectives of the feasible solutions are normalized with their minimum and maximum, each solution adds up its normalized objectives, and `delta` is the `floor(n/3)`-th smallest sum minus the smallest one, where `n` is the number of feasible solutions. With fewer than three feasible solutions `delta` keeps its value, which is 1 at the start, and an objective with a single value among them makes it `NaN`.

**The switch.** A generation belongs to phase 1 while `(delta >= 0.2 or Pf <= 1/3)` and the evaluations done when it starts are not more than 90% of the budget. As `delta` and `Pf` only change in phase 1 and the evaluations only grow, phase 2 lasts until the end once it starts. The three thresholds are `delta_threshold`, `feasible_ratio_threshold` and `phase_one_evaluation_fraction`.

**Phase 2.** The population is sorted into fronts with ToP's constrained dominance: two feasible solutions are compared by Pareto dominance, and otherwise the one with the smaller violation dominates, so every infeasible solution is behind every feasible one and infeasible solutions with the same violation share a front. The crowding distance sorts each front by each objective from the order of the population; an objective with a single value in a front gives `NaN` to its inner solutions. Each offspring is `x_1 + F (x_2 - x_3)`, where `x_1` wins a binary tournament on the front number and the crowding distance (`NaN` is the worst) and `x_2` and `x_3` are random. The next population is the best solutions of the population and the offspring by front and, in the last front, by crowding distance, `NaN` first. Every remaining tie goes to the solution that comes first.

**Variation.** Every trial vector comes from a binomial crossover of rate CR that, unlike VAMOS's `de` operator and jMetal's `DifferentialEvolutionCrossover`, does not force one variable from the mutant vector, so with CR < 1 a trial vector can be a copy of its base vector. It is then repaired (clamped to the bounds by default) and mutated (polynomial mutation by default; VAMOS's polynomial mutation is jMetal's). The configuration names the differential evolution as the crossover `de` with `cr` and `f`; any other crossover is rejected, and so are discrete and mixed problems. The `top` [tuning space](../topics/tuning.md#constrained-spaces) samples the population size, CR, F, the three thresholds and VAMOS's real-coded mutations; `vamos tune --algorithm top` tunes it.

```python
from vamos.algorithms import ToPConfig

config = ToPConfig.builder().pop_size(100).crossover("de", cr=1.0, f=0.5).mutation("pm", prob="1/n", eta=20.0).build()
```

## Budget

The initial population takes `pop_size` evaluations, so `max_evaluations` must be at least `pop_size`; `optimize`, study planning and the `vamos` command line reject a smaller budget before the run. Each generation then costs `pop_size` evaluations. Budgets are exact: when fewer evaluations are left than `pop_size`, the last generation, of either phase, creates the offspring of a full generation and evaluates only the first ones, in the order of the population. In phase 1 they face the solutions at their positions, the other solutions being kept, and `Pf` and `delta` are updated; in phase 2 the next population is selected from the population and them. The phase of a generation is decided when it starts, from the evaluations done until then, so the decisions are those of jMetal, which checks the budget after each generation and can exceed it by up to `pop_size - 1` evaluations: with `pop_size = 100` and 25,050 evaluations, VAMOS evaluates 50 offspring in its last generation where jMetal evaluates 100. The `n_gen` criterion counts generations of `pop_size` evaluations, the initial population included.

## Agreement with jMetal's port and the published results

**jMetal's integration test.** IGD of the feasible solutions of the final population (the mean distance from the points of the reference front to them) against jMetal's reference fronts, with the defaults, over 31 seeds. jMetal's own ToP (jMetal 7.10 with the pull request's classes) was run with the same settings and the seeds 1 to 31; on the seeds of the pull request's test, its runs reproduce the test's medians. The last column gives those medians, which are the `n/2`-th smallest IGD of the seeds 1 to `n`, and VAMOS's and jMetal's on the same seeds.

| Problem | Evaluations | VAMOS, median (range) | jMetal, median (range) | Mann–Whitney p | jMetal PR #525 (seeds): PR, jMetal, VAMOS |
| --- | --- | --- | --- | --- | --- |
| MW3, 15 variables | 25,000 | 1.01e-2 (7.65e-3 to 9.18e-1) | 1.20e-2 (8.44e-3 to 9.18e-1) | 0.17 | 0.0129 (1 to 15): 0.0129, 0.0108 |
| LIR-CMOP2, 30 variables | 100,000 | 2.23e-1 (1.91e-1 to 2.72e-1) | 2.22e-1 (1.53e-1 to 2.72e-1) | 0.56 | 0.227 (1 to 10): 0.227, 0.225 |
| C1-DTLZ1, 3 objectives, 7 variables | 100,000 | 2.72e-2 (2.52e-2 to 2.92e-2) | 2.71e-2 (2.44e-2 to 2.94e-2) | 0.58 | 0.0270 (1 to 10): 0.0270, 0.0272 |

Every final population is feasible, in VAMOS and in jMetal. On MW3, the population of 5 of VAMOS's 31 runs and of 7 of jMetal's converges to one end of the front, with IGD 0.918, as the pull request records for 4 of its 15 runs. The LIR-CMOP2 row uses jMetal's former 200-point front, as the pull request's test did; against the exact front that VAMOS packages, the medians are 2.73e-1 for VAMOS and for jMetal (p = 0.79).

**The CMOEA-MS paper.** The paper of CMOEA-MS [2] runs PlatEMO's ToP on MW and LIR-CMOP with D = 10, N = 100 and 200,000 evaluations, which are the defaults here. IGD is computed as above, against jMetal's reference fronts (VAMOS packages jMetal's MW11, LIR-CMOP7 and LIR-CMOP10 fronts), over 31 seeds, with jMetal's ToP run in the same way; a run without a feasible solution has an infinite IGD.

| Problem | VAMOS, median (range) | jMetal, median (range) | Mann–Whitney p | jMetal PR #525, median of 5 | Paper, mean |
| --- | --- | --- | --- | --- | --- |
| MW1 | 2.31e-3 (2.10e-3 to ∞) | 2.23e-3 (2.07e-3 to ∞) | 0.50 | 2.19e-3 | 1.37e-1 |
| MW3 | 5.85e-3 (5.56e-3 to 6.44e-3) | 5.92e-3 (5.57e-3 to 9.18e-1) | 0.96 | 5.95e-3 | 4.64e-1 |
| MW11 | 7.39e-3 (7.14e-3 to 8.24e-3) | 7.37e-3 (6.99e-3 to 9.65e-3) | 0.37 | 7.17e-3 | 7.36e-2 |
| LIR-CMOP7 | 9.06e-3 (8.36e-3 to 9.71e-3) | 8.88e-3 (8.24e-3 to 9.64e-3) | 0.28 | 8.79e-3 | 8.63e-3 |
| LIR-CMOP10 | 5.44e-3 (5.07e-3 to 5.92e-3) | 5.50e-3 (5.08e-3 to 5.98e-3) | 0.32 | 5.54e-3 | 5.59e-3 |

The medians of VAMOS and jMetal agree with each other and with the pull request's on every problem. On LIR-CMOP7 and LIR-CMOP10 they agree with the paper's means; on the MW problems they are much better, as the pull request also found: the paper's means of 0.46 on MW3 or 0.14 on MW1 suggest that many of its runs ended far from the front. On MW1 the final population of 12 of VAMOS's 31 runs and of 8 of jMetal's has no feasible solution, and of 32 and 38 of their first 100 runs (Fisher's exact test p = 0.46): phase 1 lasts until 90% of the budget with the population stuck at a violation of 1.5, and the pull request's table also has one such run out of five.

## Where VAMOS departs from jMetal's port

- **Budget:** exact, as described above; jMetal can exceed it in its last generation.
- **Random draws:** each generation draws its random numbers in blocks (in phase 1, the choice of operator for every solution, then the indices `r1`, `r2` and `r3`, then the crossover draws, then the mutation's; in phase 2, the tournament pairs, then `x_2` and `x_3`, then the crossover and the mutation), while jMetal draws them solution by solution, so the runs are not those of jMetal for the same seed; each draw has jMetal's distribution.
- **Initialization:** the initial population is created by the configured initializer (uniform random by default) from the run's generator.
- **Result:** `result.X` and `result.F` keep the feasible non-dominated solutions of the final population (with `result_mode="non_dominated"`, the default) or the whole population (`result_mode="population"`); jMetal returns the whole population, which is `result.data["population"]` in VAMOS.
- **Violation:** VAMOS's convention, `g <= 0` feasible and the violation the sum of the positive parts, which is, with the sign changed, jMetal's. jMetal adds the violated parts with a compensated sum, so with two or more violated constraints the two can differ in the last bit (by 1.1e-16 in one of the eleven phase-1 cases of the fixture), and so can the solutions whose violations are equal.

jMetal's port follows PlatEMO's implementation where it differs from the paper, and so does VAMOS: the indices of phase 1 are drawn with replacement instead of mutually different and different from `i`; F and CR are fixed instead of drawn from the paper's pools; the crossover has no `j_rand`, and the variables of DE/rand-to-best/1/bin that are not crossed come from `x_r1` instead of `x_i` (neither difference matters with CR = 1); the trial vectors of phase 1 are clamped and mutated; `delta` normalizes the objectives with the feasible solutions of the current population instead of all those found, keeps its value with fewer than three feasible solutions, and phase 1 also ends at 90% of the budget; the mean of the objectives replaces their sum, which gives the same comparisons; and phase 2 creates its offspring with DE/rand/1/bin and the polynomial mutation instead of SBX.

## References

[1] Liu, Z.-Z., and Wang, Y. (2019). [Handling Constrained Multiobjective Optimization Problems With Constraints in Both the Decision and Objective Spaces](https://doi.org/10.1109/TEVC.2019.2894743). *IEEE Transactions on Evolutionary Computation*, 23(5), 870–884.

[2] Tian, Y., Zhang, Y., Su, Y., Zhang, X., Tan, K. C., and Jin, Y. (2022). Balancing Objective Optimization and Constraint Satisfaction in Constrained Evolutionary Multiobjective Optimization. *IEEE Transactions on Cybernetics*. Tables I and II.
