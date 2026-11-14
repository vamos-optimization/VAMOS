---
description: 'CMOEA-MS: a constrained multi-objective algorithm whose fitness has two stages, its GA and DE variations, its exact budget rule and its agreement with jMetal''s port and the published results.'
---

# CMOEA-MS

The **constrained multi-objective evolutionary algorithm with multiple stages** balances the objectives and the constraints by changing how it ranks its solutions [1]. In **stage A** the objectives have the same priority as the constraints: a solution is ranked by two objectives, its density in the objective space and its constraint violation. In **stage B** the constraints come first: a smaller violation is better, and equal violations are compared by the objectives. The algorithm is in stage B while more than half of the solutions are feasible, once 10% of the budget is spent. Both stages use the fitness and the environmental selection of SPEA2. The result is the final population.

**Development after 1.1.0.** CMOEA-MS is new in the development series after VAMOS 1.1.0. It follows jMetal's port (pull request [jMetal/jMetal#526](https://github.com/jMetal/jMetal/pull/526)), which follows the authors' implementation in PlatEMO where it differs from the paper, and its constraint violation, density, fitness, selections, stage rule and differential evolution are checked against that port's own classes.

[All algorithms](../reference/algorithms.md) · [Configuration reference](../reference/api/algorithms/cmoeams.md) · [Constraints](../reference/constraints.md)

## Run an example

Use **MW3** with 15 variables (two constraints), the configuration of the paper (a population of **100**, SBX with probability 1.0 and distribution index 20, polynomial mutation with probability `1/n` and distribution index 20, and `lambda = 0.5`), **25,000 evaluations**, **NumPy** and **seed 42**.

```python
import numpy as np

from vamos import make_problem_selection, optimize
from vamos.algorithms import CMOEAMSConfig

problem = make_problem_selection("mw3").instantiate()
config = CMOEAMSConfig.default(n_var=problem.n_var)
result = optimize(
    problem,
    algorithm="cmoeams",
    algorithm_config=config,
    max_evaluations=25_000,
    engine="numpy",
    seed=42,
)
population = result.data["population"]
print(result.data["evaluations"])  # 25000
print(result.data["stage"])  # B
print(result.data["stage_b_generations"])  # 222
print(population["F"].shape)  # (100, 2)
print(bool(np.all(population["G"] <= 0)))  # True
```

`result.data["population"]` is the final population, which is the result of PlatEMO's implementation and of jMetal and can contain infeasible and dominated solutions; `result.X` and `result.F` hold its feasible non-dominated solutions. `result.data["stage"]` is the stage of the fitness of the final population, and `result.data["stage_b_generations"]` the number of generations in stage B: here, 222 of the 249 generations. The first 23 generations end before 10% of the budget, and in the next 4 half or more of the solutions of `Q` are infeasible, so those 27 are in stage A; stage B lasts from then on, although the stage changes back to A whenever half or more of the solutions of `Q` are infeasible.

## How it works

Each generation selects parents by binary tournaments on the fitness that the population had when it was selected, creates the offspring and computes the fitness of the union `Q` of the population and the offspring, in that order.

**Constraint violation.** The violation of each constraint, `max(g, 0)`, is divided by its largest value in `Q`, and the violation `CV` of a solution is the mean over the constraints; a constraint that no solution violates contributes zero. A solution is feasible when its `CV` is zero.

**The stages.** `Q` is in stage B when its ratio of feasible solutions is greater than `feasible_ratio_threshold` (the paper's `lambda`, 0.5) and at least 10% of the evaluations have been done, those of the offspring included; otherwise, and for the initial population, it is in stage A.

- **Stage A** ranks the points `[SDE, CV]` by Pareto dominance. `SDE` is the shift-based density estimation: on the objectives normalized to [0, 1] in `Q`, each other solution is shifted to the solution (its objectives that are better take the solution's values), and `SDE = 1 / (d + 2)`, where `d` is the `(floor(sqrt(|Q|)) + 1)`-th smallest distance to the shifted solutions, the solution's own distance of zero included.
- **Stage B** ranks the objectives with the constraint violation first: a smaller `CV` dominates, and equal `CV` values are compared by Pareto dominance.

**Fitness and selection.** The fitness is SPEA2's: the sum of the strengths (the number of solutions each one dominates) of the solutions that dominate a solution, plus `1 / (d + 2)`, where `d` is the cosine distance to its `floor(sqrt(|Q|))`-th nearest solution. The solutions that no other one dominates have a fitness lower than 1, and the environmental selection keeps them; if they are fewer than `pop_size`, it keeps the `pop_size` solutions of lowest fitness, and if they are more, it removes, one at a time, the solution whose sorted cosine distances between objective vectors to the remaining ones are the lexicographically smallest. The selected solutions keep their order in `Q`, every tie goes to the solution that comes first, and the binary tournament takes the lower fitness. The divisions by zero of PlatEMO are kept: an objective with a single value in `Q` gives `NaN` densities and a zero objective vector `NaN` cosine distances, and `NaN` sorts after every number, as in MATLAB.

**Variation.** The crossover chooses one of PlatEMO's two variations:

- **GA** (any crossover with two parents and two children; SBX by default): `pop_size` parents are selected, the `i`-th of the first half mates with the `i`-th of the second half, and both children are kept, the first ones before the second ones, then mutated, which gives `2 * (pop_size // 2)` offspring. Discrete and mixed encodings use their two-parent crossovers.
- **DE** (crossover `de`, real variables): for the `i`-th solution `x_i`, two tournaments select `r1` and `r2`, and the mutant `x_i + F (x_r1 - x_r2)` is crossed with `x_i`, then clamped and mutated, which gives `pop_size` offspring. `de_variant` chooses the crossover of jMetal's `DifferentialEvolutionCrossover`: `"rand_1_bin"` (PlatEMO's, the default) takes each variable from the mutant with probability CR and one at random always, as VAMOS's `de` operator does; `"rand_1_exp"` takes a block of consecutive variables, from a random one while a draw is lower than CR. PlatEMO uses CR = 1 and F = 0.5, with which both take the whole mutant.

The `cmoeams` [tuning space](../topics/tuning.md#constrained-spaces) samples the population size, `lambda`, the GA or DE variation and VAMOS's real-coded mutations; `vamos tune --algorithm cmoeams` tunes it.

```python
from vamos.algorithms import CMOEAMSConfig

config = (
    CMOEAMSConfig.builder()
    .pop_size(100)
    .crossover("de", cr=1.0, f=0.5)
    .de_variant("rand_1_bin")
    .mutation("pm", prob="1/n", eta=20.0)
    .feasible_ratio_threshold(0.5)
    .build()
)
```

## Budget

The initial population takes `pop_size` evaluations, so `max_evaluations` must be at least `pop_size`; `optimize`, study planning and the `vamos` command line reject a smaller budget before the run. A GA generation then costs `2 * (pop_size // 2)` evaluations, one fewer than `pop_size` when it is odd, and a DE generation `pop_size`. Budgets are exact: when fewer evaluations are left than offspring, the last generation creates the offspring of a full generation and evaluates the first ones, in the order of the offspring (for GA, the first children of the pairs and then the second ones; for DE, the order of the population), and `Q` is the population and those. jMetal checks the budget before each generation, so its last generation can exceed it: with `pop_size = 100` and 25,050 evaluations, VAMOS evaluates 50 offspring in its last generation where jMetal evaluates 100.

The 10% condition of stage B counts the evaluations done once the offspring of the generation are evaluated, as jMetal does. In every generation but the last, the evaluations done are those of jMetal's, and in the last one both have done at least 10% of the budget, so the stage of every generation is decided as in jMetal. The `n_gen` criterion counts generations of `pop_size` evaluations, the initial population included.

## Agreement with jMetal's port and the published results

**The port's classes.** `tools/cmoeams_reference_values.py` runs jMetal's own classes on fixed inputs: the violation, the density, the fitness of both stages, the cosine distances, the environmental selection, the truncation and the tournaments on 23 populations (two to five objectives, up to 200 solutions, with ties, duplicates, a constant objective, zero objective vectors, `NaN` and infinite values), the fitness of 9 sets of points, 324 stage decisions, jMetal's `DifferentialEvolutionCrossover` (`RAND_1_BIN` and `RAND_1_EXP`) and jMetal's SBX on fixed draws. VAMOS gives the same values, bit for bit, and the same selections in the same order; its SBX gives jMetal's children, at the positions described below.

**jMetal's integration test.** IGD of the feasible solutions of the final population (the mean distance from the points of the reference front to them) against jMetal's reference fronts, with a population of 100, 25,000 evaluations and the defaults, over 31 seeds. jMetal's own CMOEA-MS (jMetal 7.10 with the pull request's classes) was run with the same settings and the seeds 1 to 31. The last column applies the test's own check to every run: a fully feasible population whose IGD as jMetal computes it by default (on the non-dominated solutions, the square root of the sum of the squared distances divided by the number of front points) is below the test's threshold.

| Problem | VAMOS, median (range) | jMetal, median (range) | Mann–Whitney p | Runs that pass the test's check: VAMOS, jMetal |
| --- | --- | --- | --- | --- |
| MW3, 15 variables, SBX | 7.00e-3 (5.92e-3 to 1.04e-1) | 7.58e-3 (6.11e-3 to 1.08e-1) | 0.15 | 31, 31 (threshold 0.01) |
| LIR-CMOP2, 30 variables, SBX | 3.08e-1 (2.21e-1 to 3.80e-1) | 3.15e-1 (1.90e-1 to 3.56e-1) | 0.30 | 31, 31 (threshold 0.045) |
| C1-DTLZ1, 3 objectives, SBX | 2.09e-2 (2.00e-2 to 3.71e-2) | 2.06e-2 (1.98e-2 to 3.92e-2) | 0.36 | 30, 30 (threshold 0.0004) |
| MW3, 15 variables, DE | 7.74e-3 (7.05e-3 to 9.24e-3) | 7.82e-3 (6.88e-3 to 9.03e-3) | 0.35 | 31, 31 (threshold 0.0005) |

Every final population is feasible, in VAMOS and in jMetal. With the test's seed, 12, both pass the four checks. The one run of each that fails the C1-DTLZ1 check (4.9e-4 and 5.2e-4, above 4e-4) is also its worst by the mean distance; the pull request's 20 seeds had no such run.

**The paper.** The paper [1] runs CMOEA-MS in PlatEMO on MW and LIR-CMOP with D = 10, N = 100 and 200,000 evaluations, which are the defaults here. IGD is computed as above, against jMetal's reference fronts (VAMOS packages jMetal's LIR-CMOP fronts), over the seeds 1 to 100, with jMetal's CMOEA-MS run in the same way; a run without a feasible solution would count as an infinite IGD, but every final population is feasible. On the seeds 1 to 5, jMetal's runs give the medians of the pull request's table, which validates the comparison.

| Problem | VAMOS, median (range) | jMetal, median (range) | Mann–Whitney p | jMetal PR #526, median of 5 | Paper, mean |
| --- | --- | --- | --- | --- | --- |
| MW1 | 1.66e-3 (1.64e-3 to 1.72e-3) | 1.66e-3 (1.62e-3 to 1.74e-3) | 0.60 | 1.67e-3 | 1.62e-3 |
| MW3 | 4.95e-3 (4.60e-3 to 5.38e-3) | 4.94e-3 (4.67e-3 to 5.45e-3) | 0.55 | 5.07e-3 | 4.53e-3 |
| MW7 | 3.80e-3 (3.67e-3 to 6.27e-2) | 3.79e-3 (3.65e-3 to 6.24e-2) | 0.62 | 3.80e-3 | 3.82e-3 |
| LIR-CMOP7 | 7.31e-3 (6.61e-3 to 5.98e-2) | 7.39e-3 (6.66e-3 to 1.11e-1) | 0.56 | 7.21e-3 | 7.27e-3 |
| LIR-CMOP8 | 7.00e-3 (6.63e-3 to 2.22e-1) | 6.98e-3 (6.61e-3 to 1.22e-1) | 0.56 | 7.02e-3 | 7.51e-3 |
| LIR-CMOP10 | 6.13e-2 (7.61e-3 to 1.70e-1) | 6.12e-2 (8.11e-3 to 1.69e-1) | 0.32 | 1.56e-2 | 4.25e-3 |

The medians of VAMOS and jMetal agree with each other on every problem, and with the paper's means on MW1, MW3, MW7, LIR-CMOP7 and LIR-CMOP8. Some runs of both miss a part of the front, with an IGD more than twice jMetal's median: 16 of VAMOS's 100 and 9 of jMetal's on MW7, 4 and 9 on LIR-CMOP7, 3 and 1 on LIR-CMOP8 (Fisher's exact test p >= 0.20). On LIR-CMOP10 both are far from the paper's mean: 56 of VAMOS's runs and 49 of jMetal's end near 0.061, with a part of the front missing, as the pull request also found (its 5 runs range from 1.0e-2 to 6.3e-2); the pull request traces such gaps to PlatEMO's implementation, which an independent transcription of it reproduces.

## Where VAMOS departs from jMetal's port

- **Budget:** exact, as described above; jMetal can exceed it in its last generation.
- **Random draws:** each generation draws its random numbers in blocks (the tournaments, then the crossover of every pair or solution, then the mutation of every child), while jMetal draws them pair by pair or solution by solution, so the runs are not those of jMetal for the same seed; each draw has jMetal's distribution.
- **SBX and the order of the parents:** jMetal's SBX gives a variable that it does not cross to the other child, while VAMOS's leaves it with its own parent. The GA variation passes the second parent of each pair first, so each child has the distribution of jMetal's child at its position in `Q`, which decides ties; the fixture checks that VAMOS's SBX on `(second, first)` gives jMetal's children on fixed draws. A pair that is not crossed (with a crossover probability below 1) keeps its parents at swapped positions: the two children are the same as jMetal's. The other crossovers receive the parents in that order too. As both children are kept, the order matters little: with the parents in their own order, the medians of 31 runs on MW3, LIR-CMOP2 and C1-DTLZ1 (25,000 evaluations) and on MW3, LIR-CMOP7 and LIR-CMOP10 (the paper's setting) change by at most 2% (Mann–Whitney p >= 0.29).
- **Initialization:** the initial population is created by the configured initializer (uniform random by default) from the run's generator.
- **Result:** `result.X` and `result.F` keep the feasible non-dominated solutions of the final population (with `result_mode="population"`, the whole population); jMetal returns the whole population, which is `result.data["population"]` in VAMOS.
- **Violation:** VAMOS's convention, `g <= 0` feasible; jMetal's constraint values are VAMOS's with the sign changed, so the normalized violations are the same.

jMetal's port follows PlatEMO's implementation where it differs from the paper, and so does VAMOS: stage B needs a feasible ratio greater than `lambda` and 10% of the evaluations, where the paper's Algorithm 1 has no condition on the evaluations; the violation is normalized per constraint and averaged, instead of the sum of the violations; the density is computed on normalized objectives with the `(floor(sqrt(|Q|)) + 1)`-th shifted neighbour, instead of the nearest one; the density term of the fitness and the truncation use the cosine distance instead of the Euclidean distance, and the truncation uses the objectives in stage A too; and the tournaments use the fitness computed on the union of the previous generation.

## References

[1] Tian, Y., Zhang, Y., Su, Y., Zhang, X., Tan, K. C., and Jin, Y. (2022). [Balancing Objective Optimization and Constraint Satisfaction in Constrained Evolutionary Multiobjective Optimization](https://doi.org/10.1109/TCYB.2020.3021138). *IEEE Transactions on Cybernetics*, 52(9), 9559–9572.
