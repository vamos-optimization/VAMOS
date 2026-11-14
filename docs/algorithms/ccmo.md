---
description: 'CCMO: two cooperating populations for constrained multi-objective problems, its GA and DE variations, its exact budget rule and its agreement with the published results.'
---

# CCMO

The **coevolutionary framework for constrained multi-objective optimization** evolves two populations of the same size: population 1 solves the problem, and population 2 solves the same problem without its constraints [1]. The two populations cooperate only through their offspring: in each generation every population selects its survivors from itself together with the offspring of both. Population 2 explores across infeasible regions and its offspring help population 1 reach the feasible parts of the front that infeasible barriers hide.

**Development after 1.1.0.** CCMO is new in the development series after VAMOS 1.1.0. It follows the authors' reference code through jMetal's port (pull request [jMetal/jMetal#521](https://github.com/jMetal/jMetal/pull/521)), and its fitness and environmental selection are checked against that port's own classes.

[All algorithms](../reference/algorithms.md) · [Configuration reference](../reference/api/algorithms/ccmo.md) · [Constraints](../reference/constraints.md)

## Run an example

Use **MW3** (15 variables, two objectives, two constraints), the paper's configuration (two populations of **100**, SBX with probability 1.0 and distribution index 20, polynomial mutation with probability `1/n` and distribution index 20), **20,000 evaluations**, **NumPy** and **seed 42**.

```python
import numpy as np

from vamos import make_problem_selection, optimize
from vamos.algorithms import CCMOConfig

problem = make_problem_selection("mw3").instantiate()
config = CCMOConfig.default(n_var=problem.n_var)
result = optimize(
    problem,
    algorithm="ccmo",
    algorithm_config=config,
    max_evaluations=20_000,
    engine="numpy",
    seed=42,
)
population = result.data["population"]
helper = result.data["helper_population"]
print(result.data["evaluations"])  # 20000
print(population["F"].shape)  # (100, 2)
print(bool(np.all(population["G"] <= 0)))  # True
print(int(np.sum(np.any(helper["G"] > 0, axis=1))))  # 42
```

`result.X` and `result.F` hold the feasible non-dominated solutions of population 1 (all 100 here). `result.data["population"]` is the whole population 1, which is the result of the authors' code and of jMetal, and `result.data["helper_population"]` is population 2, which ignores the constraints: 42 of its 100 solutions are infeasible.

## How it works

Each population is optimized by SPEA2 with a dominance relation in which the constraint violation comes first, as in the authors' code:

- a solution dominates another one if its total violation is smaller or, when both violations are equal, if it dominates it in the Pareto sense; population 2 uses zero violations, which leaves Pareto dominance;
- the fitness is SPEA2's raw fitness (the sum of the strengths of a solution's dominators) plus the density `1 / (d_k + 2)`, where `d_k` is the distance in the objective space, without normalization, to the `k`-th nearest other solution, with `k = floor(sqrt(n))`;
- the environmental selection keeps the solutions with a fitness lower than 1, fills up with the next ones by fitness or removes the most crowded ones with SPEA2's truncation, and sorts the survivors by fitness, which drives the next mating selection;
- the mating selection is a binary tournament on that fitness; a tie goes to the solution that comes first.

The crossover chooses the variation:

| Crossover | Variation | Offspring per population and generation |
| --- | --- | --- |
| `sbx` or any other two-parent crossover (default) | GA: `pop_size` parents by tournament; the i-th of the first half mates with the i-th of the second half, and the first child of each pair is mutated | `pop_size // 2` |
| `de` with `cr` and `f` (1.0 and 0.5 by default) | DE, real variables only: for each member `x_i`, `x_i + F (r1_i - r2_i)` with binomial crossover of rate `CR`, then mutation, with `r1` and `r2` from `2 * pop_size` tournaments | `pop_size` |

```python
from vamos.algorithms import CCMOConfig

de = CCMOConfig.builder().pop_size(100).crossover("de", cr=1.0, f=0.5).mutation("pm", prob="1/n", eta=20.0).build()
```

The GA variation runs every encoding with its two-parent operators; the DE variation needs real variables. The `ccmo` [tuning space](../topics/tuning.md#constrained-spaces) samples the population size and the crossover among SBX, BLX-alpha, arithmetic and `de` (with `de_cr` and `de_f`), with VAMOS's real-coded mutations; `vamos tune --algorithm ccmo` tunes it.

## Budget

The two initial populations take `2 * pop_size` evaluations, so `max_evaluations` must be at least `2 * pop_size`; `optimize` and study planning reject a smaller budget before the run. A GA generation then costs `2 * (pop_size // 2)` evaluations and a DE generation `2 * pop_size`. Budgets are exact: when fewer evaluations are left than the last generation creates, it evaluates the first of its offspring in the order `[population 1's offspring, population 2's offspring]`, so population 2's offspring are cut first, and both populations then select from the offspring that were evaluated. The authors' code and jMetal check the budget before each generation, so their last generation can exceed it. The `n_gen` criterion counts `pop_size` evaluations per generation, as for every algorithm.

## Agreement with the published results

IGD is the mean distance from the points of the reference front to the final population 1 (the paper's definition), against jMetal's reference fronts, with the paper's configuration and 100,000 evaluations, over 31 seeds:

| Problem | Population | VAMOS, median (range) | jMetal PR #521, mean | Paper |
| --- | --- | --- | --- | --- |
| MW1 | 100 | 1.61e-3 (1.59e-3 to 1.65e-3; two runs trapped, see below) | 1.61e-3 | 1.61e-3 |
| MW3 | 100 | 4.76e-3 (4.46e-3 to 5.51e-3) | 4.88e-3 | 4.69e-3 |
| C1-DTLZ1, 3 objectives and 7 variables | 105 | 1.94e-2 (1.92e-2 to 1.98e-2) | 1.94e-2 | 1.99e-2 |

The reference fronts that VAMOS packages sample the same fronts at other points (MW1 with 4,446 points instead of jMetal's 1,505), which changes the IGD a little: the medians are then 1.48e-3, 4.60e-3 and 1.94e-2. C1-DTLZ1 runs with 7 variables, as in the paper and jMetal (`make_problem_selection("c1dtlz1", n_var=7, n_obj=3)`; VAMOS's default is 12). With 100 individuals, as in jMetal's integration test, it gives a median of 1.99e-2 (1.97e-2 to 2.02e-2), where that test reports 1.98e-2 to 2.02e-2. On MW1, two of the 31 runs ended trapped, one with no feasible solution and one at IGD 2.0e-2, and no other run of seeds 1 to 100 did. MW1's distance function is flat over most of each variable's range; while population 1 is infeasible, the violation-first ranking orders it as a single chain, which loses diversity, and a variable that settles in the flat region does not leave it. jMetal's own CCMO, run with the same configuration on 100 seeds, ended trapped once, with no feasible solution, and its median IGD over those seeds, 1.610e-3, is VAMOS's. LIR-CMOP2 with the DE variation and 50,000 evaluations gives a median IGD of 0.15 (0.05 to 0.24), where jMetal's integration test reports 0.08 to 0.24 over 15 runs.

## Where VAMOS departs from the reference code

- **Budget:** exact, as described above; the reference code can exceed it in its last generation.
- **Initialization:** each population is created by the configured initializer (uniform random by default), population 1 first, from the run's generator.
- **Result:** `result.X` and `result.F` keep the feasible non-dominated solutions of population 1 (with `result_mode="non_dominated"`, the default) or the whole population 1 (`result_mode="population"`); the reference code returns the whole population 1, which is `result.data["population"]` in VAMOS.
- **Violation:** VAMOS's convention, `g <= 0` feasible and the violation the sum of the positive parts, which is the authors' and, with the sign changed, jMetal's.

The authors' code and jMetal differ from the paper in three places, and VAMOS follows the code: the paper embeds NSGA-II with SPEA2's truncation instead of SPEA2; it treats two infeasible solutions with the same violation as non-dominated; and it selects `pop_size / 2` parents for `pop_size / 2` offspring. The paper does not describe the DE variation.

## References

[1] Tian, Y., Zhang, T., Xiao, J., Zhang, X., and Jin, Y. (2021). [A Coevolutionary Framework for Constrained Multiobjective Optimization Problems](https://doi.org/10.1109/TEVC.2020.3004012). *IEEE Transactions on Evolutionary Computation*, 25(1), 102–116.
