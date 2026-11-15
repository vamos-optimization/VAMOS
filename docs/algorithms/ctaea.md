---
description: 'C-TAEA: a convergence archive and a diversity archive for constrained multi-objective problems, its weight vectors, its exact budget rule and its agreement with jMetal''s port and the published results.'
---

# C-TAEA

The **two-archive evolutionary algorithm for constrained multi-objective optimization** keeps two archives of the same size, one solution per weight vector [1]. The convergence archive (CA) is driven by feasibility and convergence, and it is the result. The diversity archive (DA) ignores the constraints and keeps solutions in the subregions where the CA has few, so that the search can cross infeasible regions towards the parts of the front the CA has not reached. Each offspring has one parent from each archive or both from one, chosen from how many non-dominated solutions each archive holds.

**New in VAMOS 1.2.0.** C-TAEA follows jMetal's port (pull request [jMetal/jMetal#524](https://github.com/jMetal/jMetal/pull/524)), which follows PlatEMO's implementation where it differs from the paper, and its weight vectors, archive updates, mating proportions and tournament are checked against that port's own classes.

[All algorithms](../reference/algorithms.md) · [Configuration reference](../reference/api/algorithms/ctaea.md) · [Constraints](../reference/constraints.md)

## Run an example

Use **C1-DTLZ1** with three objectives and 7 variables (one constraint), the paper's configuration (100 requested weight vectors, which give **91**, SBX with probability 1.0 and distribution index 20, polynomial mutation with probability `1/n` and distribution index 20), **200 generations of 91 evaluations**, **NumPy** and **seed 42**.

```python
import numpy as np

from vamos import make_problem_selection, optimize
from vamos.algorithms import CTAEAConfig

problem = make_problem_selection("c1dtlz1", n_var=7, n_obj=3).instantiate()
config = CTAEAConfig.default(n_var=problem.n_var)
result = optimize(
    problem,
    algorithm="ctaea",
    algorithm_config=config,
    max_evaluations=91 * 200,
    engine="numpy",
    seed=42,
)
archive = result.data["population"]
diversity = result.data["diversity_archive"]
print(result.data["evaluations"])  # 18200
print(result.data["weights"].shape)  # (91, 3)
print(archive["F"].shape)  # (91, 3)
print(bool(np.all(archive["G"] <= 0)))  # True
print(int(np.sum(np.any(diversity["G"] > 0, axis=1))))  # 87
```

`result.data["population"]` is the whole CA, which is the result of PlatEMO's implementation and of jMetal, and `result.X` and `result.F` hold its feasible non-dominated solutions (all 91 here). `result.data["diversity_archive"]` is the DA: 87 of its 91 solutions are infeasible, because it ignores the constraints. `result.data["weights"]` holds the weight vectors, which a stored run keeps as its reference directions.

## How it works

**Weight vectors.** `pop_size` is the requested number of weight vectors N. They are the Das–Dennis lattice with the most divisions (at least one) that gives at most N vectors; when it has fewer divisions than objectives, an inner layer, shrunk halfway towards the centre of the simplex, adds the lattice with the most divisions that keeps the total at most N, and every component is raised to at least `1e-6`. Their number, which may be smaller than N (91 for 100 and three objectives, 100 for 100 and two), is the size of both archives and of each generation; with fewer requested vectors than objectives, it is the number of objectives. Each weight vector defines a subregion, and a solution belongs to the one whose vector has the largest cosine similarity with its objectives.

**Convergence archive.** The candidates are the CA and the offspring:

- if as many as the archive size are feasible, they are the new CA;
- if more are, the fronts of their non-dominated sorting are added until the archive is full, and then one solution at a time is removed until it fits: in the most crowded subregion (on objectives normalized with the ideal and nadir points of the set), among the solutions at the smallest distance from another one, the one with the largest Tchebycheff value;
- if fewer are, all of them are kept, and the infeasible ones are sorted by non-dominated sorting on their violation and their Tchebycheff value; the fronts are added until the archive is full, and the solutions of the last front with the largest violation are left out.

**Diversity archive.** The candidates are the DA and the offspring, and the constraints play no part. In iteration `itr = 1, 2, ...`, each subregion in turn with fewer than `itr` solutions of the CA receives candidates of its own until it has `itr` counting the CA's, each time the non-dominated one with the smallest Tchebycheff value; the iterations stop when the DA is full or no candidate is left.

**Mating selection.** The first parent of every offspring of a generation comes from the CA when the share of non-dominated solutions of `CA + DA` that belong to the CA is larger than that of the DA, and from the DA otherwise. The second parent comes from the CA with a probability equal to the share of non-dominated solutions of the CA, and from the DA otherwise. Each parent wins a binary tournament in its archive: a feasible solution beats an infeasible one; of two feasible ones, the first wins unless the second dominates it; of two infeasible ones, a third solution drawn at random wins.

**Variation.** Each generation creates one offspring per weight vector: the first child of a two-parent crossover, mutated. The crossover receives the second parent first (see [where VAMOS departs](#where-vamos-departs-from-jmetals-port)). The `ctaea` [tuning space](../topics/tuning.md#constrained-spaces) samples the requested population size and the crossover among SBX, BLX-alpha and arithmetic, with VAMOS's real-coded mutations; `vamos tune --algorithm ctaea` tunes it. Every encoding runs with its two-parent crossovers; a crossover that needs more parents, such as `pcx` or `de`, is rejected.

```python
from vamos.algorithms import CTAEAConfig

config = CTAEAConfig.builder().pop_size(100).crossover("sbx", prob=1.0, eta=20.0).mutation("pm", prob="1/n", eta=20.0).build()
```

## Budget

The initial population has one solution per weight vector, so `max_evaluations` must be at least that number: 91 for `pop_size=100` and three objectives. `optimize`, study planning and the `vamos` command line compute it from `pop_size` and the problem's objectives and reject a smaller budget before the run; a budget between it and `pop_size` is accepted. Each generation then costs one evaluation per weight vector. Budgets are exact: when fewer evaluations are left than weight vectors, the last generation creates the offspring of a full generation and evaluates only the first ones, in the order of the weight vectors, and both archives are updated with them. jMetal checks the budget after each generation, so it can exceed it by up to the number of weight vectors minus one. The `n_gen` criterion counts generations of one evaluation per weight vector, the initial population included: `("n_gen", 500)` is 45,500 evaluations with 91 vectors.

## Agreement with jMetal's port and the published results

**jMetal's integration test.** IGD+ of the whole CA against jMetal's reference fronts, with the paper's configuration and 100 requested weight vectors, over 30 seeds. jMetal's own C-TAEA (jMetal 7.10 with the pull request's classes) was run 30 times with the same configuration; the last column is the median of the pull request's calibration runs.

| Problem | Evaluations | VAMOS, median (range) | jMetal, median (range) | Mann–Whitney p | jMetal PR #524 |
| --- | --- | --- | --- | --- | --- |
| C1-DTLZ1, 3 objectives, 7 variables | 91 × 500 | 1.66e-2 (1.61e-2 to 1.97e-2) | 1.66e-2 (1.60e-2 to 2.12e-2) | 0.97 | 0.017 |
| C3-DTLZ4, 3 objectives, 12 variables | 91 × 750 | 5.60e-2 (5.39e-2 to 6.31e-2) | 5.64e-2 (5.24e-2 to 6.19e-2) | 0.89 | 0.055 |
| MW3, 15 variables | 100 × 300 | 3.84e-3 (3.17e-3 to 5.65e-3) | 3.78e-3 (3.04e-3 to 4.80e-3) | 0.30 | 0.0037 |

Every solution of every result is feasible, in VAMOS and in jMetal.

**The CMOEA-MS paper.** The paper of CMOEA-MS [2] runs PlatEMO's C-TAEA on MW and LIR-CMOP with D = 10, N = 100 and 200,000 evaluations. IGD is the mean distance from the points of the reference front to the feasible solutions of the final CA, against jMetal's reference fronts (VAMOS packages jMetal's LIR-CMOP7 front), over 31 seeds, with jMetal's C-TAEA run in the same way. Every final CA is feasible. The medians agree with jMetal's and with the pull request's within the spread of the seeds; on MW3, VAMOS's, jMetal's and the pull request's medians are 11 to 12% above the paper's mean.

| Problem | VAMOS, median (range) | jMetal, median (range) | Mann–Whitney p | jMetal PR #524, median of 5 | Paper, mean |
| --- | --- | --- | --- | --- | --- |
| MW1 | 2.02e-3 (1.86e-3 to 2.20e-3) | 2.04e-3 (1.84e-3 to 2.64e-3) | 0.64 | 2.05e-3 | 2.01e-3 |
| MW3 | 5.13e-3 (4.63e-3 to 5.82e-3) | 5.07e-3 (4.59e-3 to 5.83e-3) | 0.58 | 5.06e-3 | 4.56e-3 |
| LIR-CMOP7 | 1.74e-2 (1.48e-2 to 2.83e-2) | 1.73e-2 (1.31e-2 to 2.89e-2) | 0.54 | 1.79e-2 | 1.81e-2 |

## Where VAMOS departs from jMetal's port

- **Budget:** exact, as described above; jMetal can exceed it in its last generation.
- **Parent order:** jMetal's SBX gives a variable that it does not cross to the other child, so jMetal's first child takes those variables from the second parent; VAMOS's SBX, as Deb's code, leaves them with their own parent. VAMOS therefore passes the second parent first, which gives the kept child jMetal's distribution. With the first parent first, the median IGD+ on C3-DTLZ4 is 6.38e-2 (6.00e-2 to 6.89e-2) over 30 seeds, against jMetal's 5.64e-2 (Mann–Whitney p < 1e-10), while C1-DTLZ1 and MW3 do not change.
- **Random draws:** each tournament draws three solutions and uses the third only when the first two are infeasible, which gives the winner jMetal's distribution; the operators are VAMOS's SBX and polynomial mutation, and the runs are not those of jMetal for the same seed.
- **Initialization:** the initial population is created by the configured initializer (uniform random by default) from the run's generator.
- **Result:** `result.X` and `result.F` keep the feasible non-dominated solutions of the CA (with `result_mode="non_dominated"`, the default) or the whole CA (`result_mode="population"`); jMetal returns the whole CA, which is `result.data["population"]` in VAMOS.
- **Violation:** VAMOS's convention, `g <= 0` feasible and the violation the sum of the positive parts, which is the authors' and, with the sign changed, jMetal's.

jMetal's port follows PlatEMO's implementation where it differs from the paper, and so does VAMOS: the second parent comes from the CA with the share of non-dominated solutions of the CA alone, not the share that chooses the first parent's archive; solutions are associated with the subregions on their raw objectives, except in the truncation of the feasible solutions of the CA, while the paper normalizes before every association; the ideal point of each Tchebycheff function is that of the set it is applied to; ties go to the solution that comes first instead of at random; and a tournament between two infeasible solutions picks a third one instead of one of the two at random. The CA may hold dominated solutions, and infeasible ones while fewer feasible solutions than weight vectors have been found.

## References

[1] Li, K., Chen, R., Fu, G., and Yao, X. (2019). [Two-Archive Evolutionary Algorithm for Constrained Multiobjective Optimization](https://doi.org/10.1109/TEVC.2018.2855411). *IEEE Transactions on Evolutionary Computation*, 23(2), 303–315.

[2] Tian, Y., Zhang, Y., Su, Y., Zhang, X., Tan, K. C., and Jin, Y. (2022). Balancing Objective Optimization and Constraint Satisfaction in Constrained Evolutionary Multiobjective Optimization. *IEEE Transactions on Cybernetics*. Tables I and II.
