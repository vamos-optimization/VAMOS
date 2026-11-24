---
description: 'PPS: the push and pull search for constrained multi-objective problems in MOEA/D, its switch between stages, its epsilon level, its output set and its agreement with the published results.'
---

# PPS

The **push and pull search** [1] splits the search for the front of a constrained multi-objective problem in two stages. The **push stage** ignores the constraints, so the population crosses the infeasible regions on its way to the unconstrained front. Once the ideal and nadir points of the population stop changing, the **pull stage** pulls the population back to the feasible regions with an epsilon constraint handling whose level shrinks to zero. VAMOS implements PPS-MOEA/D, the paper's instance of the framework in MOEA/D with differential evolution. The result is the set of feasible solutions that the run keeps along the way.

**Development after 1.5.2.** PPS is ported from the paper, whose Algorithms 1 to 8 and Eqs. 3 to 8 it follows; the paper leaves some choices open, which are [listed below](#where-the-paper-leaves-a-choice). jMetal has no port of PPS, so there is no reference implementation to check it against; its runs are compared with the paper's results instead.

[All algorithms](../reference/algorithms.md) · [Configuration reference](../reference/api/algorithms/pps.md) · [Constraints](../reference/constraints.md)

## Run an example

Use **LIR-CMOP2** with 30 variables (two constraints), the paper's configuration with a population of **100** (differential evolution with CR = 1.0 and F = 0.5, polynomial mutation with probability `1/n` and distribution index 20, neighbourhoods of 30), **30,000 evaluations**, **NumPy** and **seed 42**.

```python
import numpy as np

from vamos import make_problem_selection, optimize
from vamos.algorithms import PPSConfig

problem = make_problem_selection("lircmop2").instantiate()
config = PPSConfig.default(pop_size=100, n_var=problem.n_var)
result = optimize(
    problem,
    algorithm="pps",
    algorithm_config=config,
    max_evaluations=30_000,
    engine="numpy",
    seed=42,
)
print(result.data["evaluations"])  # 30000
print(result.data["switch_generation"])  # 152
print(result.data["stage"])  # pull
print(result.data["feasible_set"]["F"].shape)  # (100, 2)
print(result.F.shape)  # (54, 2)
print(bool(np.all(result.data["G"] <= 0)))  # True
```

The push stage lasted 151 generations: at the start of generation 152, the ideal and nadir points of the population had changed by at most 0.1% over the last 20 generations, and the search switched to the pull stage until the end. `result.data["feasible_set"]` is the paper's output, the set `NS` of at most `pop_size` feasible solutions kept from every generation; `result.X` and `result.F` hold its non-dominated solutions. `result.data["population"]` is the final population, one solution per subproblem. `switch_generation` is `None` when the run did not switch.

## How it works

**Decomposition and variation.** Each of the `pop_size` subproblems has a weight vector, those of MOEA/D (uniform with two objectives, the Das-Dennis lattice of `pop_size` vectors or a file with more), and the Tchebycheff function of Eq. 7, `max_j |f_j - z*_j| / lambda_j`, where `z*` is the ideal point of every solution evaluated. A generation visits the subproblems in order. For subproblem `i`, the candidates `S` are its `neighbor_size` (`T`) nearest subproblems with probability `delta`, and the whole population otherwise. The new solution is `x_i + F (x_r2 - x_r3)`, with `r2 != r3` drawn from `S`, crossed with `x_i` by the binomial crossover of rate CR (VAMOS's `de` operator), repaired, mutated and repaired again. It updates `z*` and the largest violation seen, then replaces at most `replace_limit` (`n_r`) solutions of `S`, tried in random order. The run is steady state: each new solution is evaluated before the next one is created.

**Push stage.** A new solution replaces a solution whose Tchebycheff value is not smaller than its own, whatever their violations (Algorithm 1).

**The switch.** Before generation `k` (counted from 1), `r_k` is the largest relative change of any component of the ideal and nadir points of the population since generation `k - l`, where `l` is `switch_window`, each change divided by the absolute value of the earlier component or by `1e-6` if that is smaller (Eqs. 4 to 6). `r_k` starts at 1. While `k < Tc`, the search switches to the pull stage when `r_k <= switch_threshold` (the paper's `epsilon = 1e-3`), and the epsilon level starts at `epsilon(0)`, the largest violation of every solution evaluated so far.

**Pull stage.** While `k < Tc`, the epsilon level of each generation, from the generation of the switch on, follows Eq. 8: `(1 - tao) epsilon(k - 1)` when the ratio of feasible to infeasible solutions of the population, `rf_k`, is below `feasibility_ratio_threshold` (`alpha`), and `epsilon(0) (1 - k / Tc) ** cp` otherwise. From `k = Tc` it is 0. A new solution replaces a solution when both violations are within the epsilon level, or equal, and its Tchebycheff value is not larger; otherwise the less violated of the two wins (Algorithm 2). With an epsilon level of 0, this is Deb's feasibility rule with the Tchebycheff value.

**The output.** After each generation, the feasible solutions of `NS` and of the population, each distinct solution once, form the next `NS`: all of them when they are `pop_size` or fewer, dominated ones included; otherwise those that NSGA-II selects, by non-dominated fronts and crowding distance (Algorithm 8). The result reports the non-dominated solutions of `NS` (with `result_mode="non_dominated"`, the default); when the run found no feasible solution, the non-dominated solutions of the least violated ones of the population; or the final population (`result_mode="population"`), which `optimize` uses when it is given no `algorithm_config`, as for the other algorithms ([results and cardinality](../reference/algorithms.md#results-and-cardinality)); `result.data["feasible_set"]` is `NS` in every mode. With an external archive, the result is the archive.

| Paper | Configuration field | Default (the paper's) |
| --- | --- | --- |
| `N` | `pop_size` | 300 |
| `T` | `neighbor_size` | 30 (or `pop_size` when smaller) |
| `delta` | `delta` | 0.9 |
| `n_r` | `replace_limit` | 2 |
| `epsilon` of Eq. 4 | `switch_threshold` | 1e-3 |
| `l` | `switch_window` | 20 |
| `alpha` | `feasibility_ratio_threshold` | 0.95 |
| `tau` | `tao` | 0.1 |
| `cp` | `cp` | 2 |
| `Tc` | `decay_fraction` of the run's generations | 0.8 (800 of 1000) |
| `CR`, `F` | `crossover=("de", {"cr": ..., "f": ...})` | 1.0, 0.5 |
| `p_m`, `eta_m` | `mutation=("pm", {"prob": ..., "eta": ...})` | `1/n`, 20 |

`feasibility_ratio_threshold` bounds a ratio, so it may exceed 1: 2 asks for two thirds of the population feasible. With two or three objectives the paper's 300 subproblems have weight vectors (the lattice of 23 divisions with three); with four or more, `PPSConfig.default(n_obj=...)` takes the largest population up to 300 that has them, and `optimize` without `pop_size` takes the largest population up to its quick default that has them (190 for 200 with three objectives). `Tc` is `decay_fraction` of the run's generations, the budget divided by the population, rounded half up: the paper's `Tc = 800` with 300,000 evaluations of 300 solutions. MOEA/D's `improved_epsilon` criterion, by the same authors, sets its control generation the same way. `tao` is spelled as in that criterion and in jMetal. The variation must be differential evolution (the crossover `de`, with `cr` and `f`, which may also be spelled `CR` and `F`, but not both ways at once); any real-coded mutation and repair of VAMOS can follow it. Discrete and mixed problems are rejected.

```python
from vamos.algorithms import PPSConfig

config = PPSConfig.builder().pop_size(300).crossover("de", cr=1.0, f=0.5).mutation("pm", prob="1/n", eta=20.0).build()
```

## Budget

The initial population takes `pop_size` evaluations, so `max_evaluations` must be at least `pop_size`; `optimize`, study planning and the `vamos` command line reject a smaller budget before the run. Each generation then costs `pop_size` evaluations, one per subproblem. Budgets are exact: when the budget ends within a generation, the subproblems after the last new solution are not visited, and the result merges the population into `NS` as the end of a generation would. The `n_gen` criterion counts generations of `pop_size` evaluations, the initial population included. An `("hv", ...)` termination is checked at the end of each generation, on the rows the result reports. A live callback gets a generation every `pop_size` evaluations, and the evaluations of a generation that the budget cuts short as one more.

**The switch needs a long enough run.** The ideal and nadir points must change by at most 0.1% over 20 generations before `Tc`; a run that has not switched by then stays in the push stage, as the pseudocode switches only while `k < Tc`, and its result is the feasible solutions that the push stage met. With 10,000 evaluations of 100 solutions (`Tc` = 80), PPS does not switch on MW3 or LIR-CMOP2; on MW3, whose front the unconstrained search crosses, its result is still close to the front, while on LIR-CMOP2 it has at most a few feasible solutions. With 30,000 evaluations, as in the example, it switches. The paper's runs have 1000 generations.

## Tuning

The `pps` [tuning space](../topics/tuning.md#constrained-spaces) samples MOEA/D's neighbourhood parameters, the differential evolution's CR and F, VAMOS's real-coded mutations and the parameters of both stages, with the paper's settings as its defaults; `vamos tune --algorithm pps` tunes it.

## Where the paper leaves a choice

- **`r_k` from `k = l + 1`.** The pseudocode computes `r_k` when `k >= l`, from the points of generation `k - l`; generation `l` would need a generation 0, which its numbering, from 1, does not have.
- **`epsilon(0)`** is the largest violation of every solution evaluated before the switch, as Algorithms 3 and 5 compute it; the text describes it as the largest violation of the population.
- **`rf_k`** is what the text says, "the ratio of feasible to infeasible solutions" of the population (infinite when all are feasible), so with `alpha = 0.95` the epsilon level follows its schedule once about 48.7% of the population is feasible. MOEA/D-IEpsilon, by the same authors [2], and MOEA/D's `improved_epsilon` criterion use the fraction of feasible solutions instead. With the fraction, over the same 30 seeds as [below](#agreement-with-the-published-results), 5 of the 30 runs on LIR-CMOP11 and 11 on LIR-CMOP12 miss a part of the front (IGD from 0.06 to 0.155), and the mean IGD on LIR-CMOP10 is 2.42e-3, against the paper's 2.11e-3 and the ratio's 2.17e-3; the fraction does better only on LIR-CMOP2 (4.45e-3, against the ratio's 5.25e-3) and LIR-CMOP7 (2.81e-3, against 2.86e-3).
- **The replacement loop** stops after `n_r` replacements or when `S` is used up: the pseudocode's `or` in line 25 is an `and`.
- **`k` in Eq. 8** is the generation counter of the run, not the generations since the switch.
- **After `Tc`**, a search still in the push stage stays there, as written.
- **The union `NS ∪ P`** of line 37 counts each distinct solution once, so a solution of the population that `NS` already holds does not take two places.
- **Repair.** The new solution is clamped to the bounds after the differential evolution as well as after the mutation, as MOEA/D-DE in VAMOS and jMetal, so the mutation works within the bounds; the paper repairs it after the mutation only.
- **A zero weight** counts as `1e-4` in Eq. 7, whose `1 / lambda_j` it would make infinite; a small weight makes its term dominate, as for the neighbouring weight vectors.
- **Non-finite values.** A NaN constraint makes the violation infinite, and only finite violations count for `epsilon(0)`; the ideal and nadir points ignore NaN objectives, as Algorithms 4 and 7 do.

## Agreement with the published results

The paper's Table 1 gives the mean and standard deviation of the IGD of PPS-MOEA/D over 30 runs on LIR-CMOP1 to 14, the problems with large infeasible regions of [2]. VAMOS ran the paper's configuration on the same problems: `PPSConfig.default(n_var=30)`, 300,000 evaluations, seeds 1 to 30 and the Numba engine (the engines give the same runs). The IGD is that of `result.F`, the non-dominated solutions of `NS`, against VAMOS's reference fronts of LIR-CMOP, jMetal's (1000 points with two objectives, 9870 with three). Welch's `t` compares the two means; beyond ±2, the difference is unlikely to be chance.

| Problem | Paper, mean (std) | VAMOS, mean (std) | VAMOS, median (range) | Welch's `t` | Switch generation, median (range) |
| --- | --- | --- | --- | --- | --- |
| LIR-CMOP1 | 6.41e-3 (1.94e-3) | 8.09e-3 (1.82e-3) | 7.76e-3 (5.63e-3 to 1.32e-2) | +3.4 | 77 (44 to 105) |
| LIR-CMOP2 | 4.67e-3 (7.84e-4) | 5.25e-3 (7.41e-4) | 5.13e-3 (3.96e-3 to 7.43e-3) | +2.9 | 96 (78 to 118) |
| LIR-CMOP3 | 8.55e-3 (5.18e-3) | 6.61e-3 (2.51e-3) | 5.34e-3 (3.71e-3 to 1.11e-2) | -1.8 | 77 (44 to 105) |
| LIR-CMOP4 | 4.68e-3 (1.12e-3) | 3.73e-3 (1.07e-3) | 3.64e-3 (2.52e-3 to 8.73e-3) | -3.4 | 96 (78 to 118) |
| LIR-CMOP5 | 1.84e-3 (9.26e-5) | 1.67e-3 (7.16e-5) | 1.68e-3 (1.55e-3 to 1.81e-3) | -8.0 | 222 (169 to 366) |
| LIR-CMOP6 | 2.49e-3 (3.40e-4) | 2.00e-3 (9.32e-5) | 1.98e-3 (1.84e-3 to 2.31e-3) | -7.7 | 226 (168 to 404) |
| LIR-CMOP7 | 2.80e-3 (9.85e-5) | 2.86e-3 (5.71e-5) | 2.86e-3 (2.74e-3 to 2.95e-3) | +2.9 | 222 (169 to 366) |
| LIR-CMOP8 | 2.78e-3 (7.56e-5) | 2.79e-3 (7.48e-5) | 2.80e-3 (2.66e-3 to 2.98e-3) | +0.6 | 226 (168 to 404) |
| LIR-CMOP9 | 9.94e-2 (1.52e-1) | 1.99e-3 (1.23e-4) | 1.96e-3 (1.80e-3 to 2.30e-3) | -3.5 | 88 (77 to 109) |
| LIR-CMOP10 | 2.11e-3 (7.75e-5) | 2.17e-3 (1.67e-4) | 2.12e-3 (1.93e-3 to 2.77e-3) | +1.9 | 95 (75 to 139) |
| LIR-CMOP11 | 2.83e-3 (1.36e-3) | 1.11e-3 (9.77e-4) | 9.18e-4 (6.89e-4 to 6.17e-3) | -5.6 | 95 (75 to 139) |
| LIR-CMOP12 | 2.70e-2 (5.00e-2) | 5.42e-4 (5.14e-5) | 5.48e-4 (4.54e-4 to 6.53e-4) | -2.9 | 88 (77 to 109) |
| LIR-CMOP13 | 6.46e-2 (2.18e-3) | 6.35e-2 (1.51e-3) | 6.34e-2 (6.05e-2 to 6.67e-2) | -2.2 | 201 (163 to 280) |
| LIR-CMOP14 | 6.42e-2 (1.69e-3) | 6.29e-2 (1.24e-3) | 6.28e-2 (6.05e-2 to 6.61e-2) | -3.5 | 201 (163 to 280) |

Every run switched to the pull stage, between generations 44 and 404 of 1000, and ended with feasible solutions only: 300 non-dominated ones in most runs, and from 141 to 276 on LIR-CMOP3 and 4.

- **On 12 of the 14 problems, the means agree or VAMOS's is lower.** VAMOS's mean is within 3% of the paper's on LIR-CMOP7, 8 and 10, and below it on LIR-CMOP3 to 6, 9 and 11 to 14. The paper's runs on LIR-CMOP9 and 12 have large deviations, so some of its runs missed a part of the front, which no run of VAMOS did; on LIR-CMOP11, one run of 30 did (6.17e-3).
- **LIR-CMOP1 and 2 are the exceptions:** VAMOS's means are 26% and 12% above the paper's, with similar deviations. The cause is not known; the IGD depends on the points of the reference front, which may not be the paper's.
- **The push stage depends only on the objectives**, as it ignores the constraints: LIR-CMOP1 and 3, 2 and 4, 5 and 7, 6 and 8, 9 and 12, 10 and 11, and 13 and 14 share their objectives, and each pair switched at the same generation with every seed.
- **The runs took 92 seconds each**, the median with the Numba engine on one core, most of it in the Python code that creates and places one solution at a time.

## References

[1] Fan, Z., Li, W., Cai, X., Li, H., Wei, C., Zhang, Q., Deb, K., and Goodman, E. (2019). [Push and pull search for solving constrained multi-objective optimization problems](https://doi.org/10.1016/j.swevo.2018.08.017). *Swarm and Evolutionary Computation*, 44, 665–679.

[2] Fan, Z., Li, W., Cai, X., Huang, H., Fang, Y., You, Y., Mo, J., Wei, C., and Goodman, E. (2019). [An improved epsilon constraint-handling method in MOEA/D for CMOPs with large infeasible regions](https://doi.org/10.1007/s00500-019-03794-x). *Soft Computing*, 23, 12491–12510.
