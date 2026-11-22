# Constraints

!!! info "Stable problem-definition API"
    `make_problem` and `optimize` are Stable in VAMOS 1.5.2. The violation
    helpers are Internal. See
    [Stability and versioning](../project/stability-and-versioning.md).
    **VAMOS 1.2.0:** the internal symbolic constraint DSL
    (`vamos.foundation.constraints.dsl`), which no algorithm or guide used, is
    removed; declare constraints with `make_problem` or a `Problem` subclass.

## Define inequality constraints

VAMOS uses the convention **`g(x) <= 0` means feasible**. Convert a requirement
such as `x[0] + x[1] <= 1` into `x[0] + x[1] - 1`. Bounds belong in `bounds`;
they do not need to be repeated as inequality constraints.

This complete example uses only the core [installation](../guide/installation.md)
and public interfaces:

```python
import numpy as np

from vamos import make_problem, optimize


def objectives(x):
    return [x[0] ** 2 + x[1] ** 2, (x[0] - 1) ** 2 + (x[1] - 1) ** 2]


def constraints(x):
    return [x[0] + x[1] - 1.0]


problem = make_problem(
    objectives,
    n_var=2,
    n_obj=2,
    bounds=[(0.0, 1.0), (0.0, 1.0)],
    encoding="real",
    constraints=constraints,
    n_constraints=1,
)
result = optimize(
    problem,
    algorithm="nsgaii",
    pop_size=40,
    max_evaluations=400,
    seed=7,
    engine="numpy",
)

# Check the returned decisions against the original requirement.
G = np.array([constraints(x) for x in result.X])
feasible = np.all(G <= 0.0, axis=1)
print(f"Feasible returned solutions: {feasible.sum()}/{len(feasible)}")
print("Feasible objective vectors:", result.F[feasible])
```

NSGA-II uses feasibility-aware selection by default. A short run is not a
guarantee that every returned solution is feasible; always inspect feasibility
before interpreting a result or calculating a quality indicator.

## Shapes, vectorization, and equalities

| Evaluation mode | Input | Objectives returned | Constraints returned |
| --- | --- | --- | --- |
| Default, one decision at a time | `(n_var,)` | `(n_obj,)` | `(n_constraints,)` |
| `vectorized=True` | `(n_points, n_var)` | `(n_points, n_obj)` | `(n_points, n_constraints)` |

The objective and constraint functions must follow the same evaluation mode.
Declare `n_constraints` explicitly. In a custom `Problem.evaluate(X, out)`
implementation, put the constraint matrix in `out["G"]` using the same sign
convention.

**VAMOS 1.2.0:** constraint values from a problem that declares
`n_constraints = 0` raise `EvaluationError`. This covers a `G` written by
`evaluate` and an overridden `Problem.constraints()`. VAMOS 1.1.0 ignored them
and optimized the problem as unconstrained. Constraint values, like objectives,
must be finite.

For an equality `h(x) = 0`, choose a meaningful numerical tolerance and encode
`abs(h(x)) - tolerance <= 0`. State that tolerance and any scaling of constraint
values in the experiment: they affect what counts as feasible and how violations
are compared.

## Choose an algorithm with constraint support

Consult the [algorithm capability matrix](algorithms.md) before selecting an
algorithm. Constraint support is algorithm-specific; the presence of a
constraint function does not make every algorithm feasibility-aware. For
example, NSGA-II's `constraint_mode="none"` explicitly disables its constraint
selection behavior.

Hypervolume and other objective-space metrics do not establish feasibility.
Filter to feasible solutions first and apply the same feasibility rule to every
algorithm in a comparison.

## Choose a constraint-handling technique

**VAMOS 1.2.0:** NSGA-II, NSGA-III, SMS-EMOA, AGE-MOEA, RVEA and
SMPSO take `constraint_handling`, the technique that ranks or compares
constrained solutions. They are the techniques of jMetal's package
`util.constrainthandling`, which Evolver's constrained spaces tune:

| Technique | Parameters | Ranking |
| --- | --- | --- |
| `cdp` (default) | none | Deb's constrained-domination principle: feasible solutions first, by Pareto dominance, then infeasible ones by total violation. |
| `violated_constraints` | none | The same rule over the number of violated constraints. |
| `epsilon_constrained` | `initial_quantile`, `exponent`, `decay_fraction` | Takahama and Sakai's epsilon-constrained method: a solution whose violation is at most epsilon counts as feasible. Epsilon starts at that quantile of the violations of the first ranked population and decays to 0 at that fraction of the run. |
| `push_pull` | `change_threshold`, `window_fraction`, `decay_fraction`, `exponent`, `feasibility_threshold`, `reduction_rate` | Fan et al.'s push and pull search: Pareto dominance alone until the ideal and nadir points settle, then epsilon-constrained. |
| `stochastic_ranking` | `probability` | Runarsson and Yao's stochastic ranking: a bubble sort that compares by objectives with that probability, or always between feasible solutions. |
| `self_adaptive_penalty` | none | Woldesenbet, Yen and Tessema's self-adaptive penalty: Pareto dominance over objectives modified by the violation and the fraction of feasible solutions. |

```python
from vamos.algorithms import NSGAIIConfig

config = (
    NSGAIIConfig.builder()
    .pop_size(100)
    .crossover("sbx", prob=0.9, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .constraint_handling("epsilon_constrained", initial_quantile=0.2, exponent=5.0, decay_fraction=0.8)
    .build()
)
```

The other configurations take the same field and builder method, such as
`NSGAIIIConfig.builder().constraint_handling(...)`. Every parameter of a
technique is required, as in jMetal. The techniques that change during a run,
`epsilon_constrained`, `push_pull` and `dynamic_penalty`, follow its progress: the evaluations
before the offspring's over the evaluation budget. Each of jMetal's ranks fixed populations
as jMetal does, which `tests/engine/test_constraint_handling_techniques.py`
checks against jMetal's own implementation; the others are checked against
rankings computed by hand and the worked examples of their papers.

**Development after 1.5.2:** NSGA-II, NSGA-III, SMS-EMOA and AGE-MOEA also
take four techniques that neither jMetal nor Evolver has, and NSGA-II three
more, `two_rankings`, `infeasibility_driven` and `adaptive_tradeoff`:

| Technique | Parameters | Ranking |
| --- | --- | --- |
| `static_penalty` | `coefficient` | A static penalty: Pareto dominance over every objective plus `coefficient` times the overall violation, the sum of the positive constraint values. |
| `dynamic_penalty` | `constant`, `alpha`, `beta` | Joines and Houck's non-stationary penalty: Pareto dominance over every objective plus `(constant * k) ** alpha` times the sum of the violations raised to `beta`, where `k` is the generation. |
| `violation_objective` | optional `violation_norm`, `ratio_objective`, `cdf_alpha`, `feasible_first` | The overall violation as one more objective (Coello, 2000): Pareto dominance over the objectives and the violation; Dam and Truong's options change the violation's norm, add the ratio of violated constraints, or rank feasible solutions first (NSGA-II only). |
| `two_rankings` | none | NSGA-II only. Ma, Wang and Song's ToR: the best by `alpha * R^c + (1 - alpha) * R^p`, positions by the constrained-domination principle and by Pareto dominance, one solution per front. |
| `shift_penalty` | `constant` | Ma and Wang's ShiP: Pareto dominance over objectives that move each infeasible solution towards its local feasible nadir and add its violation. |
| `infeasibility_driven` | `proportion` | NSGA-II only. Ray, Singh, Isaacs and Smith's IDEA: the best infeasible solutions, `proportion` of the population, rank above the feasible ones, one solution per front. |
| `adaptive_tradeoff` | none | NSGA-II only. Wang, Cai, Zhou and Zeng's ATM, in Li, Wang, Yang and Cai's version for several objectives: the ranking depends on the feasible proportion of the population, one solution per front. |

```python
from vamos.algorithms import NSGAIIConfig

config = (
    NSGAIIConfig.builder()
    .pop_size(100)
    .crossover("sbx", prob=0.9, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .constraint_handling("dynamic_penalty", constant=0.5, alpha=2.0, beta=2.0)
    .build()
)
```

Their parameters must be positive. `dynamic_penalty` follows section 3.1 of
J. A. Joines and C. R. Houck, "On the use of non-stationary penalty functions
to solve nonlinear constrained optimization problems with GA's" (Proceedings
of the First IEEE Conference on Evolutionary Computation, 1994, pp. 579–584).
The paper minimizes `f + P(alpha, beta)`, with `P(alpha, beta) = (C * k) **
alpha * SVC(beta)`, where `k` is the generation number and `SVC(beta)` is the
sum of every violation raised to `beta`, of the inequalities and the
equalities, each outside a tolerance band. VAMOS adds the same penalty to every
objective, and measures each violation as it stores the constraint, `max(0,
g_j)` with `g_j <= 0` feasible, with no tolerance band: a problem that turns an
equality into inequalities sets its own tolerance.

The generation `k` counts the run's evaluations in populations, the
evaluations before the offspring's over the population size, and is at least
1, as the paper's generation count: a generational algorithm ranks its initial
population and its first replacement at `k = 1` and its `k`-th replacement at
`k`, and a steady-state one, such as SMS-EMOA or NSGA-II with fewer offspring
than its population, adds `1 / pop_size` per offspring rather than 1.

The paper tested `C = 0.5` with `(alpha, beta)` set to (1, 1), (1, 2) and
(2, 2). P(1, 2) reached no feasible solution in two of its four test cases,
where the constraints were barely violated; P(2, 2), which squares the
generation number, puts the most pressure towards feasibility; and P(1, 1) did
best on average but took longer to converge. The defaults are P(2, 2) with
`C = 0.5`. VAMOS leaves out the paper's other family, `exp(P(alpha, beta))`
with `C = 0.05`, which overflowed in the paper when the initial violations were
large.

`two_rankings` follows section III-A of Z. Ma, Y. Wang and W. Song, "A New
Fitness Function With Two Rankings for Evolutionary Constrained Multiobjective
Optimization" (IEEE Transactions on Systems, Man, and Cybernetics: Systems
51(8):5005–5016, 2021). `R^c` is each solution's position, from 1, when the
population is sorted by the levels of the constrained-domination principle and
each level by crowding distance in descending order (Algorithm 1); `R^p` is
the same with Pareto dominance, ignoring the constraints (Algorithm 2). The
fitness is `alpha * R^c + (1 - alpha) * R^p` (eq. 6), with `alpha = 0.5 + 0.5
* P_fea` (eq. 7), where `P_fea` is the feasible proportion of the population
being ranked. The paper leaves three choices open:

- solutions with the same crowding distance in a level, such as its boundary
  solutions, keep the order of the population, as in the paper's Tables I–III;
- solutions with the same fitness are ordered by `R^c`;
- the crowding distance is NSGA-II's, each difference over the objective's
  range in the level. The paper's tables sum the differences without that
  division; with their sum, VAMOS reproduces the tables' positions, and with
  NSGA-II's some positions change but the solutions each table selects do not.

The technique puts each solution in a front of its own, ordered by fitness, so
NSGA-II's replacement keeps the best by fitness and crowding distance never
decides; the tournament compares the positions in the current population. The
paper defines ToR for NSGA-II only, and NSGA-III, SMS-EMOA and AGE-MOEA reject
it: its order would replace their niching, hypervolume contributions and
survival scores.

`shift_penalty` follows section III-B of Z. Ma and Y. Wang, "Shift-Based
Penalty for Evolutionary Constrained Multiobjective Optimization and Its
Application" (IEEE Transactions on Cybernetics 53(1):18–30, 2023). With
`P_fea` the feasible proportion of the population being ranked, each objective
`f_i` of an infeasible solution moves towards its local feasible nadir `z*_i`,
the smallest `f_i` of the feasible solutions above its own, or the largest
`f_i` of the population when there is none: `S_i = f_i + c * P_fea * (z*_i -
f_i)` (eq. 11). Its overall violation is added, `P_i = CV` (eq. 12), so its
objective is `S_i + P_i` (eq. 13). Feasible solutions keep their objectives,
and so does every solution while none is feasible. `constant` is the paper's
`c`, 2 in its experiments; its sensitivity study tried 0.5 to 9 and found 2
and 3 best. The paper says that its NSGA-II's mating selection uses the
constraint violation, without defining it. VAMOS's tournament compares ShiP's
ranks of the current population, as for every technique, so while no solution
is feasible it compares by Pareto dominance alone. ShiP+, the paper's
archive-assisted algorithm, is not part of VAMOS.

`violation_objective` takes the options of T. M. Dam and L. V. Truong, "A
Generalized Framework for Multi-objective-based Constraint Handling Technique"
(International Journal of Control, Automation, and Systems 22(10):3240–3252,
2024, section 3). Each is optional, and without them the technique ranks as it
does without options:

- `violation_norm`: the violation is `V = ||v||_p^p` of the positive constraint
  values `v` (Def. 7): their sum (`l1`, the default), the sum of their squares
  (`l2_squared`) or the largest (`linf`).
- `ratio_objective`: the constraint violation ratio `R` (Def. 6), the weighted
  share of the violated constraints, is one more objective, before `V`. VAMOS
  has no weights of its own for the constraints, so they are equal, or, with
  `cdf_alpha` above 0, those of the CDF (section 3.2): `1 + cdf_alpha * f_j /
  sum_i f_i`, where `f_j` is how many solutions of the population violate
  constraint `j`. The paper uses `cdf_alpha = 0.5`; `cdf_alpha` needs
  `ratio_objective`. The paper's shortened framework with `R` and without `V`
  is not offered.
- `feasible_first`: Algorithm A's survival, a total order, so NSGA-II only. The
  feasible solutions come first, by fronts and crowding distance of the
  objectives, then the infeasible ones by `R` (with `ratio_objective`) and then
  `V`, in ascending order, as `numpy.lexsort` with `R` as the primary key;
  equal values keep the order of the population. With it, the added
  objectives order the infeasible solutions only, so it is the
  constrained-domination principle with another order of the infeasible ones,
  which the paper's results on BNH show.

The tuning spaces offer `violation_objective_norm`, `violation_objective_ratio`
(`none`, `equal` or `cdf`, whose `violation_objective_cdf_alpha` ranges over
[0.1, 2], default 0.5) and, in NSGA-II's only, `violation_objective_feasible_first`.

`infeasibility_driven` and `adaptive_tradeoff` are defined for NSGA-II, and,
like `two_rankings`, order the population totally, so that NSGA-II's
replacement keeps the selection their papers define and its tournament
compares the positions in the current population. Their fronts and crowding
distances are NSGA-II's, and rows with the same front and crowding distance
keep the order of the population. NSGA-III, SMS-EMOA and AGE-MOEA reject them.

`infeasibility_driven` follows T. Ray, H. K. Singh, A. Isaacs and W. Smith,
"Infeasibility Driven Evolutionary Algorithm for Constrained Optimization", in
E. Mezura-Montes (ed.), *Constraint-Handling in Evolutionary Optimization*
(Studies in Computational Intelligence 198, Springer, 2009, pp. 145–165):

- The violation measure (section 3, Table 1) is one more objective. For each
  constraint, the solutions that violate it rank from 1 by their violation in
  ascending order, and those that satisfy it get 0; the measure is the sum of
  the ranks. Equal violations share a rank, and the paper does not say how the
  next rank follows: VAMOS continues with the next integer (dense ranks).
- The population splits into the feasible and the infeasible solutions, and
  each set is sorted by fronts and crowding distance over the objectives and
  the measure. The best `N_inf = proportion * N` infeasible solutions come
  first, then the feasible ones, then the other infeasible ones, so the first
  `N` are Algorithm 1's selection, with section 2's rule when either set falls
  short. `N` is the population size, and `N_inf` is rounded half up, which the
  paper leaves open.
- `proportion` lies in (0, 1]; the paper uses 0.2, 20% of the population, and
  calls it "a small percentage". The tuning space offers 0.05 to 0.5: half the
  population would no longer be a small percentage.

`adaptive_tradeoff` follows section III of Y. Wang, Z. Cai, Y. Zhou and
W. Zeng, "An Adaptive Tradeoff Model for Constrained Evolutionary
Optimization" (IEEE Transactions on Evolutionary Computation 12(1):80–92,
2008), in the version for several objectives of section III-C of J.-P. Li,
Y. Wang, S. Yang and Z. Cai, "A Comparative Study of Constraint-Handling
Techniques in Evolutionary Constrained Multiobjective Optimization" (IEEE CEC
2016, pp. 4175–4182). Its violation is Li et al.'s `C` (eq. 3), the mean over
the constraints of each violation over the largest in the population, and its
phase depends on the feasible proportion of the population being ranked:

- No feasible solution (phase one): the hierarchical non-dominated selection of
  Wang et al.'s Fig. 3 over the objectives and `C`. The non-dominated solutions
  of those that remain are sorted by `C` in ascending order, and the first half
  of them come next, until none remains. Half of an odd number is rounded up,
  which the paper leaves open, since rounding down would never take the last
  solution.
- Some feasible solutions (phase two, Li et al.'s eqs. 10–15): each objective
  of an infeasible solution becomes `max(phi * f_b + (1 - phi) * f_w, f)`,
  with `f_b` and `f_w` the best and worst of the feasible solutions; each
  objective is normalized over all the solutions and `C` over the infeasible
  ones, 0 for the feasible ones, and their sums rank by fronts and crowding
  distance. A range of 0 normalizes to 0.
- All feasible (phase three): fronts and crowding distance of the objectives.

`phi` is the feasible proportion of the population that the last replacement
kept: 0 until the first replacement, as in Wang et al.'s Fig. 4, and kept in
NSGA-II's checkpoints. ATM has no parameters.

The penalized or shifted objectives, like those of `self_adaptive_penalty`, and
the violation of `violation_objective` only rank the population: crowding
distance, niching, hypervolume contributions and survival scores use the
objectives themselves. The static and dynamic penalties multiply violations,
so the coefficient that balances them against the objectives depends on the
problem's scales.

Before each replacement, the technique adapts to the union of parents and
offspring and ranks it, as Evolver's algorithms do, and its fronts take the
place of the non-dominated ones:

| Algorithm | Replacement | Tournament |
| --- | --- | --- |
| NSGA-II | Whole fronts survive; crowding distance cuts the last. | Rank, then crowding distance. |
| NSGA-III | Whole fronts survive; reference-direction niching cuts the last. | Rank, then crowding distance. |
| SMS-EMOA | Once per child, the least hypervolume contributor of the last front leaves. | Rank, then crowding distance. |
| AGE-MOEA | Whole fronts survive; the survival score cuts the last. | Rank, then survival score. |

RVEA and SMPSO take only the four techniques that judge the violation of a
single solution, as Evolver's `EffectiveViolation` does, and MOEA/D none of them
(see below): `cdp` (the total
violation), `violated_constraints` (their number), `epsilon_constrained` (the
violation, or 0 within epsilon) and `push_pull` (0 while it pushes, then as
`epsilon_constrained`). RVEA's feasibility-first survival judges feasibility
and violation by that measure, after the technique adapts to the union of
parents and offspring. In SMPSO, the technique adapts to the swarm once per
iteration and decides, as its jMetal comparator, whether a particle replaces
its personal best: the less violated position wins, and two equally violated
ones compare by Pareto dominance when both count as feasible, and also otherwise
for `cdp` and `violated_constraints`. SMPSO's leaders archive keeps comparing by
the constrained-domination principle.

The tournament compares the technique's ranks of the population, which VAMOS
ranks again, as it does with `cdp`; jMetal reuses the ranks of the replacement.
Stochastic ranking draws its uniforms from the run's generator, so a run stays
a function of its seed, and the engines give the same run. NSGA-II's checkpoints
keep the state of the technique, such as epsilon or the generation of
`dynamic_penalty`. `cdp` keeps each algorithm's own survival, so
its results are those of VAMOS 1.1.0. A technique other than `cdp` needs
`constraint_mode="feasibility"`.

**VAMOS 1.2.0:** [CCMO](../algorithms/ccmo.md) is an algorithm
designed for constrained problems rather than a technique: it ranks its main
population with the violation first and evolves a second population that
ignores the constraints, whose offspring the main population can select.
[C-TAEA](../algorithms/ctaea.md) is another: its convergence archive is
driven by feasibility, and its diversity archive ignores the constraints and
keeps solutions in the subregions where the convergence archive has few.
[ToP](../algorithms/top.md) is a third: it first solves the problem as a
constrained single-objective one, to find the feasible regions that hold the
front, and then runs NSGA-II with the constrained-domination principle.
[CMOEA-MS](../algorithms/cmoeams.md) is a fourth: its fitness ranks the
density of the solutions together with their violation while half of them or
fewer are feasible, and puts the violation first once more than half of them
are, after the first 10% of the budget.

### MOEA/D: subproblem update criteria

**VAMOS 1.2.0:** MOEA/D handles constraints where a new solution
replaces the solutions of its neighbouring subproblems, and
`subproblem_update_criterion` chooses how, among the criteria of jMetal's
package `subproblemupdate`, as Evolver's `MOEADDoubleConstrained` space does:

| Criterion | Parameters | Replacement |
| --- | --- | --- |
| `feasibility_rules` (default) | none | Deb's feasibility rules: feasible first, then the less violated, then the better aggregation value. Two infeasible solutions with the same violation keep the current one, where jMetal compares their aggregation values. |
| `improved_epsilon` | `tao`, `feasibility_ratio_threshold`, `decay_fraction` | Fan et al.'s improved epsilon method: violations within epsilon, or equal, compare by aggregation value. Epsilon shrinks by `tao` each generation while fewer than `feasibility_ratio_threshold` of the population is feasible, grows otherwise, and is 0 from `decay_fraction` of the run's generations on. |
| `violation_threshold` | none | Jan and Khanum's adaptive threshold: the less violated wins if a violation (the number of violated constraints times the total violation) exceeds the fraction of infeasible solutions times their mean; otherwise the aggregation value decides. |

```python
from vamos.algorithms import MOEADConfig

config = (
    MOEADConfig.builder()
    .pop_size(100)
    .neighbor_size(20)
    .delta(0.9)
    .replace_limit(2)
    .aggregation("tchebycheff")
    .crossover("sbx", prob=1.0, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .subproblem_update_criterion("improved_epsilon", tao=0.1, feasibility_ratio_threshold=0.95, decay_fraction=0.8)
    .build()
)
```

The criteria decide as jMetal's own, which
`tests/engine/test_subproblem_update_criteria.py` checks on sequences of
populations. As in jMetal, a criterion adapts to the population once per new
solution, before its replacements, and a generation is `pop_size` new solutions.
With a criterion other than `feasibility_rules`, MOEA/D replaces child by child
in NumPy on both engines, and its checkpoints keep the criterion's state.
`feasibility_rules` keeps MOEA/D's own replacement, so its results are those of
VAMOS 1.1.0.

## Read the feasibility of a result

**VAMOS 1.2.0:** every solution array of a result carries the
constraint values of its rows: `result.data["G"]` for `result.X`, and `"G"` in
`result.data["population"]` and in the external archive's entry. The archive's
entry is `result.data["archive"]` for most algorithms, `archive_G` next to
`archive_X` for NSGA-III and IBEA, and `result.data["external_archive"]` for
SPEA2, whose `archive` is its environmental archive. SPEA2 and SMPSO omit
them with `constraint_mode="none"`. VAMOS 1.1.0 dropped them whenever the result
came from an external archive or SMPSO's leaders, and from the population of
most algorithms.

`result.feasibility()` summarizes the top-level rows:

```python
summary = result.feasibility()
# {"n_solutions": 40, "n_feasible": 37, "min_violation": 0.0}
```

The violation of a row is the sum of the positive parts of its constraint
values. `min_violation` is 0 when a row is feasible. A run without a feasible
solution has `n_feasible == 0` and the violation of its least violated row,
which is what feasibility-first comparisons, such as those of constrained
algorithm tuning, rank such runs by.

For a fuller modeling example, read
[Solve your own problem](../guide/custom-problem.md). Contributors who need the
implementation-level strategies can consult the
[Internal constraint reference](api/internal/constraints.md); these are not
stable public imports or interchangeable algorithm configuration options.
