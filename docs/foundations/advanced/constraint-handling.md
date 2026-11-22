---
description: 'Constraint-handling techniques in VAMOS: feasibility rules, epsilon-constrained, push and pull search, stochastic ranking and the self-adaptive penalty, with their parameters, papers and the algorithms that accept each, an interactive figure and VAMOS code.'
search:
  boost: 0.5
---

# Constraint-handling techniques

Lesson [Constraints](constrained.md) explained constrained domination, the rule
that ranks feasible solutions first, and named a few alternatives. VAMOS offers
six techniques that take the place of that rule when an algorithm ranks a
constrained population: those of jMetal's package `util.constrainthandling`,
which Evolver tunes. They differ in one question: how much may a solution that
breaks the constraints gain from good objectives? This lesson answers it for
each technique, with its parameters, the paper that introduced it and the
algorithms that accept it.

In this lesson you will learn:

- how a technique plugs into an algorithm's survival and parent selection;
- the feasibility rules, and the techniques that relax them: ε-constrained,
  push and pull search and stochastic ranking;
- the self-adaptive penalty, which lets the feasible share of the population
  weigh the violation;
- which algorithms accept which technique.

## How a technique plugs in

Constraints follow VAMOS's convention, g_j(x) ≤ 0 feasible, and the overall
violation is CV(x) = Σ_j max(0, g_j(x)) (lesson
[Constraints](constrained.md#feasibility-and-violation)). A technique turns
the objectives and constraint values of a population into **fronts**. Before
each replacement, it adapts to the union of parents and offspring and ranks
it, and its fronts take the place of the non-dominated fronts. Whatever breaks
ties inside the last front still uses the objectives: crowding distance in
NSGA-II, reference-direction niching in NSGA-III, hypervolume contributions in
SMS-EMOA, the survival score in AGE-MOEA. The parent tournament compares the
technique's ranks of the current population.

A builder sets the technique with `.constraint_handling(name, **parameters)`.
Every parameter is required, as in jMetal, and a technique other than `cdp`
needs the default `constraint_mode="feasibility"`. `cdp`, the default, keeps
each algorithm's own feasibility-first survival. Two techniques change during
the run: `epsilon_constrained` and `push_pull` follow its **progress**, the
evaluations before the offspring's over the evaluation budget.

| Algorithm | Techniques it accepts |
| --- | --- |
| NSGA-II, NSGA-III, SMS-EMOA, AGE-MOEA | all six |
| RVEA, SMPSO | `cdp`, `violated_constraints`, `epsilon_constrained`, `push_pull` |
| MOEA/D | none: its `subproblem_update_criterion` is `feasibility_rules`, `improved_epsilon` or `violation_threshold` |
| SPEA2, IBEA | none: they keep their own feasibility-first selection |
| CCMO, C-TAEA, ToP, CMOEA-MS | none: they are algorithms designed for constraints |

RVEA and SMPSO compare single solutions rather than rank fronts, so they take
only the four techniques that judge the violation of one solution, as
Evolver's `EffectiveViolation` does. Their configurations state it:

```python
# Excerpt from src/vamos/engine/algorithm/config/rvea.py
    #: The techniques that judge the violation of a single solution, the only ones this algorithm takes.
    constraint_handling_techniques: ClassVar[tuple[str, ...]] = PAIRWISE_TECHNIQUES
```

## Feasibility rules

**`cdp`** (no parameters) is Deb's constrained-domination principle [1]:
feasible solutions first, ranked by Pareto dominance, then the infeasible ones
by CV, one front per value of CV, with solutions of equal CV compared by Pareto
dominance. Lesson [Constraints](constrained.md#constrained-domination)
explains it with a figure.

**`violated_constraints`** (no parameters) applies the same rule to the number
of violated constraints instead of CV, a variant that jMetal provides. A
solution that breaks one constraint badly then ranks ahead of one that breaks
two constraints slightly.

Both rank with one function, which the ε-constrained method below also uses.
After the feasible fronts, it sorts the infeasible solutions by the measure and
gives each group of equal measure its own fronts:

```python
# Excerpt from src/vamos/engine/algorithm/components/constrained_ranking.py
    order = infeasible[np.argsort(measure[infeasible], kind="stable")]
    values = measure[order]
    starts = np.flatnonzero(np.concatenate(([True], values[1:] != values[:-1])))
    ends = np.concatenate((starts[1:], [order.size]))
    for start, end in zip(starts.tolist(), ends.tolist()):
        members = order[start:end]
        if objectives_on_ties and members.size > 1:
            group_ranks = pareto_ranks(kernel, F[members])
            ranks[members] = offset + group_ranks
            offset += int(group_ranks.max()) + 1
        else:
            ranks[members] = offset
            offset += 1
```

`cdp` and `violated_constraints` compare the solutions of a group by Pareto
dominance (`objectives_on_ties`); the ε-constrained method puts the whole
group in one front.

## Relaxing feasibility

### ε-constrained

Takahama and Sakai's ε-constrained method [2] treats a solution as feasible
when its violation is at most ε, and lets ε shrink to 0 during the run, so
that slightly infeasible solutions with good objectives can guide the search
early on:

<div class="vf-formula">ε(0) = the CV at quantile θ of the first ranked population
ε(t) = ε(0) · (1 − t / T_c)^cp   for t &lt; T_c,     0 from T_c on      (t: progress)
CV ≤ ε  →  ranked with the feasible solutions, by Pareto dominance
CV > ε  →  ranked after them, by CV; equal CV shares a front</div>

The parameters of `epsilon_constrained` are `initial_quantile` (θ),
`exponent` (cp) and `decay_fraction` (T_c); VAMOS's tuning space starts from
0.2, 5 and 0.8. The quantile is taken over the sorted violations, feasible
solutions included, so when a fifth of the first population is feasible,
θ = 0.2 gives ε(0) = 0 and the technique behaves almost like `cdp` from the
start.

### Push and pull search

Fan et al.'s push and pull search [3] runs in two phases. The **push** phase
ignores the constraints and ranks by Pareto dominance alone, so the population
approaches the unconstrained front. It ends when, over the last
`window_fraction` of the run, no coordinate of the ideal point or of the nadir
point of the non-dominated solutions changes by more than `change_threshold`,
relative to its value, or when the progress reaches `decay_fraction`. The
**pull** phase is ε-constrained: ε starts at the largest violation of the
population; while the feasible share is below `feasibility_threshold`, ε
shrinks by the factor 1 − `reduction_rate` at each update, and otherwise it
decays to 0 at `decay_fraction` with `exponent`. The six parameters of
`push_pull` start in VAMOS's tuning space from 0.001, 0.02, 0.8, 2, 0.95 and
0.1.

### Stochastic ranking

Runarsson and Yao's stochastic ranking [4] sorts the population with a bubble
sort that compares neighbours either by objectives or by violation. Two
feasible solutions always compare by objectives (in VAMOS, their Pareto rank);
otherwise a coin decides, objectives with probability p_f and CV otherwise. At
most n sweeps run, and the sort stops after a sweep without swaps:

```python
# Excerpt from src/vamos/engine/algorithm/components/constrained_ranking.py
    for _ in range(n):
        swapped = False
        for j in range(n - 1):
            first, second = order[j], order[j + 1]
            if is_feasible[first] and is_feasible[second]:
                by_rank = True
            else:
                by_rank = draws[drawn] < probability
                drawn += 1
            if (rank[second] < rank[first]) if by_rank else (value[second] < value[first]):
                order[j], order[j + 1] = second, first
                swapped = True
        if not swapped:
            break
```

Consecutive feasible solutions with the same Pareto rank share a front, and
every infeasible solution is a front of its own. The only parameter of
`stochastic_ranking` is `probability` (p_f); Runarsson and Yao used 0.45, the
default of VAMOS's tuning space. The coins come from the run's random
generator, so a run stays a function of its seed.

## The self-adaptive penalty

A penalty adds a multiple of the violation to every objective and ranks the
modified objectives by Pareto dominance. Penalties are the oldest
constraint-handling technique; Coello Coello's survey [6] reviews them. A
fixed multiple must balance violations against objective values, so it depends
on the scales of the problem.

**`self_adaptive_penalty`** (no parameters) is Woldesenbet, Yen and Tessema's
self-adaptive penalty [5], which needs no such multiple. It normalizes each
objective to the population's range, f̃_i, and averages each violation over
the largest in the population, v(x), and lets r_f, the feasible share, set the
balance:

<div class="vf-formula">d_i(x) = v(x)                    if r_f = 0
d_i(x) = √( f̃_i(x)² + v(x)² )     otherwise
F_i(x) = d_i(x) + (1 − r_f) · X(x) + r_f · Y_i(x)
X(x) = 0 if r_f = 0, else v(x);      Y_i(x) = 0 if x is feasible, else f̃_i(x)</div>

With no feasible solution only the violation counts; as the feasible share
grows, an infeasible solution is judged more by its objectives. The modified
objectives only rank: crowding distance, niching, hypervolume contributions
and survival scores use the real ones.

## Compare the techniques

The figure ranks ten solutions with two objectives and two constraints. A to E
are feasible (D dominates E); F to J are infeasible: F barely (CV 0.05), I with
the best objectives and the largest violation (CV 1.10). Choose a technique
and watch the front numbers; for the self-adaptive penalty, a dashed line leads
to the point that the technique actually ranks.

1. `cdp`: A to D form front 1 and E front 2; the infeasible ones follow one by
   one, in the order of their violation: F, G, J, H, I.
2. `violated_constraints`: H, which breaks one constraint (CV 0.35), moves ahead
   of G, which breaks two (CV 0.20).
3. `epsilon_constrained` with ε = 0.25: F and G count as feasible and push B,
   C and D to front 2.
4. `push_pull` in its push phase: the constraints do not count, so I, the most
   violated solution, shares front 1 with A and H.
5. `stochastic_ranking`: draw again, and the infeasible solutions change places,
   while feasible neighbours always compare by objectives.
6. `self_adaptive_penalty`: half of the solutions are feasible, so F and G,
   slightly infeasible, share front 1 with A, C and D, and push B to front 2.

<div class="vf-widget" data-vf-widget="constraint-techniques"><p class="vf-fallback">Interactive figure: ten solutions with two objectives and two constraints, ranked into fronts by each of the six constraint-handling techniques of VAMOS, with the parameters of each technique, the objectives that the self-adaptive penalty ranks and the algorithms that accept the technique. It needs JavaScript.</p></div>

## In VAMOS

The builders check a technique before any evaluation, so a missing
parameter, a value out of range or a technique that the algorithm does not
take fails at once:

```python
from vamos.algorithms import NSGAIIConfig, RVEAConfig

attempts = {
    "NSGA-II, epsilon_constrained without parameters": lambda: NSGAIIConfig.builder().constraint_handling("epsilon_constrained"),
    "NSGA-II, stochastic_ranking with probability 1.5": lambda: NSGAIIConfig.builder().constraint_handling(
        "stochastic_ranking", probability=1.5
    ),
    "RVEA, self_adaptive_penalty": lambda: RVEAConfig.builder()
    .pop_size(91)
    .n_partitions(12)
    .alpha(2.0)
    .adapt_freq(0.1)
    .crossover("sbx", prob=1.0, eta=30.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .constraint_handling("self_adaptive_penalty")
    .build(),
}
for label, attempt in attempts.items():
    try:
        attempt()
    except ValueError as error:
        print(f"{label}:\n  {error}")
```

Output:

```text
NSGA-II, epsilon_constrained without parameters:
  constraint_handling 'epsilon_constrained' takes initial_quantile, exponent, decay_fraction; unknown: [], missing: ['initial_quantile', 'exponent', 'decay_fraction'].
NSGA-II, stochastic_ranking with probability 1.5:
  constraint_handling 'stochastic_ranking': probability must lie in [0.0, 1.0]; got 1.5.
RVEA, self_adaptive_penalty:
  constraint_handling 'self_adaptive_penalty' does not judge the violation of a single solution; this algorithm takes cdp, violated_constraints, epsilon_constrained, push_pull.
```

The techniques change what survives. LIR-CMOP1 [7] has small feasible
regions, which no random design hits. NSGA-II, with the same seed and budget,
ends with very different populations:

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import LIRCMOP

push_pull = {
    "change_threshold": 0.001,
    "window_fraction": 0.02,
    "decay_fraction": 0.8,
    "exponent": 2.0,
    "feasibility_threshold": 0.95,
    "reduction_rate": 0.1,
}
techniques = [
    ("cdp", {}),
    ("violated_constraints", {}),
    ("epsilon_constrained", {"initial_quantile": 0.2, "exponent": 5.0, "decay_fraction": 0.8}),
    ("push_pull", push_pull),
    ("stochastic_ranking", {"probability": 0.45}),
    ("self_adaptive_penalty", {}),
]
for name, parameters in techniques:
    config = (
        NSGAIIConfig.builder()
        .pop_size(100)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament")
        .constraint_handling(name, **parameters)
        .result_mode("population")
        .build()
    )
    result = optimize(LIRCMOP(1), algorithm="nsgaii", algorithm_config=config, max_evaluations=10_000, seed=1)
    feasible = np.all(result.data["G"] <= 0.0, axis=1)
    print(f"{name:<22} {feasible.sum()} feasible of {len(feasible)}, front of {len(result.front())}")
```

Output:

```text
cdp                    100 feasible of 100, front of 51
violated_constraints   100 feasible of 100, front of 11
epsilon_constrained    100 feasible of 100, front of 34
push_pull              56 feasible of 100, front of 11
stochastic_ranking     97 feasible of 100, front of 37
self_adaptive_penalty  0 feasible of 100, front of 1
```

- `cdp`, `violated_constraints` and `epsilon_constrained` end with every
  solution feasible, but keep fronts of different sizes.
- `push_pull` and `stochastic_ranking` still keep infeasible solutions at the
  end.
- `self_adaptive_penalty` ends with no feasible solution; `result.front()`
  then holds the least violated one.

One seed shows what a technique does, not which one is better. Compare
techniques over many seeds and problems, as lesson
[Comparing algorithms rigorously](performance-assessment.md) explains. The
[constraint reference](../../reference/constraints.md#choose-a-constraint-handling-technique)
lists every technique with the choices VAMOS makes where the papers leave
details open.

## Key points

- A technique ranks the union of parents and offspring into fronts, which
  replace the non-dominated fronts in survival and in the tournament; the
  tie-breaking inside a front still uses the objectives.
- `cdp` and `violated_constraints` put feasibility first; `epsilon_constrained`,
  `push_pull` and `stochastic_ranking` let infeasible solutions with good
  objectives compete, by a shrinking threshold, a first phase without
  constraints or a coin.
- `self_adaptive_penalty` ranks objectives modified by the violation, which
  the feasible share of the population weighs.
- NSGA-II, NSGA-III, SMS-EMOA and AGE-MOEA take all six; RVEA and SMPSO take
  the four techniques that judge single solutions, and MOEA/D has its own
  update criteria.
- Every parameter is required, and techniques that change during a run follow
  the evaluations over the budget.

**Next:** [CCMO](ccmo.md).

## References

[1] Deb, K. (2000). [An efficient constraint handling method for genetic algorithms](https://doi.org/10.1016/S0045-7825(99)00389-8). *Computer Methods in Applied Mechanics and Engineering*, 186(2–4), 311–338.

[2] Takahama, T., and Sakai, S. (2006). [Constrained optimization by the ε constrained differential evolution with gradient-based mutation and feasible elites](https://doi.org/10.1109/CEC.2006.1688283). *Proceedings of the 2006 IEEE Congress on Evolutionary Computation*, 1–8.

[3] Fan, Z., Li, W., Cai, X., Li, H., Wei, C., Zhang, Q., Deb, K., and Goodman, E. (2019). [Push and pull search for solving constrained multi-objective optimization problems](https://doi.org/10.1016/j.swevo.2018.08.017). *Swarm and Evolutionary Computation*, 44, 665–679.

[4] Runarsson, T. P., and Yao, X. (2000). [Stochastic ranking for constrained evolutionary optimization](https://doi.org/10.1109/4235.873238). *IEEE Transactions on Evolutionary Computation*, 4(3), 284–294.

[5] Woldesenbet, Y. G., Yen, G. G., and Tessema, B. G. (2009). [Constraint handling in multiobjective evolutionary optimization](https://doi.org/10.1109/TEVC.2008.2009032). *IEEE Transactions on Evolutionary Computation*, 13(3), 514–525.

[6] Coello Coello, C. A. (2002). [Theoretical and numerical constraint-handling techniques used with evolutionary algorithms: A survey of the state of the art](https://doi.org/10.1016/S0045-7825(01)00323-1). *Computer Methods in Applied Mechanics and Engineering*, 191(11–12), 1245–1287.

[7] Fan, Z., Li, W., Cai, X., Huang, H., Fang, Y., You, Y., Mo, J., Wei, C., and Goodman, E. (2019). [An improved epsilon constraint-handling method in MOEA/D for CMOPs with large infeasible regions](https://doi.org/10.1007/s00500-019-03794-x). *Soft Computing*, 23, 12491–12510.
