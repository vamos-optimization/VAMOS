---
description: 'Constraint-handling techniques in VAMOS: feasibility rules, epsilon-constrained, push and pull search, stochastic ranking, static, dynamic, self-adaptive and shift-based penalties, the violation as an objective and its generalization by Dam and Truong, ToR, IDEA, ATM, MCR and the ensemble of techniques, with their parameters, papers and the algorithms that accept each, an interactive figure and VAMOS code.'
search:
  boost: 0.5
---

# Constraint-handling techniques

Lesson [Constraints](constrained.md) explained constrained domination, the rule
that ranks feasible solutions first, and named a few alternatives. VAMOS offers
fifteen techniques that take the place of that rule when an algorithm ranks a
constrained population. They differ in one question: how much may a solution
that breaks the constraints gain from good objectives? This lesson answers it
for each technique, with its parameters, the paper that introduced it and the
algorithms that accept it.

In this lesson you will learn:

- how a technique plugs into an algorithm's survival and parent selection;
- the feasibility rules, and the techniques that relax them: ε-constrained,
  push and pull search and stochastic ranking;
- the penalties, from static to self-adaptive and shift-based, and the
  violation as one more objective, with Dam and Truong's options;
- the five techniques that order the population totally, ToR, IDEA, ATM, MCR
  and the ensemble, and why VAMOS offers them for NSGA-II only;
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
Every parameter is required, except the options of `violation_objective` and
the weights of `ensemble`, which have defaults, and a technique other than
`cdp` needs the default `constraint_mode="feasibility"`. `cdp`, the default,
keeps each algorithm's own feasibility-first survival. Four techniques change
during the run: `epsilon_constrained`, `push_pull`, `dynamic_penalty` and
`ensemble`, which includes a dynamic penalty, follow its **progress**, the
evaluations before the offspring's over the evaluation budget.

| Algorithm | Techniques it accepts |
| --- | --- |
| NSGA-II | all fifteen |
| NSGA-III, SMS-EMOA, AGE-MOEA | all but `two_rankings`, `infeasibility_driven`, `adaptive_tradeoff`, `multiple_constraint_ranking`, `ensemble`, and `violation_objective` with `feasible_first` |
| RVEA, SMPSO | `cdp`, `violated_constraints`, `epsilon_constrained`, `push_pull` |
| MOEA/D | none: its `subproblem_update_criterion` is `feasibility_rules`, `improved_epsilon` or `violation_threshold` |
| SPEA2, IBEA | none: they keep their own feasibility-first selection |
| CCMO, C-TAEA, ToP, CMOEA-MS | none: they are algorithms designed for constraints |

RVEA and SMPSO compare single solutions rather than rank fronts, so they take
only the techniques that judge the violation of one solution. The last five,
and `violation_objective` with `feasible_first`, belong to NSGA-II, as the
source states:

```python
# Excerpt from src/vamos/engine/algorithm/components/constraint_technique.py
#: The techniques that judge the violation of a single solution and compare pairs of solutions, which
#: RVEA's survival and SMPSO's personal bests use, as Evolver's ``EffectiveViolation``.
PAIRWISE_TECHNIQUES: tuple[str, ...] = ("cdp", "violated_constraints", "epsilon_constrained", "push_pull")

#: The techniques that their papers define for NSGA-II only. Each orders the population totally, one solution per
#: front, which would take the place of the niching, hypervolume contributions or survival scores of the others.
NSGAII_ONLY_TECHNIQUES: tuple[str, ...] = (
    "two_rankings",
    "infeasibility_driven",
    "adaptive_tradeoff",
    "multiple_constraint_ranking",
    "ensemble",
)

def nsgaii_only(name: str, params: Mapping[str, TechniqueParameter]) -> bool:
    """Whether the technique, with these parameters, is defined for NSGA-II only.

    ``violation_objective`` is, with ``feasible_first``: Dam and Truong's survival orders the population totally.
    """
    return name in NSGAII_ONLY_TECHNIQUES or (name == "violation_objective" and params.get("feasible_first") is True)
```

The first six techniques are those of jMetal's package
`util.constrainthandling`, which Evolver tunes; the other nine are not in
jMetal.

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
start:

```python
# Excerpt from src/vamos/engine/algorithm/components/constraint_jmetal.py
    def update(self, kernel: KernelBackend, F: NDArray[Any], G: NDArray[Any], progress: float) -> None:
        progress = check_progress(progress, "epsilon_constrained")
        if not self.started:
            violations = np.sort(violation(G))
            if violations.size == 0:
                raise ValueError("epsilon_constrained cannot start from an empty population.")
            self.initial_epsilon = float(violations[int(math.floor(self.initial_quantile * (violations.size - 1)))])
            self.started = True
        self.epsilon = (
            self.initial_epsilon * (1.0 - progress / self.decay_fraction) ** self.exponent if progress < self.decay_fraction else 0.0
        )
```

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
most n sweeps run, and the sort stops after a sweep without swaps. Consecutive
feasible solutions with the same Pareto rank share a front, and every
infeasible solution is a front of its own. The only parameter of
`stochastic_ranking` is `probability` (p_f); Runarsson and Yao used 0.45, the
default of VAMOS's tuning space. The coins come from the run's random
generator, so a run stays a function of its seed.

## Penalties

A penalty adds a multiple of the violation to every objective and ranks the
modified objectives by Pareto dominance. Feasible solutions keep their
objectives. The modified objectives only rank: crowding distance, niching,
hypervolume contributions and survival scores use the real ones. Penalties are
the oldest constraint-handling technique; Coello Coello's survey [6] reviews
them.

**`static_penalty`** (`coefficient` = c) ranks f_i + c · CV. The coefficient
must balance violations against objective values, so it depends on the scales
of the problem; there is no published setting, and VAMOS's tuning space starts
from 10⁶, pymoo's default, which makes any violation outweigh any difference
of objectives.

**`dynamic_penalty`** (`constant`, `alpha`, `beta`) is Joines and Houck's
non-stationary penalty [7]:

<div class="vf-formula">f_i + (C · k)^α · Σ_j max(0, g_j)^β          (k: the generation, from 1)</div>

The weight grows with the generation k, which VAMOS counts as the evaluations
so far over the population size, at least 1, so infeasible solutions are
tolerated early and rejected late. The paper's P(2, 2) with C = 0.5 is the
default of the tuning space. VAMOS measures each violation as max(0, g_j),
without the tolerance band of the paper.

**`self_adaptive_penalty`** (no parameters) is Woldesenbet, Yen and Tessema's
self-adaptive penalty [5]. It normalizes each objective to the population's
range, f̃_i, and averages each violation over the largest in the population,
v(x), and lets r_f, the feasible share, set the balance:

<div class="vf-formula">d_i(x) = v(x)                    if r_f = 0
d_i(x) = √( f̃_i(x)² + v(x)² )     otherwise
F_i(x) = d_i(x) + (1 − r_f) · X(x) + r_f · Y_i(x)
X(x) = 0 if r_f = 0, else v(x);      Y_i(x) = 0 if x is feasible, else f̃_i(x)</div>

With no feasible solution only the violation counts; as the feasible share
grows, an infeasible solution is judged more by its objectives.

**`shift_penalty`** (`constant` = c) is Ma and Wang's ShiP [10]. Each
objective of an infeasible solution moves towards its **local feasible
nadir** z\*_i, the smallest f_i of the feasible solutions above its own (the
largest f_i of the population if there is none), and its violation is added:

<div class="vf-formula">f_i  →  f_i + c · P_fea · (z*_i − f_i) + CV          (P_fea: the feasible share)</div>

The shift grows with the feasible share. With c · P_fea = 1, an infeasible
solution lands exactly on feasible values plus its violation, so a slightly
infeasible solution in a gap of the feasible front can stay non-dominated. The
paper uses c = 2, and found 2 and 3 best; feasible solutions, and every
solution while none is feasible, keep their objectives.

## The violation as an objective

**`violation_objective`** ranks (f₁, …, f_m, CV) by Pareto dominance. Coello
Coello [8] studied how to treat constraints as objectives; without options,
VAMOS uses the simplest form, one extra objective. An infeasible solution with
better objectives than every feasible one is then non-dominated, so the
population keeps a whole trade-off between objectives and violation, and fewer
of its members are feasible at the end.

Dam and Truong [15] generalized that form, and `violation_objective` takes
their options, each optional:

- `violation_norm` measures the violation V as the sum of the positive
  constraint values (`l1`, the default, which is CV), the sum of their squares
  (`l2_squared`) or the largest (`linf`).
- `ratio_objective=True` adds one more objective before V, the constraint
  violation ratio R: the weighted share of the constraints that a solution
  violates. The weights are equal or, with `cdf_alpha` above 0,
  1 + cdf_alpha · f_j / Σ_i f_i, where f_j counts the solutions that violate
  constraint j, so a constraint that many solutions violate weighs more. The
  paper uses `cdf_alpha=0.5`.
- `feasible_first=True` replaces Pareto dominance with the paper's survival:
  the feasible solutions first, by fronts and crowding distance of the
  objectives, then the infeasible ones in ascending order of R and then V. It
  amounts to constrained domination with another order of the infeasible
  solutions, and it is a total order, so it is for NSGA-II only, like the
  techniques of the next section.

## Total orders, NSGA-II only

Five techniques define a selection rather than a dominance relation. VAMOS
returns each as a **total order**, one solution per front, so NSGA-II's
replacement, which keeps whole fronts, keeps exactly the solutions the paper
selects, and crowding distance never decides. In NSGA-III, SMS-EMOA or AGE-MOEA
that order would replace their own diversity mechanism, so they reject the
five.

**`two_rankings`** (no parameters) is Ma, Wang and Song's ToR [9]. It sorts the
population twice by fronts and then by crowding distance: once with
constrained domination, giving each solution a position R^c from 1, and once
with Pareto dominance alone, giving R^p. The fitness weighs the two by the
feasible share:

<div class="vf-formula">fit = α · R^c + (1 − α) · R^p,     α = 0.5 + 0.5 · P_fea</div>

With no feasible solution both positions count equally; with all feasible,
only R^c. Equal fitness is ordered by R^c.

**`infeasibility_driven`** (`proportion`) is Ray, Singh, Isaacs and Smith's
IDEA [11], which keeps a few infeasible solutions on purpose, because the best
of them lie near the boundary of the feasible region, where constrained optima
often are. Its violation measure ranks the violations of each constraint
(1 for the smallest, 0 when it is satisfied) and adds the ranks. The feasible
and the infeasible solutions are each sorted by fronts and crowding distance
over the objectives and that measure; the best N_inf = proportion · N
infeasible solutions come first, then the feasible ones, then the other
infeasible ones. N is the population size, and the paper uses a proportion of
0.2.

**`adaptive_tradeoff`** (no parameters) is Wang, Cai, Zhou and Zeng's ATM [12],
in the version for several objectives of Li, Wang, Yang and Cai [13]. Its
violation C averages each violation over the largest in the population, and
the feasible share chooses one of three phases:

- **No feasible solution:** a hierarchical selection over the objectives and
  C. Among the non-dominated solutions of those left, the half with the
  smallest C comes next, until none is left.
- **Some feasible:** each objective of an infeasible solution becomes
  max(φ f_best + (1 − φ) f_worst, f), with the best and worst values of the
  feasible solutions; the objectives are normalized, the normalized C of the
  infeasible ones is added, and fronts and crowding distance order the sums.
  φ is the feasible share of the last population that survived.
- **All feasible:** fronts and crowding distance of the objectives.

The last two follow Fukumoto and Oyama's framework [17], which brings
techniques designed for one objective to NSGA-II through a scalar **NSGA-II
fitness**: the Pareto rank minus the crowding distance, scaled so as to keep
the order of the ranks. The papers do not say how; VAMOS scales the
finite distances of the population into [0, 1/4] and gives the extremes of each
front 1/2, so the extremes stay preferred, as in NSGA-II.

**`multiple_constraint_ranking`** (no parameters) is Garcia, de Lima, Lemonge
and Jacob's MCR [16]. It ranks the population separately by the number of
violated constraints, R_Nv, by the violation of each constraint j, R_φ^j, and,
when some solution is feasible, by the NSGA-II fitness of the objectives, R_f,
and adds the ranks:

<div class="vf-formula">fitness = R_Nv + Σ_j R_φ^j  (+ R_f when some solution is feasible)
rank of a value = 1 + the number of solutions with a smaller value</div>

Equal values share a rank, and the smallest fitness comes first. Only ranks
count, not magnitudes, so a constraint with large values does not outweigh the
others.

**`ensemble`** (optional `nds_weight`, `cdp_weight`, `mcr_weight`,
`dpf_weight`) is Vodopija, Oyama and Filipič's ensemble [18]. Four members
give each solution a quality, lower being better: NDS, the NSGA-II fitness
with Pareto dominance alone; CDP, the same with constrained domination; MCR's
fitness; and DPF, the NDS fitness plus Joines and Houck's dynamic penalty with
C = 0.5 and α = β = 2, settings that the paper does not give. Each quality is
scaled min-max into [0, 1] over the population, and the solutions are ordered
by

<div class="vf-formula">c_e = w_NDS · c_NDS + w_CDP · c_CDP + w_MCR · c_MCR + w_DPF · c_DPF</div>

The weights default to 1/4 each, as in the paper, and are normalized to sum 1,
so at least one must be above 0; a member of weight 0 takes no part. DPF's
penalty grows with the generation, so the ensemble lets infeasible solutions
compete early and pushes them back later.

## Compare the techniques

The figure ranks ten solutions with two objectives and two constraints. A to E
are feasible (D dominates E); F to J are infeasible: F barely (CV 0.05), I with
the best objectives and the largest violation (CV 1.10). Choose a technique
and watch the front numbers; for the penalties, a dashed line leads to the
point that the technique actually ranks.

1. `cdp`: A to D form front 1 and E front 2; the infeasible ones follow one by
   one, in the order of their violation: F, G, J, H, I.
2. `violated_constraints`: H, which breaks one constraint (CV 0.35), moves ahead
   of G, which breaks two (CV 0.20).
3. `epsilon_constrained` with ε = 0.25: F and G count as feasible and push B,
   C and D to front 2.
4. `static_penalty`: with coefficient 1, F and G share front 1 with A and D;
   with 5, G falls to front 3.
5. `violation_objective`: F, G, H and I join front 1; J, which F dominates
   in all three, does not.
6. `infeasibility_driven` with proportion 0.2: two infeasible solutions, F and
   H, rank above every feasible one.
7. `multiple_constraint_ranking`: H comes ahead of F. Its objectives rank
   better, and only ranks count, not that its violation, 0.35, is seven times
   F's.
8. `ensemble` at generation 1: H and F rank second and third, ahead of the
   feasible B, C and D. At generation 30 the dynamic penalty pushes both behind
   B, C and D.
9. `violation_objective` with `feasible_first`: the feasible solutions first,
   then F, G, J, H and I by CV, one per front. With `ratio_objective` as well,
   J and H, which violate one constraint, overtake G, which violates two.

<div class="vf-widget" data-vf-widget="constraint-techniques"><p class="vf-fallback">Interactive figure: ten solutions with two objectives and two constraints, ranked into fronts by each of the fifteen constraint-handling techniques of VAMOS, with the parameters and options of each technique, the objectives that penalties rank and the algorithms that accept the technique. It needs JavaScript.</p></div>

## In VAMOS

The builders check a technique when they build the configuration, so a
missing parameter or a technique that the algorithm does not take fails before
any evaluation:

```python
from vamos.algorithms import NSGAIIConfig, NSGAIIIConfig, RVEAConfig

attempts = {
    "NSGA-II, epsilon_constrained without parameters": lambda: NSGAIIConfig.builder().constraint_handling("epsilon_constrained"),
    "NSGA-III, two_rankings": lambda: NSGAIIIConfig.builder()
    .pop_size(91)
    .crossover("sbx", prob=1.0, eta=30.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .selection("random")
    .constraint_handling("two_rankings")
    .build(),
    "RVEA, static_penalty": lambda: RVEAConfig.builder()
    .pop_size(91)
    .n_partitions(12)
    .alpha(2.0)
    .adapt_freq(0.1)
    .crossover("sbx", prob=1.0, eta=30.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .constraint_handling("static_penalty", coefficient=10.0)
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
NSGA-III, two_rankings:
  constraint_handling 'two_rankings' is defined for NSGA-II only: its total order would replace this algorithm's own diversity mechanism; this algorithm takes cdp, violated_constraints, epsilon_constrained, push_pull, stochastic_ranking, self_adaptive_penalty, static_penalty, dynamic_penalty, violation_objective, shift_penalty.
RVEA, static_penalty:
  constraint_handling 'static_penalty' does not judge the violation of a single solution; this algorithm takes cdp, violated_constraints, epsilon_constrained, push_pull.
```

The techniques change what survives. NSGA-II on CTP2 [14], with the same seed
and budget, keeps different numbers of feasible solutions in its final
population:

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import CTP

techniques = [
    ("cdp", {}),
    ("static_penalty", {"coefficient": 1.0}),
    ("static_penalty", {"coefficient": 100.0}),
    ("violation_objective", {}),
    ("two_rankings", {}),
    ("infeasibility_driven", {"proportion": 0.2}),
    ("adaptive_tradeoff", {}),
    ("multiple_constraint_ranking", {}),
    ("ensemble", {}),
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
    result = optimize(CTP(2), algorithm="nsgaii", algorithm_config=config, max_evaluations=10_000, seed=1)
    feasible = np.all(result.data["G"] <= 0.0, axis=1)
    label = name + "".join(f" {key}={value:g}" for key, value in parameters.items())
    print(f"{label:<35} {feasible.sum()} feasible of {len(feasible)}, feasible front of {len(result.front())}")
```

Output:

```text
cdp                                 100 feasible of 100, feasible front of 69
static_penalty coefficient=1        32 feasible of 100, feasible front of 32
static_penalty coefficient=100      94 feasible of 100, feasible front of 79
violation_objective                 7 feasible of 100, feasible front of 7
two_rankings                        100 feasible of 100, feasible front of 64
infeasibility_driven proportion=0.2 80 feasible of 100, feasible front of 80
adaptive_tradeoff                   100 feasible of 100, feasible front of 100
multiple_constraint_ranking         100 feasible of 100, feasible front of 56
ensemble                            100 feasible of 100, feasible front of 100
```

- With coefficient 1, a violation costs little next to the objectives, and
  most of the population stays infeasible; with 100 it behaves almost like
  `cdp`.
- The violation as an objective keeps a trade-off between objectives and
  violation, and only a few of its solutions are feasible.
- IDEA keeps 20 infeasible solutions, a fifth of the population, by design.
- ToR, ATM, MCR and the ensemble end with a feasible population, as `cdp`
  does, but keep feasible fronts of different sizes.

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
- `static_penalty`, `dynamic_penalty`, `self_adaptive_penalty` and
  `shift_penalty` rank modified objectives; `violation_objective` adds the
  violation as an objective, measured as Dam and Truong's options choose.
- `two_rankings`, `infeasibility_driven`, `adaptive_tradeoff`,
  `multiple_constraint_ranking` and `ensemble`, and `violation_objective` with
  `feasible_first`, return a total order and are NSGA-II only; RVEA and SMPSO
  take the four techniques that judge single solutions, and MOEA/D has its own
  update criteria.
- Every parameter is required, except the options of `violation_objective` and
  the weights of `ensemble`, and techniques that change during a run follow
  the evaluations over the budget.

**Next:** [CCMO](ccmo.md).

## References

[1] Deb, K. (2000). [An efficient constraint handling method for genetic algorithms](https://doi.org/10.1016/S0045-7825(99)00389-8). *Computer Methods in Applied Mechanics and Engineering*, 186(2–4), 311–338.

[2] Takahama, T., and Sakai, S. (2006). [Constrained optimization by the ε constrained differential evolution with gradient-based mutation and feasible elites](https://doi.org/10.1109/CEC.2006.1688283). *Proceedings of the 2006 IEEE Congress on Evolutionary Computation*, 1–8.

[3] Fan, Z., Li, W., Cai, X., Li, H., Wei, C., Zhang, Q., Deb, K., and Goodman, E. (2019). [Push and pull search for solving constrained multi-objective optimization problems](https://doi.org/10.1016/j.swevo.2018.08.017). *Swarm and Evolutionary Computation*, 44, 665–679.

[4] Runarsson, T. P., and Yao, X. (2000). [Stochastic ranking for constrained evolutionary optimization](https://doi.org/10.1109/4235.873238). *IEEE Transactions on Evolutionary Computation*, 4(3), 284–294.

[5] Woldesenbet, Y. G., Yen, G. G., and Tessema, B. G. (2009). [Constraint handling in multiobjective evolutionary optimization](https://doi.org/10.1109/TEVC.2008.2009032). *IEEE Transactions on Evolutionary Computation*, 13(3), 514–525.

[6] Coello Coello, C. A. (2002). [Theoretical and numerical constraint-handling techniques used with evolutionary algorithms: A survey of the state of the art](https://doi.org/10.1016/S0045-7825(01)00323-1). *Computer Methods in Applied Mechanics and Engineering*, 191(11–12), 1245–1287.

[7] Joines, J. A., and Houck, C. R. (1994). [On the use of non-stationary penalty functions to solve nonlinear constrained optimization problems with GA's](https://doi.org/10.1109/ICEC.1994.349995). *Proceedings of the First IEEE Conference on Evolutionary Computation*, 579–584.

[8] Coello Coello, C. A. (2000). [Treating constraints as objectives for single-objective evolutionary optimization](https://doi.org/10.1080/03052150008941301). *Engineering Optimization*, 32(3), 275–308.

[9] Ma, Z., Wang, Y., and Song, W. (2021). [A new fitness function with two rankings for evolutionary constrained multiobjective optimization](https://doi.org/10.1109/TSMC.2019.2943973). *IEEE Transactions on Systems, Man, and Cybernetics: Systems*, 51(8), 5005–5016.

[10] Ma, Z., and Wang, Y. (2023). [Shift-based penalty for evolutionary constrained multiobjective optimization and its application](https://doi.org/10.1109/TCYB.2021.3069814). *IEEE Transactions on Cybernetics*, 53(1), 18–30.

[11] Ray, T., Singh, H. K., Isaacs, A., and Smith, W. (2009). [Infeasibility driven evolutionary algorithm for constrained optimization](https://doi.org/10.1007/978-3-642-00619-7_7). In E. Mezura-Montes (Ed.), *Constraint-Handling in Evolutionary Optimization*, Studies in Computational Intelligence 198, 145–165. Springer.

[12] Wang, Y., Cai, Z., Zhou, Y., and Zeng, W. (2008). [An adaptive tradeoff model for constrained evolutionary optimization](https://doi.org/10.1109/TEVC.2007.902851). *IEEE Transactions on Evolutionary Computation*, 12(1), 80–92.

[13] Li, J.-P., Wang, Y., Yang, S., and Cai, Z. (2016). A comparative study of constraint-handling techniques in evolutionary constrained multiobjective optimization. *Proceedings of the 2016 IEEE Congress on Evolutionary Computation*, 4175–4182.

[14] Deb, K., Pratap, A., and Meyarivan, T. (2001). [Constrained test problems for multi-objective evolutionary optimization](https://doi.org/10.1007/3-540-44719-9_20). *Evolutionary Multi-Criterion Optimization – EMO 2001*, Lecture Notes in Computer Science 1993, 284–298.

[15] Dam, T. M., and Truong, L. V. (2024). A generalized framework for multi-objective-based constraint handling technique. *International Journal of Control, Automation, and Systems*, 22(10), 3240–3252.

[16] Garcia, R. P., de Lima, B. S. L. P., Lemonge, A. C. de C., and Jacob, B. P. (2017). A rank-based constraint handling technique for engineering design optimization problems solved by genetic algorithms. *Computers and Structures*, 187, 77–87.

[17] Fukumoto, H., and Oyama, A. (2018). A generic framework for incorporating constraint handling techniques into multi-objective evolutionary algorithms. *Applications of Evolutionary Computation – EvoApplications 2018*, Lecture Notes in Computer Science 10784, 634–649.

[18] Vodopija, A., Oyama, A., and Filipič, B. (2019). Ensemble-based constraint handling in multiobjective optimization. *Proceedings of the Genetic and Evolutionary Computation Conference Companion – GECCO '19*, 2072–2075.
