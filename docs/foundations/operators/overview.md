---
description: 'What crossover and mutation do, exploration and exploitation, crossover and per-variable mutation probabilities, step-size parameters, bounds and repair, and how the encoding of a problem decides which operators VAMOS uses.'
search:
  boost: 0.5
---

# Variation operators: an overview

Every evolutionary algorithm of the previous lessons repeats the same cycle:
select parents, create offspring from them, and decide who survives. The
lessons on NSGA-II, SPEA2 or SMS-EMOA were about the first and last steps.
This section is about the middle one, **variation**: the operators that turn
parents into new candidate solutions. They are the only part of the algorithm
that creates new points in the decision space, so they decide which solutions
the search can reach at all.

In this lesson you will learn:

- what crossover and mutation do, and how they share exploration and
  exploitation;
- how a crossover probability differs from a per-variable mutation
  probability, and why 1/n is the usual choice;
- what the distribution and step-size parameters of the operators control;
- how VAMOS keeps real-valued children inside their bounds;
- how the encoding of a problem determines its operators, how VAMOS picks the
  default ones, and how to choose others.

## Where variation happens

Mating selection hands the variation step a list of parents. VAMOS groups them
by the number of parents the crossover needs (two for most operators, three for
some real-coded ones), applies the crossover to each group, mutates every
child, and, for real variables only, repairs values that left their bounds:

<div class="vf-figure"><svg class="vf-diagram" viewBox="0 0 640 150" role="img" aria-label="The variation step: selected parents go through crossover, applied to each mating group with the crossover probability, then mutation, applied to each variable with the mutation probability, then repair of out-of-bounds real values, and the offspring are evaluated."><rect class="vf-box" x="10" y="30" width="110" height="70" rx="10"/><text x="65" y="60" text-anchor="middle">Parents</text><text class="vf-diagram-small" x="65" y="80" text-anchor="middle">from selection</text><rect class="vf-box" x="140" y="30" width="110" height="70" rx="10"/><text x="195" y="60" text-anchor="middle">Crossover</text><text class="vf-diagram-small" x="195" y="80" text-anchor="middle">per group, p_c</text><rect class="vf-box" x="270" y="30" width="110" height="70" rx="10"/><text x="325" y="60" text-anchor="middle">Mutation</text><text class="vf-diagram-small" x="325" y="80" text-anchor="middle">per variable, p_m</text><rect class="vf-box" x="400" y="30" width="110" height="70" rx="10"/><text x="455" y="60" text-anchor="middle">Repair</text><text class="vf-diagram-small" x="455" y="80" text-anchor="middle">real variables</text><rect class="vf-box" x="530" y="30" width="100" height="70" rx="10"/><text x="580" y="60" text-anchor="middle">Offspring</text><text class="vf-diagram-small" x="580" y="80" text-anchor="middle">evaluated</text><path class="vf-arrow" d="M120 65H134M250 65H264M380 65H394M510 65H524"/><path class="vf-arrowhead" d="M140 65l-7-4v8zM270 65l-7-4v8zM400 65l-7-4v8zM530 65l-7-4v8z"/></svg></div>

NSGA-II, IBEA, AGE-MOEA, RVEA and the constrained algorithms run this step
through the `VariationPipeline` of `src/vamos/engine/variation/`; the others
assemble the same operators in the same order. The pipeline's two methods are
short, and the repair runs after the crossover and again after the mutation:

```python
# Excerpt from src/vamos/engine/variation/pipeline.py
    def recombine(self, parents: NDArray[np.generic], rng: np.random.Generator) -> NDArray[np.generic]:
        offspring = self.crossover_op(parents, rng)
        if self.encoding == "real":
            assert self.repair_op is not None
            offspring = self.repair_op(offspring, self.xl, self.xu, rng)
        return offspring

    def mutate(self, offspring: NDArray[np.generic], rng: np.random.Generator) -> NDArray[np.generic]:
        """Mutation of every row, then the repair of real encoding."""
        offspring = self.mutation_op(offspring, rng)
        if self.repair_op is not None:
            offspring = self.repair_op(offspring, self.xl, self.xu, rng)
        return offspring
```

## Crossover and mutation

**Crossover**, or recombination, builds children from two or more parents.
Operators for discrete variables copy values: a child bit, integer or city
position comes from one of the parents. Operators for real variables sample a
region that the parents define, such as the interval between them or a cloud
around each of them. Either way, crossover combines what the parents already
contain. Uniform crossover of two bit strings cannot set a bit that is 0 in
both parents, and the real-coded operators spread their children in proportion
to the distance between the parents, so their steps shrink when the population
converges.

**Mutation** changes a single solution. It can create values that no parent
has, and it is what keeps a converged population able to move: a flipped bit,
a reset integer, a perturbed real value or two exchanged cities.

## Exploration and exploitation

Children far from their parents **explore** regions the population has not
visited; children close to them **exploit** what the parents already found,
refining it. A search needs both, and the balance shifts during a run [1, 2].
The operators control it in two ways:

- **through the parents**: a crossover of distant parents explores, of similar
  parents exploits. This happens by itself as the population converges;
- **through their parameters**: a distribution index, a widening factor or a
  step size sets how far a child may land from its parents.

## Two kinds of probability

The two operators use their probabilities differently.

- **Crossover probability p_c, per mating.** A group of parents is recombined
  with probability p_c; otherwise its children are copies of the parents, and
  only mutation changes them. VAMOS uses 0.9 by default, as Deb et al. did for
  NSGA-II [3].
- **Mutation probability p_m, per variable.** Each variable of a child is
  mutated independently with probability p_m. With n variables a child has
  n·p_m mutated variables on average, so p_m = 1/n mutates one variable per
  child, a rate that theory and experience support for many problems [4]. VAMOS
  accepts the expression `"1/n"` (or `"k/n"`), and resolves it with the
  problem's number of variables.

<div class="vf-formula">expected mutated variables per child  =  n · p_m        (1 for p_m = 1/n)
P(a child has no mutation)            =  (1 − p_m)ⁿ     (0.36 for n = 30 and p_m = 1/30; it tends to 1/e ≈ 0.37)</div>

Some operators add a probability of their own inside a mating: SBX recombines
each variable of a crossed pair with probability 0.5, and differential
evolution takes each variable from its mutant with probability CR.

Permutation mutations are different. A swap, an inversion or a 2-opt move
changes the whole tour at once, so for them, and for the binary segment
inversion, `prob` is the probability that a **child** is mutated, not each
variable. A configuration that leaves `prob` out gets the algorithm's
fallback, and the fallbacks are not all equal: NSGA-II's default permutation
setup, `crossover("ox")` and `mutation("swap")`, crosses every pair and mutates
a child with probability 2/n:

```python
# Excerpt from src/vamos/engine/variation/helpers.py
    elif "prob" in params:
        params["prob"] = resolve_prob_expression(params["prob"], n_var, 1.0)
    else:
        if normalized == "permutation":
            params["prob"] = min(1.0, 2.0 / max(1, n_var))
        else:
            params["prob"] = 1.0 / max(1, n_var)
```

State `prob` explicitly when a comparison depends on it.

## Distribution and step-size parameters

Besides its probability, almost every operator has a parameter that sets how
far its children land. The names differ, the role is the same:

| Parameter | Operators (VAMOS names) | Default | Larger values give |
| --- | --- | --- | --- |
| distribution index η | `sbx` (η_c), `pm`, `linked_polynomial`, integer `sbx` and `pm` (η_m) | 20 in the default configurations (10 for an `sbx` that omits it) | children closer to the parents |
| widening α, β | `blx_alpha`, `blx_alpha_beta` | 0.5 | a wider interval around the parents |
| share of the range σ, γ, perturb, scale | `gaussian`, `cauchy`, `uniform`, `levy_flight` | 0.1, 0.1, 0.1, 0.01 | larger steps |
| F, CR | `de` | 0.5, 0.9 | longer difference steps; more variables from the mutant |
| index | `power_law` | 1.0 | smaller steps |
| step, σ (in units) | integer `creep`, integer `gaussian` | 1, 1.0 | larger integer steps |

The real-valued step sizes σ, γ, perturb and scale are **shares of each
variable's range** in VAMOS, so the same value means the same relative step on
a variable in [0, 1] and on one in [0, 1000]. The [real-coded
operators](real-coded.md) lesson shows how η shapes SBX and polynomial
mutation, and [More real-coded operators](real-coded-more.md) lets you change
the others and watch the children move.

## Bounds and repair

Every variable has a lower and an upper bound. Several real-coded operators
can produce values outside them: BLX widens the parents' interval beyond the
parents, Laplace, Gaussian, Cauchy and Lévy steps have no upper limit, and the
three-parent operators extrapolate. Others stay inside by construction:
bounded SBX and polynomial mutation, arithmetic and fuzzy crossover, uniform
reset, non-uniform and power mutation.

For real variables, VAMOS applies a **repair** after the crossover and after
the mutation. The default, `repair="auto"`, clips each value to its nearer
bound; the builder's `.repair(name)` selects another one:

| Name | What it does to a value outside [l, u] |
| --- | --- |
| `clip`, `clamp` | sets it to the nearer bound (the default) |
| `reflect` | mirrors it back inside, as many times as needed |
| `wrap`, `wrapping` | re-enters it from the opposite bound, for periodic variables |
| `random`, `resample` | draws a new uniform value between the bounds |
| `midpoint`, `midpoint_base` | sets it to a quarter or three quarters of the interval |
| `round` | rounds every value to the nearest integer, then clips |
| `gradient` | steps against the gradients of the constraint violations you supply |

The same out-of-bounds child repaired five ways:

```python
import numpy as np

from vamos.engine.operators import ClampRepair, MidpointBaseRepair, ReflectRepair, ResampleRepair, WrappingRepair

x = np.array([[-0.2, 0.5, 1.3]])  # one child after a real-coded operator; bounds [0, 1]
lower, upper = np.zeros(3), np.ones(3)
rng = np.random.default_rng(1)
for repair in (ClampRepair(), ReflectRepair(), WrappingRepair(), MidpointBaseRepair(), ResampleRepair()):
    print(f"{type(repair).__name__:<18}", repair(x, lower, upper, rng).round(3).tolist())
```

Output:

```text
ClampRepair        [[0.0, 0.5, 1.0]]
ReflectRepair      [[0.2, 0.5, 0.7]]
WrappingRepair     [[0.8, 0.5, 0.3]]
MidpointBaseRepair [[0.25, 0.5, 0.75]]
ResampleRepair     [[0.512, 0.5, 0.144]]
```

The choice matters when the optimum lies on a bound, as in many ZDT problems:
clipping piles repaired values exactly on the bound, reflection and resampling
spread them inside. Padhye, Mittal and Deb compare these strategies [5].
Discrete operators need no repair, because they only produce valid values; a
repair for a binary, integer, permutation or mixed problem raises
`ValueError: Repair operators are only supported for real encoding.`

The operator classes that the lessons of this section call directly live in
`vamos.engine.operators`. They are implementation modules, not part of the
[stable API](../../project/stability-and-versioning.md); a script that should
survive upgrades selects operators by name through the builders, as the last
section shows.

## The encoding decides the operators

A crossover that averages two real numbers makes no sense for a tour, and
swapping two cities makes no sense for a bit string. Each **encoding** has its
own idea of a valid value, and each operator is written for one encoding
[6]. These are the operators VAMOS implements, by the names the builders
accept (`available_crossover_methods(encoding)` and
`available_mutation_methods(encoding)` in `vamos.algorithms` list them, with
their aliases):

| Encoding | A valid value | Crossovers | Mutations | Lesson |
| --- | --- | --- | --- | --- |
| Real | a float in [l, u] | `sbx`, `blx_alpha`, `blx_alpha_beta`, `arithmetic`, `whole_arithmetic`, `laplace`, `fuzzy`, `de`, `pcx`, `undx`, `simplex` | `pm` (`polynomial`), `gaussian`, `uniform`, `uniform_reset`, `non_uniform`, `cauchy`, `levy_flight`, `power_law`, `linked_polynomial` | [Real-coded operators](real-coded.md), [More real-coded operators](real-coded-more.md) |
| Binary | 0 or 1 | `one_point`, `two_point`, `uniform`, `hux` | `bitflip`, `segment_inversion` | [Binary operators](binary.md) |
| Integer | an integer in [l, u] | `uniform`, `arithmetic`, `sbx` | `reset`, `creep`, `gaussian`, `boundary`, `pm` | [Integer operators](integer.md) |
| Permutation | each of 0 … n − 1 exactly once | `ox`, `pmx`, `cx`, `position_based`, `erx`, `aex` | `swap`, `insert`, `scramble`, `inversion`, `displacement`, `two_opt` | [Permutation operators](permutation.md) |
| Mixed | each column valid for its type | `mixed` | `mixed` | [Mixed operators](mixed.md) |

The same name can mean different operators in different encodings: `uniform`
crossover exchanges bits in a binary problem and integers in an integer one,
and `sbx` on an integer problem is the integer SBX, which rounds its children.

### How VAMOS picks the default operators

`optimize(problem, algorithm="nsgaii")` without a configuration calls
`NSGAIIConfig.default(...)` with the problem's number of variables and its
encoding, and the other algorithms' `default()` do the same. A real-coded
problem gets the algorithm's own real-coded operators; the other encodings get
the operators of one shared table:

```python
# Excerpt from src/vamos/engine/algorithm/config/base.py
    enc = normalize_encoding(encoding, default="real")
    if enc == "permutation":
        return ("ox", {}), ("swap", {"prob": 0.1})
    if enc == "binary":
        return ("uniform", {"prob": 0.9}), ("bitflip", {"prob": mut_prob})
    if enc == "integer":
        return ("sbx", {"prob": 0.9, "eta": 20.0}), ("polynomial", {"prob": mut_prob, "eta": 20.0})
    if enc == "mixed":
        return ("mixed", {"prob": 0.9}), ("mixed", {"prob": mut_prob})
```

`mut_prob` is 1/n when the number of variables is known and 0.1 otherwise.
NSGA-II writes these choices in its own `default()`, and leaves the permutation
probabilities to the fallbacks of the previous section:

```python
from vamos.algorithms import NSGAIIConfig

for encoding in ("real", "binary", "integer", "permutation", "mixed"):
    config = NSGAIIConfig.default(encoding=encoding, n_var=10)
    print(f"{encoding:<12} crossover={config.crossover}  mutation={config.mutation}")
```

Output:

```text
real         crossover=('sbx', {'prob': 0.9, 'eta': 20.0})  mutation=('pm', {'prob': 0.1, 'eta': 20.0})
binary       crossover=('uniform', {'prob': 0.9})  mutation=('bitflip', {'prob': 0.1})
integer      crossover=('sbx', {'prob': 0.9, 'eta': 20.0})  mutation=('pm', {'prob': 0.1, 'eta': 20.0})
permutation  crossover=('ox', {})  mutation=('swap', {})
mixed        crossover=('mixed', {'prob': 0.9})  mutation=('mixed', {'prob': 0.1})
```

## Choosing operators with the builder

Every configuration builder has `.crossover(name, **params)`,
`.mutation(name, **params)` and, for real problems, `.repair(name)`. The
parameters are those of the operator: `prob` and the parameters of the table
above. A name the problem's encoding does not support raises `ValueError` when
the algorithm builds its operators. The four three-parent crossovers `de`,
`pcx`, `undx` and `simplex` need an algorithm that groups parents by the
crossover's arity: NSGA-II, IBEA, AGE-MOEA and RVEA do.

Two NSGA-II configurations on ZDT1, the default operators and BLX-0.5 with
Gaussian mutation and reflection:

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import ZDT1

builders = {
    "SBX and polynomial": NSGAIIConfig.builder().crossover("sbx", prob=0.9, eta=20.0).mutation("pm", prob="1/n", eta=20.0),
    "BLX-0.5 and Gaussian": NSGAIIConfig.builder()
    .crossover("blx_alpha", prob=0.9, alpha=0.5)
    .mutation("gaussian", prob="1/n", sigma=0.05)
    .repair("reflect"),
}
for name, builder in builders.items():
    config = builder.pop_size(100).selection("tournament", size=2).build()
    print(f"{name}: {config.crossover}, {config.mutation}, repair {config.repair}")
    result = optimize(ZDT1(n_var=30), algorithm="nsgaii", algorithm_config=config, max_evaluations=10_000, seed=1)
    gap = np.abs(result.F[:, 1] - (1.0 - np.sqrt(result.F[:, 0])))  # ZDT1 front: f2 = 1 - sqrt(f1)
    print(f"  mean vertical distance to the front after 10,000 evaluations: {gap.mean():.3f}")
```

Output:

```text
SBX and polynomial: ('sbx', {'prob': 0.9, 'eta': 20.0}), ('pm', {'prob': '1/n', 'eta': 20.0}), repair auto
  mean vertical distance to the front after 10,000 evaluations: 0.025
BLX-0.5 and Gaussian: ('blx_alpha', {'prob': 0.9, 'alpha': 0.5}), ('gaussian', {'prob': '1/n', 'sigma': 0.05}), repair ('reflect', {})
  mean vertical distance to the front after 10,000 evaluations: 0.083
```

One run on one problem ranks nothing: lesson [Comparing algorithms
rigorously](../advanced/performance-assessment.md) explains how to compare
operator choices with several seeds and a statistical test.

## Key points

- Crossover combines parents and its steps shrink as the population
  converges; mutation changes one solution and can create values no parent
  has.
- The crossover probability applies per mating (0.9 by default); the mutation
  probability applies per variable, and 1/n mutates one variable per child on
  average. Permutation mutations apply it per child.
- Distribution indices, widening factors and step sizes set how far children
  land; VAMOS expresses the real-valued step sizes as shares of the range.
- Real-valued children are repaired after crossover and mutation; the default
  clips them to the bounds.
- The encoding decides which operators are valid. `XConfig.default(encoding=...)`
  takes the operators of a shared table for binary, integer, permutation and
  mixed problems, and the builders select any other supported operator by
  name.

## References

[1] Eiben, A. E., and Smith, J. E. (2015). *Introduction to Evolutionary Computing* (2nd ed.). Springer.

[2] Črepinšek, M., Liu, S.-H., and Mernik, M. (2013). Exploration and exploitation in evolutionary algorithms: A survey. *ACM Computing Surveys*, 45(3), 35.

[3] Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://doi.org/10.1109/4235.996017). *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

[4] Bäck, T. (1993). Optimal mutation rates in genetic search. *Proceedings of the Fifth International Conference on Genetic Algorithms*, 2–8.

[5] Padhye, N., Mittal, P., and Deb, K. (2015). Feasibility preserving constraint-handling strategies for real parameter evolutionary optimization. *Computational Optimization and Applications*, 62(3), 851–890.

[6] Michalewicz, Z. (1996). *Genetic Algorithms + Data Structures = Evolution Programs* (3rd ed.). Springer.

**Next:** [Real-coded operators: SBX and polynomial mutation](real-coded.md).
