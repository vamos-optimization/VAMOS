---
description: 'The other real-coded operators of VAMOS: BLX-α and BLX-αβ, arithmetic, Laplace, fuzzy, differential evolution, PCX, UNDX and SPX crossovers, and Gaussian, uniform, non-uniform, Cauchy, Lévy flight, power and linked polynomial mutations, with interactive offspring clouds.'
search:
  boost: 0.5
---

# More real-coded operators

SBX and polynomial mutation, the subject of the [previous lesson](real-coded.md),
are VAMOS's default real-coded operators, but not its only ones. VAMOS
implements ten other crossovers and eight other mutations for real variables.
They differ in the **shape** of the region where children land: an interval
around the parents, a segment between them, a cloud around each of them, a
step along a difference vector, an ellipse or a simplex. That shape decides
which directions the search can take, and whether children can leave the
bounds.

In this lesson you will learn:

- how the blend, arithmetic, Laplace and fuzzy crossovers place two children
  around two parents;
- how differential evolution, PCX, UNDX and SPX use three parents;
- how the Gaussian, uniform, non-uniform, Cauchy, Lévy flight, power and linked
  polynomial mutations perturb a solution, and which of them stay inside the
  bounds;
- how to select each of them in a VAMOS configuration.

## Children of fixed parents

Each figure below draws 600 children (or mutated copies) of the same parents in
two variables with bounds [0, 1]. Pick an operator, change its parameters and
drag the parents. Children outside the shaded square are shown in orange; in a
run, the repair of lesson [Variation operators](overview.md#bounds-and-repair)
moves them back.

<div class="vf-widget" data-vf-widget="operator-cloud" data-group="crossover" data-operator="blx_alpha"><p class="vf-fallback">Interactive figure: the children of two or three draggable parents in two variables for a chosen real-coded crossover (SBX, BLX-α, BLX-αβ, arithmetic, whole arithmetic, Laplace, fuzzy, differential evolution, PCX, UNDX or SPX) and its parameters, with the share of children outside the bounds. It needs JavaScript.</p></div>

Things to try:

1. With BLX-α, set α to 0: the children fill the rectangle spanned by the
   parents. With α = 0.5 the rectangle grows by half its size on each side, and
   with the parents near a corner some children leave the bounds.
2. Switch to SBX and compare: SBX recombines each variable with probability
   0.5, so many children keep one coordinate of a parent and line up on the
   parents' axes.
3. Arithmetic crossover puts every child on the segment between the parents;
   whole arithmetic, with its fixed weight, gives the same two children every
   time.
4. Choose differential evolution and move b: the only randomness is which
   variables come from the mutant, so with fixed parents there are just three
   possible children.
5. Choose UNDX and drag P3 away from the line P1–P2: the ellipse widens across
   that line.

## Blend crossovers: BLX-α and BLX-αβ

Eshelman and Schaffer's blend crossover [1] draws each variable of each child
uniformly from the parents' interval, widened by α times its length d on both
sides:

<div class="vf-formula">d = |p₁ − p₂|
cᵢ ~ U( min(p₁, p₂) − α·d ,  max(p₁, p₂) + α·d )      for each variable, each child independently</div>

With α = 0 the children stay between the parents, and the population's spread
can only shrink. With α = 0.5, the most common value and the default of
`blx_alpha`, the widened interval is twice as long as the parents' one, so a
child is as likely to land beyond the parents as between them.

`blx_alpha_beta` widens the interval by α below the smaller value and by β
above the larger one. The published BLX-αβ [2] extends towards the better
parent and needs the parents' fitness; VAMOS's variant, like jMetal's, is
positional, so α ≠ β pushes the population towards one bound. Both default to
0.5, which makes it BLX-0.5. The core of `blx_alpha`:

```python
# Excerpt from src/vamos/engine/operators/impl/real/crossover_blend.py
        lo = np.minimum(p1, p2)
        hi = np.maximum(p1, p2)
        span = hi - lo
        lower = lo - self.alpha * span
        upper = hi + self.alpha * span
        width = upper - lower
        active[:, 0, :] = lower + self._rand("blx_rand1", lower.shape, rng) * width
        active[:, 1, :] = lower + self._rand("blx_rand2", lower.shape, rng) * width
```

## Arithmetic crossovers

Arithmetic crossover [3] mixes the parents with one weight λ per mating, so
both children lie on the segment between the parents, symmetric about its
midpoint, and never leave the bounds:

<div class="vf-formula">c₁ = λ p₁ + (1 − λ) p₂
c₂ = (1 − λ) p₁ + λ p₂          λ ~ U[0, 1) for `arithmetic`;  λ = α (default 0.5) for `whole_arithmetic`</div>

With a fixed α, `whole_arithmetic` is deterministic: α = 0.5 gives the
midpoint twice, and repeated matings pull the population together.

## Laplace and fuzzy crossovers

Deep and Thakur's Laplace crossover [4] moves each parent by a multiple of the
parents' distance, drawn from a Laplace distribution of location a and scale b:

<div class="vf-formula">β = a + b · sign(u − ½) · ln(1 − 2|u − ½|)      u ~ U(0, 1)
cᵢ = pᵢ + β · |p₁ − p₂|</div>

The distribution has heavy tails, so most children stay near their parent and
a few go far. The paper uses one β for both children, which places them
symmetrically; VAMOS draws a β for every variable of every child. Defaults:
a = 0, b = 0.5.

Fuzzy recombination [5] draws each variable from a triangular distribution.
VAMOS's variant uses one triangle over the parents' interval widened by d times
its length on each side (d = 0.5 by default), clipped to the bounds, with its
mode at the parents' midpoint; the paper uses two triangles, one centred on
each parent.

All six scale their steps by the parents' distance. In the first figure, drag
P1 and P2 to the same height: whichever of these operators you choose, every
child keeps that value of x₂. Then move both parents near the right edge: BLX
and Laplace children leave the bounds, the arithmetic and fuzzy children do
not.

## Differential evolution

Storn and Price's differential evolution (DE) [6] creates a **mutant** by
adding a scaled difference of two solutions to a third one, then takes each
variable of the child from the mutant with probability CR, and at least one:

<div class="vf-formula">v = x + F · (a − b)
cⱼ = vⱼ   if rⱼ &lt; CR or j = j_rand,      cⱼ = xⱼ   otherwise</div>

The difference a − b adapts the step to the population: large while it is
spread, small when it converges. In VAMOS, `de` takes groups of three parents
[x, a, b] and returns **one child** per group, with F = 0.5 and CR = 0.9 by
default; it ignores `prob` and recombines every group. MOEA/D's default real
variation is also differential evolution, built by its own policy.

```python
# Excerpt from src/vamos/engine/operators/impl/real/crossover_differential.py
        target = groups[:, 0, :]
        mutant = target + self.F * (groups[:, 1, :] - groups[:, 2, :])
        trial = target.copy()
        cross_mask = rng.random((n_groups, n_vars)) < self.CR
        cross_mask[np.arange(n_groups), rng.integers(n_vars, size=n_groups)] = True
        trial[cross_mask] = mutant[cross_mask]
```

## Multi-parent crossovers: PCX, UNDX and SPX

Three operators sample around a group of three parents, with distributions
that rotate with the parents instead of following the coordinate axes.

**PCX**, Deb, Anand and Joshi's parent-centric crossover [7], centres one child
on each parent. For parent xⱼ, with g the centroid of the group and d = g − xⱼ,
the child moves along d with a normal weight and, orthogonally to d, with
normal noise scaled by D, the mean distance of the other two parents to the
line through xⱼ along d:

<div class="vf-formula">c = xⱼ + w·d + noise,      w ~ N(0, σ_ζ²),  noise ~ N(0, (σ_η·D)²) orthogonal to d</div>

Defaults: σ_η = σ_ζ = 0.1. Three parents give three children.

**UNDX**, Ono and Kobayashi's unimodal normal distribution crossover [8],
places two children symmetrically about the midpoint m of the first two
parents. The spread along d = p₂ − p₁ is set by ζ, and the spread across it by
η and the distance D of the third parent to that line [9]:

<div class="vf-formula">c = m ± ( ξ·d + D · Σᵢ ηᵢ eᵢ ),      ξ ~ N(0, ζ²),  ηᵢ ~ N(0, η²/n)</div>

with eᵢ an orthonormal basis orthogonal to d. Defaults: ζ = 0.5, η = 0.35, the
values Kita, Ono and Kobayashi recommend [9]. UNDX applies `prob` per group.

**SPX**, Tsutsui, Yamamura and Higuchi's simplex crossover [10], expands the
triangle of the three parents about their centroid O by ε and draws each child
uniformly inside it:

<div class="vf-formula">c = O + ε · ( Σₖ wₖ xₖ − O ),      w ~ Dirichlet(1, 1, 1),  ε = √(m + 1) = 2 by default</div>

PCX and SPX ignore `prob` and recombine every group. The loop of PCX and the
whole of SPX:

```python
# Excerpt from src/vamos/engine/operators/impl/real/crossover_multi_parent.py
        centroid = groups.mean(axis=1)
        for j in range(n_parents):
            index_parent = groups[:, j, :]
            d = centroid - index_parent
            others = np.delete(groups, j, axis=1) - index_parent[:, None, :]
            perpendicular = _remove_component(others, d[:, None, :])
            mean_distance = np.linalg.norm(perpendicular, axis=-1).mean(axis=1)
            noise = rng.normal(0.0, 1.0, size=(n_groups, n_vars)) * (self.sigma_eta * mean_distance)[:, None]
            noise = _remove_component(noise, d)
            w = rng.normal(0.0, self.sigma_zeta, size=(n_groups, 1))
            offspring[:, j, :] = index_parent + w * d + noise
        epsilon = np.sqrt(k + 1.0) if self.epsilon is None else self.epsilon
        centroid = groups.mean(axis=1, keepdims=True)
        weights = rng.dirichlet(np.ones(k), size=(n_groups, k))
        points = np.einsum("gck,gkv->gcv", weights, groups)
        return np.asarray(centroid + epsilon * (points - centroid), dtype=np.float64)
```

In the first figure, DE draws one child per group from x = P1, a = P2 and
b = P3; PCX gives three children, one around each parent; UNDX gives two,
symmetric about the midpoint of P1 and P2; and SPX gives three, inside the
expanded triangle.

## Mutations

The second figure applies a mutation to one parent. Each variable is mutated
with the probability of the last slider, so at 0.5 (1/n for two variables)
a quarter of the copies change both variables, half change one and lie on the
lines through the parent, and a quarter stay at the parent.

<div class="vf-widget" data-vf-widget="operator-cloud" data-group="mutation" data-operator="gaussian"><p class="vf-fallback">Interactive figure: mutated copies of a draggable parent in two variables for a chosen real-coded mutation (polynomial, Gaussian, uniform, uniform reset, non-uniform, Cauchy, Lévy flight, power or linked polynomial), its parameters and the probability per variable. It needs JavaScript.</p></div>

**Gaussian and Cauchy.** Gaussian mutation, the classic operator of evolution
strategies [11], adds normal noise; Cauchy mutation, proposed for fast
evolutionary programming [12], adds noise with much heavier tails, so most of
its steps are small and some are very long:

<div class="vf-formula">gaussian:  x′ = x + σ (u − l) · N(0, 1)          σ = 0.1 by default
cauchy:    x′ = x + γ (u − l) · C,  C standard Cauchy    γ = 0.1 by default</div>

**Uniform and uniform reset.** `uniform` moves a variable by
U(−perturb, perturb) times its range (perturb = 0.1); `uniform_reset` replaces
it with a uniform value between its bounds, forgetting the parent.

**Non-uniform.** Michalewicz's non-uniform mutation [3] moves a variable
towards one of its bounds, chosen with probability ½, by a share of the
distance y to that bound that shrinks as the run progresses:

<div class="vf-formula">x′ = x ± y · (1 − r^((1 − t/T)^b))</div>

The operator in VAMOS does not receive the run's progress t/T and uses 0, as
the [algorithm reference](../../reference/algorithms.md) notes: the step is y (1 − r), uniform between the parent and the
bound, and the parameter `perturbation` (b) has no effect. The value never
leaves the bounds.

```python
# Excerpt from src/vamos/engine/operators/impl/real/mutation.py
        rand = self._rand("nu_rand", X.shape, rng)
        # (1 - t / T) ** b with the progress t / T unknown, and therefore 0.
        np.power(rand, (1.0 - self.progress) ** self.perturbation, out=rand)
        np.subtract(1.0, rand, out=delta)
        direction_rand = self._rand("nu_direction", X.shape, rng)
        distance = np.where(direction_rand <= 0.5, self.upper - X, self.lower - X)
        update = delta * distance
```

**Lévy flight.** `levy_flight` draws heavy-tailed steps with Mantegna's
algorithm [13], the steps of Lévy-flight metaheuristics [14]: a normal
numerator over a power of a normal denominator,

<div class="vf-formula">x′ = x + scale · (u − l) · σ_u N(0, 1) / |N(0, 1)|^(1/β)
σ_u = [ Γ(1 + β) sin(πβ/2) / ( Γ((1 + β)/2) β 2^((β − 1)/2) ) ]^(1/β)</div>

with β = 1.5 and scale = 0.01 by default. A smaller β gives heavier tails.

**Power mutation.** Deep and Thakur's power mutation [15] moves a variable by a
share s of its distance to one bound, towards the lower bound when its
relative position t is below a uniform number r:

<div class="vf-formula">s = u^index,   t = (x − l)/(u − l)
x′ = x − s (x − l)   if t &lt; r,      x′ = x + s (u − x)   otherwise</div>

It never leaves the bounds; `index` = 1 by default, and a larger index gives
smaller steps. The paper writes t = (x − l)/(u − x), which drifts towards the
upper bound; VAMOS and jMetal use the relative position.

```python
# Excerpt from src/vamos/engine/operators/impl/real/mutation.py
        strength = np.power(rng.random((n_ind, n_var)), self.index)
        position = (X - self.lower) / self._span_safe
        downward = position < rng.random((n_ind, n_var))
        mutated = np.where(downward, X - strength * (X - self.lower), X + strength * (self.upper - X))
        np.clip(mutated, self.lower, self.upper, out=mutated)
```

**Linked polynomial.** Zille, Ishibuchi, Mostaghim and Nojima's linked
polynomial mutation [16], designed for problems with many variables, is the
bounded polynomial mutation of the previous lesson with **one** random number
per individual for all its mutated variables. The step of each variable still
depends on its distance to the bounds, but they all move in the same
direction. In the figure, with probability 1, the copies form a curve instead
of a cloud.

```python
# Excerpt from src/vamos/engine/operators/impl/real/mutation.py
        # One draw per individual, shared by all of its mutated variables.
        u = active_rng.random((n_ind, 1))
            rnd_delta=np.broadcast_to(u, (n_ind, n_var)),
```

In the second figure, choose Cauchy and drag the parent towards a corner:
some copies land far outside the bounds, and the run's repair would move them
back. Power and non-uniform mutation never leave the square. With linked
polynomial mutation and probability 1, every copy moves both variables in the
same direction.

## Using them in a run

Every operator of this lesson is selected by name in a builder, with the
parameters of its class (`prob` replaces `prob_crossover` and
`prob_mutation`). The three-parent crossovers need an algorithm that groups
parents by the crossover's arity, such as NSGA-II. Five crossovers on ZDT1,
with polynomial mutation:

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import ZDT1

problem = ZDT1(n_var=30)
crossovers = {
    "sbx": {"prob": 0.9, "eta": 20.0},
    "blx_alpha": {"prob": 0.9, "alpha": 0.5},
    "de": {"F": 0.5, "CR": 0.9},
    "pcx": {"sigma_eta": 0.1, "sigma_zeta": 0.1},
    "undx": {"prob": 0.9},
}
for name, params in crossovers.items():
    config = (
        NSGAIIConfig.builder()
        .pop_size(100)
        .crossover(name, **params)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    )
    result = optimize(problem, algorithm="nsgaii", algorithm_config=config, max_evaluations=10_000, seed=1)
    gap = np.abs(result.F[:, 1] - (1.0 - np.sqrt(result.F[:, 0])))
    print(f"{name:<10} mean distance to the front {gap.mean():.3f}")
```

Output:

```text
sbx        mean distance to the front 0.025
blx_alpha  mean distance to the front 0.011
de         mean distance to the front 0.275
pcx        mean distance to the front 0.193
undx       mean distance to the front 0.179
```

This is one seed on one problem with a small budget, so it ranks nothing; it
only shows that the choice changes the result under the same budget. Note that
`de` returns one child per group of three parents, where the two-parent
crossovers return two children per pair.

The nine mutations, with SBX and the parameters of this lesson:

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import NSGAIIConfig, available_mutation_methods
from vamos.problems import ZDT1

print("mutations:", ", ".join(available_mutation_methods("real")))
mutations = {
    "pm": {"eta": 20.0},
    "gaussian": {"sigma": 0.1},
    "uniform": {"perturb": 0.1},
    "uniform_reset": {},
    "non_uniform": {},
    "cauchy": {"gamma": 0.1},
    "levy_flight": {"beta": 1.5, "scale": 0.01},
    "power_law": {"index": 1.0},
    "linked_polynomial": {"eta": 20.0},
}
problem = ZDT1(n_var=30)
for name, params in mutations.items():
    config = (
        NSGAIIConfig.builder()
        .pop_size(100)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation(name, prob="1/n", **params)
        .selection("tournament", size=2)
        .build()
    )
    result = optimize(problem, algorithm="nsgaii", algorithm_config=config, max_evaluations=10_000, seed=1)
    gap = np.abs(result.F[:, 1] - (1.0 - np.sqrt(result.F[:, 0])))
    print(f"{name:<18} mean distance to the front {gap.mean():.3f}")
```

Output:

```text
mutations: cauchy, gaussian, levy_flight, linked_polynomial, non_uniform, pm, polynomial, power_law, uniform, uniform_reset
pm                 mean distance to the front 0.025
gaussian           mean distance to the front 0.004
uniform            mean distance to the front 0.007
uniform_reset      mean distance to the front 0.148
non_uniform        mean distance to the front 0.021
cauchy             mean distance to the front 0.003
levy_flight        mean distance to the front 0.034
power_law          mean distance to the front 0.005
linked_polynomial  mean distance to the front 0.016
```

The same caution applies. ZDT1's optimum lies on the lower bound of x₂ … x₃₀,
and the default repair clips every value that a Gaussian or Cauchy step pushes
below it exactly onto that bound (lesson
[Variation operators](overview.md#bounds-and-repair)), which favours the
mutations that leave the bounds on this problem. Uniform reset forgets the
parent's value, so it cannot refine a solution.

## Key points

- BLX-α samples an interval widened by α around the parents and can leave the
  bounds; arithmetic crossover samples the segment between them; Laplace
  crossover adds heavy-tailed multiples of the parents' distance; VAMOS's fuzzy
  crossover samples a triangle with its mode at the midpoint.
- Differential evolution adds F (a − b) to x and returns one child; PCX, UNDX
  and SPX sample around three parents with distributions that rotate with
  them.
- Gaussian, Cauchy, uniform and Lévy mutations take steps proportional to the
  range and can leave the bounds; uniform reset, non-uniform and power
  mutation cannot.
- VAMOS's non-uniform mutation uses the progress t/T = 0, and its linked
  polynomial mutation shares one random number across all the variables of an
  individual.
- Every operator is selected by name in a builder; three-parent crossovers need
  an algorithm that groups parents by arity.

## References

[1] Eshelman, L. J., and Schaffer, J. D. (1993). Real-coded genetic algorithms and interval-schemata. In *Foundations of Genetic Algorithms 2*, 187–202. Morgan Kaufmann.

[2] Eshelman, L. J., Mathias, K. E., and Schaffer, J. D. (1997). Crossover operator biases: Exploiting the population distribution. *Proceedings of the Seventh International Conference on Genetic Algorithms*.

[3] Michalewicz, Z. (1996). *Genetic Algorithms + Data Structures = Evolution Programs* (3rd ed.). Springer.

[4] Deep, K., and Thakur, M. (2007). A new crossover operator for real coded genetic algorithms. *Applied Mathematics and Computation*, 188(1), 895–911.

[5] Voigt, H.-M., Mühlenbein, H., and Cvetković, D. (1995). Fuzzy recombination for the breeder genetic algorithm. *Proceedings of the Sixth International Conference on Genetic Algorithms*.

[6] Storn, R., and Price, K. (1997). [Differential evolution – A simple and efficient heuristic for global optimization over continuous spaces](https://doi.org/10.1023/A:1008202821328). *Journal of Global Optimization*, 11(4), 341–359.

[7] Deb, K., Anand, A., and Joshi, D. (2002). [A computationally efficient evolutionary algorithm for real-parameter optimization](https://doi.org/10.1162/106365602760972767). *Evolutionary Computation*, 10(4), 371–395.

[8] Ono, I., and Kobayashi, S. (1997). A real-coded genetic algorithm for function optimization using unimodal normal distribution crossover. *Proceedings of the Seventh International Conference on Genetic Algorithms*.

[9] Kita, H., Ono, I., and Kobayashi, S. (1999). Multi-parental extension of the unimodal normal distribution crossover for real-coded genetic algorithms. *Proceedings of the 1999 Congress on Evolutionary Computation*.

[10] Tsutsui, S., Yamamura, M., and Higuchi, T. (1999). Multi-parent recombination with simplex crossover in real coded genetic algorithms. *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO 1999)*.

[11] Bäck, T., and Schwefel, H.-P. (1993). [An overview of evolutionary algorithms for parameter optimization](https://doi.org/10.1162/evco.1993.1.1.1). *Evolutionary Computation*, 1(1), 1–23.

[12] Yao, X., Liu, Y., and Lin, G. (1999). [Evolutionary programming made faster](https://doi.org/10.1109/4235.771163). *IEEE Transactions on Evolutionary Computation*, 3(2), 82–102.

[13] Mantegna, R. N. (1994). Fast, accurate algorithm for numerical simulation of Lévy stable stochastic processes. *Physical Review E*, 49(5), 4677–4683.

[14] Yang, X.-S. (2010). *Nature-Inspired Metaheuristic Algorithms* (2nd ed.). Luniver Press.

[15] Deep, K., and Thakur, M. (2007). A new mutation operator for real coded genetic algorithms. *Applied Mathematics and Computation*, 193(1), 211–230.

[16] Zille, H., Ishibuchi, H., Mostaghim, S., and Nojima, Y. (2016). Mutation operators based on variable grouping for multi-objective large-scale optimization. *IEEE Symposium Series on Computational Intelligence (SSCI 2016)*.

**Next:** [Binary operators](binary.md).
