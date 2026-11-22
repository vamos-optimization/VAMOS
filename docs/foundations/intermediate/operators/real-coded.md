---
description: 'How NSGA-II creates offspring for real variables: simulated binary crossover (SBX) and polynomial mutation, their distribution indices and bounded forms, interactive offspring distributions and how to configure them in VAMOS.'
search:
  boost: 0.5
---

# Real-coded operators: SBX and polynomial mutation

Selection decides **who** reproduces; variation decides **what** the children
look like. For real-valued decision variables, NSGA-II uses simulated binary
crossover (SBX) and polynomial mutation, both designed by Deb and co-authors.
Their behaviour is governed by one number each, the *distribution index* η.

In this lesson you will learn:

- how SBX generates two children from two parents, and what η_c controls;
- how polynomial mutation perturbs a variable, and what η_m controls;
- how both operators respect the variable bounds;
- how to configure them in VAMOS, and which alternatives it offers.

## Exploration and exploitation

Offspring close to their parents *exploit* what the parents already found;
offspring far from them *explore* new regions. Early in a run exploration
helps to locate the front; later exploitation refines it. SBX and polynomial
mutation make this trade-off with a probability distribution centred on the
parents, whose width is controlled by η and, for SBX, by the distance between
the parents themselves.

## Simulated binary crossover (SBX)

Before real-coded algorithms became common, genetic algorithms encoded real
variables as bit strings and used single-point crossover. Deb and Agrawal [1]
derived the distribution of children that single-point crossover produces on
such strings and reproduced it directly on real numbers, hence *simulated
binary* crossover.

For one variable with parent values p₁ ≤ p₂, draw u uniformly in [0, 1] and
compute a *spread factor* β_q:

<div class="vf-formula">β_q = (2u)^(1/(η_c + 1))                 if u ≤ 0.5
β_q = (1 / (2(1 − u)))^(1/(η_c + 1))     otherwise

c₁ = ½ [ (p₁ + p₂) − β_q (p₂ − p₁) ]
c₂ = ½ [ (p₁ + p₂) + β_q (p₂ − p₁) ]</div>

Its key properties:

- **The children are symmetric around the parents' mean**, so crossover does
  not bias the population in any direction.
- **The spread is proportional to the distance between the parents.** When
  the population is diverse, children are diverse; when it converges, children
  stay close. This makes SBX *self-adaptive*.
- **η_c sets the shape.** A large η_c (20 is the usual default) gives children
  near their parents; a small η_c (2 to 5) spreads them widely.

The formula above ignores the bounds of the variable. NSGA-II's reference
implementation, and VAMOS, use a **bounded** version that rescales the
distribution so that no child falls outside [x^L, x^U]. For each child, the
probability mass that would fall beyond the nearer bound is removed: β_q is
computed from αu instead of 2u, with the threshold 1/α instead of 0.5, where
α = 2 − β^−(η_c+1), β = 1 + 2 (p₁ − x^L)/(p₂ − p₁) for c₁ and
β = 1 + 2 (x^U − p₂)/(p₂ − p₁) for c₂.

In NSGA-II each pair of parents is crossed with probability 0.9 (VAMOS'
default). Within a crossed pair, each variable is recombined with probability
0.5, and the two children swap the variable with probability 0.5.

Move the parents and change η_c. The histogram shows where 40,000 children of
those two parents land, for one variable bounded by [0, 1]:

<div class="vf-widget" data-vf-widget="variation" data-mode="sbx"><p class="vf-fallback">Interactive figure: the distribution of SBX offspring for two parents and a distribution index. It needs JavaScript.</p></div>

Try placing both parents close to a bound (for example 0.02 and 0.08) with a
small η_c: the bounded version squeezes the distribution instead of producing
values outside [0, 1].

## Polynomial mutation

Mutation perturbs single variables of a child. Polynomial mutation [2] draws
the perturbation from a polynomial distribution centred on the current value,
again controlled by a distribution index η_m. The bounded form used by
NSGA-II's reference code, and VAMOS, scales the perturbation by the distance
to each bound [3]. With y the current value in [y^L, y^U],
δ₁ = (y − y^L)/(y^U − y^L), δ₂ = (y^U − y)/(y^U − y^L) and u uniform in [0, 1]:

<div class="vf-formula">δ_q = [ 2u + (1 − 2u)(1 − δ₁)^(η_m+1) ]^(1/(η_m+1)) − 1               if u ≤ 0.5
δ_q = 1 − [ 2(1 − u) + 2(u − 0.5)(1 − δ₂)^(η_m+1) ]^(1/(η_m+1))       otherwise

y' = y + δ_q (y^U − y^L)</div>

Each variable is mutated with probability p_m. The usual choice,
p_m = 1/n for n variables, mutates one variable per child on average; VAMOS
writes it as `prob="1/n"`.

<div class="vf-widget" data-vf-widget="variation" data-mode="pm"><p class="vf-fallback">Interactive figure: the distribution of polynomially mutated values for a parent value and a distribution index. It needs JavaScript.</p></div>

## In VAMOS

The builder selects the operators by name and sets their parameters; these are
the values of `NSGAIIConfig.default(...)` for real variables:

```python
from vamos.algorithms import NSGAIIConfig, available_crossover_methods, available_mutation_methods

config = (
    NSGAIIConfig.builder()
    .pop_size(100)
    .crossover("sbx", prob=0.9, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .selection("tournament", size=2)
    .build()
)
print(config.crossover)
print(config.mutation)
print("crossovers:", ", ".join(available_crossover_methods()))
print("mutations:", ", ".join(available_mutation_methods()))
```

Output:

```text
('sbx', {'prob': 0.9, 'eta': 20.0})
('pm', {'prob': '1/n', 'eta': 20.0})
crossovers: arithmetic, blx_alpha, blx_alpha_beta, de, fuzzy, laplace, pcx, sbx, simplex, undx, whole_arithmetic
mutations: cauchy, gaussian, levy_flight, linked_polynomial, non_uniform, pm, polynomial, power_law, uniform, uniform_reset
```

With `engine="numba"`, both operators run as compiled code and give the same
children as with `engine="numpy"` for the same seed.

## Other encodings

SBX and polynomial mutation are for real variables. The other real-coded
operators of VAMOS are the subject of [More real-coded operators](real-coded-more.md),
and [Binary](binary.md), [integer](integer.md), [permutation](permutation.md)
and [mixed](mixed.md) problems have their own lessons.

## Key points

- SBX produces two children symmetric around their parents; their spread grows
  with the distance between the parents and shrinks as η_c grows.
- Polynomial mutation perturbs a variable with a spread controlled by η_m;
  with p_m = 1/n it changes one variable per child on average.
- The bounded forms of both operators keep every value inside its bounds.
- VAMOS' defaults are SBX (0.9, η_c = 20) and polynomial mutation (1/n, η_m = 20).

**Next:** [More real-coded operators](real-coded-more.md).

## References

[1] Deb, K., and Agrawal, R. B. (1995). Simulated binary crossover for continuous search space. *Complex Systems*, 9(2), 115–148.

[2] Deb, K., and Goyal, M. (1996). A combined genetic adaptive search (GeneAS) for engineering design. *Computer Science and Informatics*, 26(4), 30–45.

[3] Deb, K., and Deb, D. (2014). [Analysing mutation schemes for real-parameter genetic algorithms](https://doi.org/10.1504/IJAISC.2014.059280). *International Journal of Artificial Intelligence and Soft Computing*, 4(1), 1–28.
