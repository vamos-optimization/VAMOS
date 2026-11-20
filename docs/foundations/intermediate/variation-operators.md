---
description: 'I4. How NSGA-II creates offspring for real variables: simulated binary crossover (SBX) and polynomial mutation, their distribution indices and bounded forms, interactive offspring distributions and the VAMOS implementation.'
search:
  boost: 0.5
---

# I4. Variation: SBX and polynomial mutation

Selection decides **who** reproduces; variation decides **what** the children
look like. For real-valued decision variables, NSGA-II uses simulated binary
crossover (SBX) and polynomial mutation, both designed by Deb and co-authors.
Their behaviour is governed by one number each, the *distribution index* η.

In this lesson you will learn:

- how SBX generates two children from two parents, and what η_c controls;
- how polynomial mutation perturbs a variable, and what η_m controls;
- how both operators respect the variable bounds;
- how VAMOS implements and configures them, and which alternatives it offers.

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

The arithmetic of both operators lives in
`src/vamos/foundation/kernel/operator_primitives.py`. SBX works on a tensor of
parent pairs, so every pair and every variable of a generation is processed by
the same array expressions. This excerpt computes the first child, c₁, with
the bounded spread factor:

```python
# Excerpt from src/vamos/foundation/kernel/operator_primitives.py
    pair_mask = rng.random(n_pairs) <= prob_crossover
    idx = np.flatnonzero(pair_mask)
    if idx.size == 0:
        return offspring

    parent1 = offspring[idx, 0, :]
    parent2 = offspring[idx, 1, :]
    base1 = parent1.copy()
    base2 = parent2.copy()
    eps = 1.0e-14

    y1 = np.minimum(parent1, parent2)
    y2 = np.maximum(parent1, parent2)
    diff = y2 - y1
    valid = diff > eps
    if prob_var >= 1.0:
        active = valid
    else:
        active = valid & (rng.random(parent1.shape) <= prob_var)
    if not np.any(active):
        return offspring

    xl = lower_arr.reshape(1, -1)
    xu = upper_arr.reshape(1, -1)
    rand = rng.random(parent1.shape)
    betaq = np.empty_like(parent1)

    beta_valid = 1.0 + (2.0 * (y1 - xl) / diff.clip(min=eps))
    beta_valid = np.maximum(beta_valid, eps)
    alpha = 2.0 - np.power(beta_valid, -(eta + 1.0))
    alpha = np.maximum(alpha, eps)
    term = rand <= (1.0 / alpha)
    inv_eta = 1.0 / (eta + 1.0)
    betaq[term] = np.power(rand[term] * alpha[term], inv_eta)
    betaq[~term] = np.power(1.0 / (2.0 - rand[~term] * alpha[~term]), inv_eta)
    c1 = 0.5 * ((y1 + y2) - betaq * diff)
```

`pair_mask` applies the crossover probability per pair, `active` the
per-variable probability (`prob_var`, 0.5), and a variable whose parents are
equal is left unchanged (`valid`). The second child uses the upper bound in the
same way, and a final mask swaps the children's values with probability 0.5.

Polynomial mutation follows the two branches of the formula:

```python
# Excerpt from src/vamos/foundation/kernel/operator_primitives.py
    delta1 = (y - yl) / span
    delta2 = (yu - y) / span
    deltaq = np.empty_like(y)
    mut_pow = 1.0 / (eta + 1.0)

    lower_branch = rnd <= 0.5
    if np.any(lower_branch):
        rnd_low = rnd[lower_branch]
        xy_low = 1.0 - delta1[lower_branch]
        val_low = 2.0 * rnd_low + (1.0 - 2.0 * rnd_low) * np.power(xy_low, eta + 1.0)
        deltaq[lower_branch] = np.power(val_low, mut_pow) - 1.0

    upper_branch = ~lower_branch
    if np.any(upper_branch):
        rnd_high = rnd[upper_branch]
        xy_high = 1.0 - delta2[upper_branch]
        val_high = 2.0 * (1.0 - rnd_high) + 2.0 * (rnd_high - 0.5) * np.power(xy_high, eta + 1.0)
        deltaq[upper_branch] = 1.0 - np.power(val_high, mut_pow)

    mutated = y + deltaq * span
    np.clip(mutated, yl, yu, out=mutated)
```

With `engine="numba"`, compiled kernels perform the same arithmetic and must
draw the random numbers in the same order and shape; the
[backend guide](../../dev/add_backend.md) states that contract.

## Other encodings

SBX and polynomial mutation are for real variables. Binary, integer,
permutation and mixed problems need operators that produce valid values of
their type, such as bit-flip mutation, order crossover or swap mutation.
`NSGAIIConfig.default(..., encoding=...)` chooses suitable ones, and
[Decision representations](../../tutorials/encodings.md) shows each encoding
at work.

## Key points

- SBX produces two children symmetric around their parents; their spread grows
  with the distance between the parents and shrinks as η_c grows.
- Polynomial mutation perturbs a variable with a spread controlled by η_m;
  with p_m = 1/n it changes one variable per child on average.
- The bounded forms of both operators keep every value inside its bounds.
- VAMOS' defaults are SBX (0.9, η_c = 20) and polynomial mutation (1/n, η_m = 20).

**Next:** [I5. Elitist survival and the complete generation](elitist-survival.md).

## References

[1] Deb, K., and Agrawal, R. B. (1995). Simulated binary crossover for continuous search space. *Complex Systems*, 9(2), 115–148.

[2] Deb, K., and Goyal, M. (1996). A combined genetic adaptive search (GeneAS) for engineering design. *Computer Science and Informatics*, 26(4), 30–45.

[3] Deb, K., and Deb, D. (2014). [Analysing mutation schemes for real-parameter genetic algorithms](https://doi.org/10.1504/IJAISC.2014.059280). *International Journal of Artificial Intelligence and Soft Computing*, 4(1), 1–28.
