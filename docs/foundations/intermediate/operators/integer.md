---
description: 'Variation operators for bounded integer variables: uniform, arithmetic and integer SBX crossover, random reset, creep, Gaussian, boundary and integer polynomial mutation, and how VAMOS rounds and bounds their results.'
search:
  boost: 0.5
---

# Integer operators

An integer variable takes a whole value between two bounds, both included: a
number of machines, a count of items, a level from 0 to 10. Two families of
operators serve it. Some only **copy or draw integers**, so their results are
valid by construction. Others reuse a real-coded idea, such as SBX or
polynomial mutation, compute in floating point and **round**. How they round,
and what they do at the bounds, decides whether the search stays unbiased and
whether small steps survive at all.

In this lesson you will learn:

- how uniform, arithmetic and integer SBX crossover recombine integers;
- how random reset, creep, Gaussian, boundary and integer polynomial mutation
  change them;
- how each operator rounds its result and keeps it inside the bounds, and why
  rounding to the nearest integer matters;
- how VAMOS selects them for an integer problem.

## Crossovers

- **Uniform crossover** [1] exchanges each variable between the two children
  with probability 0.5, as for bits. Every value comes from a parent.
- **Arithmetic crossover** [2] draws one weight λ from [0, 1) per mating and
  gives the complementary children λ p₁ + (1 − λ) p₂ and (1 − λ) p₁ + λ p₂,
  rounded to the nearest integer. Both lie between the parents.
- **Integer SBX** runs the bounded SBX of lesson
  [Real-coded operators](real-coded.md) on the integer values, as floats, then
  clamps each child to the bounds and rounds it to the nearest integer. As in
  real-coded SBX, each variable of a crossed pair is recombined with
  probability 0.5 [3].

## Mutations

Each variable is mutated independently with probability p_m (1/n by default).

- **Random reset** replaces the value with a uniform integer between the bounds,
  both included [4].
- **Creep** adds +step or −step, with equal probability, and clips the result
  to the bounds [4]. `step` is 1 by default.
- **Gaussian** adds normal noise N(0, σ²), rounds and clips. Here σ is in the
  variable's units, 1.0 by default, unlike the real-coded Gaussian mutation,
  whose σ is a share of the range.
- **Boundary** sets the value to its lower or its upper bound, with equal
  probability.
- **Integer polynomial mutation** applies the bounded polynomial mutation [5]
  to the value as a float, clamps it and rounds it to the nearest integer.

## Rounding and bounds

| Operator | VAMOS name | Rounding | Bounds |
| --- | --- | --- | --- |
| uniform crossover | `uniform` | none: copies parent values | inside by construction |
| arithmetic crossover | `arithmetic` (`blend`) | nearest integer | between the parents |
| integer SBX | `sbx` | nearest integer, after clamping | bounded SBX, then clamped |
| random reset | `reset` (`random_reset`) | none: draws an integer | uniform in [l, u] |
| creep | `creep` | none: integer steps | clipped |
| Gaussian | `gaussian` | nearest integer | clipped |
| boundary | `boundary` | none | sets l or u |
| integer polynomial | `pm` (`polynomial`) | nearest integer, after clamping | bounded mutation, then clamped |

"Nearest integer" uses round-half-to-even, so an exact .5 goes to the even
neighbour; a continuous operator produces an exact .5 with probability zero.

### Why the nearest integer

Rounding to the nearest integer keeps SBX symmetric: the children of 3 and 4
still have mean 3.5. jMetal and jMetalPy [6] truncate the children, which
sends every fraction towards zero: their children of 3 and 4 have mean 3.25.
The figure computes 20,000 integer SBX crossovers of two parents in [0, 10]
for the variables that are recombined, with either rule:

<div class="vf-widget" data-vf-widget="integer-sbx"><p class="vf-fallback">Interactive figure: the distribution of integer SBX children of two parents in [0, 10] for a distribution index, rounded to the nearest integer as VAMOS does or truncated as jMetal does, with the children's mean. It needs JavaScript.</p></div>

Things to try:

1. Keep parents 3 and 4 and η = 20, and switch to truncation: the children's
   mean drops from 3.5 to about 3.0, because almost every child lies between 3
   and 4 and truncation sends it to 3.
2. Set η to 2: the distribution widens, and some children reach 1 or 6.
3. Move both parents far apart, to 1 and 9: the spread grows with the distance,
   as in real-coded SBX.

Rounding also **absorbs small steps**. With parents 3 and 4 and η = 20, the
figure shows every child rounding back to 3 or 4: integer SBX then only
exchanges the parents' values. With η = 2, 94.1% of the children still equal a
parent. Polynomial mutation behaves the same way. For the value 5 in [0, 10],
the formula of lesson [Real-coded operators](real-coded.md#polynomial-mutation)
gives a step δ_q·10 with |δ_q| > 0.05, the half unit that changes the rounded
value, with probability 0.95^(η+1) (the bound terms are negligible here): 34%
for η = 20 and 74% for η = 5. Two of three "mutations" with the default η leave
the value unchanged. On integer variables with few values, a smaller η, random
reset or creep makes the mutation probability mean what it says.

## In VAMOS

On an integer problem the builder names select the integer operators, and
every operator receives the problem's bounds; the table above shows how each
one rounds and clips. Creep and Gaussian mutation, for example, can leave a
value at a bound unchanged even when they mutate it, because a step outside is
clipped back.

### Selecting them in a configuration

The builder names are `uniform`, `arithmetic` (`blend`) and `sbx` for the
crossovers, and `reset` (`random_reset`), `creep`, `gaussian`, `boundary` and
`pm` (`polynomial`) for the mutations; `step`, `sigma` and `eta` are passed as
parameters. On an integer problem these names select the integer operators,
not the real-coded ones of the same name. `NSGAIIConfig.default(encoding="integer")`
uses integer SBX (0.9, η = 20) and integer polynomial mutation (1/n, η = 20).
Three mutations on `int_alloc`, twenty allocations from 0 to 10:

```python
from vamos import make_problem_selection, optimize
from vamos.algorithms import NSGAIIConfig, available_crossover_methods, available_mutation_methods

print("crossovers:", ", ".join(available_crossover_methods("integer")))
print("mutations:", ", ".join(available_mutation_methods("integer")))

problem = make_problem_selection("int_alloc").instantiate()
print("variables:", problem.n_var, "bounds:", int(problem.xl[0]), "to", int(problem.xu[0]))
for mutation, params in (("pm", {"eta": 20.0}), ("reset", {}), ("creep", {"step": 1})):
    config = (
        NSGAIIConfig.builder()
        .pop_size(60)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation(mutation, prob="1/n", **params)
        .selection("tournament", size=2)
        .build()
    )
    result = optimize(problem, algorithm="nsgaii", algorithm_config=config, max_evaluations=6_000, seed=1)
    integral = bool((result.X == result.X.round()).all())
    print(f"{mutation:<6} lowest f1: {result.F[:, 0].min():.3f}, lowest f2: {result.F[:, 1].min():.3f}, integral: {integral}")
```

Output:

```text
crossovers: arithmetic, blend, sbx, uniform
mutations: boundary, creep, gaussian, pm, polynomial, random_reset, reset
variables: 20 bounds: 0 to 10
pm     lowest f1: 25.949, lowest f2: -109.702, integral: True
reset  lowest f1: 24.248, lowest f2: -117.497, integral: True
creep  lowest f1: 27.523, lowest f2: -110.067, integral: True
```

`int_alloc` minimizes a total cost (f1) and the negative of a sum of
diminishing returns (f2); in this single run, random reset reached the lowest
values of both. The encodings tutorial runs the same problem with uniform
crossover and reset mutation: [Decision representations](../../../tutorials/encodings.md#bounded-integers).

## Key points

- Uniform crossover, random reset, creep and boundary mutation produce integers
  directly; arithmetic crossover, integer SBX, Gaussian and integer polynomial
  mutation compute in floats and round to the nearest integer.
- Every integer operator returns values inside the bounds: by construction, by
  clamping before rounding, or by clipping after the step.
- Rounding to the nearest integer keeps SBX unbiased, where truncation pulls
  children towards zero.
- Rounding absorbs steps smaller than half a unit, so on small ranges integer
  SBX and polynomial mutation with η = 20 often reproduce their parents.
- VAMOS's integer default is integer SBX (0.9, 20) with integer polynomial
  mutation (1/n, 20).

## References

[1] Syswerda, G. (1989). Uniform crossover in genetic algorithms. *Proceedings of the Third International Conference on Genetic Algorithms*.

[2] Michalewicz, Z. (1996). *Genetic Algorithms + Data Structures = Evolution Programs* (3rd ed.). Springer.

[3] Deb, K., and Agrawal, R. B. (1995). Simulated binary crossover for continuous search space. *Complex Systems*, 9(2), 115–148.

[4] Eiben, A. E., and Smith, J. E. (2015). *Introduction to Evolutionary Computing* (2nd ed.). Springer.

[5] Deb, K., and Goyal, M. (1996). A combined genetic adaptive search (GeneAS) for engineering design. *Computer Science and Informatics*, 26(4), 30–45.

[6] Benítez-Hidalgo, A., Nebro, A. J., García-Nieto, J., Oregi, I., and Del Ser, J. (2019). jMetalPy: A Python framework for multi-objective optimization with metaheuristics. *Swarm and Evolutionary Computation*, 51, 100598.

**Next:** [Permutation operators](permutation.md).
