---
description: 'Variation for mixed problems: how a problem declares its real, integer, categorical and permutation columns in a mixed spec, how VAMOS applies a suitable operator to each part, and how to change the operator of each part.'
search:
  boost: 0.5
---

# Mixed operators

Many design problems mix kinds of decisions: a diameter (real), a number of
supports (integer), a material (categorical), perhaps an order of operations
(a permutation). Each kind needs the operators of its own lesson [3], and a
categorical variable needs care of its own: its values are labels, so averaging
material 0 and material 2 into material 1 has no meaning. Algorithms for such
problems, from Deb and Goyal's GeneAS [1] to mixed-integer evolution
strategies [2], apply a different operator to each kind of variable. VAMOS
does so with one encoding, `mixed`, and a description of which column is
which, the **mixed spec**. Its mixed crossover and mutation split every row by
that description and apply a suitable operator to each part.

In this lesson you will learn:

- how a mixed problem declares its variables in the mixed spec;
- which operator VAMOS applies to the real, integer, categorical and
  permutation parts by default;
- how to change the operator, its parameters and its probability for each
  part;
- how VAMOS keeps every column valid, and where its defaults differ from the
  single-encoding operators.

## The mixed spec

A mixed problem sets `encoding = "mixed"`, bounds `xl` and `xu` for every
column, and a dictionary `mixed_spec` with the column indices of each kind and
their domains:

| Key | Meaning |
| --- | --- |
| `real_idx`, `real_lower`, `real_upper` | real columns and their bounds |
| `int_idx`, `int_lower`, `int_upper` | integer columns and their bounds, both included |
| `cat_idx`, `cat_cardinality` | categorical columns and their number of categories; the values are 0 … cardinality − 1 |
| `perm_idx` | columns that together hold a permutation of 0 … k − 1 |

The indices of different kinds must not overlap. A decision row is still a
single float array: integer and categorical columns hold whole numbers, and
the evaluation reads them as indices. A small problem with one column of each
of the first three kinds, solved with NSGA-II's default operators for mixed
problems:

```python
import numpy as np

from vamos import Problem, optimize

MATERIALS = ["steel", "aluminium", "titanium"]
DENSITY = np.array([7.8, 2.7, 4.5])  # toy values, not engineering data
STIFFNESS = np.array([210.0, 70.0, 115.0])
PRICE = np.array([1.0, 2.0, 6.0])


class ToyShaft(Problem):
    """A diameter (real), a number of supports (integer) and a material (categorical)."""

    encoding = "mixed"

    def __init__(self):
        self.n_var = 3
        self.n_obj = 2
        self.xl = np.array([0.05, 2.0, 0.0])
        self.xu = np.array([0.30, 8.0, 2.0])  # the categorical column spans 0 .. cardinality - 1
        self.mixed_spec = {
            "real_idx": np.array([0]),
            "int_idx": np.array([1]),
            "cat_idx": np.array([2]),
            "real_lower": np.array([0.05]),
            "real_upper": np.array([0.30]),
            "int_lower": np.array([2]),
            "int_upper": np.array([8]),
            "cat_cardinality": np.array([3]),
        }

    def objectives(self, X):
        d, supports, material = X[:, 0], X[:, 1], X[:, 2].astype(int)
        cost = 100 * PRICE[material] * DENSITY[material] * d**2 + 0.5 * supports
        deflection = 1e-3 / (STIFFNESS[material] * d**4 * supports)
        return np.column_stack([cost, deflection])


result = optimize(ToyShaft(), algorithm="nsgaii", max_evaluations=3_000, seed=1)
order = np.argsort(result.F[:, 0])
for i in order[:: max(1, len(order) // 5)]:
    d, supports, material = result.X[i]
    print(f"d = {d:.3f}, supports = {int(supports)}, {MATERIALS[int(material)]:<9} cost {result.F[i, 0]:.2f}, deflection {result.F[i, 1]:.4f}")
```

Output:

```text
d = 0.050, supports = 2, aluminium cost 2.35, deflection 1.1429
d = 0.056, supports = 2, aluminium cost 2.72, deflection 0.7053
d = 0.064, supports = 5, steel     cost 5.65, deflection 0.0585
d = 0.171, supports = 8, steel     cost 26.80, deflection 0.0007
d = 0.252, supports = 8, steel     cost 53.42, deflection 0.0001
```

The cheapest designs use aluminium, the stiffer ones steel; titanium, which is
dominated in this toy, does not appear. Every support count and material is a
valid whole number, because the operators produced only valid values.

## The default operators of each part

`optimize` gave this problem `NSGAIIConfig.default(encoding="mixed")`: the
`mixed` crossover with probability 0.9 and the `mixed` mutation with
probability 1/n. For each crossed pair, the mixed crossover applies:

- to the **real** part, the arithmetic crossover of lesson
  [More real-coded operators](real-coded-more.md#arithmetic-crossovers), with
  one weight λ per pair, so the children lie between the parents;
- to the **integer and categorical** parts, uniform crossover: each value is
  exchanged with probability 0.5 and never combined;
- to the **permutation** part, order crossover (OX).

The mixed mutation applies the mutation probability to every variable of the
real, integer and categorical parts, and to every child for the permutation
part:

- **real**: Gaussian noise with a standard deviation of 10% of the range, then
  clipping to the bounds;
- **integer**: random reset to a uniform integer between the bounds;
- **categorical**: random reset to a uniform category, 0 to cardinality − 1;
- **permutation**: swap mutation.

The crossover's split, and the real, integer and categorical mutations:

```python
# Excerpt from src/vamos/engine/operators/impl/_mixed_crossover.py
        if real_idx.size:
            child1[real_idx], child2[real_idx] = _complementary(p1[real_idx], p2[real_idx], rng)
        if int_idx.size or cat_idx.size:
            swap_positions = np.concatenate([int_idx, cat_idx])
            if swap_positions.size:
                swap_cols = swap_positions[rng.random(swap_positions.size) < 0.5]
                child1[swap_cols] = p2[swap_cols]
                child2[swap_cols] = p1[swap_cols]
```

```python
# Excerpt from src/vamos/engine/operators/impl/_mixed_mutation.py
    if real_idx.size:
        span = np.maximum(real_upper - real_lower, 1e-6)
        noise = rng.normal(scale=0.1 * span, size=(X.shape[0], real_idx.size))
        mask = rng.random((X.shape[0], real_idx.size)) <= prob
        proposed = np.clip(X[:, real_idx] + noise, real_lower, real_upper)
        X[:, real_idx] = np.where(mask, proposed, X[:, real_idx])
            rand_vals = rng.integers(int_lower, int_upper + 1, size=(X.shape[0], int_idx.size), dtype=np.int32)
                cats[:, j] = rng.integers(0, int(cat_cardinality[j]), size=X.shape[0], dtype=np.int32)
```

These defaults are not the defaults of the single encodings: a purely integer
problem gets integer SBX and polynomial mutation, but the integer part of a
mixed problem gets uniform crossover and random reset. In a crossed pair, for
example, the real columns of both children lie on the segment between the
parents, while each integer and categorical column of a child holds one of
the parents' values; a mutation then resets an integer or a category to any
value of its domain, or moves a real value by a Gaussian step.

## Changing the operator of each part

The `mixed` crossover and mutation read optional **segment settings**, passed
as parameters of the builder's `crossover("mixed", ...)` and
`mutation("mixed", ...)`. The pipeline adds them to the problem's mixed spec
before it calls the operators:

```python
# Excerpt from src/vamos/engine/variation/strategies.py
    def _merged_spec(self, params: dict[str, Any]) -> dict[str, Any]:
        base_spec = getattr(self.ctx.problem, "mixed_spec", None)
        if base_spec is None:
            raise ValueError("Mixed-encoding problems must define 'mixed_spec'.")
        merged = dict(base_spec)
        for key, value in params.items():
            if key == "prob":
                continue
            merged[key] = value
        return merged
```

| Part | Crossover settings | Mutation settings |
| --- | --- | --- |
| real | `real_crossover`: `arithmetic` (default) or `mean`; `real_crossover_prob` | `real_mutation`: `gaussian` (default), `reset` or `polynomial`; `real_mutation_sigma` (absolute) or `real_mutation_sigma_factor` (share of the range, 0.1); `real_mutation_eta` (20); `real_mutation_prob` |
| integer | `int_crossover`: `uniform` (default), `arithmetic` or `sbx`; `int_crossover_eta` (20); `int_crossover_prob` | `int_mutation`: `reset` (default), `creep` or `polynomial`; `int_mutation_step` (1); `int_mutation_eta` (20); `int_mutation_prob` |
| categorical | `cat_crossover`: `uniform`; `cat_crossover_prob` | `cat_mutation`: `reset`; `cat_mutation_prob` |
| permutation | `perm_crossover`: a permutation crossover name (`ox` by default); `perm_crossover_prob` | `perm_mutation`: a permutation mutation name (`swap` by default); `perm_mutation_prob` |

A part's probability defaults to the operator's `prob`. Once any of these
settings is given (other than the permutation operator names), each part
draws its own crossover decision per pair, instead of one decision for the
whole pair. The integer `sbx` and `polynomial` options round to the nearest
integer as in lesson [Integer operators](integer.md); `mean` gives both
children the parents' mean, which halves the variance of the real part at every
mating. Two configurations on `mixed_design`, three columns of each kind:

```python
import numpy as np

from vamos import make_problem_selection, optimize
from vamos.algorithms import NSGAIIConfig

problem = make_problem_selection("mixed_design").instantiate()
print({key: value.tolist() for key, value in problem.mixed_spec.items() if key.endswith("idx")})
settings = {
    "default": ({}, {}),
    "custom": ({"int_crossover": "sbx", "int_crossover_eta": 20.0}, {"real_mutation": "polynomial", "int_mutation": "creep"}),
}
for name, (crossover_settings, mutation_settings) in settings.items():
    config = (
        NSGAIIConfig.builder()
        .pop_size(40)
        .crossover("mixed", prob=0.9, **crossover_settings)
        .mutation("mixed", prob="1/n", **mutation_settings)
        .selection("tournament", size=2)
        .build()
    )
    result = optimize(problem, algorithm="nsgaii", algorithm_config=config, max_evaluations=2_000, seed=1)
    discrete = np.concatenate([problem.mixed_spec["int_idx"], problem.mixed_spec["cat_idx"]])
    integral = bool((result.X[:, discrete] == result.X[:, discrete].round()).all())
    inside = bool(((result.X >= problem.xl) & (result.X <= problem.xu)).all())
    print(f"{name:<8} {len(result.F)} solutions, discrete columns integral: {integral}, within bounds: {inside}")
```

Output:

```text
{'real_idx': [0, 1, 2], 'int_idx': [3, 4, 5], 'cat_idx': [6, 7, 8]}
default  27 solutions, discrete columns integral: True, within bounds: True
custom   20 solutions, discrete columns integral: True, within bounds: True
```

Both configurations keep every integer and categorical column whole and every
column inside its bounds. The encodings tutorial runs `mixed_design` end to end
and checks the same properties: [Decision representations](../../tutorials/encodings.md#mixed-designs).
Every built-in algorithm except ToP accepts mixed problems; SMPSO does so
through the mixed mutation, as the [algorithm reference](../../reference/algorithms.md#capability-matrix)
lists.

## Key points

- A mixed problem sets `encoding = "mixed"` and a `mixed_spec` with the indices
  and domains of its real, integer, categorical and permutation columns.
- The `mixed` crossover applies arithmetic crossover to the real part, uniform
  exchange to the integer and categorical parts and OX to the permutation part;
  categories are never averaged.
- The `mixed` mutation applies Gaussian steps of 10% of the range to the real
  part, random reset to the integer and categorical parts, and swap to the
  permutation part.
- Segment settings such as `int_crossover="sbx"` or `real_mutation="polynomial"`,
  passed to the builder, change the operator, its parameters and its
  probability for each part.
- The defaults of a mixed problem's integer part differ from those of a purely
  integer problem: uniform crossover and random reset instead of integer SBX
  and polynomial mutation.

## References

[1] Deb, K., and Goyal, M. (1996). A combined genetic adaptive search (GeneAS) for engineering design. *Computer Science and Informatics*, 26(4), 30–45.

[2] Li, R., Emmerich, M. T. M., Eggermont, J., Bäck, T., Schütz, M., Dijkstra, J., and Reiber, J. H. C. (2013). Mixed integer evolution strategies for parameter optimization. *Evolutionary Computation*, 21(1), 29–64.

[3] Eiben, A. E., and Smith, J. E. (2015). *Introduction to Evolutionary Computing* (2nd ed.). Springer.

**Next section:** [Many-objective optimization](../advanced/many-objective.md).
