# MOEA/D configuration

Configure `moead` through `vamos.algorithms.MOEADConfig`. This page is the parameter reference, not a tutorial about the algorithm.

**Stable public API.** The [1.x compatibility policy](../../../project/stability-and-versioning.md) defines the supported surface; import from the public facade shown below.

```python
from vamos.algorithms import MOEADConfig
```

[Configuration overview](index.md) · [Algorithm contracts](../../algorithms.md) · [Optimization](../optimization.md)

**VAMOS 1.3.0:** the resolved spec of a MOEA/D run records where its
weight vectors come from, in `algorithm.resolution.weight_vectors`, beside the
population in `population.initial_size`: `{"source": "das_dennis", "divisions": 12, "vectors": 91}`,
`{"source": "packaged_file", "file": "W3D_100.dat", "vectors": 100}`, `{"source": "file", "file": ...}`
for a file of `weight_vectors(path=...)`, or `{"source": "uniform", ...}` with two objectives.
A run whose `weight_vectors(path=...)` directory lacks the file generates the
lattice and saves it there; a file that holds exactly that lattice is recorded as
`das_dennis`, so every seed of a study records the same source whether or not an
earlier run wrote the file.
A complete lattice comes before a packaged file of the same size, so 91 and 120
individuals with three objectives use the lattice although VAMOS ships
`W3D_91.dat` and `W3D_120.dat`. Once normalized, `W3D_120.dat` differs from the
lattice by about `1e-16`, so a tool that reads the file can drift from VAMOS with
the same seed. `vamos tune` records the population a trial ran
when it differs from the sampled one (see [Tuning](../../../topics/tuning.md#populations-that-depend-on-the-number-of-objectives)).

**Development after 1.4.0:** every weight file that VAMOS ships loads, from
`optimize` and from the packaged directory that the command line passes as
`weight_vectors(path=...)`. Fourteen of the 31 did not:

- `W4D_100.dat` held its 100 vectors twice, as the jMetalPy file it comes from
  (jMetal has no such file), so VAMOS skipped it, and MOEA/D with four
  objectives and 100 to 119 individuals ran the 84 vectors of the next smaller
  lattice; it now runs the file's 100 vectors, and the command line, which
  failed, runs them too.
- `W4D_165.dat` is comma-separated in jMetal, whose own reader expects
  whitespace; VAMOS ships it space-separated, and reads either separator in
  every weight file.
- jMetal rounds its files: the rows of `W3D_500.dat` to `W3D_1200.dat` and of
  `W5D_1000.dat` to `W5D_2500.dat` sum to one within `2e-6`, and `W3D_136.dat`
  has components of `-5e-17`. A file's rows may now depart from the simplex by
  `1e-5` (`1e-6` before), and a negative component within it reads as zero; the
  rows are not renormalized, so a file that loaded before gives the same vectors.

With `optimize`, a complete lattice still comes before a packaged file of the
same size, and the populations of the other sizes are unchanged.

**Development after 1.4.0:** `weight_vectors(divisions=H1, inner_divisions=H2)`
runs Deb and Jain's two-layer lattice, the boundary lattice of `H1` divisions
and the lattice of `H2` shrunk by half towards the centre of the simplex, with
`comb(H1 + m - 1, m - 1) + comb(H2 + m - 1, m - 1)` vectors for `m` objectives:
156 with eight objectives, `H1 = 3` and `H2 = 2`, the size of jMetal's
`W8D_156.dat` (which stacks the two lattices without shrinking the inside one,
so it repeats the eight axes). A single lattice with many objectives has every
vector on the boundary of the simplex and grows fast (eight objectives: 120
vectors with 3 divisions, 330 with 4). The resolved spec records
`{"source": "two_layer", "divisions": 3, "inner_divisions": 2, "vectors": 156}`.
NSGA-III's `reference_directions` and RVEA's `reference_vectors` take the same
keys (see [reference and weight vectors](../../algorithms.md#reference-and-weight-vectors)).
A `weight_vectors(path=...)` that does not exist now raises `ConfigurationError`
when the run starts; VAMOS 1.4.0 created it as a directory and saved the
lattice there. A directory that exists but lacks the file still gets the
lattice, saved there with a `RuntimeWarning`.

**VAMOS 1.4.0:** the options of Evolver's MOEA/D. Each is off by
default, so a configuration that leaves them out runs exactly as in VAMOS 1.3.0.

| Option | Values | Reference |
| --- | --- | --- |
| `subproblem_order` (field and builder method) | `"random_permutation"` (default), `"cyclic"` | Evolver's `subProblemIdGenerator`: jMetal's `RandomPermutationCycle` and `CyclicIntegerSequence` |
| `scheme` of `crossover("de", cr=..., f=..., scheme=...)` | `"rand_1_bin"` (default), `"rand_1_exp"`, `"rand_2_bin"`, `"rand_2_exp"` | jMetal's `DifferentialEvolutionCrossover`: `RAND_1_BIN`, `RAND_1_EXP`, `RAND_2_BIN`, `RAND_2_EXP` |
| `base` of `crossover("de", ..., base=...)` | `"current"` (default), `"random"` | Evolver's MOEA/D, whose mating pool holds random parents only (`selectCurrentSolution = false`) |
| `aggregation(...)` | `"augmented_tchebycheff"` (`rho`, default `1e-4`), `"inverted_pbi"` (`theta`, default `0.1`), `"inverse_weight_tchebycheff"` | jMetal's `AugmentedTschebyscheff`, `InvertedPenaltyBoundaryIntersection` and `ModifiedTschebyscheff` |
| `normalize` and `epsilon` of any `aggregation(...)` | `normalize=False` (default) or `True`; `epsilon > 0`, default `1e-6`, only with `normalize=True` | jMetal's `normalizeObjectives` and `epsilon` (Evolver's `epsilonParameterForNormalization`) |

The definitions follow jMetal 7.8, the version Evolver uses:

- **Order of the subproblems.** `"random_permutation"` visits a new random permutation of
  the subproblems in each cycle. `"cyclic"` visits 0, 1, ..., `pop_size - 1` and starts
  again, and draws no random number for it. A checkpoint keeps the order and its cursor;
  a cyclic run refuses the checkpoint of a random order. `to_dict()` leaves out the
  default order, so a run that does not set it keeps the resolved specification, and the
  task id, that VAMOS 1.3.0 gives it.
- **DE schemes.** The child starts as the subproblem's current solution `x`. The mutant
  is `x + F (x_r1 - x_r2)` with rand/1 and `x + F (x_r1 - x_r2) + F (x_r3 - x_r4)`
  with rand/2: its base is the current solution, as in jMetal's `MOEADDEBuilder` and in
  MOEA/D-DE. With `base="random"` the base is one more random parent, `x_r3` (rand/1) or
  `x_r5` (rand/2), as in Evolver's MOEA/D, whose mating pool holds three or five random
  parents; jMetal's `DifferentialEvolutionCrossover` takes the last as the base, and the
  child still starts as the current solution. The random parents are distinct and come
  from the neighbourhood (probability `delta`) or the population, so the neighbourhood must
  hold them: `neighbor_size >= 2` for rand/1, 3 with a random base, 4 for rand/2 and 5
  with a random base. The binomial
  crossover takes each variable from the mutant when its draw is lower than `cr`, and one
  variable at random always. The exponential crossover takes a random variable and the next
  ones, cyclically, while a draw is lower than `cr`, up to all of them; it is CMOEA-MS's
  `rand_1_exp`. The repair then clips the child to the bounds.
- **Augmented Tchebycheff.** `max_i w_i |f_i - z*_i| + rho sum_i w_i |f_i - z*_i|`, where
  a zero weight counts `1e-4 |f_i - z*_i|`. This is the formula of `modified_tchebycheff`,
  whose `rho` defaults to `1e-3`, and which Evolver does not have.
- **Inverse-weight Tchebycheff** (`inverse_weight_tchebycheff`, no parameter). jMetal's
  `ModifiedTschebyscheff`, which Evolver calls `modifiedTschebyscheff`:
  `max_i |f_i - z*_i| / w_i`, where a zero weight counts `1e-7 |f_i - z*_i|`. Up to a positive
  factor per weight vector, which leaves the comparisons of a subproblem unchanged, it is the
  Tchebycheff function with the weights that Qi et al. (2014, Eq. 3) call the
  WS-transformation, `(1 / w_i) / sum_j (1 / w_j)`.
- **Inverted PBI** (Sato, 2014). With the displacement `u = z^nad - f` from the nadir
  point, `d1 = |u . w| / ||w||` and `d2 = ||u - d1 w / ||w|| ||`, the value is
  `theta d2 - d1`, which is minimized (Sato's IPBI maximizes `d1 - theta d2`).
- **Normalization.** `normalize=True` divides the displacement of each objective by
  `z^nad_i - z*_i + epsilon`: `(f_i - z*_i) / (z^nad_i - z*_i + epsilon)`, which the weighted
  sum uses in place of `f_i`, and `(z^nad_i - f_i) / (z^nad_i - z*_i + epsilon)` for the
  inverted PBI. The nadir point is jMetal's estimate: the largest value of each objective
  among the non-dominated solutions of the initial population and of every evaluated child.
  Solutions compare by their constraint violation first (feasible before infeasible, then
  the less violated) and by Pareto dominance at equal violation. The ideal and nadir points
  are updated with each child before its replacements, or with the whole batch when
  `batch_size > 1`. The inverted PBI reads the nadir point with or without normalization.
  Evolver always normalizes it, so its `invertedPenaltyBoundaryIntersection` is
  `inverted_pbi` with `normalize=True`. A checkpoint keeps the non-dominated set, in
  `extra["nadir_front"]` and `extra["nadir_front_cv"]`.

The compiled neighbourhood update computes every aggregation, with and without
normalization. The compiled one-offspring step of `engine="numba"` covers the cyclic order
and the augmented and inverse-weight Tchebycheff aggregations. The other DE schemes, the
random base, the normalization and the inverted PBI keep the reference step, so both
engines give the same results.

The tuning name table (`vamos.engine.tuning.racing.parameter_names`) reads Evolver's
`modifiedTschebyscheff` as `inverse_weight_tchebycheff` and `augmentedTschebyscheff` as
`augmented_tchebycheff`, and writes them back; `to_evolver` refuses `modified_tchebycheff`,
which Evolver does not have.

::: vamos.algorithms.MOEADConfig
    options:
      heading_level: 2
      show_root_heading: true
      show_root_full_path: false
      show_source: false

## Builder methods

Use `MOEADConfig.builder()` to obtain the fluent builder; do not import the private builder class directly. The methods below are generated from that returned object, so their signatures are not maintained as a second manual parameter table.

::: vamos.engine.algorithm.config.moead._MOEADConfigBuilder
    options:
      heading_level: 2
      show_root_heading: false
      show_root_full_path: false
      show_source: false

## Populations with weight vectors

**Development after 1.4.0:** `moead_populations(n_obj, max_pop_size=1000)`
lists the populations that MOEA/D runs without weight options, as
`MOEADConfig.default(pop_size=...)` and `optimize` build them: every size from
2 with one or two objectives, and with more the sizes of the complete lattices
and of the packaged jMetal files. With four objectives up to 200 they are 4, 10,
20, 35, 56, 84, 100, 120 and 165; with six up to 300, 6, 21, 56, 126 and 252;
with eight up to 400, 8, 36, 120, 156 and 330. Other sizes need a file or a two-layer
lattice.

```python
from vamos.algorithms import moead_populations

print(moead_populations(4, max_pop_size=200))  # (4, 10, 20, 35, 56, 84, 100, 120, 165)
```

::: vamos.algorithms.moead_populations
    options:
      heading_level: 3
      show_root_heading: true
      show_root_full_path: false
      show_source: false
