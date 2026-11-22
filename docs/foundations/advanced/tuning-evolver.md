---
description: 'Meta-optimization of multi-objective algorithms: configuring an MOEA with an MOEA, quality indicators as objectives and a front of configurations, how the VAMOS tuning spaces mirror the spaces of Evolver, and how a configuration written with Evolver becomes a point of a VAMOS space, with VAMOS code.'
search:
  boost: 0.5
---

# Evolver and meta-optimization

The tuners of the previous lessons reduce the quality of a configuration to
one number, a mean hypervolume. But a multi-objective result has several
quality aspects (lesson [What makes a good approximation?](../basic/quality-aspects.md)),
and a configuration that converges well may spread poorly. **Evolver** [1],
from the jMetal project, takes this seriously: it configures a
multi-objective algorithm with another multi-objective algorithm, using
several quality indicators as objectives, and returns a front of
configurations. VAMOS designed its tuning spaces to match Evolver's, so that
configurations can move between the two tools.

In this lesson you will learn:

- meta-optimization: a base level, a meta level, and quality indicators as
  objectives;
- how the VAMOS spaces mirror Evolver's, and where they differ;
- how a configuration tuned or published with Evolver becomes a point of a
  VAMOS space, and back;
- what VAMOS does not do yet.

Evolver is not a backend of `vamos tune`. It is a separate Java tool, built on
jMetal; VAMOS exchanges configurations with it.

## Configuring an MOEA with an MOEA

Meta-optimization has two levels:

- the **base level** is the algorithm being configured, NSGA-II for example,
  run on a set of training problems;
- the **meta level** is a multi-objective algorithm whose solutions are
  configurations of the base algorithm. Evaluating one meta-solution means
  running the base algorithm with that configuration on the training problems
  and measuring the results.

The objectives of the meta level are **quality indicators** of the base
results (lesson [Quality indicators](../basic/quality-indicators.md)), such as
the additive ε-indicator and IGD+, both minimized and computed against a
reference front. Each indicator captures a different aspect: ε+ the worst
gap to the reference front, IGD+ the average gap. Since the meta level is
multi-objective, its result is not one configuration but a **front of
configurations**, each the best compromise between the indicators that the
search found. A user who cares most about the worst case picks the
configuration with the lowest ε+; one who cares about the average picks the
lowest IGD+.

A configuration is a tree (lesson
[Automatic algorithm configuration](tuning.md#spaces-as-grammars)), so the
meta level needs an encoding. VAMOS's VAMOS-CONFIG design [2] describes the
two that a meta-optimizer can use on a VAMOS grammar:

- a **flat encoding**, one real gene in [0, 1] per parameter of the space in
  depth-first order; decoding reads only the genes of active parameters, and a
  categorical gene of k values takes the value of index min(⌊u·k⌋, k − 1).
  Continuous meta-optimizers such as SMPSO can search it;
- a **derivation tree**, the configuration itself, with a crossover that
  exchanges subtrees rooted at the same parameter and a mutation that changes
  one node and regrows a changed production.

VAMOS 1.5.2 has neither as a meta-optimization feature: the design plans them
for a later phase, with a tree solution type of their own.

### A sketch with VAMOS

The idea fits in a few lines of the public VAMOS API. The meta problem below
has three genes, the flat encoding of three real parameters of NSGA-II's
space: the crossover probability in [0.6, 1.0], SBX's distribution index from
5 to 40, and the mutation probability factor in [0.25, 3.0]. Evaluating a
meta-solution runs NSGA-II on ZDT1 with that configuration and returns two
objectives, ε+ and IGD+ against ZDT1's Pareto front; the meta level is NSGA-II
too:

```python
import numpy as np

from vamos import make_problem, optimize
from vamos.algorithms import NSGAIIConfig

t = np.linspace(0.0, 1.0, 200)
ZDT1_FRONT = np.column_stack([t, 1.0 - np.sqrt(t)])


def epsilon_plus(A, R):
    return float(np.max(np.min(np.max(A[:, None, :] - R[None, :, :], axis=2), axis=0)))


def igd_plus(A, R):
    gaps = np.maximum(A[None, :, :] - R[:, None, :], 0.0)
    return float(np.mean(np.min(np.linalg.norm(gaps, axis=2), axis=1)))


def decode(u):
    """Flat encoding: one gene in [0, 1] per parameter, scaled to the parameter's range in NSGA-II's space."""
    return 0.6 + 0.4 * u[0], 5.0 + 35.0 * u[1], 0.25 + 2.75 * u[2]


def base_level(u):
    probability, eta, factor = decode(u)
    config = NSGAIIConfig.builder().pop_size(20).crossover("sbx", prob=probability, eta=eta).mutation("pm", prob=factor / 30, eta=20.0).build()
    result = optimize("zdt1", algorithm="nsgaii", algorithm_config=config, max_evaluations=2000, seed=1)
    return [epsilon_plus(result.F, ZDT1_FRONT), igd_plus(result.F, ZDT1_FRONT)]


meta_problem = make_problem(base_level, n_var=3, n_obj=2, xl=0.0, xu=1.0, encoding="real", name="nsgaii_configuration")
meta_config = NSGAIIConfig.builder().pop_size(8).crossover("sbx", prob=0.9, eta=20.0).mutation("pm", prob="1/n", eta=20.0).build()
meta = optimize(meta_problem, algorithm="nsgaii", algorithm_config=meta_config, max_evaluations=40, seed=1)
print(f"configurations on the front: {len(meta.F)}")
for label, k in (("lowest eps+", 0), ("lowest IGD+", 1)):
    i = int(np.argmin(meta.F[:, k]))
    probability, eta, factor = decode(meta.X[i])
    print(f"{label}: eps+ {meta.F[i, 0]:.4f}, IGD+ {meta.F[i, 1]:.4f}; crossover probability {probability:.3f}, SBX index {eta:.1f}, mutation factor {factor:.2f}")
```

Output:

```text
configurations on the front: 2
lowest eps+: eps+ 0.1748, IGD+ 0.1646; crossover probability 0.931, SBX index 18.5, mutation factor 1.76
lowest IGD+: eps+ 0.1755, IGD+ 0.1438; crossover probability 0.943, SBX index 18.8, mutation factor 1.80
```

The meta run evaluates 40 configurations, each with one run of 2,000
evaluations, and ends with a front of two: one is better in ε+, the other in
IGD+. The sketch is deliberately small. A real meta-optimization uses several
training problems, several seeds per configuration, the whole space with its
categorical and conditional parameters, and a meta budget large enough for
the front to mean something; it is the same experimental discipline as any
tuning campaign (lesson
[Automatic algorithm configuration](tuning.md#training-validation-and-test-instances)).
This is the idea of Evolver, not Evolver itself.

## VAMOS spaces and Evolver's spaces

VAMOS keeps its own spaces, even where they coincide with Evolver's, and does
not read Evolver's space files; instead, the two vocabularies are aligned so
that a **configuration** can be exchanged [2].

### Names

The built-in spaces name their parameters after Evolver's, in snake_case:
Evolver's `sbxDistributionIndex` is VAMOS's `sbx_distribution_index`, and
`neighborhoodSize` is `neighborhood_size`. A few names differ more: Evolver's
`createInitialSolutions` is `initial_solutions`, `selectionTournamentSize` is
`tournament_size`, and the `CR` and `F` of differential evolution are
`de_crossover_rate` and `de_scale_factor`. VAMOS still accepts the earlier
VAMOS name of each parameter, such as `crossover_eta` for
`sbx_distribution_index`, in configurations saved by earlier releases.

Values are translated as well: Evolver's `SBX`, `blxAlpha`, `UNDC`,
`linkedPolynomial`, `levyFlight`, `tschebyscheff`,
`penaltyBoundaryIntersection`, `latinHypercubeSampling` or `oppositionBased`
are VAMOS's `sbx`, `blx_alpha`, `undc`, `linked_polynomial`, `levy_flight`,
`tchebycheff`, `pbi`, `lhs` and `obl`, and Evolver's `createInitialSolutions
default` is VAMOS's `auto`, the configuration's automatic choice.

### Spaces that follow Evolver's

The spaces were extended release by release so that a configuration tuned or
published with Evolver is a point of them [3]:

- **MOEA/D** (Evolver's `MOEADDouble`): `neighborhood_size` from 5 to 50,
  `neighborhood_selection_probability` in [0, 1], a `variation` node that
  chooses differential evolution (with Evolver's four DE schemes) or crossover
  and mutation, the order of the subproblems, the aggregations of Evolver's
  MOEA/D (Tchebycheff, weighted sum, PBI, augmented Tchebycheff, inverted PBI,
  and jMetal's modified Tchebycheff, which VAMOS calls
  `inverse_weight_tchebycheff`), and the normalization of the objectives, as
  Evolver's `normalizeObjectives`;
- **SMS-EMOA** (`SMSEMOADouble`) and MOEA/D: the crossovers, mutations and
  initializations of Evolver's spaces that VAMOS runs for them, and the SBX
  and polynomial distribution indices in Evolver's range, from 5 to 400;
- **NSGA-III** (`NSGAIIIDouble`): the mating selection, random or a tournament
  of size 2 to 10;
- **AGE-MOEA**: Evolver's `replacement`, AGE-MOEA's survival or AGE-MOEA-II's;
- the **constrained** spaces of NSGA-II, NSGA-III, SMS-EMOA, AGE-MOEA, RVEA,
  SMPSO and MOEA/D: the constraint-handling techniques of jMetal with the
  ranges of Evolver's constrained spaces, such as `NSGAIIDoubleConstrained`;
- **CCMO** and **CMOEA-MS**: a `variation` node, as Evolver's.

### Where they differ

The exchange translates names and values; it does not reconcile structure:

- **The population** is not tuned on either side: Evolver passes it to the
  algorithm, and VAMOS treats it as a run setting.
- **The mutation probability factor** lies in [0.25, 3.0] in VAMOS and in
  [0, 2] in Evolver; VAMOS rejects a factor of 0.
- **Some choices exist on one side only.** Evolver's NSGA-III also offers
  Boltzmann, ranking and stochastic universal sampling selection, which VAMOS's
  NSGA-III does not run, and Evolver's Cauchy initialization has no VAMOS
  counterpart. VAMOS's `modified_tchebycheff` (the augmented formula with
  ρ = 10⁻³) and the constraint-handling techniques that jMetal does not have
  have no Evolver form, and the translation to Evolver's terms refuses them
  rather than write a configuration that Evolver would read differently.
- **Some parameters are structured differently**, such as the external
  archive, which VAMOS writes as one choice, `external_archive`, where Evolver
  has `algorithmResult` and `archiveType`, or Evolver's per-operator repair
  strategies. They keep their own names.

## From Evolver to VAMOS, and back

Evolver writes a configuration as a line of `--name value` pairs, as in its
`VAR_CONF.txt` files. VAMOS reads such a line into its own space with
`configuration_from_command_line(space, line, evolver=True)`, which
translates every name and value and then checks that the result is a valid
derivation of the grammar: every active parameter has a value inside its
domain, and nothing else is set.

The translation reads the whole line at once, so it also adds the settings
that Evolver implies without writing them: its inverted PBI always normalizes
the objectives, and its MOEA/D draws the base of the DE mutant at random. Take
a line that sets NSGA-II with Evolver's names:

```text
--crossover SBX --crossoverProbability 0.95 --sbxDistributionIndex 12.5 --mutation polynomial --mutationProbabilityFactor 1.4 --polynomialMutationDistributionIndex 31.0 --selection tournament --selectionTournamentSize 4 --createInitialSolutions latinHypercubeSampling
```

It reads into the real-coded NSGA-II space as follows:

| Evolver | VAMOS |
| --- | --- |
| `--crossover SBX` | `crossover = sbx` |
| `--crossoverProbability 0.95` | `crossover_probability = 0.95` |
| `--sbxDistributionIndex 12.5` | `sbx_distribution_index = 12.5` |
| `--mutation polynomial` | `mutation = polynomial` |
| `--mutationProbabilityFactor 1.4` | `mutation_probability_factor = 1.4` |
| `--polynomialMutationDistributionIndex 31.0` | `polynomial_mutation_distribution_index = 31.0` |
| `--selection tournament` | `selection = tournament` |
| `--selectionTournamentSize 4` | `tournament_size = 4` |
| `--createInitialSolutions latinHypercubeSampling` | `initial_solutions = lhs` |
| not in the line | `offspring_ratio`, `external_archive` and `repair` |

The last row matters: the VAMOS space has three active parameters that the
line does not set, and the reader refuses the line until they are given,
with an error that names the first of them, `offspring_ratio`. Adding them at
their defaults (`--offspring_ratio 1.0 --external_archive none --repair auto`),
which are the algorithm's defaults, gives a valid point of the space,
identified by the space's name and hash. It runs as an ordinary
`NSGAIIConfig`, with the population added as a run setting: a tournament of
size 4, SBX with probability 0.95 and index 12.5, polynomial mutation with
index 31 and probability factor 1.4, and Latin hypercube initialization. The
public builder writes it directly, with the factor as the probability
`"1.4/n"`, and `optimize` runs it:

```python
from vamos import optimize
from vamos.algorithms import NSGAIIConfig

config = (
    NSGAIIConfig.builder()
    .pop_size(100)  # a run setting, not part of the configuration
    .selection("tournament", size=4)
    .initializer("lhs")
    .crossover("sbx", prob=0.95, eta=12.5)
    .mutation("polynomial", prob="1.4/n", eta=31.0)
    .build()
)
result = optimize("zdt1", algorithm="nsgaii", algorithm_config=config, max_evaluations=10_000, seed=1)
print(f"{len(result.F)} solutions, f1 from {result.F[:, 0].min():.3f} to {result.F[:, 0].max():.3f}")
```

Output:

```text
100 solutions, f1 from 0.000 to 1.000
```

From there it is an ordinary VAMOS configuration, which a study can compare
with the default configuration and a run manifest records.

The way back is `to_command_line(evolver=True)`, which writes the active
parameters in the depth-first order of the grammar, with Evolver's names and
values where Evolver has them. The three parameters that Evolver structures
differently keep VAMOS's names in that line.

The reader is strict. A value outside VAMOS's range, a parameter that is
inactive for the values given, or a parameter that the space does not have,
such as Evolver's `algorithmResult`, raises a `ParameterSpaceError` that names
it, instead of being dropped silently. Translating such a parameter, or
deciding to leave it out, is the user's choice, and the validation makes it
explicit.

## What VAMOS does and does not do

| | VAMOS 1.5.2 |
| --- | --- |
| Spaces aligned with Evolver's names, values and ranges | yes, the built-in grammars |
| Read and write Evolver's configuration lines | yes, in the experimental grammar module (`configuration_from_command_line`, `to_command_line(evolver=True)`) |
| Read Evolver's space files | no, by design: VAMOS keeps its own spaces |
| Tune with one scalar score (racing, irace, Optuna, SMAC3, BOHB) | yes, `vamos tune` (experimental) |
| Meta-optimize with several indicators and return a front of configurations | no; the flat and tree encodings are planned |

A configuration found by Evolver can thus be run, compared and replayed in
VAMOS, and a configuration tuned by `vamos tune` can be written in Evolver's
terms. Comparing such configurations fairly is the subject of the next
lesson.

## Key points

- Meta-optimization configures a multi-objective algorithm with another one:
  the meta level searches configurations, scored by several quality
  indicators of the base algorithm's results, and returns a front of
  configurations.
- Evolver, from the jMetal project, implements this approach; it is a
  separate tool, not a backend of `vamos tune`.
- The VAMOS spaces use Evolver's parameter names in snake_case and offer the
  choices and ranges of Evolver's spaces for MOEA/D, SMS-EMOA, NSGA-III,
  AGE-MOEA, CCMO, CMOEA-MS and the constrained algorithms, with documented
  differences.
- `configuration_from_command_line(..., evolver=True)` turns an Evolver
  configuration line into a validated point of a VAMOS space;
  `to_command_line(evolver=True)` writes one back.
- VAMOS 1.5.2 does not meta-optimize; its design plans flat and tree
  encodings of the grammars for that.

**Next:** [Comparing algorithms rigorously](performance-assessment.md).

## References

[1] The jMetal project. Evolver. Source repository: [https://github.com/jMetal/Evolver](https://github.com/jMetal/Evolver).

[2] The VAMOS developers. [VAMOS-CONFIG design: parameter spaces as grammars](../../dev/vamos-config-design.md). VAMOS developer documentation.

[3] The VAMOS developers. [Hyperparameter tuning](../../topics/tuning.md). VAMOS documentation.
