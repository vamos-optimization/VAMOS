---
description: 'Automatic algorithm configuration: why parameters matter, configuration spaces with categorical, numerical and conditional parameters, instances, seeds and budgets, the hypervolume score and its scale, training and test instances, and how to report a tuned configuration, with the VAMOS tuning spaces.'
search:
  boost: 0.5
---

# Automatic algorithm configuration

Every lesson so far ran its algorithm with published settings: SBX with
η = 20, polynomial mutation with probability 1/n, a population of 100. Those
settings are reasonable compromises, not optima, and the operator lessons
showed how much a single parameter changes the offspring (lesson
[Real-coded operators](../intermediate/operators/real-coded.md)). **Automatic algorithm
configuration**, or tuning, treats the choice of settings as an optimization
problem of its own: find the configuration that performs best on a class of
problems, within a fixed computational budget [1, 2].

This lesson sets up the problem. The next three explain how it is solved:
racing, model-based search, and multi-objective meta-optimization.

In this lesson you will learn:

- why parameters matter, and what an algorithm configuration problem contains;
- configuration spaces with categorical, numerical and conditional
  parameters, and how VAMOS writes them as grammars;
- the roles of training instances, seeds and the two budgets;
- the performance measure, here the hypervolume, and why its scale matters;
- why a tuned configuration must be tested on instances it was not tuned on,
  and how to report it.

!!! warning "Experimental in VAMOS 1.5.2"
    Tuning and the `vamos tune` command are experimental: they can change
    incompatibly in a minor release, and VAMOS has no public programmatic
    interface for tuning. These lessons show tuning through interactive
    figures, the `vamos tune` command and examples that use the public VAMOS
    API. The [tuning guide](../../topics/tuning.md) is the reference for the
    command and its options.

## Why parameters matter

A configurable algorithm is a family of algorithms. NSGA-II with SBX and
NSGA-II with BLX-α crossover are two members of the family; so are two runs
with SBX at η = 5 and η = 40. On a given class of problems and budget, some
members are much better than others, and the best member for one class is
rarely the best for another. The **algorithm configuration problem** asks for
the member that optimizes the expected performance over a distribution of
problem instances [2, 3]. It has four ingredients:

1. a **configuration space** Θ, the settings that may change and their
   domains;
2. a set of **instances**, problems drawn from the class of interest;
3. a **performance measure** m(θ, π, s), the quality of a run of configuration
   θ on instance π with seed s;
4. a **budget**: what one run may spend, and what the whole search may spend.

The goal is the configuration θ* that maximizes the expected performance
E[m(θ, π, s)] over instances and seeds. The expectation cannot be computed;
it is estimated by running configurations, and every run is expensive. Each
of the next lessons is a strategy to spend those runs well.

Tuning is not the same as comparing algorithms (lesson
[Comparing algorithms rigorously](performance-assessment.md)), but the two
meet: a fair comparison gives every algorithm the same tuning effort, or none
[4].

## Configuration spaces

A configuration space mixes three kinds of parameters:

- **categorical** parameters choose between unordered values: the crossover
  operator, the selection scheme, the initialization;
- **numerical** parameters take values in an interval, integer or real, on a
  linear or logarithmic scale: a crossover probability in [0.6, 1.0], a
  tournament size in {2, …, 10};
- **conditional** parameters exist only for some values of another parameter:
  SBX's distribution index matters only when the crossover is SBX, and
  BLX-α's α only when it is BLX-α.

Conditional parameters make the space a tree rather than a box. A
configuration that sets η for a run without SBX describes nothing; two
configurations that differ only in an inactive parameter are the same
algorithm. A tuner that ignores the structure wastes runs on such
duplicates.

### Spaces as grammars

VAMOS writes each space as a strict **grammar with defaults**, a YAML document
under the schema `vamos.parameter-space` 1.0.0. A categorical parameter
chooses between productions, one per value, and the production of a value
holds the parameters that the value activates; `global_sub_parameters` are
active whatever the value. Numerical parameters are terminals with a `range`,
both ends included. This is the crossover of NSGA-II's real-coded space:

```yaml
crossover:
  type: categorical
  default: sbx
  role: operator
  global_sub_parameters:
    crossover_probability: {type: real, default: 0.9, range: [0.6, 1.0], role: operator_rate}
  values:
    sbx:
      conditional_parameters:
        sbx_distribution_index: {type: real, default: 20.0, range: [5.0, 40.0], role: operator_rate}
    blx_alpha:
      conditional_parameters:
        crossover_alpha: {type: real, default: 0.5, range: [0.0, 1.0], role: operator_rate}
    arithmetic: {}
```

The file continues with seven more crossovers. The tree that it describes:

<div class="vf-figure"><svg class="vf-diagram" viewBox="0 0 640 210" role="img" aria-label="The crossover node of NSGA-II's space: crossover_probability is active for every value; sbx activates sbx_distribution_index, blx_alpha activates crossover_alpha, arithmetic activates nothing, and seven more values have their own parameters."><rect class="vf-box" x="220" y="10" width="200" height="56" rx="10"/><text x="320" y="34" text-anchor="middle">crossover</text><text class="vf-diagram-small" x="320" y="54" text-anchor="middle">categorical, 10 values, default sbx</text><rect class="vf-box" x="10" y="130" width="116" height="66" rx="10"/><text x="68" y="152" text-anchor="middle">every value</text><text class="vf-diagram-small" x="68" y="172" text-anchor="middle">crossover_probability</text><text class="vf-diagram-small" x="68" y="188" text-anchor="middle">real in [0.6, 1.0]</text><rect class="vf-box" x="136" y="130" width="116" height="66" rx="10"/><text x="194" y="152" text-anchor="middle">sbx</text><text class="vf-diagram-small" x="194" y="172" text-anchor="middle">sbx_distribution_index</text><text class="vf-diagram-small" x="194" y="188" text-anchor="middle">real, 5 to 40</text><rect class="vf-box" x="262" y="130" width="116" height="66" rx="10"/><text x="320" y="152" text-anchor="middle">blx_alpha</text><text class="vf-diagram-small" x="320" y="172" text-anchor="middle">crossover_alpha</text><text class="vf-diagram-small" x="320" y="188" text-anchor="middle">real in [0, 1]</text><rect class="vf-box" x="388" y="130" width="116" height="66" rx="10"/><text x="446" y="152" text-anchor="middle">arithmetic</text><text class="vf-diagram-small" x="446" y="172" text-anchor="middle">no parameters</text><rect class="vf-box" x="514" y="130" width="116" height="66" rx="10"/><text x="572" y="152" text-anchor="middle">7 more values</text><text class="vf-diagram-small" x="572" y="172" text-anchor="middle">pcx, undc, laplace, …</text><text class="vf-diagram-small" x="572" y="188" text-anchor="middle">their own parameters</text><path class="vf-arrow" d="M320 66V96M68 96H572M68 96V124M194 96V124M320 96V124M446 96V124M572 96V124"/><path class="vf-arrowhead" d="M68 130l-4-7h8zM194 130l-4-7h8zM320 130l-4-7h8zM446 130l-4-7h8zM572 130l-4-7h8z"/></svg></div>

Three rules of the schema give the tree its shape [5]:

- **One parent per parameter.** A parameter hangs from a single production
  or node; there are no condition expressions. A parameter that applies to
  two values is duplicated under each, with its own name: polynomial and
  linked polynomial mutation have an index each.
- **A default for every parameter**, inside its domain. The defaults of a
  space form its **default configuration**, and a test checks for every
  shipped space that it runs exactly as the algorithm's `default()`
  configuration, by comparing seeded runs.
- **Only configurations that run.** The schema has no forbidden
  combinations: a space offers only what its algorithm and encoding
  implement.

A **configuration** is a derivation of the grammar: a value for every
categorical node on its branch, and for every parameter that the chosen
productions activate. VAMOS packages 53 such spaces, one per algorithm and
encoding (`nsgaii.yaml`, `moead_permutation.yaml`, `nsgaii_constrained.yaml`,
…), and `vamos tune --algorithm` selects one. The population size is not a
parameter of any space: it is a setting of the run, `--pop-size`, 100 by
default.

### Sampling a configuration

The real-coded NSGA-II space, `nsgaii.yaml`, has 33 parameters: 9
categorical, 1 integer and 23 real. Seven hang from the root:
`offspring_ratio`, `selection`, `external_archive`, `initial_solutions`,
`crossover`, `mutation` and `repair`. A random configuration is drawn from
the root down: every parameter of the root gets a value, and each value drawn
for a categorical parameter brings the parameters of its production, which
get values in turn. A value is drawn only for the parameters that the choices
above it activate. The default configuration makes every choice at its
default, and it activates 12 parameters.

A smoke run of `vamos tune` shows such configurations. `--smoke` is a cheap
check of the command: it clamps each run to 30 evaluations and a population of
15, the search to four trials and two seeds, and skips the validation and
test stages. The random backend draws its trials in this way, and
`best_config_active.json` holds the active parameters of the best one:

```bash
vamos tune --instances zdt1,zdt2,zdt3 --algorithm nsgaii --backend random --smoke \
  --output-dir results/tuning --name smoke_random
cat results/tuning/smoke_random/best_config_active.json
```

```text
{
  "offspring_ratio": 0.25,
  "selection": "sus",
  "initial_solutions": "random",
  "crossover": "whole_arithmetic",
  "crossover_probability": 0.8198374750692238,
  "wa_alpha": 0.027559113243068367,
  "mutation": "levy_flight",
  "mutation_probability_factor": 1.729894111353015,
  "levy_beta": 0.9945975747486382,
  "levy_scale": 0.07905444163941203,
  "repair": "wrap"
}
```

This configuration activates 11 of the 33 parameters. It draws the whole
arithmetic crossover, so `wa_alpha` is active and `sbx_distribution_index` is
not; the Lévy-flight mutation brings its two parameters; stochastic universal
sampling (`sus`) has no tournament, so `tournament_size` is absent. So is
`external_archive`: `vamos tune` searches the space without it, so that every
candidate is scored on the same final population. A smoke run checks the
command, the evaluator and the output files; with 30 evaluations per run its
scores support no conclusion.

The mutation probability is a **relative** parameter:
`mutation_probability_factor` in [0.25, 3.0] is divided by the number of
variables when the configuration is built, so one configuration means the
same on a problem with 10 variables and on one with 30. A space does not know
the problem; settings that depend on it are written relative to it.

## Instances, seeds and budgets

A configuration is good or bad *on a class of problems*. Tuning therefore
needs **training instances**, problems that represent the class: ZDT and DTLZ
problems if the class is "bi-objective continuous benchmarks", a set of
scheduling instances if it is "my scheduling problem with 50 to 200 jobs".
Tuning on one instance finds the best configuration for that instance.

Each run is stochastic, so each instance is also run with several **seeds**.
One run of one configuration on one instance with one seed is one
**experiment**; an instance and a seed together form a **block**, and two
configurations evaluated on the same block can be compared directly, as
paired samples.

Two budgets must not be confused:

- the **run budget**, the evaluations of one run of the configured algorithm:
  `vamos tune --budget`, 5,000 by default;
- the **tuning budget**, the experiments or trials the whole search may
  spend: `--tune-budget`, 200 by default, which counts experiments for the
  racing backends and trials for the model-based ones.

Their product, roughly, is the cost of tuning. A configuration tuned for
5,000 evaluations is not necessarily good for 50,000: the run budget is part
of the problem class [2].

In `vamos tune`, `--instances` lists the problems and `--n-seeds` (5 by
default) sets the number of seeds per configuration. The seeds are derived
from `--seed`, which also seeds the search itself, so changing `--seed`
changes both; record the two together.

## The performance measure

A multi-objective run returns a set of points, and a tuner needs one number
per experiment. The usual choice is a unary quality indicator, most often the
hypervolume (lesson [Quality indicators](../basic/quality-indicators.md)),
which rewards convergence and spread at once, and which tuners of
multi-objective algorithms commonly maximize [6]. The scores of a
configuration on its blocks are then **aggregated**, usually by their mean.

`vamos tune` scores every experiment by the hypervolume and maximizes it. It
takes the final population, removes infeasible solutions, keeps the
non-dominated ones and computes their hypervolume (SMPSO is scored on its
leaders archive on unconstrained problems). The reference point is
`--ref-point`, by default 10 in every objective, the same for every training
instance. `--aggregate-mode` combines the scores of the blocks by their
`mean` (the default), `median`, 25th percentile (`p25`) or 10th percentile
(`p10`); the percentiles favour configurations that are never bad.

### Scale and normalization

The hypervolume is measured in the units of the objectives, so its scale
depends on the problem and on the reference point. Averaged over instances,
a problem whose hypervolume is around 100 outweighs one whose hypervolume is
around 1: a configuration that gains 1% on the first and loses half of its
quality on the second looks better. Two remedies are common:

- **normalize** each problem's objectives, for example to [0, 1] with the
  ideal and nadir points of its reference front, before computing the
  hypervolume, so that every problem contributes on the same scale;
- **rank** the configurations within each block and compare ranks instead of
  raw values, as F-race does (lesson [Racing and irace](tuning-racing.md)).

`vamos tune` does not normalize: it computes the hypervolume in the original
objective units, with one reference point for all instances. Choose a
reference point that every training instance can reach and that is
meaningful for all of them, and check the scores of a few runs before a long
campaign: a point outside the box contributes nothing, and an unreachable
reference point gives every configuration the same score of 0. The `mean`
aggregation then weighs problems by their hypervolume scale, while the
racing backend's tests use ranks and weigh every block the same.

The public API is enough to check the scores before a campaign. The example
runs the default configuration of NSGA-II with the run budget and the
population of `vamos tune` (5,000 evaluations, 100 individuals) on the six
problems of the campaign in the next section, with three seeds, and computes
the hypervolume of each run with the default reference point:

```python
from vamos import optimize
from vamos.ux.api import HVIndicator  # computes the hypervolume with moocore (the compute extra)

for problem in ("zdt1", "zdt2", "zdt3", "dtlz1", "dtlz2", "wfg1"):
    runs = optimize(problem, algorithm="nsgaii", max_evaluations=5000, seed=[1, 2, 3])  # the default configuration
    n_obj = runs[0].F.shape[1]
    hv = HVIndicator(reference_point=[10.0] * n_obj)  # vamos tune's default --ref-point
    print(f"{problem:<6} {n_obj} objectives, hypervolume of seeds 1 to 3: " + ", ".join(f"{hv.compute(run.F).value:.1f}" for run in runs))
```

Output:

```text
zdt1   2 objectives, hypervolume of seeds 1 to 3: 96.5, 95.4, 96.2
zdt2   2 objectives, hypervolume of seeds 1 to 3: 88.6, 88.4, 87.1
zdt3   2 objectives, hypervolume of seeds 1 to 3: 102.7, 103.1, 103.2
dtlz1  3 objectives, hypervolume of seeds 1 to 3: 720.8, 945.0, 905.4
dtlz2  3 objectives, hypervolume of seeds 1 to 3: 999.3, 999.3, 999.2
wfg1   3 objectives, hypervolume of seeds 1 to 3: 659.5, 657.8, 662.8
```

Every score is positive, so every problem reaches the reference point. The
scales differ, though: with 10 in every objective, the box of a
three-objective problem is ten times larger than that of a bi-objective one,
and so are its scores, and two seeds of DTLZ1 differ by more than twice the
whole score of a ZDT problem. Averaged by `mean`, DTLZ1, DTLZ2 and WFG1 decide
the ranking of the configurations almost on their own.

Failed runs need a score too. In `vamos tune`, a run that raises or returns a
score that is not finite counts as failed and scores `--failure-score`, 0 by
default, which is never better than a valid front.

## Training, validation and test instances

The tuned configuration is selected on its training instances, so its
training score is optimistically biased: among many configurations, the one
that looks best has partly been lucky. With few instances, the tuner can also
**overfit**: it finds settings that exploit the quirks of the training
problems and generalize poorly, as a model fitted to its training data does
[3, 7]. The remedy is the one of machine learning: estimate the quality of
the tuned configuration on **instances it was not tuned on**, and on seeds it
was not tuned with.

`vamos tune --instances` splits the problems into training, validation and
test sets: `--split-strategy suite_stratified` (the default) spreads each
benchmark suite over the three, `--train-frac` and `--validation-frac` (0.6
and 0.2) set their sizes, and `--split-seed` (42) fixes the split.
Validation (`--run-validation`, on by default) re-evaluates the five best
configurations of the search on the validation problems, and the test stage
(`--run-test`, off by default) evaluates the selected configurations on the
test problems. `--validation-seeds` and `--test-seeds` give those stages
their own seeds.

```bash
vamos tune \
  --instances zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1 \
  --algorithm nsgaii \
  --backend racing \
  --budget 5000 \
  --tune-budget 2000 \
  --validation-seeds 1001:1011 \
  --run-test \
  --test-seeds 2001:2011 \
  --output-dir results/tuning \
  --name nsgaii_racing
```

The test result answers the question of the campaign. If it is fed back into
another round of tuning, it becomes training data and a new test set is
needed.

## Reporting a tuned configuration

A configuration without its protocol cannot be reproduced or judged. Report:

- the configured algorithm, its **space** and the VAMOS version that ships
  it, and the configuration with every active parameter;
- the training instances and seeds, and the split;
- the run budget and the tuning budget, with the backend and its options;
- the performance measure: the indicator, its reference point or
  normalization, the result set it is computed on, and the aggregation;
- the result on held-out instances and seeds, compared with the default
  configuration under the same protocol.

`vamos tune` writes most of this to its output directory:
`best_config_active.json` (the active parameters of the returned
configuration), `tuning_summary.json` (the arguments, the budgets, the
aggregation, the score of the returned configuration and the number of failed
evaluations), `tuning_history.json` and `tuning_history.csv` (every
configuration and its score), and `split_instances.csv` and
`split_seeds.json` (the split). The default configuration of every space runs
as the algorithm's default, so it is the natural baseline.

A configuration of a VAMOS space can also be written as a document of the
schema `vamos.parameter-configuration` 1.0.0, which names the space by its
name, schema version and the SHA-256 of its canonical form, so that a reader
knows exactly which space the values belong to.

The same configuration can be written as a flat mapping of its values, the
form of `best_config_active.json`, and as a command line in the style of
Evolver's `VAR_CONF.txt` files (lesson [Evolver and meta-optimization](tuning-evolver.md)).

## Key points

- Algorithm configuration searches a configuration space for the setting that
  performs best on a class of instances within a run budget, spending a
  tuning budget of experiments.
- Spaces mix categorical, numerical and conditional parameters; VAMOS writes
  them as grammars with a default for every parameter, whose default
  configuration runs as the algorithm's default.
- An experiment is one run of one configuration on one block (an instance and
  a seed); configurations are compared on the same blocks.
- `vamos tune` maximizes the hypervolume of the feasible non-dominated final
  population, with one reference point and no normalization; raw means weigh
  problems by their scale, ranks do not.
- Tuned configurations overfit their training instances: report them with
  the space, instances, seeds, budgets and measure, and test them on
  held-out instances and seeds.

**Next:** [Racing and irace](tuning-racing.md).

## References

[1] Eiben, A. E., and Smit, S. K. (2011). [Parameter tuning for configuring and analyzing evolutionary algorithms](https://doi.org/10.1016/j.swevo.2011.02.001). *Swarm and Evolutionary Computation*, 1(1), 19–31.

[2] Hutter, F., Hoos, H. H., Leyton-Brown, K., and Stützle, T. (2009). [ParamILS: An automatic algorithm configuration framework](https://doi.org/10.1613/jair.2861). *Journal of Artificial Intelligence Research*, 36, 267–306.

[3] Birattari, M. (2009). [*Tuning Metaheuristics: A Machine Learning Perspective*](https://doi.org/10.1007/978-3-642-00483-4). Studies in Computational Intelligence 197. Springer.

[4] Bartz-Beielstein, T., et al. (2020). [Benchmarking in optimization: Best practice and open issues](https://arxiv.org/abs/2007.03488). arXiv:2007.03488.

[5] The VAMOS developers. [VAMOS-CONFIG design: parameter spaces as grammars](../../dev/vamos-config-design.md). VAMOS developer documentation.

[6] Bezerra, L. C. T., López-Ibáñez, M., and Stützle, T. (2016). [Automatic component-wise design of multiobjective evolutionary algorithms](https://doi.org/10.1109/TEVC.2015.2474158). *IEEE Transactions on Evolutionary Computation*, 20(3), 403–417.

[7] López-Ibáñez, M., Dubois-Lacoste, J., Pérez Cáceres, L., Birattari, M., and Stützle, T. (2016). [The irace package: Iterated racing for automatic algorithm configuration](https://doi.org/10.1016/j.orp.2016.09.002). *Operations Research Perspectives*, 3, 43–58.
