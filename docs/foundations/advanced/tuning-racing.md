---
description: 'Racing for algorithm configuration: F-race with the Friedman test and Conover post-hoc comparisons, eliminating configurations block by block, iterated racing in irace with sampling models, elitism and restarts, and how to run them with the racing and irace backends of VAMOS, with an interactive race and VAMOS examples.'
search:
  boost: 0.5
---

# Racing and irace

Lesson [Automatic algorithm configuration](tuning.md) set up the problem: many
candidate configurations, many instances and seeds, and a run for every
pair. Evaluating every configuration on every block wastes most of the runs
on configurations that are clearly bad after a few blocks. **Racing**
evaluates the candidates block by block and drops a candidate as soon as the
evidence says it is worse than the best, so the runs go to the candidates that
are still in doubt [1].

In this lesson you will learn:

- F-race: the Friedman test on the ranks of each block, and the post-hoc
  comparison that eliminates configurations;
- why ranks make blocks on different scales weigh the same;
- iterated racing: sampling models, elitism and restarts, as irace implements
  them;
- how to use all of this with the `racing` and `irace` backends of
  `vamos tune`.

## Racing

Racing comes from model selection in machine learning, where it chose among
models by testing them on more and more data points [1]. For configurations,
a race goes like this:

1. Start with a set of candidate configurations, all **alive**.
2. Take the next block, an instance and a seed, and run every alive
   configuration on it.
3. Once enough blocks have been played, test whether the alive configurations
   differ; if they do, eliminate those that are significantly worse than the
   best.
4. Stop when one configuration is left, the blocks run out, or the tuning
   budget is spent; return the best survivor.

Every alive configuration plays the same blocks, so their results form a
matrix with one row per configuration and one column per block, and the
comparison is paired: two configurations are compared on the same instances
with the same seeds.

## F-race: Friedman and Conover

**F-race** [1] uses the Friedman test [2], a non-parametric test for paired
data with more than two treatments. Each block **ranks** the alive
configurations, 1 for the best, with average ranks for ties. If all the
configurations were equivalent, every configuration would have the same
expected rank in every block, (N + 1)/2. The test measures how far the **rank
sums** R_j, the sum over the B blocks of configuration j's ranks, are from
their expected value:

<div class="vf-formula">R_j = Σ_b r_bj                                   (rank sum of configuration j over B blocks)
T = 12 Σ_j ( R_j − B(N + 1)/2 )²  /  ( B N (N + 1) − Σ_ties (t³ − t) / (N − 1) )
p = P( χ²_{N−1} ≥ T )                           (T follows a χ² law with N − 1 degrees of freedom)</div>

N is the number of alive configurations, and the sum over ties corrects for
blocks in which t configurations share a rank. When p is below the
significance level α, the configurations differ, and a **post-hoc** comparison
decides which ones to eliminate. VAMOS, like irace, uses Conover's comparison
of rank sums [3]: a configuration survives if its rank sum is within the
critical difference of the best rank sum.

<div class="vf-formula">D = t_{1−α/2, (B−1)(N−1)} · √( 2 ( B Σ_b Σ_j r_bj² − Σ_j R_j² ) / ( (B − 1)(N − 1) ) )
keep j   ⇔   R_j − R_best ≤ D                     (the best rank sum is always kept)</div>

With two configurations left, the Friedman test is replaced by a paired
Wilcoxon signed-rank test [4] on their score differences.

### Why ranks

The scores of different instances can lie on very different scales: a
hypervolume of 90 on one problem, of 1.2 on another. A test on raw scores, or
a mean, is dominated by the problems with the largest values and the largest
noise. Ranks give every block the same weight, whatever its scale, which is
what the user usually means by "better on most instances". VAMOS's tuning
guide records the effect: on scores where half of the blocks were 100 times
larger, the paired t-tests of VAMOS 1.2.0 eliminated no configuration out of
eight on ten blocks, against about three for F-race.

### When to test

A test on two or three blocks has little power and can eliminate a good
configuration by bad luck. F-race therefore waits for a minimum number of
blocks before its first test: irace's `firstTest`, 5 by default [5], and the
same in VAMOS.

## Try it: a race block by block

The figure races eight configurations, C1 to C8, on four instances and three
seeds, twelve blocks in all. The scores are synthetic hypervolumes: each
configuration has a true quality, each instance its own scale (instance 2
scores four times instance 1, and instance 3 half of it), and every run adds
noise. Each circle is the rank of a configuration on a block among the
configurations alive at that time; a dashed line after a block is a test, and
a cross marks an elimination. Step through the race with **Next block**, the
slider, or the arrow keys after focusing the plot, and change α, `first_test`
and the noise.

<div class="vf-widget" data-vf-widget="frace-race"><p class="vf-fallback">Interactive figure: a race of eight configurations on twelve blocks (four instances, three seeds). After the fifth block a Friedman test eliminates three configurations; after the eleventh and twelfth blocks two more tests eliminate four more, and the race returns the remaining configuration, C8, after 73 runs instead of 96. It needs JavaScript.</p></div>

In the default race, nothing happens during the first four blocks. After
block 5, the Friedman statistic is T = 23.87 with p = 0.0012 < 0.05; the best
rank sum is 12 (C4) and Conover's critical difference is 10.01, so C5, C6 and
C7, with rank sums 36, 34 and 32, are eliminated. C2 survives with 22, just
inside 12 + 10.01. The next five tests find no significant difference among
the five survivors (p from 0.22 down to 0.062). After block 11, p = 0.022:
C2 and C3 are eliminated, and after block 12 the test on C1, C4 and C8
eliminates C1 and C4. The race returns C8, the configuration with the best
true quality, after 73 runs instead of 96.

Compare the columns of the table at the end: C4 has a higher mean score than
C1 (1.543 against 1.535) although its true quality is lower, because its
noise on the large-scale instance 2 happened to be favourable. The rank sums
are not fooled by that. Raise the noise, and the first elimination comes
later and the race spends more runs; lower α to 0.01, and the race keeps more
configurations.

## The test in VAMOS

The `racing` backend of `vamos tune` runs the test of this lesson. It ranks
the configurations in each block so that rank 1 is always the best, whether
the score is maximized, as the hypervolume is, or minimized. It tests after
every block, as soon as every alive configuration has played the same blocks
and at least five of them, or `--min-blocks-before-elimination` blocks (3 by
default) if that is more, so the first test comes after the fifth block. With
three or more alive configurations it applies the Friedman test and Conover's
comparison; with two, the Wilcoxon test. The figure's race follows the same
steps, and a test of the VAMOS documentation checks that it eliminates the
same configurations after the same blocks as the racing backend.

## Iterated racing

A single race can only choose among the candidates it starts with. With 33
parameters, 20 random candidates cover the space poorly. **Iterated racing**
[6, 5] repeats races and moves the candidates towards the good regions:

1. sample candidates from a **sampling model** of the space;
2. race them, with the survivors of the previous race;
3. keep the best survivors as **elites** and update the model around them;
4. repeat until the tuning budget is spent.

irace, the R package that implements iterated racing [5], samples each new
candidate from one of the elites. A numerical parameter is drawn from a
truncated normal distribution centred on the elite's value, whose standard
deviation shrinks over the iterations; a categorical parameter is drawn from
a discrete distribution that moves its probability towards the elite's value.
The search starts broad and concentrates around the elites as the budget is
spent.

Two refinements matter in practice [5]:

- **Elitism.** The elites enter the next race with their previous results. A
  new candidate must first be evaluated on the blocks the elites have already
  played, so an elite is never eliminated on less evidence than it was
  selected with.
- **Restarts.** When the sampling models have converged so far that new
  candidates are almost copies of their elites, irace reinitializes the models
  partially (a soft restart), so the search does not stall.

By default, irace derives the number of iterations from the number of
parameters and the number of candidates of each iteration from the budget
that remains, so the user sets only the total number of experiments.

## Racing in `vamos tune`

`vamos tune` has two racing backends.

### The `racing` backend

`--backend racing` runs one F-race, in VAMOS, with the core installation:

- the candidates are the space's default configuration and
  `--initial-configs − 1` random configurations of the space (20 in all by
  default);
- the blocks are the training instances and seeds, in a shuffled order that
  plays every instance with one seed before the next seed;
- `--tune-budget` (200 by default) is the maximum number of experiments, and
  each run gets `--budget` evaluations;
- the tests use `--alpha` (0.05) and start after
  max(5, `--min-blocks-before-elimination`) blocks;
- the race returns the survivor with the best aggregated score over the
  blocks that all survivors played.

```bash
vamos tune \
  --instances zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1 \
  --algorithm nsgaii \
  --backend racing \
  --budget 5000 \
  --tune-budget 1000 \
  --initial-configs 20 \
  --alpha 0.05
```

A smaller race runs in a few seconds. It tunes NSGA-II on ZDT1 alone, with
ten training seeds, eight candidates, runs of 1,000 evaluations with a
population of 20, at most 80 experiments, one worker, and no validation or
statistical finisher:

```bash
vamos tune --problem zdt1 --algorithm nsgaii --backend racing \
  --budget 1000 --pop-size 20 --n-seeds 10 --initial-configs 8 --tune-budget 80 --n-jobs 1 \
  --no-run-validation --no-run-statistical-finisher \
  --output-dir results/tuning --name race_zdt1
```

Its output, without the lines that print the output folder and the five
files it saves:

```text
Only one instance provided; train/validation/test share instance but use disjoint seeds.
Backend: requested=racing effective=racing
Jobs: requested=1 resolved=1
Tune budget: 80
Split sizes (instances): train=1 validation=1 test=1
Split sizes (seeds): train=10 validation=10 test=10
[racing] Stage 0: instance 0, seed idx 8, alive=8
[racing] Stage 1: instance 0, seed idx 4, alive=8
[racing] Stage 2: instance 0, seed idx 7, alive=8
[racing] Stage 3: instance 0, seed idx 0, alive=8
[racing] Stage 4: instance 0, seed idx 1, alive=8
[racing] Reached minimum survivors, stopping early.
[racing] Best score=94.930628 after stage 5.
--- Tuning complete ---
Best configuration:
  offspring_ratio: 0.25
  selection: boltzmann
  initial_solutions: obl
  crossover: blx_alpha_beta
  crossover_probability: 0.7049253361767398
  blxab_alpha: 0.7503646726300526
  blxab_beta: 0.2804087579860399
  mutation: uniform_reset
  mutation_probability_factor: 2.9470272994534064
  repair: reflect
```

With one problem, the training, validation and test stages share ZDT1 and
use disjoint seeds. The race plays the training seeds in a shuffled order.
The eight candidates, the space's default configuration and seven random
draws, all run on the first five blocks; the first test, after block 5, finds
one candidate significantly better than every other and eliminates the other
seven, and the race stops with one survivor after 40 of its 80 experiments.
`tuning_history.json` records each candidate with its score and blocks: the
default configuration, SBX with polynomial mutation, had the second-best mean
hypervolume, 90.79 against 94.93, and was eliminated with the rest. The
result holds for ZDT1 at 1,000 evaluations, and nothing more (lesson
[Automatic algorithm configuration](tuning.md#training-validation-and-test-instances)).

The returned configuration is an ordinary NSGA-II configuration, and the
public builder writes it from `best_config_active.json`: the offspring ratio
0.25 of a population of 20 is 5 offspring, and the mutation probability factor
becomes the probability `"2.947/n"`, which VAMOS divides by the number of
variables. The example runs it and the default configuration of the space
with the settings of the race, on ten seeds that the race did not play:

```python
import numpy as np

from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.ux.api import HVIndicator  # computes the hypervolume with moocore (the compute extra)

# The configuration of best_config_active.json, rounded to three decimals.
tuned = (
    NSGAIIConfig.builder()
    .pop_size(20)
    .offspring_size(5)  # offspring_ratio 0.25 of the population
    .selection("boltzmann")
    .initializer("obl")
    .crossover("blx_alpha_beta", prob=0.705, alpha=0.750, beta=0.280)
    .mutation("uniform_reset", prob="2.947/n")  # mutation_probability_factor, divided by n
    .repair("reflect")
    .build()
)
default = NSGAIIConfig.default(pop_size=20, n_var=30)  # the default configuration of the space

hv = HVIndicator(reference_point=[10.0, 10.0])
for name, config in (("default", default), ("tuned", tuned)):
    runs = optimize("zdt1", algorithm="nsgaii", algorithm_config=config, max_evaluations=1000, seed=list(range(2001, 2011)))
    scores = [hv.compute(run.F).value for run in runs]
    print(f"{name:<8} mean hypervolume {np.mean(scores):.2f}, worst {min(scores):.2f}")
```

Output:

```text
default  mean hypervolume 90.63, worst 85.26
tuned    mean hypervolume 94.84, worst 94.08
```

On the new seeds the tuned configuration keeps its lead: a mean hypervolume
of 94.84 against 90.63, and a worst run better than the default's mean.

Two variants change the elimination. `--no-use-statistical-tests` replaces the
tests by a rank-based rule: after `--min-blocks-before-elimination` blocks,
each block eliminates the worst `--elimination-fraction` (25%) of the alive
configurations by aggregated score. `--multi-fidelity` turns the race into
successive halving over increasing run budgets (`--fidelity-levels`,
1,000, 3,000 and 10,000 evaluations by default): every candidate runs at the
first level, and only the best fraction by mean score (`--fidelity-promotion-ratio`,
0.3) moves to the next. It runs no statistical test, and its levels, not
`--budget`, set the run budgets. Lesson
[Model-based tuning](tuning-model-based.md) explains successive halving.

The `racing` backend is not iterated racing: it does not sample new
candidates around the elites. The `irace` backend does.

### The `irace` backend

`--backend irace` runs the R package irace itself, through `Rscript`. It
needs R and the CRAN package irace 4.4.4 or newer (`install.packages("irace")`
in R), and no Python extra:

```bash
vamos tune \
  --instances zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1 \
  --algorithm nsgaii \
  --backend irace \
  --budget 5000 \
  --tune-budget 2000 \
  --irace-option nbIterations=3
```

VAMOS writes an irace scenario and translates the space: the grammar becomes
irace's parameter file, with a condition `parent == "value"` for every
parameter of a production and `node %in% c(...)` for a global
sub-parameter below the root, and the default configuration is the first of
irace's initial configurations. irace calls back into the running `vamos
tune` process for every experiment, and the scores are negated, because irace
minimizes. The options map as follows:

- `--tune-budget` is irace's `maxExperiments`, as for `racing`, and
  `--n-jobs` its `parallel`;
- by default (`--irace-instance-seeds task`) each training instance/seed pair
  is an irace instance of a deterministic race, so every experiment uses the
  seeds of the task; with `irace`, irace draws the seeds itself;
- `--irace-option KEY=VALUE` adds a scenario option, such as `nbIterations=3`,
  and may be repeated;
- the scenario, `irace.log` and `irace.Rdata` stay in `<output>/irace`, or in
  `--irace-work-dir`.

The result is irace's first final elite. irace needs many experiments even
for a short search: the real-coded NSGA-II space needs at least 728 with
irace's default number of iterations, so plan `--tune-budget` accordingly.
irace rounds real values to four decimal places by default. The backend has
one run budget, so it rejects `--fidelity-levels`, and it has no wall-clock
limit, so it rejects `--timeout-seconds`.

## Key points

- Racing evaluates candidates block by block and eliminates those that are
  significantly worse, so the runs go to the candidates still in doubt.
- F-race ranks the alive configurations in every block, tests the rank sums
  with the Friedman test, and keeps the configurations within Conover's
  critical difference of the best; with two left, a Wilcoxon test decides.
- Ranks give every block the same weight whatever the scale of its scores.
- Iterated racing (irace) samples new candidates around the elites from
  models that concentrate over time, keeps the elites between races, and
  restarts the models when they converge.
- `vamos tune --backend racing` runs one F-race in VAMOS, first test after
  five blocks; `--backend irace` runs irace through R, with `--tune-budget`
  as `maxExperiments`.

**Next:** [Model-based tuning: SMAC3, BOHB and Optuna](tuning-model-based.md).

## References

[1] Birattari, M., Stützle, T., Paquete, L., and Varrentrapp, K. (2002). A racing algorithm for configuring metaheuristics. In *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO 2002)*, 11–18. Morgan Kaufmann.

[2] Friedman, M. (1937). [The use of ranks to avoid the assumption of normality implicit in the analysis of variance](https://doi.org/10.1080/01621459.1937.10503522). *Journal of the American Statistical Association*, 32(200), 675–701.

[3] Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley.

[4] Wilcoxon, F. (1945). [Individual comparisons by ranking methods](https://doi.org/10.2307/3001968). *Biometrics Bulletin*, 1(6), 80–83.

[5] López-Ibáñez, M., Dubois-Lacoste, J., Pérez Cáceres, L., Birattari, M., and Stützle, T. (2016). [The irace package: Iterated racing for automatic algorithm configuration](https://doi.org/10.1016/j.orp.2016.09.002). *Operations Research Perspectives*, 3, 43–58.

[6] Balaprakash, P., Birattari, M., and Stützle, T. (2007). [Improvement strategies for the F-Race algorithm: Sampling design and iterative refinement](https://doi.org/10.1007/978-3-540-75514-2_9). In *Hybrid Metaheuristics (HM 2007)*, Lecture Notes in Computer Science 4771, 108–122. Springer.
