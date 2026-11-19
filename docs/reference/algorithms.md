---
description: Compare VAMOS algorithms, decision encodings, constraint support, archives, and computation backends.
---

# Algorithms and backends

Choose a built-in algorithm through its public identifier in `optimize()`. Start with the [NSGA-II guide](../algorithms/nsgaii.md) for a complete executable example and an explanation of the result. The configuration pages below document exact interfaces; they are not performance rankings.

## Algorithm catalogue

| Algorithm | Identifier | Teaching guide | Configuration reference |
| --- | --- | --- | --- |
| NSGA-II | `nsgaii` | [Run and understand NSGA-II](../algorithms/nsgaii.md) | [NSGAIIConfig](api/algorithms/nsgaii.md) |
| NSGA-III | `nsgaiii` | [Reference directions and DTLZ2](../algorithms/nsgaiii.md) | [NSGAIIIConfig](api/algorithms/nsgaiii.md) |
| MOEA/D | `moead` | [Decomposition and ZDT1](../algorithms/moead.md) | [MOEADConfig](api/algorithms/moead.md) |
| SMS-EMOA | `smsemoa` | [Run and interpret SMSEMOA](../algorithms/smsemoa.md) | [SMSEMOAConfig](api/algorithms/smsemoa.md) |
| SPEA2 | `spea2` | [Run and interpret SPEA2](../algorithms/spea2.md) | [SPEA2Config](api/algorithms/spea2.md) |
| IBEA | `ibea` | [Run and interpret IBEA](../algorithms/ibea.md) | [IBEAConfig](api/algorithms/ibea.md) |
| SMPSO | `smpso` | [Run and interpret SMPSO](../algorithms/smpso.md) | [SMPSOConfig](api/algorithms/smpso.md) |
| AGE-MOEA | `agemoea` | [Run and interpret AGEMOEA](../algorithms/agemoea.md) | [AGEMOEAConfig](api/algorithms/agemoea.md) |
| RVEA | `rvea` | [Run and interpret RVEA](../algorithms/rvea.md) | [RVEAConfig](api/algorithms/rvea.md) |
| CCMO | `ccmo` | [Two populations for constrained problems](../algorithms/ccmo.md) | [CCMOConfig](api/algorithms/ccmo.md) |
| C-TAEA | `ctaea` | [Two archives for constrained problems](../algorithms/ctaea.md) | [CTAEAConfig](api/algorithms/ctaea.md) |
| ToP | `top` | [Two phases for constrained problems](../algorithms/top.md) | [ToPConfig](api/algorithms/top.md) |
| CMOEA-MS | `cmoeams` | [Two stages for constrained problems](../algorithms/cmoeams.md) | [CMOEAMSConfig](api/algorithms/cmoeams.md) |

To query the installed registry, use `available_algorithms()` from `vamos.algorithms`; see [discovery](api/discovery.md). **All nine built-in identifiers of VAMOS 1.1.0 and their public configuration classes are Stable** under the [1.x stability policy](../project/stability-and-versioning.md). **VAMOS 1.2.0:** `ccmo`, `ctaea`, `top`, `cmoeams`, `CCMOConfig`, `CTAEAConfig`, `ToPConfig` and `CMOEAMSConfig` are new and not yet covered by the 1.0 compatibility baseline. Stable API status is a compatibility commitment, not evidence that every combination of algorithm, operator and problem has been validated.

## Capability matrix

This table describes the **VAMOS 1.4.0 package**. Changes since 1.0.0 are listed below. “R / B / I / P / M” means real, binary, integer, permutation and mixed encodings. An encoding requires compatible variation operators and, for mixed problems, the problem's variable specification; a real-coded default configuration does not convert a discrete problem.

| Algorithm | Encodings | Constraint-aware survival in 1.4.0 | Archive | Backends | API status |
| --- | --- | --- | --- | --- | --- |
| NSGA-II | R / B / I / P / M | Feasibility-based | Optional external result archive | NumPy / Numba / MooCore | Stable |
| NSGA-III | R / B / I / P / M | Feasibility-first with reference-direction niching ([details](#constraint-handling-in-110)) | Optional external result archive | NumPy / Numba / MooCore | Stable |
| MOEA/D | R / B / I / P / M | Feasibility-based replacement | Optional external result archive | NumPy / Numba / MooCore | Stable |
| SMS-EMOA | R / B / I / P / M | Feasibility-based | Optional external result archive | NumPy / Numba / MooCore | Stable |
| SPEA2 | R / B / I / P / M | Feasibility-based | Internal environmental archive; optional external result archive | NumPy / Numba / MooCore | Stable |
| IBEA | R / B / I / P / M | Feasibility-based | Optional external result archive | NumPy / Numba / MooCore | Stable |
| SMPSO | R / M | Feasibility-based leaders/personal-best selection | Internal leaders archive; optional external result archive | NumPy / Numba / MooCore | Stable |
| AGE-MOEA | R / B / I / P / M | Feasibility-based when enabled | Optional external result archive | NumPy / Numba / MooCore | Stable |
| RVEA | R / B / I / P / M | Feasibility-based when enabled | Optional external result archive | NumPy / Numba / MooCore | Stable |
| CCMO | R / B / I / P / M (DE variation: R) | Violation-first SPEA2 selection in population 1; population 2 ignores constraints | Optional external result archive | NumPy / Numba / MooCore | New in 1.2.0 |
| C-TAEA | R / B / I / P / M (two-parent crossovers) | Convergence archive driven by feasibility; diversity archive ignores constraints | Optional external result archive | NumPy / Numba / MooCore | New in 1.2.0 |
| ToP | R (differential evolution) | Feasibility rule on the mean objective in phase 1; constrained-domination NSGA-II in phase 2 | Optional external result archive | NumPy / Numba / MooCore | New in 1.2.0 |
| CMOEA-MS | R / B / I / P / M (two-parent crossovers; DE variation: R) | SPEA2 fitness on [density, normalized violation] in stage A; violation first in stage B | Optional external result archive | NumPy / Numba / MooCore | New in 1.2.0 |

**Evidence and scope.** The encoding families are exercised by repository operator-combination tests and encoding integration tests. Historical real-coded, unconstrained ZDT1 smoke runs were checked for all nine algorithms on each of the three backends against the installed 1.0.0 package. That is not a validation of the full encoding × constraints × archive × backend Cartesian product, nor a convergence or speed benchmark. Custom operators and untested combinations require a representative check before a large study. SMPSO rejects standalone binary, integer and permutation encodings; its mixed path requires mixed-variable mutation.

**VAMOS 1.2.0:** every `XConfig.default()` takes an optional `encoding`, and `optimize(problem, algorithm=...)` passes the problem's encoding to it. Discrete and mixed problems then get the operators of the shared encoding table: uniform crossover and bit-flip mutation for binary, OX and swap for permutations, SBX (probability 0.9) and polynomial mutation for integers, and the mixed operators for mixed problems. In VAMOS 1.1.0 only NSGA-II and MOEA/D did this, so 22 of the 33 supported combinations of the nine algorithms with `bin_knapsack`, `tsp6`, `int_alloc` and `mixed_design` failed with an error about `sbx` or `pm`; integer runs of IBEA, AGE-MOEA and RVEA, which worked, now use the table's SBX(0.9, 20) instead of their real-coded crossover. Every `default()` also accepts `pop_size=None` (100, or the lattice size for MOEA/D, NSGA-III and RVEA), and `builder().pop_size()` rejects anything but a positive integer.

**VAMOS 1.3.0:** three crossovers no longer bias the search or
collapse its diversity. The integer SBX (`sbx` on integer variables, and
`int_crossover="sbx"` in mixed problems) rounds each child to the nearest
integer. jMetal and jMetalPy truncate it with `int()`, which pulls the children
towards zero: parents 3 and 4 got children with mean 3.25 instead of 3.5. The
integer arithmetic crossover (`arithmetic`, also `blend`) and the real part of
the mixed crossover, which is the default for mixed problems, draw a random
weight `lam` for each mated pair and give the complementary children
`lam * p1 + (1 - lam) * p2` and `(1 - lam) * p1 + lam * p2`. In VAMOS 1.2.1
and earlier both children were the parents' mean, which halved the variance of those
variables at every mating; `real_crossover="mean"` still asks for the mean.
`blx_alpha_beta` defaults to `alpha = beta = 0.5`, jMetal's values. It extends
the parents' interval by `alpha` below the smaller value and by `beta` above the
larger one, not towards the better parent as in the published operator, so the
former defaults of 0.75 and 0.25 drove an unselected population towards the
lower bounds.

**VAMOS 1.2.0:** the default real-coded crossover of every algorithm that uses SBX is SBX with probability 0.9, as in Deb et al. (2002), jMetal and pymoo, in `XConfig.default()`, in the CLI and in `build_algorithm`. Each algorithm keeps its distribution index: 20, 30 for NSGA-III and RVEA, and 15 for AGE-MOEA. VAMOS 1.1.0 used probability 1.0 in NSGA-II, NSGA-III, SMS-EMOA, SPEA2, IBEA and RVEA, so their default runs follow a different trajectory now; `.crossover("sbx", prob=1.0, eta=...)` reproduces the earlier setting. MOEA/D (differential evolution by default) and SMPSO are unchanged.

Constraints use `G <= 0`, with positive residuals representing violation. Consult the [constraint reference](constraints.md) for residual construction; an accepted `constraint_mode` field alone does not establish constraint enforcement. NSGA-III now uses the same feasibility-first contract in both survival and parent selection.

### Constraint handling in 1.1.0

VAMOS 1.1.0 includes the following constraint corrections.

- `constraint_mode` accepts only `"feasibility"` (the default) and `"none"`.
  Every built-in algorithm treated any other value, including `"penalty"` and
  `"epsilon"`, exactly like `"feasibility"`; such values are now rejected with
  `ValueError` instead of being accepted without effect.
- NSGA-III applies feasibility-first survival: when more solutions are feasible
  than survive, reference-direction niching selects among the feasible ones;
  otherwise every feasible solution survives and the least violated infeasible
  ones fill the population. Its parent tournament ranks feasible solutions
  first as well.
- IBEA penalized infeasible solutions in the wrong direction, so its
  environmental selection removed feasible solutions before infeasible ones.
  It now removes infeasible solutions first, in decreasing order of violation.
- With `result_mode="non_dominated"`, every algorithm returns the
  non-dominated feasible solutions whenever the final population contains a
  feasible one, and the least violated solutions otherwise. NSGA-II, NSGA-III,
  MOEA/D, SMS-EMOA, IBEA and SPEA2 used to return non-dominated solutions
  regardless of feasibility; AGE-MOEA, RVEA and SMPSO already did this.
- SPEA2 runs with `constraint_mode="none"` on constrained problems; it used to
  fail after the first generation.

Runs of constrained problems with these algorithms therefore differ from 1.0.2
runs with the same seed. Unconstrained runs are unchanged.

### SPEA2 environmental selection

**VAMOS 1.2.0:** SPEA2's environmental selection and mating follow
Zitzler, Laumanns and Thiele (2001) and jMetal.

- When the non-dominated solutions exceed the archive, the truncation operator
  repeatedly removes the solution whose distances to the others, sorted in
  ascending order, are lexicographically smallest. VAMOS 1.1.0 compared only the
  distance to the k-th nearest neighbour and broke ties by row order, so it
  could remove an extreme solution of a crowded pair.
- When they fall short, the dominated solutions with the best fitness
  `F = R + 1 / (sigma_k + 2)` fill the archive. VAMOS 1.1.0 added whole raw
  fitness levels and truncated the last one by distance.
- The mating tournament compares the raw fitness and the k-th nearest-neighbour
  distance that each archive member had among the previous archive and the
  offspring, not values recomputed over the new archive.
- With `result_mode="non_dominated"` and no external archive, the result keeps
  only the non-dominated members of the archive, feasible ones first. VAMOS
  1.1.0 returned the whole archive, dominated solutions included.

`k_neighbors` defaults to 1, as in jMetal; the paper uses the square root of
the population plus archive size. The `knn` pruning of external archives uses
the same truncation operator.

With jMetal's settings (population and archive 100, SBX 0.9, 30 000
evaluations, 10 seeds), the hypervolume goes from 0.7259 to 0.7326 on DTLZ2 and
from 0.6665 to 0.6832 on C2-DTLZ2, against 0.7354 and 0.6837 for jMetal 7.9's
SPEA2; ZDT1 does not change significantly. The remaining DTLZ2 gap comes from
jMetal's mating tournament, which compares by dominance alone and picks at random
between non-dominated solutions, whereas VAMOS compares the paper's fitness.
SPEA2 results from VAMOS 1.1.0 and earlier differ with the same seed.

### IBEA indicator corrections in 1.1.0

- With `indicator="hypervolume"`, IBEA used the hypervolume-difference
  indicator with the opposite orientation. It removed the solutions with the
  largest exclusive hypervolume first and kept dominated ones. It now uses
  `I_HD` as defined by Zitzler and Künzli (2004) and as in jMetal, with the
  adaptive scaling of their IBEA: objectives are normalized over the population
  and the indicator is divided by its largest magnitude. Its final fronts match
  jMetal 7.9's IBEA on ZDT1, DTLZ2 and DTLZ1 within seed variation.
  Hypervolume-based IBEA results from VAMOS 1.0.2 and earlier are not valid;
  rerun them.
- An exponential indicator contribution that exceeded the float64 range
  counted as zero, so a solution dominated by a wide margin looked like the
  best one. When a contribution would overflow, selection works in the log
  domain: each fitness is recomputed from the remaining solutions after every
  removal, and feasibility ranks still come first. Overflow therefore no
  longer changes the removal order; solutions whose fitness differs by less
  than floating-point resolution are ordered as in runs without overflow, and
  as in jMetal and jMetalPy. The epsilon indicator remains unscaled, as in
  jMetalPy, and runs without overflow are unchanged.

**VAMOS 1.2.0:** IBEA accepts the `indicator` values `"eps"`,
`"epsilon"`, `"additive_epsilon"` and `"hypervolume"`, in any letter case, and
rejects any other value with `ConfigurationError`. VAMOS 1.1.0 ran every other
value with the epsilon indicator without warning, `"hv"` and `"HV"` included,
so runs configured with `"hv"` used the epsilon indicator.

### AGE-MOEA survival score

**VAMOS 1.2.0:** AGE-MOEA follows its author's PlatEMO code in two
places.

- The binary mating tournament compares rank, then the survival score: the
  diversity and proximity score on the first front and the inverse distance to
  the ideal point on later fronts. VAMOS 1.1.0 used NSGA-II crowding distance
  instead.
- The front that is truncated to fill the population is scored by proximity to
  the ideal point and cut by that score. VAMOS 1.1.0, like pymoo 0.6.2, left it
  unscored and kept its last rows, which are usually offspring.

The rest follows pymoo, with PlatEMO's distance normalization. Unlike PlatEMO,
later fronts are measured from the ideal point in the same translated and
normalized coordinates as the first front, so the selection does not depend on
where the objectives' origin lies; objectives are not rounded to 6 decimals
before non-dominated sorting, the geometry exponent is capped at 20 and the
default variation is SBX (probability 0.9, eta 15) with polynomial mutation.

Over 30 seeds with population 100, the hypervolume goes from 0.8496 to 0.8545
on ZDT1, from 0.7426 to 0.7443 on DTLZ2 and from 0.1880 to 0.1892 on DTLZ1,
within seed variation of jMetal 7.9's AGE-MOEA (0.8545, 0.7446 and 0.1890).
AGE-MOEA results from VAMOS 1.1.0 and earlier differ with the same seed.

### AGE-MOEA-II

**Development after 1.4.0:** `AGEMOEAConfig.builder().variant("agemoea_ii")` runs
AGE-MOEA-II (Panichella, GECCO 2022). Its survival differs from AGE-MOEA's in two
steps of the first-front score, which are those that jMetal's
`AGEMOEA2EnvironmentalSelection`, written by the author, overrides:

- the exponent `p` of the front is the one whose Minkowski norm of the
  non-extreme point nearest to the ideal point is 1, found by Newton-Raphson
  from `p = 1` (precision 0.001, at most 100 steps), instead of a closed form
  from the point nearest to the diagonal;
- the diversity uses geodesic distances on the estimated `L_p` manifold: both
  points and their midpoint are projected on it, and the distance is the
  length of the path through the projected midpoint.

Everything else, the extreme points, the normalization, the scores of later
fronts and the mating tournament, is AGE-MOEA's above, with that `p`. The
exponents and distances agree with jMetal's on every case of
`tests/engine/data/agemoea2_jmetal_values.json`, and so do the survival scores
on the fronts where AGE-MOEA's shared steps agree with jMetal's. They do not
agree where AGE-MOEA's extreme points or its tie-breaking (on exactly linear
fronts) depart from jMetal, as described above for AGE-MOEA. As in AGE-MOEA,
`p` is capped at 20, which pymoo's AGE-MOEA-II also does and jMetal does not; a
point at the ideal point is not projected. pymoo takes the straight Euclidean
distance when `0.95 < p < 1.05`; VAMOS, like jMetal, always goes through the
midpoint, and both coincide when `p = 1`.

The tuning spaces of AGE-MOEA choose it with `replacement: agemoea_ii`, the
name of Evolver's parameter (`agemoeaII` in Evolver).

### Where the variants depart from their references

These are declared VAMOS choices, not defects; report them when comparing with
published results or with another framework.

- **NSGA-III parents:** **Development after 1.4.0:** no longer a departure.
  They are drawn at random by default, as Deb and Jain (2014) and pymoo do
  (under constraints, by a binary tournament on the violation, as Jain and
  Deb's constrained NSGA-III), and `selection("tournament")` is jMetal's
  NSGA-III tournament on rank. VAMOS 1.4.0 and earlier always chose them by a
  binary tournament on non-dominated rank and crowding distance, as NSGA-II,
  whatever the configured selection, so NSGA-III results of those versions
  are not results of the published algorithm. See
  [NSGA-III](../algorithms/nsgaiii.md#variants-and-their-effect).
- **NSGA-III nadir estimate:** when an intercept of the extreme-point
  hyperplane lies beyond the worst point found so far, VAMOS uses the worst
  point of the first front for every objective, as pymoo up to 0.4 did; pymoo
  0.6 replaces only those objectives by the worst point.
- **SMS-EMOA contributions** are exact, with no Monte Carlo or HypE
  approximation, so each step's cost grows exponentially with the number of
  objectives. See [SMS-EMOA](../algorithms/smsemoa.md#interpret-the-result-and-its-limits).
- **MOEA/D** defaults to differential evolution with PBI aggregation;
  `aggregation("tchebycheff")` reproduces jMetal's MOEA/D-DE.
  **VAMOS 1.3.0:** `MOEADConfig.default(variant="de")` gives MOEA/D-DE and
  `variant="sbx"` jMetal's MOEA/D with SBX and PBI; VAMOS 2.0 will make
  `"de"` the default. See [MOEA/D](../algorithms/moead.md#what-the-configuration-means).

**VAMOS 1.3.0:** three real-coded mutations scale their steps by
the range of each variable.

- `non_uniform` is Michalewicz's mutation, as in jMetal: a mutated variable
  moves towards one of its bounds, chosen at random, by
  `y * (1 - r ** ((1 - t / T) ** b))`, where `y` is the distance to that bound.
  The operator does not know the run's progress `t / T`, so it uses 0: the step
  is `y * (1 - r)`, and `perturbation` (`b`) has no effect. The tuning spaces
  no longer offer `nonuniform_perturbation`. In VAMOS 1.2.0 the step was
  `(1 - r ** b)` times the whole range, with a random sign, so 25% to 41% of the
  mutated genes left their bounds and the repair moved them onto the bound.
- `gaussian` and `cauchy` read `sigma` and `gamma` as fractions of each
  variable's range, as pymoo does, and `gaussian` defaults to `sigma = 0.1`.
  In VAMOS 1.2.0 both were in the variables' units, so the same value was
  negligible on a range of 1000 and large on a range of 0.01, and `gaussian`
  had no default. On variables with a range of 1, such as those of DTLZ and
  most of ZDT, they behave as before.

### Choose an initial experiment

| Need | Starting point | Check before increasing the budget |
| --- | --- | --- |
| General population-based Pareto search | [NSGA-II tutorial](../algorithms/nsgaii.md) | Encoding, operators and feasibility convention |
| Scalar subproblems that cooperate through neighbours | [MOEA/D tutorial](../algorithms/moead.md) | Objective scales, aggregation and exact weight/population count |
| Reference-direction coverage for several objectives | [NSGA-III tutorial](../algorithms/nsgaiii.md) | Exact direction/population count; unconstrained problem |
| Hypervolume-based environmental selection | SMS-EMOA | Objective count, reference point and indicator cost |
| Strength/density-based archive search | SPEA2 | Internal archive size versus separate result archive |
| Indicator-based fitness | IBEA | Indicator choice, scaling and its parameters |
| Particle-swarm search | SMPSO | Real/mixed representation and leaders archive; release notes below |
| Alternative many-objective diversity mechanisms | AGE-MOEA or RVEA | Geometry/reference vectors and explicit constraint mode |
| Constrained problems whose feasible regions are hard to reach | [CCMO](../algorithms/ccmo.md) | Budget of at least `2 * pop_size`; GA or DE variation |
| Constrained problems where infeasible regions guide the search to the feasible front | [C-TAEA](../algorithms/ctaea.md) | Population set by its weight vectors (91 for 100 and three objectives); budget of at least that size |
| Constrained problems whose feasible regions are small or far from the unconstrained front | [ToP](../algorithms/top.md) | Real variables; budget of at least `pop_size`; phase 1 can take up to 90% of the budget |
| Constrained problems where the objectives and the constraints must be balanced | [CMOEA-MS](../algorithms/cmoeams.md) | Budget of at least `pop_size`; GA or DE variation; `feasible_ratio_threshold` (lambda) switches the stage |

These are starting points for a study, not objective-count cut-offs or performance rankings. Compare algorithms using explicit budgets, repeated seeds and the same problem formulation.

<span id="algorithms-internal"></span>

## Results and cardinality

Use an explicit result mode when output cardinality matters. `population` returns the final population (or swarm); `non_dominated` selects a non-dominated result source. The latter need not have `pop_size` rows. NSGA-III's configuration builder defaults to `population`; the other builders default to `non_dominated`. The encoding-aware automatic setup and explicit builders need not select the same result mode, so report the resolved configuration.

**VAMOS 1.2.0:** without an explicit `algorithm_config`, `optimize`, studies and `vamos tune` run SMPSO in `non_dominated` mode, so its result is the leaders archive, as `SMPSOConfig.default()` already chose. VAMOS 1.1.0 and earlier forced `population` there and returned the final swarm, which is not elitist: on ZDT1 with 10 000 evaluations, 91 of its 200 rows were dominated.

For algorithms with a supported external result archive, enabling it makes that archive the default result source unless `population` is requested explicitly. Inspect `result.data["population"]` and `result.data["archive"]` separately. An internal SPEA2 environmental archive or SMPSO leaders archive influences search; an external result archive serves a different purpose. See [stopping and archives](../experiment/stopping_and_archive.md).

**VAMOS 1.3.0:** an unbounded external archive returns every non-dominated solution the run found, thousands on long runs, whereas jMetal's `BestSolutionsArchive` and Evolver return a subset as large as the population. `external_archive(capacity=None, final_selection=..., final_size=...)` makes the result report such a subset when the run ends:

```python
from vamos.algorithms import NSGAIIConfig

config = (
    NSGAIIConfig.builder()
    .pop_size(100)
    .crossover("sbx", prob=0.9, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .selection("tournament", size=2)
    .external_archive(capacity=None, final_selection="crowding", final_size=100)
    .build()
)
```

`final_selection` is `"none"` (the default: the whole archive, as before), `"farthest"`, `"crowding"`, `"hv_greedy"` or `"kmeans"`, the selections of `top_k(method=...)`; `kmeans` is seeded with the archive's `rng_seed`, which without a value derives from the run seed (see [Reproducibility](../experiment/stopping_and_archive.md#reproducibility)). `final_size` defaults to the population size. `result.X` and `result.F` hold the subset, the archive stays whole in `result.data["archive"]` (`archive_X`/`archive_F` for NSGA-III and IBEA, `external_archive` for SPEA2), and `result.data["archive_subset"]` records the selection, the size and the indices of the selected rows in the archive. A saved run stores the subset as its result and the whole archive under `archive/` (SPEA2: `external_archive/`, since its `archive/` is the environmental archive), and its resolved specification records `final_selection` and `final_size`, so a replay selects the same subset. The subset also applies to a bounded archive larger than `final_size`, not with `result_mode="population"`, and not to hypervolume termination or live callbacks, which see the whole archive during the run.

**Development after 1.4.0:** an unbounded external archive no longer filters itself once per generation. Once it holds feasible, finite rows, it keeps the rows each generation offers and merges them, with moocore's non-dominated filter when moocore is installed, only when something reads the archive (the result, a live callback in `population_archive` mode, hypervolume termination, SMPSO's leaders) or when the kept rows outnumber twice the archive. One merge gives the same rows, in the same order, as one update per generation, so results do not change; a run that reads its archive only at the end filters it once. Over the 200 000 evaluations of an NSGA-II run, the archive took 0.1 s instead of 5.0 s on CRE21 (7 000 rows), 0.16 s instead of 11.5 s on RCM08 and 0.28 s instead of 85 s on RCM25 (112 000 rows). Before the first feasible row, and after rows with non-finite feasible objectives, the archive is updated at once, as before.

NSGA-III and RVEA require a population compatible with the configured reference directions. MOEA/D requires one weight vector per population member. For a simplex lattice with `p` divisions and `m` objectives, use `comb(p + m - 1, m - 1)`; for three objectives and 12 divisions, that is **91**. Explicit incompatible cardinalities fail; they are not warning-only cases. Follow the tutorials before changing objective count or population size.

**VAMOS 1.2.0:** NSGA-III needs at least one individual per reference direction (N >= H) instead of exactly one, as Deb and Jain, jMetal and pymoo do; 100 individuals with 12 divisions and three objectives run with 91 directions, and fewer than 91 fail. Without `divisions`, NSGA-III uses the complete Das-Dennis lattice with the most directions that fit in `pop_size`; VAMOS 1.1.0 and earlier cut the next larger lattice to `pop_size` directions and lost the extreme direction of the first objective. With two objectives, NSGA-III and RVEA default to `pop_size - 1` divisions, or 99 without `pop_size`, instead of 6. RVEA and MOEA/D still need exactly one individual per vector; RVEA then keeps one survivor per occupied vector, so its population can end smaller than `pop_size` (see [RVEA](../algorithms/rvea.md)).

**VAMOS 1.2.0:** `optimize` and study planning check each built-in algorithm's smallest population and number of objectives before the run and raise `ConfigurationError` (or, when planning, `UnresolvedStudyTaskError` with `INVALID_POPULATION_SIZE` or `UNSUPPORTED_OBJECTIVE_COUNT`) that names the minimum. NSGA-II, IBEA, AGE-MOEA, MOEA/D, NSGA-III and RVEA need `pop_size >= 2` (for NSGA-II and IBEA the selection decides: a tournament needs as many individuals as its size, 2 by default, and random, Boltzmann, ranking and SUS selection one); SMS-EMOA, AGE-MOEA, NSGA-III and RVEA need at least two objectives. The table is `vamos.engine.algorithm.catalog.ALGORITHM_MINIMUMS`. VAMOS 1.1.0 failed later with an error from the parent tournament or from moocore, and AGE-MOEA ran a single-objective problem.

**VAMOS 1.2.0:** with `pop_size`, `NSGAIIIConfig.default()` uses the complete lattice with the most directions that fit in it, for any number of objectives (four divisions, 70 directions, for 100 individuals and five objectives, which used to fail with six divisions and 210 directions); `RVEAConfig.default()` uses the lattice of exactly `pop_size` vectors and otherwise fails at once naming the nearest valid sizes, such as 91 or 105 for three objectives. Without `pop_size`, both still use 12 divisions for three objectives and 6 for four or more.

### Reference and weight vectors

**Development after 1.4.0:** MOEA/D, NSGA-III and RVEA take their vectors from the same kind of option, as jMetal and Evolver configure them from the same weight files, so that the three algorithms can start from the same vectors: MOEA/D's `weight_vectors`, NSGA-III's `reference_directions` and RVEA's new `reference_vectors`, a mapping (or builder method) with these keys:

| Key | Vectors |
| --- | --- |
| `path` | A file with one vector per line, its components separated by whitespace or commas, or a directory with jMetal's `W{n_obj}D_{pop_size}.dat` files, such as the jMetal files VAMOS ships in `vamos/resources/weights`. A path that does not exist raises `ConfigurationError` before any evaluation; so does a directory without the file, for NSGA-III and RVEA. |
| `divisions` | The complete Das-Dennis lattice of that many divisions, `comb(H + m - 1, m - 1)` vectors with `m` objectives. RVEA's `n_partitions` is the same lattice. |
| `divisions` and `inner_divisions` | Deb and Jain's two-layer lattice: the boundary lattice of `divisions` and the lattice of `inner_divisions` shrunk by half towards the centre of the simplex (`w / 2 + 1 / (2 m)`, as pymoo's multi-layer directions and PlatEMO's `UniformPoint`), boundary rows first. With eight objectives, `divisions=3` and `inner_divisions=2` give 120 + 36 = 156 vectors. A single lattice of few divisions has every vector on the boundary of the simplex when the objectives are many. |

`inner_divisions` needs `divisions` and excludes `path`, and an unknown key raises `ConfigurationError`. A plain string is a `path`. The inside layer must add directions: layers that share one (two objectives with `divisions=4` and `inner_divisions=2`, whose three inside directions are on the boundary lattice) raise `ConfigurationError`, as a repeated vector is a niche that never gets its own individual; Deb and Jain's usual settings never do: when `inner_divisions` is smaller than the number of objectives `m`, every inside direction has a coordinate `1 / (2 m)`, which a boundary layer of fewer than `2 m` divisions does not have. The file or lattice is the population: MOEA/D and RVEA need exactly one individual per vector, NSGA-III at least one. MOEA/D keeps its other rules: with one or two objectives its weights are uniform unless `inner_divisions` is given, a directory without the file gets the lattice of exactly `pop_size` vectors and saves it there with a `RuntimeWarning` (jMetalPy's behaviour), and without options it runs the populations that [`moead_populations`](api/algorithms/moead.md#populations-with-weight-vectors) lists. Files may depart from the simplex by jMetal's rounding (`1e-5`) and are not renormalized.

The resolved spec of a run records where the vectors come from: MOEA/D always, in `algorithm.resolution.weight_vectors`, now also `{"source": "two_layer", "divisions": 3, "inner_divisions": 2, "vectors": 156}`; NSGA-III and RVEA when the configuration names a path, in `algorithm.resolution.reference_directions` (`{"source": "file", "file": "W3D_100.dat"}`) and `algorithm.resolution.reference_vectors` (with `"vectors"`, the population). Their lattices are already stated by the configuration, so a run without a path keeps the resolved spec, and the task id, of VAMOS 1.4.0, and runs and studies stored by 1.4.0 still verify and resume exactly. When a run starts, a `path` is bound to what the run reads: a directory becomes its `W{n_obj}D_{pop_size}.dat`, and `sha256` records the digest of the file's vectors (`sha256:` and the SHA-256 of their shape and float64 values, so a file rewritten with other separators keeps it). The resolved configuration holds both, the records come from it alone, so verification needs neither the file nor the machine it was on, and replay refuses, before any evaluation, a file whose vectors have changed instead of running another experiment. This binds MOEA/D's and NSGA-III's files too, which VAMOS 1.4.0 replayed with whatever the file held; a MOEA/D directory without the file, whose run generates the lattice, is not bound. A `sha256` given in a configuration pins the file in the same way.

### The command line and `optimize`

A top-level `vamos` run and `vamos.optimize` build a run separately, so with the
same problem, algorithm, population, budget and seed, and no algorithm
configuration, they resolve the same configuration except for these
differences, which `tests/experiment/test_cli_api_parity.py` keeps listed:

- Result mode: the command line returns `non_dominated` for AGE-MOEA, CCMO,
  CMOEA-MS, C-TAEA, IBEA, MOEA/D, RVEA, SMS-EMOA, SPEA2 and ToP, where
  `optimize` returns `population`. NSGA-II, NSGA-III and SMPSO agree.
- NSGA-III: the command line derives the reference directions from the
  population when the run starts; `optimize` records the lattice's divisions.
- MOEA/D: the command line reads the packaged weight files and records their
  directory, a path inside the installation; `optimize` records the divisions.
  **VAMOS 1.3.0:** with three or more objectives,
  `algorithm.resolution.weight_vectors` records the weights each path runs: a
  packaged jMetal file, such as `W3D_91.dat`, on the command line, and the
  Das-Dennis lattice with `optimize`.
- NSGA-II's offspring size and SMS-EMOA's reference point are written out by
  one path and left to their defaults by the other, with the same effect.

**VAMOS 1.3.0:** NSGA-III's SBX has distribution index 30 on the
command line too, as in Deb and Jain (2014), pymoo and the MOEA Framework, and
as `NSGAIIIConfig.default()` already had (jMetal's NSGA-III examples keep its
general SBX default, 20). VAMOS 1.2.1 and earlier used 20 on the command line,
the general SBX default of VAMOS too. Command-line NSGA-III runs on
real-coded problems therefore follow a different trajectory;
`--nsga3-crossover-eta 20` reproduces the earlier setting.

When the two must agree, set these options explicitly on both and compare the
resolved configuration that each run records. Both paths are stable, so their
defaults stay until one execution path replaces them in 2.0.

### Termination criteria

**VAMOS 1.2.0:** every algorithm's `run(problem, termination=...)`
and `initialize`, and `optimize(termination=..., algorithm_config=...)`, accept
the same three criteria:
`("max_evaluations", n)`, the total number of evaluations;
`("n_gen", g)`, `g` generations of `pop_size` evaluations, the initial
population included, so `g * pop_size` evaluations; and
`("hv", {"target_value": t, "reference_point": r, "max_evaluations": n})`,
which stops once the hypervolume of the result source against `r` reaches `t`,
within `n` evaluations. Counts must be positive integers: `("max_evaluations",
150.9)` raises `ValueError`, while an integral float such as `1e4` is accepted.
Every result has `hv_reached`, whether the HV target stopped the run; NSGA-III,
SMS-EMOA and IBEA keep `hv_converged` as an alias until VAMOS 2.0. In VAMOS
1.1.0, AGE-MOEA and RVEA accepted `n_gen` but not `hv`, the other seven accepted
`hv` but not `n_gen`, `run()` truncated a fractional budget, and NSGA-III,
SMS-EMOA and IBEA reported only `hv_converged`, so the CLI never recorded that
their HV target was reached.

**VAMOS 1.3.0:** `optimize(termination=("max_time", seconds))`
stops a run of any built-in algorithm once `seconds` of wall-clock time have
passed since it started, checked after each generation (MOEA/D after each
`pop_size` evaluations, SMS-EMOA after each step), so a run can overrun the
limit by one generation. `("max_time", {"seconds": s, "max_evaluations": n})`
also stops it after `n` evaluations, whichever comes first; without
`max_evaluations` there is no evaluation budget. Schedules that measure progress
against the budget, such as ToP's phases, CMOEA-MS's stages and the
constraint-handling techniques that relax over the run, then barely advance, so
give them `max_evaluations`. A live callback still receives every generation, and
its `should_stop` still stops the run first. `result.meta["time_limit_reached"]`
says whether the time limit stopped the run, and `load_result` restores it from
the manifest. **A run limited by time is not
reproducible**, on another machine or on the same one: how many evaluations
fit in the time depends on the hardware and its load. Its resolved spec records
`vamos.termination:max_time@1` with `{"seconds": s, "max_evaluations": n}`
(`null` without a cap) and `determinism.declared: false`; its manifest's
`replayability` gives the reason `wall_clock_termination`, and
`outcome.termination_reason` is `max_time` or, when the cap stopped it,
`max_evaluations`. `verify_run` reports it as not exactly replayable and
`reproduce` refuses it; rerun `("max_evaluations", n)` with its
`outcome.evaluations` for a run with the same budget. Only `optimize` takes this
criterion; an algorithm's own `run()` and `initialize()` do not.

**VAMOS 1.2.0:** all nine algorithms implement the same
step-by-step protocol: `initialize(problem, termination, seed)` evaluates the
initial population; repeat `ask()`, evaluate the returned decision vectors, and
`tell(...)` the objectives (and constraints) until `should_terminate()`; then
`result()` returns what `run()` would. With the same seed, the result equals
`run()`'s. `tell` returns `True` once the HV target is reached. In VAMOS 1.1.0,
NSGA-II and MOEA/D had no public `initialize`, `should_terminate` or `result`,
and `tell` of NSGA-III, SMS-EMOA and IBEA always returned `False`.

### Changes included in 1.0.1

- **SMPSO:** `archive_size` controls search leaders, `.external_archive(...)` configures a separate result archive, and explicit `population` mode returns the final swarm. Version 1.0.0 returned the leaders archive without honoring these independent result controls.
- **AGE-MOEA and RVEA:** constraint-aware survival and constraint-aligned results are included when constraint handling is enabled. Version 1.0.0 did not enforce constraints in survival. `constraint_mode="none"` remains an explicit opt-out.
- **RVEA:** continue to use a numeric mutation probability such as `1.0 / problem.n_var`. Its variation setup still does not resolve the `"1/n"` shorthand in 1.0.1.
- **NSGA-III:** the constraint limitation above remains in 1.0.1.

These are implementation fixes under the existing stability policy. Record the package version and, for a source checkout, its commit. See the [1.0.1 release notes](../project/release-notes-1.0.1.md) before comparing or replaying older experiments.

## Optional baselines

- PyMOO NSGA-II (real and permutation), jMetalPy NSGA-II (real and permutation), PyGMO NSGA-II.
- Enabled via `--include-external`; install with `python -m pip install "vamos-optimization[research]==1.1.0"`. These third-party integrations are **Experimental**, outside the stable VAMOS algorithm contract.

**VAMOS 1.2.0:** these baselines are removed, with `--include-external`, `--external-problem-source` and the `research` extra. VAMOS depends on no other optimization framework.

## Backends

- NumPy (default): vectorized CPU kernels.
- Numba: JIT acceleration for supported kernels (set `VAMOS_USE_NUMBA_VARIATION=1` for permutation/binary/integer variation).
- MooCore: accelerated kernels via `moocore`. Install optional backends with `python -m pip install "vamos-optimization[compute]==1.4.0"`; see [installation](../guide/installation.md).

### Backend roles

| Backend | Status | Best use |
|---------|--------|----------|
| `numpy` | Stable | Exact reference backend and deterministic default. |
| `numba` | Optional backend | JIT implementations of supported kernels; includes compilation cost on first use. |
| `moocore` | Optional backend | Native non-dominated sorting/hypervolume operations where available; not every algorithm step is accelerated. |

**VAMOS 1.2.0:** the three backends follow the same trajectory for
every built-in algorithm with the same seed. The tests compare NumPy with Numba
and NumPy with MooCore.
- **Tournaments:** with a tournament size above two, the tournament keeps the
  best rank when crowding distances are NaN or -inf, and uses fractional ranks
  as given. VAMOS 1.1.0 could let a worse rank win in NumPy and truncated ranks
  to integers in both NumPy and Numba.
- **MooCore survival:** it returns arrays the caller owns; it returned views of
  buffers that the next survival overwrote. The undocumented
  `VAMOS_MOOCORE_HV_CONTRIB` variable, which switched it to
  hypervolume-contribution truncation, is removed.

### Automatic engine selection

VAMOS 1.1.0 expands the automatic Numba preference to all nine built-in
algorithms. Version 1.0.2 applied it only to NSGA-II and MOEA/D.

| Request | Effective engine |
| --- | --- |
| Omitted / `None` | NumPy, preserving the stable 1.x default |
| `"auto"`, any of the nine built-in algorithms | Numba if its dependency and backend load and JIT is enabled; otherwise NumPy |
| `"auto"`, external/plugin/unknown algorithm | NumPy; no compatibility claim for custom algorithms |
| `"numpy"`, `"numba"`, `"moocore"` | Exactly the requested engine; unavailable engines raise an error |

**VAMOS 1.2.0:** an algorithm class built directly without a
kernel, such as `NSGAII(config)`, also uses NumPy. VAMOS 1.1.0 used Numba there
whenever Numba was installed. The backends follow the same trajectory, so only
the speed changes; pass `kernel=NumbaKernel()` to keep the compiled kernels.

Pass `engine="auto"` to `optimize`, `engine="auto"` in `StudySpec`, or
`--engine auto` on the CLI. It does not install optional packages, benchmark the
workload, choose MooCore, compile kernels during selection, or consume random
draws. Missing or incompatible optional imports fall back before optimization;
the effective engine is recorded in result metadata and the canonical resolved
specification. Diagnostics for load failures are available through VAMOS logging.
`NUMBA_DISABLE_JIT=1` also selects NumPy in auto mode.

After selection, compilation or execution errors propagate; VAMOS does not
restart an optimization under another engine. Retrying could repeat objective
side effects or consume a different random stream. The NumPy single-pair SBX
and MOEA/D optional shortcuts also tolerate unloadable Numba installations.

An explicit `auto` request is retained in the requested configuration. Saved
runs and created studies freeze the **resolved** engine: exact replay, study
resume, and retry do not rerun auto selection. If that stored engine becomes
unavailable, they fail rather than silently substitute NumPy. A newly resolved
auto configuration can select differently in another environment; pin
`engine="numpy"` or `engine="numba"` and record the environment when that matters.
Existing 1.0 artifacts and schema versions are unchanged. Expanding the
environment-dependent auto heuristic does not change the stable omitted default
or promise cross-environment bitwise reproducibility.

**VAMOS 1.2.0:** the NumPy engine sorts two-objective populations
into fronts in O(N log N) by a sweep over the first objective, instead of
comparing every pair of rows, with the same ranks and fronts (ties and
duplicated rows included). Sorting 1000 rows takes about 0.4 ms instead of
33 ms. Three or more objectives still compare every pair, in O(M N²); for large
populations there, prefer `engine="auto"` or `engine="numba"`.

Numba is a throughput preference, not a guarantee of a faster first call.
Compare an empty JIT cache in a fresh process, a fresh process with a populated
disk cache, and repeated runs with warm JIT separately. The reproducible
[benchmark harness](https://github.com/vamos-optimization/VAMOS/tree/main/experiments/numba_acceleration)
supports `numpy`, `numba`, and `auto`; see also
[Measure backend performance](../tutorials/measure-performance.md).

### Numba engine coverage

`engine="numba"` replaces selected numerical steps with compiled kernels. Each
kernel is written to reproduce the NumPy reference: the same random draws in the
same order, the same tie-breaking and the same floating-point operations. The
repository's parity tests check that the built-in algorithms follow identical
trajectories under both engines for the same seed in the same environment;
cross-backend bitwise equality nevertheless remains outside the
[stability promise](../project/stability-and-versioning.md). Selecting an engine
never compiles problem evaluation.

| Algorithm | Compiled with `engine="numba"` | Stays in NumPy/Python |
| --- | --- | --- |
| NSGA-II | Non-dominated ranking and crowding, survival, tournament selection, SBX, polynomial mutation; the whole one-offspring steady-state step in its supported configuration (below) | Feasibility ordering, archives, steady-state front maintenance outside that configuration |
| NSGA-III | Ranking and crowding, tournament selection (**Development after 1.4.0:** where configured, and the violation tournament of random parents under constraints), SBX, polynomial mutation | Random parent draws, normalization, association to reference directions, niching (it shuffles with the run generator inside the loop) |
| MOEA/D | Differential-evolution children, SBX, polynomial mutation, neighbourhood replacement; the whole one-offspring step (see below) | Weight geometry; parent sampling and repair outside the compiled one-offspring step |
| SMS-EMOA | The steady-state step (VAMOS 1.2.0, see below): child variation and clipping, front insertion, two-objective hypervolume contributions, crowding, feasibility-first removal; tournament selection where configured | Parent sampling draws, hypervolume contributions for three or more objectives (MooCore when installed), archives |
| SPEA2 | SBX, polynomial mutation, lexicographic archive truncation | Strength/raw fitness and density (vectorized NumPy) |
| IBEA | Tournament selection, SBX, polynomial mutation, indicator-based removal | Indicator matrix |
| SMPSO | Turbulence mutation, leaders-archive crowding truncation | Velocity/position updates (vectorized NumPy), deduplication |
| AGE-MOEA | Non-dominated ranking, tournament selection, SBX, polynomial mutation, greedy survival scores | Front normalization and pairwise Minkowski distances |
| RVEA | SBX, polynomial mutation | Angle-penalized survival and reference-vector adaptation (vectorized NumPy) |

**VAMOS 1.2.0:** steady-state NSGA-II (`.steady_state()`, one
offspring per step) runs each step as a few compiled calls under
`engine="numba"`: the binary tournament, SBX on the mating pair, clip repair,
polynomial mutation, the insertion of the evaluated child into the cached
fronts, crowding and the truncation back to the population size. Random values
are still drawn from the run generator in the NumPy order, and the truncated
front is still ordered with `np.argsort`, so the trajectory is the NumPy
engine's. This covers unconstrained runs with binary tournament selection, SBX,
polynomial mutation and the default clip repair. Constrained problems, other
selection, crossover, mutation or repair operators, genealogy tracking,
immigration, parent-selection filters and non-breeding indices keep the
NumPy/Python steady-state path, as does any step with non-finite objective
values. External archives and hypervolume termination behave the same on both
paths.

**VAMOS 1.2.0:** with `batch_size=1` (the default), MOEA/D under
`engine="numba"` compiles each whole step: parent selection, DE or SBX,
polynomial mutation, the clip repair, the ideal point and the neighbourhood
replacement. Python only draws the random numbers from the run generator, in the
order of the NumPy step, and evaluates the problem, so trajectories stay
identical. The compiled step covers real-coded runs with DE or SBX crossover,
polynomial mutation and the clip repair (the defaults), every aggregation,
constraints, external archives, live callbacks and HV termination. Genealogy
tracking, `batch_size > 1`, other operators or repairs, other array types, and
platforms where compiled variation stays disabled (see below) keep the
per-operator path described in the table.

**VAMOS 1.4.0:** the compiled MOEA/D step also covers
`subproblem_order="cyclic"` and the `augmented_tchebycheff` and
`inverse_weight_tchebycheff` aggregations. The DE schemes other than rand/1/bin,
the random DE base, the normalization of the objectives and the `inverted_pbi`
aggregation keep the per-operator path, whose neighbourhood update
is compiled, so both engines give the same results (see the
[MOEA/D configuration](api/algorithms/moead.md)).

Compiled SBX and polynomial mutation are enabled only after a one-time check
that compiled `pow` matches `np.power` bit for bit on the platform (some NumPy
builds use vendor SIMD power). Otherwise the Numba engine keeps the NumPy
operators. Compiled ranking, survival and archive truncation fall back to NumPy
for non-finite objective values.

**VAMOS 1.4.0:** the compiled power takes the same shortcuts as
`np.power` for a scalar exponent, which a run-time probe detects: the exponent 2
as `x * x` (NumPy 2) and, with NumPy 2.4, the exponent 0.5 as `sqrt(x)` and -1
as `1 / x`. With NumPy 2.4 the check failed on these exponents and the Numba
engine kept the NumPy operators; it now keeps its compiled variation, with the
same results.

**VAMOS 1.2.0:** under `engine="numba"`, SMS-EMOA runs each
steady-state step (one child) in two compiled calls for the child and one for
survival, instead of per-step NumPy and Python code. Parents and every
variation draw still come from the run generator in the reference order, and
the population, the cached ranks and crowding distances and the reference point
evolve bit for bit as in the NumPy reference. The step covers real encoding
with SBX, polynomial mutation and the clip repair (`repair="auto"`), random or
tournament parent selection, duplicate elimination, constraints, fixed or
adaptive reference points, external archives and live callbacks.
Two-objective contributions and constraint-violation sums are compiled only
after a one-time check against the installed reference (MooCore or the NumPy
fallback, and NumPy's summation order); three or more objectives keep the
reference contribution call. Other operators and repairs, genealogy tracking,
kernels that override the ranking or contribution hooks, a platform where
compiled `pow` differs from `np.power`, and non-finite objective values keep
the reference step. VAMOS 1.1.0 ran front insertion and contributions in Python
under every engine. **VAMOS 1.3.0:** jMetal's reference point rule
(`reference_point(rule="jmetal")`) also keeps the reference step, with the
same results under every engine.

Two compiled paths depend on whether Numba is installed rather than on the
engine: the MOEA/D neighbourhood update (disable with
`VAMOS_USE_NUMBA_MOEAD=0`) and SBX applied to a single mating pair (MOEA/D,
SMS-EMOA, steady-state NSGA-II). Both reproduce the NumPy results, so they
change speed, not trajectories. Compare engines in an environment without
Numba when you need a pure NumPy timing reference.

## Probability shorthand

- Operator probabilities accept either a numeric value or the string literal `"1/n"`.
- `"1/n"` resolves to `1.0 / n_var` in every algorithm's variation path. In VAMOS 1.1.0, AGE-MOEA and RVEA required the numeric equivalent; see below.
- Example: `NSGAIIConfig.builder().mutation("pm", prob="1/n", eta=20.0)`

**VAMOS 1.2.0:** every algorithm, AGE-MOEA and RVEA included,
resolves operator probabilities with one interpreter. It accepts a number in
[0, 1] or an expression `"k/n"`, which means `k / n_var` capped at 1; case and
spaces do not matter (`"1/N"`, `" 2 / n "`). Anything else, a number outside
[0, 1] included, raises `ValueError` when the algorithm builds its
operators. A mutation `prob_factor` sets `factor / n_var` and must be
positive. In VAMOS 1.1.0,
AGE-MOEA and RVEA failed on `"1/n"` after evaluating the initial population;
NSGA-II, IBEA and SMPSO rejected `"1/N"`; MOEA/D and the binary, integer and
permutation builders replaced an unreadable string such as `"0,5"` with
`1/n` and clipped numbers to [0, 1]; and `prob_factor` made NSGA-II fail.

**VAMOS 1.2.0:** the typed configurations check their values
whenever they are built (`builder().build()`, `from_dict()` or the
constructor) and raise `ConfigurationError`, a `ValueError`, before anything
is evaluated:

- operator probabilities (`prob`, `cr`) as above, `eta` finite and at least 0,
  `prob_factor` and `mutation_prob_factor` positive, and a tournament `size`
  that is a positive integer;
- population, archive, batch and partition sizes, `neighbor_size`,
  `replace_limit`, `k_neighbors` and `offspring_size` positive integers;
- IBEA's `kappa` finite and positive, MOEA/D's `delta` in [0, 1], RVEA's
  `alpha` at least 0 and `adapt_freq` in [0, 1], SMPSO's `inertia`, `c1` and
  `c2` at least 0 and `vmax_fraction` positive, `result_mode` either
  `non_dominated` or `population`, and a MOEA/D aggregation MOEA/D implements.

The MOEA/D builder also rejects a `neighbor_size` larger than `pop_size`.
`MOEADConfig.default()`, the CLI and the tuning space now use
`min(20, pop_size)` and `min(neighbor_size, pop_size)`, which is the
neighbourhood MOEA/D already used. `from_dict()` rejects unknown keys with a
suggestion (`pop_sise` → `pop_size`) instead of ignoring them, and restores the
operator pairs that JSON turns into lists, so
`from_dict(json.loads(cfg.to_json())) == cfg`. In VAMOS 1.1.0, SBX with
probability 1.7 or `eta=-5`, `kappa` at or below 0 (which inverts IBEA's
selection) and `delta=1.5` ran without warning, and a `result_mode` or
aggregation typo failed only after the initial population was evaluated.

**VAMOS 1.3.0:** `mutation_prob_factor` reaches every algorithm
whose configuration has it: MOEA/D, NSGA-III, SMS-EMOA, SPEA2, AGE-MOEA and
RVEA now set the mutation probability to `factor / n_var` (at most 1), as
NSGA-II and IBEA did. In VAMOS 1.2.0 those six ran with the mutation's own
`prob`. The fields that still have no effect warn; see
[stable configuration](../project/stability-and-versioning.md#stable-configuration).

For a measured first-call/subsequent-call example, follow [Measure backend performance](../tutorials/measure-performance.md).

## Comparative benchmarking

- Kernel-focused benchmarks: `python tools/benchmark_kernels.py --smoke --output artifacts/performance/kernel_smoke.json`
- VAMOS vs pymoo seeded comparisons: `python tools/benchmark_compare_pymoo.py --output artifacts/performance/pymoo_comparison.json --markdown artifacts/performance/pymoo_comparison.md`. **VAMOS 1.2.0:** install pymoo yourself first (`python -m pip install -c constraints/ci.txt pymoo` in a source checkout); no VAMOS extra installs it.

## Live visualization

The visualization surface is **Experimental**. Enable `--live-viz` to stream Pareto fronts during runs (`--live-viz-interval`, `--live-viz-max-points`). Saves a `live_pareto.png` at run end.

**VAMOS 1.2.0:** the live window plots the current population,
dominated and infeasible solutions included, and its title says so. It rescales
its axes at every update, labels them with the plotted objectives, and samples
the populations larger than `--live-viz-max-points` with a seeded generator
instead of the global `random` state. VAMOS 1.1.0 and earlier kept the axes of
the first generation, so a converging population soon moved out of view.
