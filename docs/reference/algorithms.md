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

To query the installed registry, use `available_algorithms()` from `vamos.algorithms`; see [discovery](api/discovery.md). **All nine built-in identifiers and their public configuration classes are Stable** under the [1.x stability policy](../project/stability-and-versioning.md). Stable API status is a compatibility commitment, not evidence that every combination of algorithm, operator and problem has been validated.

## Capability matrix

This table describes the **VAMOS 1.1.0 package**. Changes since 1.0.0 are listed below. “R / B / I / P / M” means real, binary, integer, permutation and mixed encodings. An encoding requires compatible variation operators and, for mixed problems, the problem's variable specification; a real-coded default configuration does not convert a discrete problem.

| Algorithm | Encodings | Constraint-aware survival in 1.1.0 | Archive | Backends | API status |
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

**Evidence and scope.** The encoding families are exercised by repository operator-combination tests and encoding integration tests. Historical real-coded, unconstrained ZDT1 smoke runs were checked for all nine algorithms on each of the three backends against the installed 1.0.0 package. That is not a validation of the full encoding × constraints × archive × backend Cartesian product, nor a convergence or speed benchmark. Custom operators and untested combinations require a representative check before a large study. SMPSO rejects standalone binary, integer and permutation encodings; its mixed path requires mixed-variable mutation.

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

These are starting points for a study, not objective-count cut-offs or performance rankings. Compare algorithms using explicit budgets, repeated seeds and the same problem formulation.

<span id="algorithms-internal"></span>

## Results and cardinality

Use an explicit result mode when output cardinality matters. `population` returns the final population (or swarm); `non_dominated` selects a non-dominated result source. The latter need not have `pop_size` rows. NSGA-III's configuration builder defaults to `population`; the other builders default to `non_dominated`. The encoding-aware automatic setup and explicit builders need not select the same result mode, so report the resolved configuration.

For algorithms with a supported external result archive, enabling it makes that archive the default result source unless `population` is requested explicitly. Inspect `result.data["population"]` and `result.data["archive"]` separately. An internal SPEA2 environmental archive or SMPSO leaders archive influences search; an external result archive serves a different purpose. See [stopping and archives](../experiment/stopping_and_archive.md).

NSGA-III and RVEA require a population compatible with the configured reference directions. MOEA/D requires one weight vector per population member. For a simplex lattice with `p` divisions and `m` objectives, use `comb(p + m - 1, m - 1)`; for three objectives and 12 divisions, that is **91**. Explicit incompatible cardinalities fail; they are not warning-only cases. Follow the tutorials before changing objective count or population size.

### Changes included in 1.0.1

- **SMPSO:** `archive_size` controls search leaders, `.external_archive(...)` configures a separate result archive, and explicit `population` mode returns the final swarm. Version 1.0.0 returned the leaders archive without honoring these independent result controls.
- **AGE-MOEA and RVEA:** constraint-aware survival and constraint-aligned results are included when constraint handling is enabled. Version 1.0.0 did not enforce constraints in survival. `constraint_mode="none"` remains an explicit opt-out.
- **RVEA:** continue to use a numeric mutation probability such as `1.0 / problem.n_var`. Its variation setup still does not resolve the `"1/n"` shorthand in 1.0.1.
- **NSGA-III:** the constraint limitation above remains in 1.0.1.

These are implementation fixes under the existing stability policy. Record the package version and, for a source checkout, its commit. See the [1.0.1 release notes](../project/release-notes-1.0.1.md) before comparing or replaying older experiments.

## Optional baselines

- PyMOO NSGA-II (real and permutation), jMetalPy NSGA-II (real and permutation), PyGMO NSGA-II.
- Enabled via `--include-external`; install with `python -m pip install "vamos-optimization[research]==1.1.0"`. These third-party integrations are **Experimental**, outside the stable VAMOS algorithm contract.

## Backends

- NumPy (default): vectorized CPU kernels.
- Numba: JIT acceleration for supported kernels (set `VAMOS_USE_NUMBA_VARIATION=1` for permutation/binary/integer variation).
- MooCore: accelerated kernels via `moocore`. Install optional backends with `python -m pip install "vamos-optimization[compute]==1.1.0"`; see [installation](../guide/installation.md).

### Backend roles

| Backend | Status | Best use |
|---------|--------|----------|
| `numpy` | Stable | Exact reference backend and deterministic default. |
| `numba` | Optional backend | JIT implementations of supported kernels; includes compilation cost on first use. |
| `moocore` | Optional backend | Native non-dominated sorting/hypervolume operations where available; not every algorithm step is accelerated. |

### Automatic engine selection

VAMOS 1.1.0 expands the automatic Numba preference to all nine built-in
algorithms. Version 1.0.2 applied it only to NSGA-II and MOEA/D.

| Request | Effective engine |
| --- | --- |
| Omitted / `None` | NumPy, preserving the stable 1.x default |
| `"auto"`, any of the nine built-in algorithms | Numba if its dependency and backend load and JIT is enabled; otherwise NumPy |
| `"auto"`, external/plugin/unknown algorithm | NumPy; no compatibility claim for custom algorithms |
| `"numpy"`, `"numba"`, `"moocore"` | Exactly the requested engine; unavailable engines raise an error |

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
| NSGA-II | Non-dominated ranking and crowding, survival, tournament selection, SBX, polynomial mutation | Steady-state incremental front maintenance for three or more objectives, feasibility ordering, archives |
| NSGA-III | Ranking and crowding, tournament selection, SBX, polynomial mutation | Normalization, association to reference directions, niching (it shuffles with the run generator inside the loop) |
| MOEA/D | Differential-evolution children, SBX, polynomial mutation, neighbourhood replacement | Parent sampling, repair, weight geometry |
| SMS-EMOA | SBX, polynomial mutation; ranking and tournament selection where configured | Front insertion, hypervolume contributions (MooCore when installed) |
| SPEA2 | SBX, polynomial mutation, k-th-distance truncation | Strength/raw fitness and density (vectorized NumPy) |
| IBEA | Tournament selection, SBX, polynomial mutation, indicator-based removal | Indicator matrix |
| SMPSO | Turbulence mutation, leaders-archive crowding truncation | Velocity/position updates (vectorized NumPy), deduplication |
| AGE-MOEA | Ranking and crowding, tournament selection, SBX, polynomial mutation, greedy survival scores | Front normalization and pairwise Minkowski distances |
| RVEA | SBX, polynomial mutation | Angle-penalized survival and reference-vector adaptation (vectorized NumPy) |

Compiled SBX and polynomial mutation are enabled only after a one-time check
that compiled `pow` matches `np.power` bit for bit on the platform (some NumPy
builds use vendor SIMD power). Otherwise the Numba engine keeps the NumPy
operators. Compiled ranking, survival and archive truncation fall back to NumPy
for non-finite objective values.

Two compiled paths depend on whether Numba is installed rather than on the
engine: the MOEA/D neighbourhood update (disable with
`VAMOS_USE_NUMBA_MOEAD=0`) and SBX applied to a single mating pair (MOEA/D,
SMS-EMOA, steady-state NSGA-II). Both reproduce the NumPy results, so they
change speed, not trajectories. Compare engines in an environment without
Numba when you need a pure NumPy timing reference.

## Probability shorthand

- Operator probabilities accept either a numeric value or the string literal `"1/n"`.
- `"1/n"` resolves to `1.0 / n_var` in supported variation paths. RVEA in 1.1.0 still requires the numeric equivalent, as noted above.
- Example: `NSGAIIConfig.builder().mutation("pm", prob="1/n", eta=20.0)`

For a measured first-call/subsequent-call example, follow [Measure backend performance](../tutorials/measure-performance.md).

## Comparative benchmarking

- Kernel-focused benchmarks: `python tools/benchmark_kernels.py --smoke --output artifacts/performance/kernel_smoke.json`
- VAMOS vs pymoo seeded comparisons: `python tools/benchmark_compare_pymoo.py --output artifacts/performance/pymoo_comparison.json --markdown artifacts/performance/pymoo_comparison.md`

## Live visualization

The visualization surface is **Experimental**. Enable `--live-viz` to stream Pareto fronts during runs (`--live-viz-interval`, `--live-viz-max-points`). Saves a `live_pareto.png` at run end.
