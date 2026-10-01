# Algorithms and backends

Choose a built-in algorithm through its public identifier in `optimize()`. Start with the [NSGA-II guide](../algorithms/nsgaii.md) for a complete executable example and an explanation of the result. The configuration pages below document exact interfaces; they are not performance rankings.

## Algorithm catalogue

| Algorithm | Identifier | Teaching guide | Configuration reference |
| --- | --- | --- | --- |
| NSGA-II | `nsgaii` | [Run and understand NSGA-II](../algorithms/nsgaii.md) | [NSGAIIConfig](api/algorithms/nsgaii.md) |
| NSGA-III | `nsgaiii` | [Reference directions and DTLZ2](../algorithms/nsgaiii.md) | [NSGAIIIConfig](api/algorithms/nsgaiii.md) |
| MOEA/D | `moead` | [Decomposition and ZDT1](../algorithms/moead.md) | [MOEADConfig](api/algorithms/moead.md) |
| SMS-EMOA | `smsemoa` | Use the configuration reference | [SMSEMOAConfig](api/algorithms/smsemoa.md) |
| SPEA2 | `spea2` | Use the configuration reference | [SPEA2Config](api/algorithms/spea2.md) |
| IBEA | `ibea` | Use the configuration reference | [IBEAConfig](api/algorithms/ibea.md) |
| SMPSO | `smpso` | Use the configuration reference | [SMPSOConfig](api/algorithms/smpso.md) |
| AGE-MOEA | `agemoea` | Use the configuration reference | [AGEMOEAConfig](api/algorithms/agemoea.md) |
| RVEA | `rvea` | Use the configuration reference | [RVEAConfig](api/algorithms/rvea.md) |

To query the installed registry, use `available_algorithms()` from `vamos.algorithms`; see [discovery](api/discovery.md). **All nine built-in identifiers and their public configuration classes are Stable** under the [1.x stability policy](../project/stability-and-versioning.md). Stable API status is a compatibility commitment, not evidence that every combination of algorithm, operator and problem has been validated.

## Capability matrix

This table describes the **VAMOS 1.0.1 package**. Changes since 1.0.0 are listed below. “R / B / I / P / M” means real, binary, integer, permutation and mixed encodings. An encoding requires compatible variation operators and, for mixed problems, the problem's variable specification; a real-coded default configuration does not convert a discrete problem.

| Algorithm | Encodings | Constraint-aware survival in 1.0.1 | Archive | Backends | API status |
| --- | --- | --- | --- | --- | --- |
| NSGA-II | R / B / I / P / M | Feasibility-based | Optional external result archive | NumPy / Numba / MooCore | Stable |
| NSGA-III | R / B / I / P / M | **No**; retains `G`, but survival ranks objectives | Optional external result archive | NumPy / Numba / MooCore | Stable; constraint limitation |
| MOEA/D | R / B / I / P / M | Feasibility-based replacement | Optional external result archive | NumPy / Numba / MooCore | Stable |
| SMS-EMOA | R / B / I / P / M | Feasibility-based | Optional external result archive | NumPy / Numba / MooCore | Stable |
| SPEA2 | R / B / I / P / M | Feasibility-based | Internal environmental archive; optional external result archive | NumPy / Numba / MooCore | Stable |
| IBEA | R / B / I / P / M | Feasibility-based | Optional external result archive | NumPy / Numba / MooCore | Stable |
| SMPSO | R / M | Feasibility-based leaders/personal-best selection | Internal leaders archive; optional external result archive | NumPy / Numba / MooCore | Stable |
| AGE-MOEA | R / B / I / P / M | Feasibility-based when enabled | Optional external result archive | NumPy / Numba / MooCore | Stable |
| RVEA | R / B / I / P / M | Feasibility-based when enabled | Optional external result archive | NumPy / Numba / MooCore | Stable |

**Evidence and scope.** The encoding families are exercised by repository operator-combination tests and encoding integration tests. Historical real-coded, unconstrained ZDT1 smoke runs were checked for all nine algorithms on each of the three backends against the installed 1.0.0 package. That is not a validation of the full encoding × constraints × archive × backend Cartesian product, nor a convergence or speed benchmark. Custom operators and untested combinations require a representative check before a large study. SMPSO rejects standalone binary, integer and permutation encodings; its mixed path requires mixed-variable mutation.

Constraints use `G <= 0`, with positive residuals representing violation. Consult the [constraint reference](constraints.md) for residual construction; an accepted `constraint_mode` field alone does not establish constraint enforcement. In particular, NSGA-III's environmental selection uses objective values even though it carries constraint arrays.

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
- Enabled via `--include-external`; install with `python -m pip install "vamos-optimization[research]==1.0.1"`. These third-party integrations are **Experimental**, outside the stable VAMOS algorithm contract.

## Backends

- NumPy (default): vectorized CPU kernels.
- Numba: JIT acceleration for supported kernels (set `VAMOS_USE_NUMBA_VARIATION=1` for permutation/binary/integer variation).
- MooCore: accelerated kernels via `moocore`. Install optional backends with `python -m pip install "vamos-optimization[compute]==1.0.1"`; see [installation](../guide/installation.md).

### Backend roles

| Backend | Status | Best use |
|---------|--------|----------|
| `numpy` | Stable | Exact reference backend and deterministic default. |
| `numba` | Optional backend | JIT implementations of supported kernels; includes compilation cost on first use. |
| `moocore` | Optional backend | Native non-dominated sorting/hypervolume operations where available; not every algorithm step is accelerated. |

## Probability shorthand

- Operator probabilities accept either a numeric value or the string literal `"1/n"`.
- `"1/n"` resolves to `1.0 / n_var` in supported variation paths. RVEA in 1.0.1 still requires the numeric equivalent, as noted above.
- Example: `NSGAIIConfig.builder().mutation("pm", prob="1/n", eta=20.0)`

## Comparative benchmarking

- Kernel-focused benchmarks: `python tools/benchmark_kernels.py --smoke --output artifacts/performance/kernel_smoke.json`
- VAMOS vs pymoo seeded comparisons: `python tools/benchmark_compare_pymoo.py --output artifacts/performance/pymoo_comparison.json --markdown artifacts/performance/pymoo_comparison.md`

## Live visualization

The visualization surface is **Experimental**. Enable `--live-viz` to stream Pareto fronts during runs (`--live-viz-interval`, `--live-viz-max-points`). Saves a `live_pareto.png` at run end.
