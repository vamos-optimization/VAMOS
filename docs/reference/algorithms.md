# Algorithms and backends

Choose a built-in algorithm through its public identifier in `optimize()`. Start with the [NSGA-II guide](../algorithms/nsgaii.md) for a complete executable example and an explanation of the result. The configuration pages below document exact interfaces; they are not performance rankings.

## Algorithm catalogue

| Algorithm | Identifier | Teaching guide | Configuration reference |
| --- | --- | --- | --- |
| NSGA-II | `nsgaii` | [Run and understand NSGA-II](../algorithms/nsgaii.md) | [NSGAIIConfig](api/algorithms/nsgaii.md) |
| NSGA-III | `nsgaiii` | Use the configuration reference | [NSGAIIIConfig](api/algorithms/nsgaiii.md) |
| MOEA/D | `moead` | Use the configuration reference | [MOEADConfig](api/algorithms/moead.md) |
| SMS-EMOA | `smsemoa` | Use the configuration reference | [SMSEMOAConfig](api/algorithms/smsemoa.md) |
| SPEA2 | `spea2` | Use the configuration reference | [SPEA2Config](api/algorithms/spea2.md) |
| IBEA | `ibea` | Use the configuration reference | [IBEAConfig](api/algorithms/ibea.md) |
| SMPSO | `smpso` | Use the configuration reference | [SMPSOConfig](api/algorithms/smpso.md) |
| AGE-MOEA | `agemoea` | Use the configuration reference | [AGEMOEAConfig](api/algorithms/agemoea.md) |
| RVEA | `rvea` | Use the configuration reference | [RVEAConfig](api/algorithms/rvea.md) |

Additional teaching pages will be linked when their examples have been validated. To query the installed registry, use `available_algorithms()` from `vamos.algorithms`; see [discovery](api/discovery.md). Compatibility commitments are defined in the [stability policy](../project/stability-and-versioning.md).

<span id="algorithms-internal"></span>

## Implementation notes

- NSGA-II: continuous, permutation, binary, integer, mixed; supports archive, adaptive operators, HV early-stop.
  - `result_mode` accepts only `non_dominated` (default) or `population`.
  - External archive configuration (`.external_archive(...)`) becomes the default result source unless you explicitly set `result_mode="population"`.
  - When archive is enabled, results still include `result.data["archive"]` alongside `result.data["population"]`.
  - Supported external-archive prune policies are `crowding`, `hv`, `mc_hv`, `knn`, `maxmin`, and `ref_dirs`.
  - `hv` uses exact hypervolume contributions in 2D and exact higher-dimensional contributions when `moocore` is available; `mc_hv` keeps the Monte Carlo approximation path.
- NSGA-III: many-objective real/binary/integer; reference direction support. Matching `pop_size` to the number of reference directions is recommended (with divisions p: `comb(p + n_obj - 1, n_obj - 1)`); mismatches emit a warning unless strict enforcement is enabled.
- MOEA/D: real/binary/integer; aggregation methods (tchebycheff, weighted sum, pbi). Defaults align with jMetalPy (PBI aggregation, DE crossover CR=1.0/F=0.5, packaged weight vectors for n_obj > 2).
  - `result_mode` accepts `non_dominated` (default) or `population`.
  - External archive configuration (`.external_archive(...)`) becomes the default result source unless you explicitly set `result_mode="population"`.
- SMS-EMOA: real/binary/integer; adaptive reference points.
  - `result_mode` accepts `non_dominated` (default) or `population`.
  - External archive configuration (`.external_archive(...)`) becomes the default result source unless you explicitly set `result_mode="population"`.
- SPEA2: real/binary/integer with constraint handling.
- IBEA: epsilon or hypervolume indicator variants.
- SMPSO: real-coded, archive support.
- AGE-MOEA: adaptive geometry estimation for many-objective search.
  - `result_mode` accepts `non_dominated` (default) or `population`.
  - External archive configuration (`.external_archive(...)`) becomes the default result source unless you explicitly set `result_mode="population"`.
- RVEA: reference-vector guided many-objective search.
  - `result_mode` accepts `non_dominated` (default) or `population`.
  - External archive configuration (`.external_archive(...)`) becomes the default result source unless you explicitly set `result_mode="population"`.

Optional baselines (install extras)
-----------------------------------

- PyMOO NSGA-II (real and permutation), jMetalPy NSGA-II (real and permutation), PyGMO NSGA-II.
- Enabled via `--include-external` and extras `research`.

Backends
--------

- NumPy (default): vectorized CPU kernels.
- Numba: JIT acceleration for supported kernels (set `VAMOS_USE_NUMBA_VARIATION=1` for permutation/binary/integer variation).
- MooCore: accelerated kernels via `moocore` (install `compute` extra).

Backend capability matrix
-------------------------

| Backend | Status | Best use |
|---------|--------|----------|
| `numpy` | Stable | Exact reference backend and deterministic default. |
| `numba` | Stable optional | Faster core kernels: mutation, tournament selection, and MOEA/D neighborhood updates. |
| `moocore` | Stable optional | Hypervolume and related quality-indicator acceleration. |

Probability shorthand
---------------------

- Operator probabilities accept either a numeric value or the string literal `"1/n"`.
- `"1/n"` resolves to `1.0 / n_var` at runtime and is the recommended mutation default for many encodings.
- Example: `NSGAIIConfig.builder().mutation("pm", prob="1/n", eta=20.0)`

Comparative benchmarking
------------------------

- Kernel-focused benchmarks: `python tools/benchmark_kernels.py --smoke --output artifacts/performance/kernel_smoke.json`
- VAMOS vs pymoo seeded comparisons: `python tools/benchmark_compare_pymoo.py --output artifacts/performance/pymoo_comparison.json --markdown artifacts/performance/pymoo_comparison.md`

Live visualization
------------------

Enable `--live-viz` to stream Pareto fronts during runs (`--live-viz-interval`, `--live-viz-max-points`). Saves a `live_pareto.png` at run end.
