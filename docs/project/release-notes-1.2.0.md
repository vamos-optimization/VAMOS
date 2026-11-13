# VAMOS 1.2.0 release notes

Released YYYY-MM-DD.

VAMOS 1.2.0 brings many built-in algorithms, operators, problems, reference
fronts, indicators and statistics back to their published definitions. These
scientific corrections change numerical results: read
[which results change](#which-results-change) before you compare new runs with
runs from earlier versions. The release also adds 115 constrained benchmark
problems, jMetal's constraint-handling techniques, faster Numba paths for the
steady-state algorithms, and stricter checks that reject inputs earlier
versions ignored or misapplied. The stable
1.x API, the omitted-engine NumPy default, and the 1.0.0 run and study schemas
remain the compatibility baseline.

The package metadata now serves users of VAMOS: the `research` extra and the
third-party framework integrations are removed, and `analysis` and `all`
install only what users need. VAMOS 1.2.0 supports Python 3.10 to 3.14 and is
declared Production/Stable; it is the last minor release that supports
Python 3.10.

## Which results change

A built-in component that departs from its published definition is a defect,
and its fix ships even when it changes results; see
[scientific corrections](stability-and-versioning.md#scientific-corrections).
The [changelog](https://github.com/vamos-optimization/VAMOS/blob/main/CHANGELOG.md)
lists each correction with its pull request, whose description gives the
measured effect. Rerun the experiments that use any of the following before
comparing them with results of earlier versions.

| Component | What changed | Results affected |
| --- | --- | --- |
| Default SBX | Probability 0.9 instead of 1.0 in NSGA-II, NSGA-III, SMS-EMOA, SPEA2, IBEA and RVEA, as in Deb et al. (2002), jMetal and pymoo | Runs with the default crossover; `.crossover("sbx", prob=1.0, eta=...)` reproduces 1.1.0 |
| IBEA | Default `kappa` 0.05; the epsilon indicator is normalized as in adaptive IBEA | Every IBEA run with the default `kappa` or the epsilon indicator |
| AGE-MOEA | Mating and survival follow the author's PlatEMO code | Every AGE-MOEA run |
| MOEA/D | Neighbours replaced in random order; exact resume from checkpoints | Every MOEA/D run, and resumed runs |
| NSGA-III | Complete Das-Dennis lattice without `divisions` (the CLI and specification default); N >= H; offspring offered to the archive | Runs without explicit `divisions`, and archives |
| RVEA | One survivor per occupied reference vector; parents drawn by permutation | Every RVEA run; populations can be smaller than `pop_size` |
| SMPSO | jMetal's coefficient sampling, personal-best rule and leader insertion; `optimize`, studies and `vamos tune` return the leaders archive by default | Every SMPSO run, and SMPSO tuning scores |
| SMS-EMOA | Normalized reference point; a fixed user `vector`; a generation is `pop_size` children; offspring offered to the archive | Default SMS-EMOA runs, HV-stopped runs and archives |
| SPEA2 | Lexicographic truncation and fitness-based filling, as in Zitzler et al. (2001); non-dominated result | Every SPEA2 run, and the `knn` archive pruning |
| Crowding and tournaments | An objective constant within a front is skipped; larger tournaments keep the best rank with NaN crowding; MooCore survival no longer overwrites earlier results | Runs in which these cases occur |
| Generations and budgets | Generation 0 is notified; NSGA-II immigration and the `obl` initializer count their evaluations | `hv_convergence` stops, callbacks, and runs with immigration or `obl` |
| Operators | `pcx`, `undx`, `simplex`, `linked_polynomial`, `power_law`, `insert`, `scramble`, `GradientRepair`, `ReflectRepair`, `WrappingRepair`, `scatter` and `obl` follow their definitions; overrides of a default operator keep its other parameters | Runs that use them or override a default operator |
| Problems | CEC2009 UF3 domain and UF9 distance; LSMOP1–4 and LSMOP9; RWA7 and RWA8; RE24 and RE25 domains; CF1 defaults to 10 variables | Results on these problems |
| Reference fronts | Regenerated ZDT, DTLZ, WFG1, WFG2, LSMOP9, C1-DTLZ3, UF6, UF9, DC2-DTLZ, RE, RWA3, RWA9 and RWA10 samples; fronts by number of objectives | IGD, IGD+, epsilon and normalized HV computed with them |
| Stopping | The `("hv", ...)` criterion, `--hv-threshold` and `hv_convergence` use fixed references | HV-stopped runs |
| Selection and constraints | `front()`, `best()` and `top_k()` rank feasible solutions first; archive and leader results keep their constraint values; `top_k()` defaults to `source="auto"`; `top_k(method="kmeans")` is deterministic | Selected solutions of constrained results |
| CLI configuration | `--nsga3-*` options, every algorithm-block field and partial `stopping` or `archive` overrides reach the run | CLI runs that set them |
| Statistics | Per-problem, feasible-only references in `vamos bench` and `vamos ablation`; ablation overrides applied; rank-sum tests; Nemenyi critical values; IGD+ of infeasible runs and merged fronts in Studio; the knee point; `vamos profile` timings and HV | Benchmark and ablation tables, critical-difference diagrams, Studio comparisons, knee selections and profiles |
| Archives and indicators | Exact hypervolume contributions without MooCore; explicit reference points; exact deduplication; SPEA2 truncation in `knn` pruning; the archive hook stops re-offering pruned rows | Pruned archives, and HV values without MooCore or without a reference point |
| Tuning | Racing's final choice among survivors and elitist restarts; single-fidelity model backends; SMAC3 and BOHB budgets and incumbents; Optuna pruning; failed evaluations; `best_score` | Tuning results, budgets and summaries |
| Run provenance | Wheels inside Git projects, editable installs, packaged data, user problems, re-saved results and CLI requests are recorded correctly | Task IDs and verification outcomes of those runs |

## New problems and APIs

- 115 constrained problems: LIR-CMOP1–14, DAS-CMOP1–9, CEC 2009 CF2–CF10,
  ZXH-CF1–16, the CRE and RCM real-world suites, C3-DTLZ4, and eight classic
  problems as jMetal defines them. The [problems catalog](../reference/problems.md)
  lists them with the origin of every packaged front.
- `OptimizationResult.feasibility()`, a `feasible_only` keyword on the
  selection helpers, and `hv_reached` in the results of every algorithm.
- NSGA-II and MOEA/D complete the ask/tell protocol with `initialize`,
  `should_terminate` and `result`, and every algorithm accepts the same
  `max_evaluations`, `n_gen` and `hv` termination criteria.
- The experimental tuning gains an `irace` backend through `Rscript`, and
  Studio's default LLM provider gets a `gemini` extra.
- NSGA-II, NSGA-III, SMS-EMOA and AGE-MOEA offer jMetal's constraint-handling
  techniques through `constraint_handling`, with matching `*_constrained`
  tuning spaces.

## Stricter validation

Typed algorithm configurations validate their values when they are built.
`optimize`, study planning and the CLI reject arguments, configuration keys,
dimensions, operators, checkpoints and undeclared constraint output that 1.1.0
ignored, applied in part or failed on later, and they check each algorithm's minimum population and number
of objectives before running. Stochastic components used directly without a
`numpy.random.Generator` raise `ConfigurationError` instead of drawing from an
unseeded one. Every error of the stable API derives from `VAMOSError`. Code
that passed such inputs to 1.1.0 now receives a `ConfigurationError` or a CLI
usage error that names the field to correct.

## Performance

With `engine="numba"`, the one-offspring steps of steady-state NSGA-II,
SMS-EMOA and MOEA/D run as compiled kernels, three to six times faster per
evaluation on ZDT1, with trajectories identical to the NumPy engine. The NumPy
engine sorts two-objective fronts in O(N log N), and `import vamos` loads the
experiment layer only on first use.

## Optional dependencies

- The `research` extra and the experimental third-party NSGA-II baselines
  (`--include-external`, `--external-problem-source`, and the `pymoo_nsga2`,
  `jmetalpy_nsga2` and `pygmo_nsga2` algorithm names) are removed. VAMOS no
  longer depends on another optimization framework; install one yourself to
  compare against it.
- `analysis` installs only what VAMOS imports (pandas, matplotlib, plotly and
  scikit-learn). Notebooks need `[analysis,examples]`.
- `all` installs the user extras without the `dev` and `docs` tooling.

## Python support

VAMOS 1.2.0 is tested on Python 3.10 to 3.14, including Python 3.13 and 3.14
on Linux, Windows and macOS, and its package metadata declares the development
status Production/Stable. Python 3.10 reached its end of life in October 2026:
VAMOS 1.2 is the last minor release that supports it, and VAMOS 1.3.0 will
require Python 3.11 or newer.

## Deprecations and removals

- `best("knee")` and `top_k(method="knee")` warn; use `balanced_sum`, the rule
  they always applied, or `vamos.ux.api.knee_point_scores`. The alias will be
  removed in VAMOS 2.0.
- `allow_ref_expand` is ignored with a `FutureWarning` in the HV termination
  criterion and in `hypervolume()`.
- Internal modules without callers and experimental tuning helpers without
  internal use are removed; none was part of the public API.

## Installation and remaining scope

```bash
python -m pip install --upgrade "vamos-optimization==1.2.0"
python -m pip install --upgrade "vamos-optimization[compute]==1.2.0"
```

Multi-worker study coordination, full-source typing debt, independent
conformance validation of RVEA and SMPSO, and the `welded_beam` variant remain
documented limitations. See [known limitations](known-limitations.md),
[stability and versioning](stability-and-versioning.md),
[the 1.1.0 release notes](release-notes-1.1.0.md), and
[release verification](../release_smoke.md).
