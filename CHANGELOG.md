# Changelog

All notable public changes to VAMOS are documented here.

## [Unreleased]

### Fixed

- IBEA scales its epsilon indicator as adaptive IBEA does (Zitzler and Künzli,
  2004), as PISA and jMetal do, and as VAMOS 1.1.0 already scaled the
  hypervolume indicator. The objectives are normalized over the population, and
  the indicator is divided by its largest magnitude.
  - **What went wrong unscaled:** an objective with a much wider range than the
    others decided every epsilon value, and the population collapsed toward
    that objective's minimum. On RE21 (f1 about 10^3, f2 about 10^-2), the
    default IBEA ended with a single distinct solution.
  - **New guarantees:** a positive affine change of the units of any objective
    leaves IBEA's selection unchanged, and fitness exponents stay within
    `[-1/kappa, 1/kappa]`.
  - **Effect on results:** epsilon-based runs differ from 1.1.0 runs with the
    same seed. As with the hypervolume indicator, far-dominated solutions
    widen the normalization range, which slows convergence on multimodal
    problems such as ZDT4 and DTLZ1.

## [1.1.0] - 2026-10-23

### Added

- A Studio **Statistical Comparison** page: per-problem Mann-Whitney tests
  against a pivot algorithm with Holm or Benjamini-Hochberg correction and
  Vargha-Delaney A12 effect sizes, Friedman average ranks with a Nemenyi
  critical-difference diagram and Holm post-hoc tests, distributions, median
  fronts, empirical attainment surfaces, runtimes, and LaTeX, CSV, and SAES
  exports, for hypervolume and, where a reference front exists, IGD+.
- A Studio **Problems & Algorithms** page lists the registered problems with
  their encoding, dimensions and shipped reference fronts, and the configurable
  parameters of each algorithm with their VAMOS defaults, and opens either in
  Run Experiment.
- Studio's sidebar shows the VAMOS and Python versions, the installed kernel
  backends, and which optional extras are installed.
- Explore Results filters fronts to feasible solutions and to objective ranges,
  and opens a stored run's settings in Run Experiment to run it again.
- Run Experiment lets you choose the number of variables and, where the
  problem allows it, the number of objectives.

### Changed

- Explicit `engine="auto"` now prefers loadable, JIT-enabled Numba for all nine
  built-in algorithms. Missing or incompatible optional installations fall back
  to NumPy before execution. The stable omitted-engine default remains NumPy;
  explicit backend requests and stored replay/study backends never substitute.
- Canonical requested configurations retain an explicit `engine="auto"` while
  resolved configurations continue to store the effective backend. Schemas and
  optional compute requirements are unchanged.
- Studio's Batch Comparison creates a durable study for each batch and runs it
  in a background process that survives closing the browser or restarting
  Studio. The page checks the plan before creating the study, follows its
  progress with an estimate of the time left, stops, resumes and retries it,
  lists the batches and studies of the Studio folder, and opens a finished
  batch in Explore Results.
- Studio's Run Experiment and Problem Builder no longer block the Studio server
  while they run. Run Experiment shows its progress, generation, and front as
  the run advances. Both pages have a **Stop** button, and leaving a page stops
  its run.

### Fixed

- Studio stop requests address a unique worker token; workers exit themselves,
  so a reused process ID cannot target an unrelated process. Wildcard remote
  binds accept the computer's host names and addresses, and live runs record
  the final generation once.
- Preview timeouts retain their timeout diagnosis when the worker exits at the
  deadline or receives the operating system's CPU-limit signal; unrelated
  crashes before the deadline remain distinct worker errors.
- Stopping a Studio batch unwinds active file writes before acknowledging the
  stop, so temporary attempt files do not prevent loading and resuming it.
- The registered `cec2009_cf1` and `welded_beam` problems declare their
  constraints, so algorithms enforce them; they used to run as unconstrained
  problems. CF1 also stores its CEC2009 constraint with the VAMOS sign
  (`G <= 0` is feasible; it was inverted) and uses the CEC2009 domain
  `[0, 1]^n` instead of `[-1, 1]` for variables 2 to n.
- IBEA's constraint penalty raised the fitness of infeasible solutions, so its
  environmental selection removed feasible solutions first. Infeasible
  solutions are now removed first, in decreasing order of violation.
- IBEA with `indicator="hypervolume"` used the hypervolume-difference indicator
  with the opposite orientation, so it removed the solutions with the largest
  exclusive hypervolume first and kept dominated ones; on ZDT1 and DTLZ2 its
  final populations never reached the reference box. It now follows Zitzler and
  Künzli's `I_HD` with adaptive scaling, as jMetal does, and matches jMetal's
  IBEA on ZDT1, DTLZ2 and DTLZ1. Rerun hypervolume-based IBEA experiments from
  earlier versions.
- IBEA counted an exponential indicator contribution beyond the float64 range
  as zero, which made a solution dominated by a wide margin look best (with
  `kappa=0.05`, once another solution was better by more than about 35.5 in
  every objective). Selection then works in the log domain, recomputing each
  fitness after every removal and keeping feasible solutions ahead of
  infeasible ones, so overflow no longer changes which solutions survive; runs
  without overflow are unchanged.
- NSGA-III enforces constraints: its survival keeps feasible solutions before
  infeasible ones and its parent tournament ranks them first, instead of
  ranking objective values only.
- With `result_mode="non_dominated"`, NSGA-II, NSGA-III, MOEA/D, SMS-EMOA,
  IBEA and SPEA2 return the non-dominated feasible solutions whenever the final
  population has any, as AGE-MOEA, RVEA and SMPSO already did, instead of
  returning infeasible solutions that dominate feasible ones.
- `constraint_mode` rejects values other than `"feasibility"` and `"none"`.
  `"penalty"`, `"epsilon"` and any other value were accepted and behaved exactly
  like `"feasibility"` in every algorithm.
- SPEA2 runs with `constraint_mode="none"` on constrained problems instead of
  failing with an `IndexError` after the first generation.
- The model-based tuning backends `optuna`, `bohb_optuna`, `smac3` and `bohb`
  honor conditional parameters, as `random` and `racing` already did for a
  parameter's own conditions. They
  suggested every parameter, so trials recorded values that had no effect and
  SMAC3 and BOHB modelled inactive dimensions. They now sample a parameter
  only when its conditions hold, resolve parents first, and give SMAC3 and BOHB
  native ConfigSpace conditions; unsupported conditions, unknown parents and
  dependency cycles fail before sampling. In real-valued operator spaces
  `mutation_eta` is active only for polynomial mutations, so a seeded MOEA/D
  `random` or `racing` search proposes different candidates than 1.0.2.
- Native BOHB tuning starts without an unsupported `random_state` argument and
  sends plain Python values to its workers. Its configuration generator draws
  from a private stream derived from the tuning seed instead of NumPy's global
  state, so a single-worker search is reproducible and the caller's global
  random stream is left untouched.
- `random` and `racing` no longer return a parameter whose parent is inactive.
  They still draw every value in declaration order, so their seeded values do
  not change.
- Refresh the reviewed dependency set, the canonical mypy 2.4.0 environment,
  and its exact diagnostic baseline. Pair PyMdown 12.3 with mkdocstrings 1.0.8
  on Python 3.11+, retaining the compatible documentation pair on Python 3.10.
- NumPy's optional single-pair SBX and MOEA/D neighbourhood shortcuts tolerate
  unloadable Numba installations, so they do not defeat automatic fallback.
- `vamos studio` serves the Studio application again instead of an empty page,
  and honours `--study-dir`.
- Studio algorithm settings now apply to runs: the controls come from each
  algorithm's tuning configuration space for the problem encoding, only the
  fields you change are applied on top of the VAMOS default configuration, and
  every difference is listed before the run.
- Studio runs NSGA-III, RVEA, and MOEA/D with three or more objectives with the
  population size their reference directions require.
- The Problem Builder's full experiment runs the problem you defined, templates
  bring their bounds and constraints, and generated scripts include constraints
  and any customized configuration.
- Batch Comparison reports a normalized hypervolume instead of the largest
  first-objective value, lists every failed run, and exports CSV and LaTeX
  through the browser.
- Explore Results reports studies it cannot load instead of showing demo data,
  overlays fronts per problem for any number of objectives, and exports the
  ranking it displays.
- The Problem Builder API-key field is used for the OpenAI and Anthropic
  providers.
- Studio's **Run Experiment** page is reachable from the navigation.

## [1.0.2] - 2026-10-18

### Added

- With `engine="numba"`, the built-in SBX crossover and polynomial mutation
  operators use compiled arithmetic in every algorithm that builds them
  (NSGA-II, NSGA-III, MOEA/D, SMS-EMOA, SPEA2, IBEA, SMPSO turbulence,
  AGE-MOEA and RVEA). Results are bitwise identical to the NumPy operators.
- With `engine="numba"`, the iterative survival loops of AGE-MOEA (greedy
  diversity scores), SPEA2 (k-th-distance truncation), IBEA (indicator-based
  removal) and the crowding-distance archive (SMPSO leaders and crowding
  external archives) run as compiled kernels that reproduce the NumPy loops
  exactly, including first-index tie-breaking. Non-finite inputs keep the
  NumPy path.

### Changed

- The Numba kernel ranks populations with three or more objectives by
  sequential-search non-dominated sorting over a lexicographic presort, and
  NSGA-II survival computes crowding only for the front it truncates. Ranks,
  crowding and survivor order are unchanged.
- NSGA-II reuses the generation-end ranking of an unchanged population for the
  next parent selection, and constrained selection no longer computes an
  unconstrained ranking it then discards. Trajectories are unchanged.
- MOEA/D samples the parents of a single offspring without vectorized
  bookkeeping, builds genealogy parent indices only when genealogy is tracked,
  and, with `engine="numba"`, computes its differential-evolution children with
  a compiled kernel fed by the same random draws. Trajectories are unchanged.
- The documentation is reorganized into learning paths, adds fundamental,
  study-to-report and per-algorithm tutorials with recorded outputs, and
  lists what the Numba engine compiles for each algorithm.

### Fixed

- Single-mating-pair SBX, which uses compiled arithmetic whenever Numba is
  installed, now consumes random draws exactly like the NumPy implementation.
  Steady-state runs (MOEA/D, SMS-EMOA, steady-state NSGA-II) no longer depend
  on whether Numba is installed when parents are identical.
- `NumbaKernel.nsga2_survival` returns arrays owned by the caller. It used to
  return views of reusable buffers, so a later call with the same kernel (for
  example a second run of the same algorithm object) overwrote previously
  returned populations and results.
- `NumbaKernel.polynomial_mutation` reproduces NumPy for `eta == 1`, and
  `NumbaKernel.sbx_crossover` now uses compiled arithmetic instead of
  delegating to NumPy.
- The Numba binary tournament now draws tie-break coins exactly like the
  NumPy reference, one coin per fully tied tournament. Algorithms that use
  tournament selection (NSGA-II, NSGA-III, IBEA, AGE-MOEA, and SMS-EMOA with
  tournament selection) follow the same trajectory with `engine="numba"` and
  `engine="numpy"` for the same seed; Numba-engine runs therefore differ from
  VAMOS 1.0.1 runs.

## [1.0.1] - 2026-10-01

### Fixed

- NSGA-II now applies its configured steady-state mode, including the Numba
  path, instead of silently retaining generational behavior.
- SMPSO honors `result_mode` and the external result archive independently
  from its internal leader archive and swarm dynamics.
- AGE-MOEA and RVEA honor constraint handling during selection and survival.
- CLI tuning compares hypervolume from a consistent final-population source,
  checks constrained-algorithm support before starting, and separates persisted
  Optuna studies using the corrected scoring contract.
- The experimental Dask evaluation backend discovers an active client and
  preserves explicit failure and serial-fallback behavior.

### Changed

- Installation, troubleshooting, cookbook recipes, algorithm capabilities,
  problem descriptions, and MOEA/D and NSGA-III tutorials were checked and
  expanded; the documentation homepage includes real optimization results.
- Release preparation supports subsequent patch versions while preserving
  published annotated tags and validating one frozen set of distribution bytes
  through TestPyPI and PyPI.
- Canonical typing gates use the reviewed mypy 2.3.1 environment; CI actions
  and release tooling include the updates merged since 1.0.0.

### Removed

- The retired legacy multilingual documentation site under `/website/`, and
  the `mkdocs-static-i18n` dependency that only it used, from the `docs`,
  `dev`, and `all` extras.
- Internal audit pages from the published documentation, including the
  carried-forward 1.0.0 documentation archive.
- The architecture decision records (`docs/dev/adr/`) and their published
  Architecture Decisions section, including from the 1.0.0 documentation
  archive, whose links to them are removed. Their canonical rules are
  maintained in Architecture Health and the linked contracts.

## [1.0.0] - 2026-09-06

VAMOS 1.0.0 is the first official public release and the first compatibility
baseline. Earlier version strings and Git tags were internal pre-public
development markers; they are not prior public releases.

### Added

- A stable top-level Python API for optimization, built-in problems, and typed
  algorithm configuration objects.
- Nine built-in multi-objective algorithms: AGE-MOEA, IBEA, MOEA/D, NSGA-II,
  NSGA-III, RVEA, SMPSO, SMS-EMOA, and SPEA2.
- NumPy as the deterministic reference backend, with optional Numba kernel
  acceleration and MooCore indicator acceleration.
- A canonical `RunManifest` lifecycle through `save_result`, `load_run`,
  `load_result`, `verify_run`, and exact same-environment built-in `reproduce`.
- A canonical `StudyManifest` lifecycle through `StudySpec`, `plan_study`,
  `create_study`, `load_study`, and the `Study.run`, `inspect`, `summarize`,
  `cancel`, `resume`, and `retry` operations.
- Persisted failure policies, cooperative cancellation, interrupted-state
  reconciliation, bounded retry, `StudyReport`, and `StudySummary` projections.
- Stable CLI contracts for top-level optimization, run inspection,
  verification and replay, and durable study planning and execution.
- Guides for installation, first optimization, the run lifecycle, replay,
  durable studies, Studio's trust boundary, stability, and known limitations.
- Human contributor guidance and scoped `AGENTS.md` instructions for automated
  coding agents working within the repository architecture.

### Changed

- Package metadata, runtime version, citation data, documentation, built
  artifacts, and examples now derive from the single `1.0.0` release version.
- The public compatibility policy now distinguishes stable, experimental, and
  internal surfaces and defines the 1.x deprecation and security policies.
- Studio is explicitly experimental. It binds to loopback by default, requires
  an explicit remote-binding opt-in, and requires per-code-revision consent
  before reviewed Python is executed as trusted local code.

### Fixed

- Evaluation-budget, seed ownership, backend selection, and artifact integrity
  behavior are covered by focused compatibility and replay tests.
- Loading, inspection, and verification reject unsafe paths, unsafe NPZ data,
  malformed manifests, and excessive input before executing recorded code.
- Study planning and recovery preserve deterministic task identity and a
  single-writer journal across interruption, retry, and resume.

### Compatibility and known limitations

- Semantic-versioning commitments begin with this release. No reader or shim
  is provided for internal pre-public artifacts.
- Exact replay is limited to reconstructable built-in components in the same
  materially relevant environment and backend; it does not install missing
  dependencies or execute recorded custom Python.
- Durable study mutation is single-owner. Concurrent writers, distributed
  study execution, and cross-process cancellation are not supported in 1.0.0.
- Cross-backend bitwise equality is not promised. Optional dependencies and
  third-party integrations may have narrower platform support.
- Studio, tuning, visualization, provider integrations, plugin interfaces, and
  non-stable CLI commands remain experimental.

[Unreleased]: https://github.com/vamos-optimization/VAMOS/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.1.0
[1.0.2]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.0.2
[1.0.1]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.0.1
[1.0.0]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.0.0
