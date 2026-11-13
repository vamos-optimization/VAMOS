# Changelog

All notable public changes to VAMOS are documented here.

## [Unreleased]

## [1.2.0] - 2026-11-14

VAMOS 1.2.0 corrects built-in algorithms, operators, problems, reference
fronts, indicators and statistics that departed from their published
definitions. Under the [scientific-corrections
policy](docs/project/stability-and-versioning.md#scientific-corrections), these
corrections ship in a minor release even when they change numerical results:
rerun experiments that use anything under "New defaults that change results" or
"Corrections that change results" before comparing them with earlier versions.
Each entry names its pull request, whose description gives the measured effect;
the [release notes](docs/project/release-notes-1.2.0.md) summarize what to
rerun. Run and study artifacts keep the 1.0.0 schemas.

### Added

- Problems: 115 constrained problems, most with reference fronts: LIR-CMOP1–14,
  DAS-CMOP1–9, CEC 2009 CF2–CF10, ZXH-CF1–16, the CRE and RCM real-world
  suites, C3-DTLZ4 and eight classic jMetal problems (`binh2`, `constrex`,
  `srinivas`, `tanaka`, `osyczka2`, `golinski`, `water`, `car_side_impact`);
  fronts for C-DTLZ, DC-DTLZ and MW; `DCDTLZ` and `DASCMOP` parameters
  (#216–#221, #224).
- Results: `OptimizationResult.feasibility()`; `feasible_only` on `front()`,
  `best()`, `top_k()` and `top_k_report()` (`False` restores 1.1.0's
  selection); `hv_reached` in every algorithm's results (`hv_converged` stays
  an alias until 2.0); `StudyResult.std(name, ddof=0)`; Holm-adjusted
  `p_adjusted` and `significant` in `pairwise_wilcoxon`
  (#148, #175, #211, #216, #253).
- Algorithms: `initialize`, `should_terminate` and `result` on `NSGAII` and
  `MOEAD`, completing the ask/tell protocol; `optimize(..., termination=...)`
  without an `algorithm_config`; `VAMOS_DISABLE_PLUGINS=1` (#235, #253, #254).
- Tuning: an `irace` backend (R irace >= 4.4.4 through `Rscript`) for
  `ModelBasedTuner` and `vamos tune --backend irace` (#108).
- `vamos create-problem --force`, and a `gemini` extra for the Studio Problem
  Builder's default provider (#172, #231).

- NSGA-II's `constraint_handling`: jMetal's techniques `cdp` (the default),
  `violated_constraints`, `epsilon_constrained`, `push_pull`,
  `stochastic_ranking` and `self_adaptive_penalty`, which rank fixed
  populations as jMetal does, and the tuning space `nsgaii_constrained` with
  the ranges of Evolver's `NSGAIIDoubleConstrained` (#242).
- `constraint_handling` in NSGA-III, SMS-EMOA and AGE-MOEA, with the same six
  techniques, as in Evolver's bases of these algorithms, and the tuning spaces
  `nsgaiii_constrained`, `smsemoa_constrained` and `agemoea_constrained`
  (#256).
- `constraint_handling` in RVEA and SMPSO, with the four of these techniques
  that judge a single solution, as in Evolver, and the tuning spaces
  `rvea_constrained` and `smpso_constrained` (#257).
- MOEA/D's `subproblem_update_criterion`, with jMetal's criteria
  `feasibility_rules` (the default), `improved_epsilon` and
  `violation_threshold`, which decide as jMetal's do, and the tuning space
  `moead_constrained` (#258).

- `vamos --version` prints the installed version (#273).

### Changed

#### New defaults that change results

- SBX probability 0.9 instead of 1.0 in NSGA-II, NSGA-III, SMS-EMOA, SPEA2,
  IBEA and RVEA, as in Deb et al. (2002), jMetal and pymoo; `.crossover("sbx",
  prob=1.0, eta=...)` restores 1.1.0 (#184).
- IBEA's default `kappa` is 0.05, the value of Zitzler and Künzli (2004),
  instead of 1.0 (#115).
- SMS-EMOA's reference point is `nadir + offset * (nadir - ideal)`, and HV
  archive pruning without `hv_ref_point` uses `nadir + (nadir - ideal)`, so
  neither depends on objective units (#193).
- NSGA-III accepts N >= H; with `pop_size`, NSGA-III defaults to the largest
  complete lattice that fits and RVEA to the lattice of exactly `pop_size`
  vectors; integer IBEA, AGE-MOEA and RVEA runs use the encoding table's
  SBX(0.9, 20) (#130, #162).
- `cec2009_cf1` defaults to the CEC2009 dimension, 10 variables instead of 30
  (#186).
- The `simplex` crossover's `epsilon` defaults to `sqrt(m + 1)` instead of 0.5
  (#156).
- Archives deduplicate objectives exactly (`objective_tolerance=0.0` instead of
  `1e-10`), which matters only below a scale of about 1e-9 (#143).
- Without fidelity levels, model-based tuning evaluates each trial once at the
  full budget, and `vamos tune --backend racing` races statistically unless
  `--multi-fidelity` is given (#128, #155).

#### Stricter validation

VAMOS 1.1.0 ignored or misapplied these inputs; adjust code that relied on
that.

- Algorithm configurations validate their values when built, `from_dict()`
  rejects unknown keys, and IBEA rejects unknown `indicator` names (`"hv"` ran
  the epsilon indicator) (#167).
- `optimize` rejects arguments it ignored (dimensions with a problem instance,
  `max_evaluations` contradicting `termination`, another algorithm's
  configuration, `GenericAlgorithmConfig` for a built-in algorithm, empty seed
  lists) and checks the problem and each algorithm's minimum population and
  objective count; study plans reject unknown `algorithm_configs` keys,
  parallel evaluation and negative seeds (#146, #203, #213, #253).
- Mismatched operators and initializers, unreadable probabilities, impossible
  or ignored dimensions, flat objective arrays and undeclared constraint output
  fail early instead of running something else (#150, #152, #177, #186, #203).
- Budgets must be positive integers, all algorithms accept `max_evaluations`,
  `n_gen` and `hv`, and unusable checkpoints raise instead of restarting
  silently; stochastic components used directly need a `numpy.random.Generator`
  (#198, #211, #265).
- The CLI rejects invalid values, duplicate keys, dropped algorithm-block
  fields and unbuildable `stopping` or `archive` blocks before any run, and
  `--validate-config` runs every check (#159, #169, #181, #202).
- HV indicator classes require a `reference_point`, and `hypervolume()` ignores
  points outside it instead of raising (#187).
- Every stable-API error derives from `VAMOSError`; an unknown problem raises
  `InvalidProblemError`, still a `KeyError` (#234).
- When every tuning evaluation fails, the tuners raise `TuningFailedError` and
  `vamos tune` exits with 1 (#208).
- An algorithm built without a kernel uses NumPy, with the same trajectory
  (#237).

#### Performance and tooling

- With `engine="numba"`, steady-state NSGA-II, SMS-EMOA and MOEA/D run compiled
  one-offspring steps, three to six times faster and bit-identical to NumPy;
  NumPy sorts two-objective fronts in O(N log N); `import vamos` is lazy;
  studies capture the environment once (#120–#122, #238, #250, #262).
- Packaging and documentation: no loose tests in the sdist; documentation trees
  are kept as release assets and GitHub Pages redirects to
  `vamos-optimization.org`; the Studio guide documents its LLM providers
  (#119, #231, #260).
- Typing uses mypy 2.4.1 with a baseline tied to type-affecting pins, and
  constraints, CI actions and release tooling include the updates merged since
  1.1.0 (#118).

- Python 3.13 and 3.14 are tested and declared, and the status is
  "Production/Stable". VAMOS 1.2 is the last minor release for Python 3.10;
  1.3.0 will require 3.11 or newer (#266).

- `analysis` installs only pandas, matplotlib, plotly and scikit-learn;
  notebook packages move to `examples`, so notebooks need
  `[analysis,examples]`. `all` installs the user extras without the `dev` and
  `docs` tooling (#268).

<!-- PENDING(UX-25 #276, fix/studio-small-fixes): drop this block if that pull request is not merged. -->
- `plot_critical_distance` no longer accepts `higher_is_better`, which had no
  effect (#276).
<!-- /PENDING -->

- Documentation: the README works as the PyPI description, the CLI guide says
  printed hypervolumes compare only within one command, the study guides' retry
  recipe checks for failed work first, and the navigation gains Reference →
  Support & Versions (#267).

### Deprecated

- `best("knee")` and `top_k(method="knee")` warn: `knee` always meant
  `balanced_sum`; the alias goes in 2.0 (#148).
- `allow_ref_expand` is ignored with a `FutureWarning` in HV termination and
  `hypervolume()`; it goes in 2.0 (#129, #187).

### Removed

- Unused internal modules, none public: `vamos.engine.hyperheuristics`,
  `vamos.engine.adaptation`, `vamos.engine.algorithm.components.base`,
  `vamos.experiment.study.nsga2_diagnostics`, `vamos.experiment.studio`,
  `vamos.foundation.core.api`, the `vamos.foundation.constraints` strategies,
  `vamos.ux.mcdm` and `vamos.ux.stats` (#160).
- Unused experimental tuning helpers: `vamos.engine.tuning.convergence`, racing
  `save_checkpoint`/`load_checkpoint`,
  `AlgorithmConfigSpace.to_unit_vector`/`from_unit_vector` (#236).
- `VAMOS_MOOCORE_HV_CONTRIB`, `MooCoreKernel(use_hv_contrib=...)` and
  `ProfileResult.memory_mb` (#180, #183).

- The `research` extra and the Experimental third-party baselines
  (`--include-external`, `--external-problem-source`, `pymoo_nsga2`,
  `jmetalpy_nsga2`, `pygmo_nsga2`, `vamos.experiment.external`), and the unused
  `vamos_contrib` package (#268).

### Fixed

#### Corrections that change results

Unless stated otherwise, every run that uses the corrected component changes.

- AGE-MOEA uses PlatEMO's tournament and truncation and now matches jMetal 7.9
  (#142).
- IBEA normalizes objectives for the epsilon indicator, as adaptive IBEA does
  (#115).
- MOEA/D replaces neighbours in random order, and resumes from checkpoints
  exactly (#176, #198).
- NSGA-III, the `ref_dirs` archive and `top_k(method="reference_directions")`
  use the complete Das-Dennis lattice; the truncated one lost an extreme
  direction. Two-objective NSGA-III and RVEA defaults use `pop_size - 1`
  divisions (#130).
- RVEA keeps one survivor per occupied vector and draws parents by permutation,
  so populations can be smaller than `pop_size` (#178).
- SMPSO uses jMetal's coefficients, personal-best rule and leader insertion,
  and returns its leaders archive by default from `optimize`, studies and
  `vamos tune` (#163, #195).
- SMS-EMOA keeps a user reference `vector` fixed and counts a generation as
  `pop_size` children (#193, #228).
- SPEA2 uses the truncation, filling and mating of Zitzler et al. (2001), and
  returns only non-dominated members (#151).
- NSGA-III and SMS-EMOA offer every evaluated offspring to the external archive
  (#228).
- Crowding distance skips constant objectives; larger tournaments keep the best
  rank with NaN crowding; MooCore survival no longer overwrites earlier results
  (#180, #264).
- Every algorithm notifies generation 0, which `hv_convergence` and callbacks
  now see (#196).
- NSGA-II immigration and the `obl` initializer count their evaluations, and
  `obl` prefers feasible candidates (#170, #205).
- `pcx`, `undx`, `simplex`, `linked_polynomial` and `power_law` follow their
  published definitions (#136, #156).
- The `insert` and `scramble` mutations, `GradientRepair`, `ReflectRepair` and
  `WrappingRepair` at infinity, and the `scatter` initializer follow their
  definitions (#263).
- Operator overrides keep the default operator's other parameters, such as
  MOEA/D's `cr` (#184).
- CEC2009 UF3 searches `[0, 1]^n`, and UF9 uses the CEC2009 squared distance
  (#104, #131).
- LSMOP1–4 use the linear linkage and LSMOP9 the normalization of Cheng et al.
  (2017) (#133).
- RWA7 and RWA8 use the normalized domain, and RE24 and RE25 the reference
  domains (#132, #140).
- Regenerated fronts (ZDT, DTLZ, WFG1–2, LSMOP9, C1-DTLZ3, UF6, UF9, DC2-DTLZ,
  RE, RWA3, RWA9, RWA10) change IGD, IGD+, epsilon and normalized HV
  (#140, #157).
- Automatic reference fronts match the number of objectives, and ZDT6 and ZDT3
  fronts are exact (#168, #216).
- `front()`, `best()` and `top_k()` rank feasible solutions first, and archive
  results keep their constraints; `top_k()` defaults to `source="auto"`
  (#148, #216).
- Without MooCore, HV pruning and greedy HV selection use exact contributions;
  HV pruning with three or more objectives needs MooCore (#134, #187).
- `knn` pruning uses SPEA2's truncation, and the archive hook no longer
  re-offers pruned rows (#151, #243).
- `knee_point_scores` picks the knee on the ideal side of the chord (#137), and
  `top_k(method="kmeans")` is deterministic (#265).
- HV termination uses the user's fixed reference point, and `--hv-threshold`
  and `hv_convergence` a fixed near-nadir one (#129, #164).
- `--nsga3-*` options, algorithm-block fields and partial `stopping` or
  `archive` overrides reach the run (#159, #169, #202).
- `vamos bench` and `vamos ablation` use per-problem, feasible-only references,
  rank-sum tests and one orientation per indicator, and ablation variants apply
  their overrides (#125, #126, #147, #149).
- The Nemenyi critical difference uses the right critical value; Studio counts
  infeasible runs as the worst IGD+ and merges only feasible non-dominated
  fronts (#123, #124, #165).
- `vamos profile` warms up, repeats timings and uses a problem-scaled HV
  reference (#183).
- Tuning: racing returns a survivor; elitist restarts work; SMAC3 and BOHB
  budgets, `max_trials` and incumbents are right; Optuna prunes early; failures
  get the failure score; `best_score` is the returned configuration's
  (#127, #128, #138, #155, #208, #236).
- Provenance: wheels inside Git projects, editable installs, packaged data,
  user problems (`custom_python`), re-saved results and CLI requests are
  recorded correctly, which changes their task IDs and verification
  (#145, #166, #188).

#### Other fixes

- Algorithms: encoding-aware defaults (22 of 33 combinations failed),
  `default(pop_size=None)`, `spx`, `oxd` and `"1/n"` everywhere, `tell()`
  validation and HV result, user-callback errors, genealogy, JSON round-trips,
  plugin collisions (#152, #158, #162, #167, #196, #212, #235, #248, #254).
- Problems and evaluation: problems without `n_constraints` in the CLI,
  `make_problem` errors, model-training errors, the multiprocessing pool and
  `timeout`, Dask addresses and errors (#135, #153, #179, #210, #234).
- Studies and artifacts: crash-safe study writes, sanitized failure causes
  without retrying deterministic failures, typed resource limits and column
  checks, stale locks and Windows path lengths (#161, #171, #182, #194, #261).
- Command line: RVEA partitions, failed runs in a series, output collisions,
  `create-problem`, `study summarize`, `bench` inputs and `--only-report`,
  ablation folders, warnings, hints and logs
  (#144, #147, #153, #172, #173, #197, #199, #249).
- Analysis, Studio and Assist: plot axes and labels, `explore_result_front`
  hover data, integer Batch Comparison and replay, one runner per study,
  Problem Builder scripts and LLM drafts, assist folder safety
  (#165, #174, #185, #206, #209).
- Tuning and documentation: MOEA/D with three or more objectives, `bohb_optuna`
  seeds, resumed Optuna budgets, `--smoke`, and corrected archive, DC-DTLZ and
  RE22 notes (#140, #193, #214, #216, #230, #236).

- The C-DTLZ problems accept as many variables as objectives, as jMetal and
  the STEP suite do (#269).

- `vamos --config` rejects an `algorithm`, `engine`, `eval_strategy`,
  `external_archive_pruning`, `problem_set` or `experiment` that the
  command-line option does not accept, also with `--validate-config`, with
  exit code 2; an unknown `algorithm` ran nothing and exited with 0 (#270).
- Built-in problems create `out["F"]` when the caller has not allocated it
  (#271).
- `vamos zoo` and `vamos check` report unknown problems and failures, and
  derived tables warn about indicators skipped without moocore (#273).
- `mixed_design` clips categories instead of wrapping them (#274).
<!-- PENDING(UX-25 #276, fix/studio-small-fixes): drop this line if that pull request is not merged. -->
- Studio opens runs with 64-bit seeds, uses the chosen significance level for
  the Nemenyi difference, searches studies in the background, and keeps
  Matplotlib's backend; `reduce_objectives` no longer depends on units (#276).
<!-- /PENDING -->

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

[Unreleased]: https://github.com/vamos-optimization/VAMOS/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.2.0
[1.1.0]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.1.0
[1.0.2]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.0.2
[1.0.1]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.0.1
[1.0.0]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.0.0
