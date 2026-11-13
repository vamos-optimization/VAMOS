# Changelog

All notable public changes to VAMOS are documented here.

## [Unreleased]

## [1.2.0] - YYYY-MM-DD

VAMOS 1.2.0 corrects built-in algorithms, operators, problems, reference
fronts, indicators and statistics that departed from their published
definitions. Under the [scientific-corrections
policy](docs/project/stability-and-versioning.md#scientific-corrections), these
corrections ship in a minor release even when they change numerical results.
Each entry says which results change and how; rerun affected experiments before
comparing them with results of earlier versions. Run and study artifacts keep
the 1.0.0 schemas.

### Added

- The LIR-CMOP1–14 and DAS-CMOP1–9 constrained suites (`lircmop1`–`lircmop14`,
  `dascmop1`–`dascmop9`), with jMetal's reference fronts.
  `vamos.problems.DASCMOP` takes DAS-CMOP's difficulty triplet.
- The CEC 2009 constrained problems CF2–CF10 (`cec2009_cf2`–`cec2009_cf10`),
  with reference fronts, and the ZXH-CF1–16 suite (`zxh_cf1`–`zxh_cf16`), with
  analytic reference fronts at three objectives.
- The CRE (`cre21`–`cre51`) and RCM (`rcm01`–`rcm50`) constrained real-world
  suites, with reference fronts for 54 of them.
- Eight classic constrained problems as jMetal defines them, with reference
  fronts: `binh2`, `constrex`, `srinivas`, `tanaka`, `osyczka2`, `golinski`,
  `water` and `car_side_impact`.
- The C3-DTLZ4 problem (`c3dtlz4`), and reference fronts at two and three
  objectives for C3-DTLZ4, C-DTLZ and DC-DTLZ (including C3-DTLZ1 and DC3-DTLZ,
  which had none) and MW4, MW8 and MW14, as well as fronts for MW9–MW13.
- `vamos.problems.DCDTLZ` takes DC-DTLZ's `a`, `b`, `g_factor` and
  `constraint_g_scale`; jMetal's and STEP's DC-DTLZ are `a=3, b=0.5,
  constraint_g_scale=1`. DC-DTLZ accepts as few as M − 1 variables.
- `OptimizationResult.feasibility()` summarizes the feasibility of a result:
  its rows, its feasible rows and its smallest constraint violation.
- A `feasible_only` keyword on `OptimizationResult.front()`, `best()`,
  `top_k()` and `top_k_report()`. `feasible_only=False` restores the
  objective-only selection of VAMOS 1.1.0 (see Fixed).
- `hv_reached` in the results of NSGA-III, SMS-EMOA, IBEA, AGE-MOEA and RVEA,
  so every algorithm reports whether its HV target stopped the run, and the CLI
  and run manifests record it. `hv_converged` remains an alias in NSGA-III,
  SMS-EMOA and IBEA until VAMOS 2.0.
- `NSGAII` and `MOEAD` have public `initialize`, `should_terminate` and
  `result` methods, completing the ask/tell protocol of the other built-in
  algorithms. Driven step by step with the same seed, they give the same result
  as `run`.
- `optimize(..., termination=...)` no longer needs an `algorithm_config` (VAMOS
  1.1.0 raised a false `TypeError: optimize() got an unexpected keyword
  argument 'termination'`), and `StudyResult.std(name, ddof=0)` takes `ddof`
  for the sample standard deviation.
- `ModelBasedTuner(backend="irace")` runs the R package irace (>= 4.4.4)
  through `Rscript` with the same `tuner.run(eval_fn)` contract as the other
  model-based backends. The `ParamSpace`, including its conditions, is
  translated to an irace parameter file; each experiment reaches the evaluation
  function in the calling process; `max_trials` times the task's instance/seed
  pairs sets `maxExperiments` unless `irace_max_experiments` is given; initial
  configurations and extra scenario options are supported.
- `vamos tune --backend irace`, with `--irace-rscript`,
  `--irace-instance-seeds`, `--irace-work-dir` and repeatable `--irace-option
  KEY=VALUE`. `--tune-budget` sets irace's `maxExperiments`; the irace
  scenario, log and `irace.Rdata` are kept in `<output>/irace`.
- `vamos create-problem --force` overwrites an existing output file.
- A `gemini` extra installs the SDK of the Studio Problem Builder's default LLM
  provider, and the missing-SDK errors of the three providers name their extra.
- The environment variable `VAMOS_DISABLE_PLUGINS=1` stops the algorithm
  registry from loading installed `vamos.algorithms` plugins on first use;
  `discover_algorithm_plugins()` still loads them on request.

<!-- PENDING(EXP-28, fix/cli-quality-details): drop this block if that pull request is not merged. -->
- `vamos --version` prints the installed version.
<!-- /PENDING -->

### Changed

#### Scientific defaults

- The default real-coded crossover of NSGA-II, NSGA-III, SMS-EMOA, SPEA2, IBEA
  and RVEA is SBX with probability 0.9, as in Deb et al. (2002), jMetal and
  pymoo, and as AGE-MOEA and the SBX operator already used; each algorithm
  keeps its distribution index. VAMOS 1.1.0 used probability 1.0 in those six
  algorithms, both in `XConfig.default()` and in the CLI and `build_algorithm`
  defaults. Their runs with the default crossover are not comparable with runs
  of earlier versions; pass `.crossover("sbx", prob=1.0, eta=...)` to reproduce
  the previous setting.
- IBEA's default `kappa` is 0.05, the value of Zitzler and Künzli (2004), PISA,
  jMetal and PlatEMO, and the middle of VAMOS's IBEA tuning range. With
  indicator values normalized to `[-1, 1]` (see Fixed), `kappa=1.0` left almost
  no selection pressure. With the default configuration and 20 000 evaluations
  (final-front hypervolume, mean of 3 seeds), the hypervolume goes from 0.210
  to 0.419 on DTLZ2 with two objectives, from 0.253 to 0.713 on ZDT4 and from
  0.206 to 0.470 on DTLZ1. Configurations that set `kappa` explicitly are
  unaffected.
- NSGA-III accepts a population larger than its set of reference directions
  (N >= H), as Deb and Jain (2014), jMetal and pymoo do, and still rejects a
  smaller one: `pop_size=100` with 12 divisions and three objectives runs with
  91 directions instead of failing. Study planning accepts the same
  populations. RVEA and MOEA/D still need one individual per vector.
- With `pop_size`, `NSGAIIIConfig.default()` uses the complete Das-Dennis
  lattice with the most directions that fit in it, for any number of
  objectives, and `RVEAConfig.default()` the lattice of exactly `pop_size`
  vectors, failing at once with the nearest valid sizes otherwise (91 or 105
  for three objectives). Both used 12 divisions for three objectives and 6 for
  more, whatever `pop_size` was. NSGA-III with five objectives and
  `pop_size=100` used 6 divisions (210 directions, more than the population)
  and failed; it now uses 4 (70 directions). With three objectives,
  `pop_size=150` and `200` get 15 and 18 divisions instead of 12; sizes up to
  104 keep 12.
- Integer problems run with IBEA, AGE-MOEA and RVEA by default use the encoding
  table's SBX (probability 0.9, eta 20) and polynomial mutation instead of each
  algorithm's real-coded SBX (1.0 with eta 20 for IBEA, 0.9 with eta 15 for
  AGE-MOEA, 1.0 with eta 30 for RVEA), so their results change.
- SMS-EMOA measures hypervolume contributions against `nadir + offset *
  (nadir - ideal)` of the population and the new child, so
  `reference_point(offset=...)` (default 1.0) is a fraction of each objective's
  range, the reference `1 + offset` of the normalized objectives. It was added
  to the maximum in objective units, so rescaling one objective changed which
  solutions survived. Runs with the default reference change trajectory: with
  6000 evaluations and seeds 1 to 5, the mean HV of default SMS-EMOA goes from
  0.5955 to 0.6009 on ZDT1 and from 0.7559 to 0.7565 on DTLZ2.
- The `hv` and `mc_hv` archive pruning without `hv_ref_point` measures the
  contributions against `nadir + (nadir - ideal)` of the archived points
  instead of the largest value seen plus 1 in objective units. The pruning no
  longer depends on the objective units and keeps the extreme points: with one
  objective scaled by 100, it used to drop an extreme point.
- The default `epsilon` of the `simplex` crossover is `sqrt(m + 1)` (2.0 for
  the three parents VAMOS uses) instead of 0.5, and the `simplex_epsilon`
  tuning range is [1.0, 3.0] instead of [0.1, 1.0].
- `cec2009_cf1` and `vamos.problems.CEC2009CF1` default to 10 variables, the
  dimension of the CF problems in the CEC2009 technical report and PlatEMO;
  `n_var` still selects another one. The `CEC2009_UF_CF_curved` benchmark suite
  also runs CF1 with 10 variables. Default CF1 runs of earlier versions used 30
  variables and are not comparable with CEC2009 results.
- Bounded and unbounded archives deduplicate objective vectors by exact
  equality by default, as jMetal does: `objective_tolerance` defaults to `0.0`
  instead of `1e-10` in `ExternalArchiveConfig`, in `archive.external` and in
  the archive classes, including the SMPSO leader archive. The tolerance was an
  absolute Chebyshev distance, so with objectives on a scale of about `1e-9` or
  smaller distinct non-dominated points were merged: SMPSO on ZDT1 with
  objectives multiplied by `1e-11` kept a single leader. At ordinary scales the
  results do not change. `objective_tolerance` stays configurable in
  `archive.external` and in the external archive builder.
- The hypervolume indicators `HVIndicator`, `HVContributionsIndicator`,
  `HVApproxIndicator` and `WHVRectIndicator` require an explicit
  `reference_point`; without one they raise `ValueError`. VAMOS 1.1.0 used
  `max(front) + 1` of the measured front, so the value depended on the front
  itself: adding a dominated point could take the hypervolume from 3 to 15.
- `hypervolume()` and `hypervolume_contributions()` ignore points outside the
  reference box, as MooCore and `compute_hypervolume` do, instead of raising
  `ValueError`.

#### Tuning

- Without fidelity levels (`budget_levels=None`, or `vamos tune` without
  `--fidelity-levels`), the `optuna`, `bohb_optuna`, `smac3` and `bohb`
  backends run a single fidelity: every trial is evaluated once, at the full
  per-run budget, on every instance/seed pair. They used to invent a schedule:
  Optuna evaluated each trial at `b/3`, `2b/3` and `b`, about `2b` evaluations
  per pair, and SMAC3 and BOHB ran Hyperband from 1 evaluation. `smac3` now
  uses SMAC's `HyperparameterOptimizationFacade` in that case, and `bohb`
  samples one configuration per iteration. With `budget_per_run=900`, 60
  trials, 2 seeds and one instance, `optuna` goes from 360 runs at 300, 600 and
  900 evaluations to 120 runs at 900. `random`, `racing` and `irace`, and
  searches that give fidelity levels, are unaffected.
- `vamos tune --backend racing` runs racing's statistical race by default, at
  `--budget` per run, using `--alpha`, `--use-statistical-tests`,
  `--elimination-fraction` and `--min-blocks-before-elimination`. Pass
  `--multi-fidelity` for the multi-fidelity schedule, which is successive
  halving by mean score and ignores those options; it now warns when one of
  them is set. The other backends keep `--multi-fidelity` on by default. The
  "Split-based tuning" example of the tuning guide with `--backend racing`
  stopped with `RuntimeError: Multi-fidelity tuning finished without a valid
  configuration` without evaluating anything; it now races at 5000 evaluations
  per run within its 200-run budget.
- When every tuning evaluation fails, `RacingTuner` and `RandomSearchTuner`
  raise `TuningFailedError`, and `vamos tune` exits with code 1 and the first
  failure instead of reporting an arbitrary configuration with code 0.

#### Validation and API

- The typed algorithm configurations validate their values when they are built,
  by the builder, `from_dict()` or the constructor, and raise
  `ConfigurationError` before any evaluation: operator probabilities, `eta`,
  sizes, IBEA's `kappa`, MOEA/D's `delta` and aggregation, RVEA's `alpha` and
  `adapt_freq`, SMPSO's coefficients and `result_mode`. `from_dict()` rejects
  unknown keys with a suggestion instead of ignoring them. MOEA/D defaults, CLI
  runs and tuning assignments record `min(neighbor_size, pop_size)`, the
  neighbourhood MOEA/D already used.
- `optimize` raises `ConfigurationError` for arguments it used to ignore or
  apply in part: `n_var`, `n_obj` or `problem_kwargs` with a problem instance
  (VAMOS 1.1.0 ran the instance's dimensions and recorded the requested ones);
  `problem_kwargs` other than `n_var` and `n_obj` with a registered name;
  `max_evaluations` with a `termination` that does not stop at the same budget
  (1.1.0 silently used `termination`); the configuration class of another
  algorithm (1.1.0 ran MOEA/D with an `NSGAIIConfig`); `GenericAlgorithmConfig`
  with a built-in algorithm, whose extra keys are not recorded, so the run
  could not be replayed; and an empty seed sequence. Study plans reject
  `algorithm_configs` keys that name no algorithm of the study, also a key in
  another letter case such as `"NSGAII"`, with `INVALID_STUDY_SPEC`; 1.1.0
  dropped the configuration.
- A CLI algorithm block rejects `pop_size`, `pop_size_auto`, `offspring_size`,
  `external_archive`, `track_genealogy` and the Python callbacks, naming the
  config key to use instead. Configs that set them fail validation instead of
  ignoring them.
- `RunArtifactError` and `StudyError`, the bases of the run-artifact and study
  errors, also derive from `VAMOSError` and are exported by `vamos.exceptions`,
  so `except VAMOSError` catches every error of the stable API. A study error
  in the CLI outside `vamos study`, such as an ablation that cannot be planned,
  prints `Error: ...` and exits with code 2, like other VAMOS errors, instead
  of ending with a traceback. The stability policy lists the stable exception
  classes; seven exported classes that VAMOS never raises are planned for
  removal in 2.0.
- An algorithm class built without a kernel, such as `NSGAII(config)`, uses the
  NumPy kernel, as `vamos.optimize` does without `engine`; VAMOS 1.1.0 used
  Numba whenever it was installed. The backends follow the same trajectory, so
  results do not change; pass `kernel=NumbaKernel()` for the compiled kernels.
  Importing the MooCore backend no longer imports Numba.
- Stochastic components built or called without a generator raise
  `ConfigurationError` instead of drawing from an unseeded
  `np.random.default_rng()`: the parent-selection classes, the real
  initializers, `UniformMutation` and `LinkedPolynomialMutation`,
  `NumbaKernel.sbx_crossover` and `polynomial_mutation`, the racing tuner's
  `ParamSpace.sample()` and `select_top_k_kmeans`. `optimize`, the CLI, studies
  and tuning already pass their generators and are unaffected.
- `pairwise_wilcoxon` results carry `p_adjusted` (Holm over the pairs of the
  call) and `significant` (`p_adjusted <= alpha`); `alpha` was ignored and only
  raw p-values, which do not change, were returned. `friedman_test` raises
  `ValueError` for fewer than 3 algorithms, fewer than 2 problems or NaN
  scores, and `vamos bench` skips Friedman for a metric with missing
  per-problem means.

#### Performance

- Steady-state NSGA-II (`.steady_state()`) with `engine="numba"` runs each
  one-offspring step as a few compiled calls (binary tournament, SBX, clip
  repair, polynomial mutation, insertion into the cached fronts with
  incrementally sorted crowding, truncation). Random draws, tie-breaking and
  the ordering of the truncated front are unchanged, so trajectories stay
  identical to `engine="numpy"`. Median time per evaluation (10 000
  evaluations, population 100) goes from 235–248 to 43–47 µs on ZDT1 and from
  285–292 to 60–63 µs on DTLZ2 with three objectives. Constrained problems,
  other operators or selections, genealogy, immigration and parent filters keep
  the previous path.
- SMS-EMOA runs its steady-state step with compiled kernels under
  `engine="numba"`: two compiled calls build the child and one inserts it into
  the cached fronts, removes the least contributor and updates ranks and
  crowding distances. Parents and every variation draw still come from the run
  generator in the reference order, so results are bit-identical to VAMOS 1.1.0
  and to the NumPy engine. Median time per evaluation (population 100, 10 000
  evaluations) goes from 219.6 to 36.2 µs on ZDT1, from 200.2 to 84.9 µs on
  DTLZ2 with three objectives and from 152.9 to 68.3 µs on C1-DTLZ1. Other
  operators, genealogy tracking and non-finite objectives keep the reference
  step, and three or more objectives keep the MooCore (or NumPy) contribution
  call.
- MOEA/D's default steady-state mode (`batch_size=1`) is about three times
  faster with `engine="numba"`: each step runs in compiled code while the
  random draws still come from the run's generator in the NumPy order, so
  results are bit-identical to the NumPy engine. A run drops from 73.6 to 22.6
  µs per evaluation on ZDT1 (30 variables, 100 subproblems, 5000 evaluations)
  and from 89.4 to 37.5 µs on DTLZ2 (3 objectives, 91 subproblems). Genealogy
  tracking, `batch_size > 1` and operators other than DE or SBX with polynomial
  mutation and the clip repair keep the previous path.
- The NumPy engine sorts two-objective populations into non-dominated fronts in
  O(N log N) instead of comparing every pair of rows, with the same fronts and
  ranks. Sorting 1000 rows takes about 0.4 ms instead of 33 ms.
- `import vamos`, and any import of a VAMOS module, no longer imports the
  experiment layer and the algorithms: the names of `vamos` and `vamos.api`
  load on first use. `import vamos.foundation.encoding` goes from 321 VAMOS
  modules (about 2 s) to 4 (0.04 s), which also shortens the start of every
  process-pool worker. The public names, `from vamos import *` and attribute
  access to subpackages such as `vamos.api` after `import vamos` are unchanged.
- `Study.run()`, `resume()` and `retry()` capture the current environment once
  per operation instead of once per verified run (about 0.3 s per run). Before
  running anything, `resume()` and `retry()` compare every earlier run with
  that capture without re-verifying the run artifacts that loading the study
  has just verified. The environment rule, its errors and the integrity check
  of every run are unchanged.

#### Packaging, documentation and tooling

- The source distribution no longer includes 28 loose test files from `tests/`
  without the rest of the suite. The test suite runs from the repository.
- The Studio guide lists each LLM provider's extra, default model, model
  variable and API key, and what is sent to the provider.
- Documentation publication keeps the published `docs/<version>/` trees in a
  durable store. After its live checks pass, the Cloudflare production workflow
  attaches the portal as `vamos-docs-portal-<version>.tar.gz`, with a `.sha256`
  file, to the GitHub Release of that version; release assets do not expire,
  unlike the 90-day Actions artifacts that the workflow still uploads. The
  workflow takes the archive from `archive_release` (a release tag) or from a
  prior production run, requires exactly one of them, and refuses to build when
  the release of the published version does not exist.
  `tools/build_release_docs.py --archive-from` accepts a `.tar.gz` portal
  archive as well as a directory.
- `https://vamos-optimization.github.io/VAMOS/` no longer serves a stale copy
  of the 1.0.0 documentation. It redirects to `https://vamos-optimization.org/`
  and keeps the rest of an old address; both of its pages declare `.org` as
  canonical. The `docs.yml` workflow deploys only this redirect, on manual
  dispatch.
- The canonical typing environment uses mypy 2.4.1, and the typing baseline
  depends only on the constraint pins that can change mypy diagnostics (mypy,
  typing-extensions, and the runtime and `[dev]` distributions that `src/vamos`
  imports), so bumps of other pins no longer require regenerating
  `typing/mypy-baseline.json`. The reviewed constraints pin numba 0.68.1,
  pytest 9.2.0, ruff 0.17.4 and setuptools 84.2.1; CI actions, the publication
  action, the documentation browser checks and the release tooling
  (cyclonedx-bom 8.0.1) include the updates merged since 1.1.0.

<!-- PENDING(REPO-21 #266, ci/python-support): drop this block if that pull request is not merged. -->
- VAMOS declares and tests Python 3.13 and 3.14, on Linux, Windows and macOS.
  The package metadata declares the development status "Production/Stable",
  which matches the 1.x compatibility commitments, instead of "Beta".
- Python 3.10 reached its end of life in October 2026. VAMOS 1.2 is the last
  minor release that supports it; VAMOS 1.3.0 will require Python 3.11 or
  newer.
<!-- /PENDING -->

<!-- PENDING(user-extras, build/user-oriented-extras): drop this block if that pull request is not merged. -->
- `analysis` installs only what VAMOS imports: pandas, matplotlib, plotly and
  scikit-learn. seaborn, networkx, IPython, ipykernel and ipywidgets move to
  `examples`, so notebooks need `[analysis,examples]`; nbformat and nbconvert
  are only in `dev`.
- `all` installs the user extras (`compute`, `analysis`, `tuning`, `studio`,
  `examples`, `openai`, `anthropic`, `gemini`) and no longer the `dev` and
  `docs` tooling.
<!-- /PENDING -->

<!-- PENDING(UX-25, fix/studio-small-fixes): drop this block if that pull request is not merged. -->
- `plot_critical_distance` no longer accepts `higher_is_better`, which had no
  effect: average ranks always give rank 1 to the best algorithm.
<!-- /PENDING -->

<!-- PENDING(REPO-23, docs/readme-and-guides): drop this block if that pull request is not merged. When it merges, also add release-notes-1.2.0.md to Reference > Support & Versions in mkdocs.yml. -->
- The README, which is also the PyPI project description, uses absolute links,
  so its banner and links work on PyPI, and its contributor link points to
  `AGENTS.md`. Its multi-seed example compares the seeds by hypervolume against
  a fixed reference point; it compared the evaluation count, which is the same
  in every run.
- The CLI guide explains that the hypervolume printed after a run uses a
  reference point derived from the fronts of that command, so the values are
  comparable only within one command.
- The study guides check `inspect().retryable_failed_work` before calling
  `retry(failed_only=True)`. The previous recipe stopped with
  `RetryNotAllowedError` (`TERMINAL_TASK_NOT_RETRYABLE`; exit code 4 in `vamos
  study retry`) on a study without failed tasks.
- The documentation navigation lists the stability policy, the known
  limitations, the documentation versions and the release notes under Reference
  → Support & Versions.
<!-- /PENDING -->

### Deprecated

- `best("knee")` and `top_k(method="knee")` emit a `FutureWarning`. `knee` has
  always been an alias of `balanced_sum`, the smallest sum of objectives
  normalized over the front, not a knee point. The alias will be removed in
  VAMOS 2.0; use `balanced_sum` for the same rule, or
  `vamos.ux.api.knee_point_scores` for a geometric knee.
- The `allow_ref_expand` key of the `("hv", {...})` termination criterion and
  the `allow_ref_expand` argument of `hypervolume()` and
  `hypervolume_contributions()` are ignored and emit a `FutureWarning`; they
  will be rejected or removed in VAMOS 2.0.

### Removed

- Internal modules with no importer: `vamos.engine.hyperheuristics`, whose
  indicator reward was also wrong, `vamos.experiment.study.nsga2_diagnostics`,
  the constraint-handling strategies of `vamos.foundation.constraints`
  (`ConstraintHandlingStrategy` and its three subclasses), and the internal
  modules `vamos.engine.adaptation`, `vamos.experiment.studio`,
  `vamos.foundation.core.api`, `vamos.ux.mcdm`, `vamos.ux.stats` and
  `vamos.engine.algorithm.components.base`. None of them was part of the public
  API; code that imported them directly should use the public facades instead.
- The experimental tuning helpers without internal use:
  `vamos.engine.tuning.convergence` (`ConvergenceProfile`,
  `ConvergenceDiagnostic`, `TuningCallback`, `diagnose`, `score_with_profile`),
  `save_checkpoint` and `load_checkpoint` of the racing I/O, and
  `AlgorithmConfigSpace.to_unit_vector` and `from_unit_vector`. `TwoPhaseTuner`
  stays.
- The undocumented `VAMOS_MOOCORE_HV_CONTRIB` variable and the
  `MooCoreKernel(use_hv_contrib=...)` parameter: MooCore survival follows the
  NumPy trajectory.

<!-- PENDING(user-extras, build/user-oriented-extras): drop this block if that pull request is not merged. -->
- The `research` extra (pymoo, jMetalPy, DEAP, Platypus, pygmo) and the
  Experimental third-party NSGA-II baselines it served: `--include-external`,
  `--external-problem-source`, the `pymoo_nsga2`, `jmetalpy_nsga2` and
  `pygmo_nsga2` algorithm names, and `vamos.experiment.external`. VAMOS no
  longer depends on any other optimization framework; install one yourself to
  compare against it.
- The `vamos_contrib` package (an unused pymoo problem adapter) from the
  distribution.
<!-- /PENDING -->

### Fixed

#### Algorithms

- AGE-MOEA follows its author's PlatEMO code in mating and survival, where it
  followed pymoo 0.6.2. The binary mating tournament breaks rank ties with the
  AGE survival score instead of NSGA-II crowding distance. The front that is
  truncated to fill the population is scored by proximity to the ideal point,
  measured in the first front's ideal-centred, normalized coordinates, and cut
  by that score; it was left unscored and cut by row index, so it kept its last
  rows, which are usually offspring. Over 30 seeds (population 100, default
  variation), the hypervolume goes from 0.8496 to 0.8545 on ZDT1 (10 000
  evaluations), from 0.7426 to 0.7443 on DTLZ2 (20 000) and from 0.1880 to
  0.1892 on DTLZ1 (30 000), with a five times smaller spread on DTLZ1. jMetal
  7.9's AGE-MOEA gives 0.8545, 0.7446 and 0.1890; the differences are within
  seed variation, whereas VAMOS 1.1.0 was significantly worse on all three.
  AGE-MOEA runs of earlier versions differ with the same seed.
- IBEA scales its epsilon indicator as adaptive IBEA does (Zitzler and Künzli,
  2004), as PISA and jMetal do, and as VAMOS 1.1.0 already scaled the
  hypervolume indicator: the objectives are normalized over the population, and
  the indicator is divided by its largest magnitude. Unscaled, an objective
  with a much wider range than the others decided every epsilon value, and the
  population collapsed toward that objective's minimum; on RE21 (f1 about 10^3,
  f2 about 10^-2) the default IBEA ended with a single distinct solution. A
  positive affine change of the units of any objective now leaves IBEA's
  selection unchanged, and fitness exponents stay within `[-1/kappa, 1/kappa]`.
  Epsilon-based runs differ from 1.1.0 runs with the same seed. As with the
  hypervolume indicator, far-dominated solutions widen the normalization range,
  which slows convergence on multimodal problems such as ZDT4 and DTLZ1.
- IBEA rejects `indicator` values other than `eps`, `epsilon`,
  `additive_epsilon` and `hypervolume` (any letter case); it ran every other
  value, `"hv"` included, with the epsilon indicator.
- MOEA/D visits a new solution's neighbours in random order, as MOEA/D-DE and
  jMetal do; it always started from the subproblem itself. Every MOEA/D run
  changes. `MOEADConfig.default` and `optimize` with three or more objectives
  use the packaged jMetal weights when no complete lattice has `pop_size`
  vectors, so 100 individuals with three objectives run instead of failing.
- NSGA-III without `divisions`, the `ref_dirs` pruning of the external archive
  and `top_k(method="reference_directions")` use the complete Das-Dennis
  lattice with the most directions that fit in the population, capacity or `k`,
  which matches pymoo's `get_reference_directions("das-dennis", M,
  n_partitions=H)`. They cut the next larger lattice to that size in
  lexicographic order, which always dropped the corner of the first objective,
  including the direction (1, 0, 0): with 100 individuals and three objectives,
  100 of the 105 directions of 13 divisions. This is the default NSGA-III path
  of the CLI and of experiment specifications. On DTLZ2 with three objectives,
  20 000 evaluations and seeds 0 to 2, the CLI NSGA-III run reached at most f1
  = 0.984 to 0.987 against 1.0 for the other objectives; it now reaches all
  three extremes, and the HV with reference (1.1, 1.1, 1.1) goes from 0.738,
  0.743 and 0.742 to 0.743, 0.744 and 0.745. The `ref_dirs` archive and `top_k`
  now keep the closest member of each occupied direction and fill the remaining
  slots by max-min distance and by perpendicular distance, respectively, so
  their selection no longer depends on the order of the objective columns.
  NSGA-III with explicit `divisions` or a direction file, MOEA/D and the
  `maxmin` archive give bit-identical results.
- `NSGAIIIConfig.default()` and `RVEAConfig.default()` with two objectives use
  `pop_size - 1` divisions, or 99 without `pop_size`, as
  `MOEADConfig.default()` does. They used 6 divisions, so `optimize("zdt1",
  algorithm="nsgaiii")` and `algorithm="rvea"` ran with a population of 7, and
  `pop_size=100` failed.
- RVEA keeps one survivor per occupied reference vector, as its article,
  PlatEMO and pymoo do, and draws parents by permutation; it filled the
  population up to `pop_size` in vector order, favouring the first vectors, and
  drew parents with replacement. Every RVEA run changes, and its population and
  result can have fewer rows than `pop_size`.
- SMPSO follows jMetal, the authors' implementation, in three details where it
  followed jMetalPy. The random factors r1 and r2 and the coefficients drawn
  from `[c1, c1 + 1]` and `[c2, c2 + 1]` are used as drawn; they were rounded
  to one decimal, which put 5% of the draws at r = 0. With constraints, a
  personal best is replaced unless it is better than the new position (feasible
  against infeasible, a smaller violation, or dominance when both are feasible
  or equally infeasible), the same rule as without constraints; constrained
  runs required strict dominance, so a non-dominated new position never
  replaced the personal best. The leaders archive receives the particles one at
  a time and removes the most crowded leader whenever it overflows, as jMetal's
  `leaders.add`; it truncated each batch at once. SMPSO runs change trajectory:
  with 10 000 evaluations, the mean HV on ZDT4 goes from 0.8636 (standard
  deviation 0.0162) to 0.8701 (0.0007) over 10 seeds, and on ZDT1 from 0.7826
  to 0.7861 over 30 seeds, within the seed-to-seed variation.
- `optimize`, studies and `vamos tune` return (and score) SMPSO's leaders
  archive when no `algorithm_config` is given, as `SMPSOConfig.default()` and
  jMetal do. They forced `result_mode="population"` and returned the final
  swarm, which is not elitist: on ZDT1 (10 000 evaluations, seed 1), 91 of the
  200 returned rows were dominated, and now none is. SMPSO tuning scores on
  unconstrained problems change too; persistent Optuna studies for SMPSO use
  the new suffix `__vamos_cli_leaders_hv_v1`.
- `SMSEMOAConfig.builder().reference_point(vector=...)` uses the vector as
  given and keeps it fixed, and points outside its box contribute 0. Before,
  the vector was raised to the population maximum plus `offset` and, with the
  default `adaptive=True`, replaced after the first step: `vector=[1.1, 1.1]`
  on ZDT1 ended at about `[1.67, 3.75]`. Combining `vector` with
  `adaptive=True` now raises `ValueError`, as does a negative or non-finite
  `offset`.
- SMS-EMOA counts a generation as `pop_size` children, as steady-state NSGA-II
  does. It counted every child: `result["generation"]` was the number of
  children, and live callbacks, hooks such as `hv_convergence` and the `("hv",
  ...)` target ran after every evaluation. They now run once per generation, so
  an HV-terminated run may use up to `pop_size - 1` more evaluations within its
  budget.
- NSGA-III and SMS-EMOA offer every evaluated offspring to the external
  archive. VAMOS 1.1.0 archived only the survivors, so a non-dominated
  offspring removed by NSGA-III's niching or as SMS-EMOA's least hypervolume
  contributor was lost. Archives, and results drawn from them, can contain more
  points.
- SPEA2's environmental selection and mating follow Zitzler, Laumanns and
  Thiele (2001) and jMetal. When the non-dominated solutions exceed the
  archive, the truncation operator removes the solution whose sorted distances
  to the others are lexicographically smallest; it compared only the distance
  to the k-th nearest neighbour and broke ties by row order, so of a crowded
  pair it could remove the extreme solution. When they fall short, the
  dominated solutions with the best fitness `F = R + 1 / (sigma_k + 2)` fill
  the archive, instead of whole raw-fitness levels with the last one truncated
  by distance. The mating tournament uses the raw fitness and density each
  member had in the union of archive and offspring. With jMetal's settings
  (population and archive 100, 30 000 evaluations, 10 seeds), the hypervolume
  goes from 0.7259 to 0.7326 on DTLZ2 and from 0.6665 to 0.6832 on C2-DTLZ2,
  against 0.7354 and 0.6837 for jMetal 7.9; ZDT1 does not change significantly.
  SPEA2 runs of earlier versions differ with the same seed.
- SPEA2 with `result_mode="non_dominated"` (its default) and no external
  archive returns only the non-dominated members of its archive. It returned
  the whole archive, which holds dominated solutions whenever the first front
  does not fill it: on DTLZ1 with 1000 evaluations, 55 of its 100 rows.
- Crowding distance skips an objective that is constant within a front, as
  jMetal and pymoo do. VAMOS 1.1.0 gave an infinite distance to the two
  solutions that came first and last when sorting by that objective, which
  depended only on their order, and protected them in NSGA-II survival,
  tournaments and `crowding` archive pruning. Fronts of one or two solutions
  keep infinite distances. Runs with a constant objective are not comparable
  with VAMOS 1.1.0.
- Tournaments with a size above two keep the best rank when crowding distances
  are NaN or -inf and use fractional ranks as given; NumPy could let a worse
  rank win, and both NumPy and Numba truncated ranks to integers. The MooCore
  kernel's survival returns arrays the caller owns instead of views of buffers
  that the next survival overwrote.
- NSGA-II immigration evaluates candidates without objectives through the run's
  evaluation backend, counts them against the evaluation budget and evaluates
  only the candidates that can enter the population; it evaluated them outside
  the budget at every immigration event. Runs with such immigrants run fewer
  offspring evaluations for the same budget.
- An exception raised by user code that an algorithm calls stops the run with
  that exception: a live callback's `on_generation` or `should_stop()`, and
  NSGA-II's `generation_callback`, `parent_selection_filter` and immigration
  `skip_if`. These failures were logged at debug level while the run carried
  on, and MOEA/D, SPEA2 and SMPSO then called `on_generation` a second time
  with the whole population. A `parent_selection_filter` that takes only the
  state is still called with the state alone.
- Every algorithm first notifies generation 0, the evaluated initial
  population, and then counts generations from 1. NSGA-III, SMS-EMOA, IBEA and
  AGE-MOEA started at generation 1 and RVEA at 2, so an `hv_convergence` stop
  or a live callback saw one sample fewer. `LiveCallbackMode` lists the values
  that NSGA-II accepts, `nd_only`, `population` and `population_archive`,
  instead of `"all"`, which failed at run time.
- The ask/tell `tell()` of IBEA, SPEA2, SMPSO and NSGA-II checks the evaluation
  result before it changes any state, as AGE-MOEA and RVEA already did. A
  missing `G` on a constrained problem with constraint handling active, or `F`
  or `G` with the wrong number of rows, raises a `ValueError` that names the
  problem, and the pending offspring and the evaluation count stay as they
  were. IBEA failed with `IndexError` when `G` was missing, and also when `G`
  was given with `constraint_mode="none"`, which now ignores `G`; SPEA2 filled
  a missing `G` with zeros, so infeasible offspring counted as feasible; SMPSO
  counted the evaluations and dropped the pending particles before reading the
  result; NSGA-II accepted only objects with an `F` attribute, dropped the
  pending offspring first and, with a constrained population and a result
  without `G`, silently fell back to unconstrained survival. Valid runs give
  the same results.
- `tell` of NSGA-III, SMS-EMOA and IBEA returns `True` once the HV target of an
  `("hv", ...)` termination is reached; it always returned `False`.
- Genealogy tracking in IBEA, MOEA/D and SPEA2 follows the survivors of each
  generation; VAMOS 1.1.0 kept the initial population's ids, so
  `result["genealogy"]` reported that no offspring reached the final front.
  NSGA-III and IBEA record both parents of each SBX child; they recorded one.
  `build_algorithm(..., track_genealogy=True)`, the CLI's `--track-genealogy`
  and the experiment runner support genealogy in NSGA-II, MOEA/D, SMS-EMOA,
  NSGA-III, SPEA2, IBEA and SMPSO; only NSGA-II received the flag, and the CLI
  rejected every other algorithm. AGE-MOEA and RVEA, which record no genealogy,
  are rejected with a clear error.
- `optimize(..., checkpoint=...)` with NSGA-II or MOEA/D raises
  `ConfigurationError` when the checkpoint cannot be restored, for example a
  different population size, missing `X` or `F`, a constrained population
  without `G`, an unknown random state, malformed MOEA/D subproblem state, or
  an archive that does not fit; VAMOS 1.1.0 logged a warning and silently
  started a fresh run. Checkpoints store the external archive's constraint
  values (`archive_G`), and resuming restores them; restored archive rows lost
  them and counted as feasible. Archives reject rows with constraint values
  when they hold none, and rows without them when they hold some, instead of
  filling the gap with zeros. MOEA/D with its default configuration resumed
  from a checkpoint continues exactly as the uninterrupted run; 1000 + 1000
  evaluations on ZDT1 diverged from an uninterrupted 2000-evaluation run.
- Every built-in algorithm accepts the same termination criteria:
  `("max_evaluations", n)`, `("n_gen", g)` (`g * pop_size` evaluations, the
  initial population included) and `("hv", {...})`; in VAMOS 1.1.0, AGE-MOEA
  and RVEA rejected `hv` and the other seven rejected `n_gen`. `run()` rejects
  a fractional, boolean or non-positive budget with `ValueError` instead of
  truncating it (`("max_evaluations", 150.9)` ran 150 evaluations). Resuming
  NSGA-II or MOEA/D with `max_evaluations` below the checkpoint's evaluations
  raises `ConfigurationError`: `max_evaluations` is the total budget of the
  resumed run.
- `optimize` and study planning check each built-in algorithm's smallest
  population and number of objectives before running: NSGA-II, IBEA, AGE-MOEA,
  MOEA/D, NSGA-III and RVEA need `pop_size >= 2`, and SMS-EMOA, AGE-MOEA,
  NSGA-III and RVEA need at least two objectives. `optimize` raises
  `ConfigurationError` naming the minimum, also for the `pop_size` of an
  explicit `algorithm_config`; planning raises `UnresolvedStudyTaskError` with
  `INVALID_POPULATION_SIZE` or `UNSUPPORTED_OBJECTIVE_COUNT`. VAMOS 1.1.0
  failed later with an error from the parent tournament or from moocore,
  planned such study tasks as ready, and ran AGE-MOEA on a single-objective
  problem.
- `optimize(problem, algorithm=...)` and every `XConfig.default()` follow the
  problem's encoding. `default()` takes an optional `encoding`, and discrete
  and mixed problems get the operators of the shared encoding table (uniform
  crossover and bit-flip mutation for binary, OX and swap for permutations, SBX
  with probability 0.9 and polynomial mutation for integers, the mixed
  operators for mixed problems). Only NSGA-II and MOEA/D used the encoding: 22
  of the 33 supported combinations of the nine algorithms with `bin_knapsack`,
  `tsp6`, `int_alloc` and `mixed_design` failed with errors such as
  `Unsupported SMSEMOA crossover 'sbx' for binary encoding`; all 33 now run.
- `build_default_algorithm_config(..., pop_size=None)` and every
  `XConfig.default(pop_size=None)` give a positive population: 100, or the
  lattice size for MOEA/D, NSGA-III and RVEA. NSGA-II raised `TypeError` and
  SMS-EMOA returned a configuration with `pop_size=None`.
  `builder().pop_size()` rejects anything but a positive integer with a
  `ConfigurationError`.
- `from_dict(json.loads(cfg.to_json()))` equals `cfg`: operator pairs that JSON
  returns as lists become tuples again.
- An algorithm plugin whose name is that of a built-in algorithm is ignored
  with a `RuntimeWarning`; VAMOS 1.1.0 ignored it silently.

#### Operators and initialization

- The multi-parent crossovers `pcx`, `undx` and `simplex` follow their
  published definitions; none is a default. `pcx` is the parent-centric
  crossover of Deb, Anand and Joshi (2002), as Deb's C code and pymoo implement
  it: the step along the direction to the centroid is a zero-mean normal draw
  instead of a fixed bias of 0.15·|d|, the orthogonal noise spans all n − 1
  directions instead of the plane of the three parents, each of the three
  children is centred on its own parent, and `sigma_zeta` scales the step along
  that direction and `sigma_eta` the orthogonal noise, as in the paper (their
  roles were swapped). `undx` is the unimodal normal distribution crossover of
  Ono and Kobayashi (1997): one normal `xi` per pair of children, normal
  orthogonal terms with variance `eta**2 / n`, and a basis orthogonal to the
  parents' direction; VAMOS 1.1.0 reproduced jMetal's per-coordinate variant,
  whose spread along that direction was a quarter of the reference. `simplex`
  (SPX of Tsutsui, Yamamura and Higuchi, 1999) samples uniformly in the
  parents' simplex expanded by `epsilon`, which preserves the parents' mean and
  covariance; it used non-uniform weights and contracted the simplex, so each
  application kept 4 % of the population's variance and diversity collapsed
  within a few generations. With NSGA-II on ZDT1 (population 100, 10 000
  evaluations, three seeds, mean HV with reference [1.1, 1.1]), `pcx` goes from
  0.000 to 0.615, `undx` from 0.143 to 0.712, and `simplex` from 0.000 to
  0.034.
- The global operator registry no longer maps `spx` to the real simplex
  crossover. For binary encoding, `spx` remains the one-point crossover alias.
- `linked_polynomial` mutation draws one random number per individual and
  shares it among that individual's mutated variables, using the bounded
  polynomial formula, as Zille et al. (2016) and jMetal's
  `LinkedPolynomialMutation` do. It drew a single number for the whole batch,
  so every child of a generation moved by exactly the same step, and its
  unbounded formula pushed genes outside the bounds (with x = 0.99 every
  mutated gene went to 1.062 before repair).
- `power_law` mutation follows the power mutation of Deep and Thakur (2007) as
  jMetal implements it: strength `s = u**index`, a direction chosen from the
  gene's relative position in its range, and a step scaled by the distance to
  the chosen bound. It scaled the step by the whole range with a random sign,
  so with x = 0.5 and the default index 75 % of the mutated genes left the
  domain and ended on a bound after repair. A larger `index` now gives smaller
  steps, as the docstring said, and a non-positive `index` is rejected. Results
  with `linked_polynomial` or `power_law` change; neither is a default. On ZDT1
  (NSGA-II, population 100, 10 000 evaluations, five seeds) the mean
  hypervolume goes from 0.866 to 0.857 with `linked_polynomial` and from 0.868
  to 0.867 with `power_law`: ZDT1's optimum lies on the lower bound, which the
  old out-of-bounds steps hit after repair.
- The permutation `insert` mutation moves the chosen gene to either side; it
  only moved genes to the right, so from `[0, 1, 2]` it never produced `[2, 0,
  1]`. The permutation `scramble` mutation shuffles a segment of at least two
  genes; a one-gene segment made a quarter of its mutations no-ops.
- `GradientRepair` counts only violated constraints, `max(g, 0)`; a satisfied
  constraint offset a violated one and left the point unrepaired.
  `ReflectRepair` and `WrappingRepair` send `+inf` and `-inf` to the upper and
  lower bounds instead of returning NaN.
- The `scatter` initializer combines a reference set of 20 Latin-hypercube
  points when `pop_size` is larger; with `pop_size >= 20` it returned the plain
  Latin-hypercube sample. Runs with these operators or this initializer are not
  comparable with VAMOS 1.1.0.
- The opposition-based initializer (`initializer: obl`) counts its `2 *
  pop_size` candidate evaluations against the budget in every algorithm, and
  the chosen candidates keep their objective and constraint values instead of
  being evaluated again. VAMOS 1.1.0 evaluated the candidates outside the
  budget: NSGA-II on ZDT1 with `pop_size=20` and `max_evaluations=200`
  evaluated 240 rows. A budget below `2 * pop_size` raises `ValueError`. The
  initializer keeps feasible candidates first, then the best by rank and
  crowding distance; it ignored the constraints and discarded feasible
  candidates for infeasible ones. Results of runs with `initializer: obl` are
  not comparable with VAMOS 1.1.0.
- NSGA-III, SMS-EMOA and SPEA2 reject the three-parent crossovers `pcx`,
  `undx`, `simplex` and `de`, and MOEA/D rejects `pcx`, `undx` and `simplex`,
  with a clear `ValueError` before the first evaluation. These algorithms mate
  parents two by two, so the run evaluated the initial population and then
  failed with `parents must have shape (n_matings, 3, n_vars)`. NSGA-II, IBEA,
  AGE-MOEA and RVEA, which group parents by the crossover's arity, still run
  them, and MOEA/D still runs `de`. The NSGA-III, SMS-EMOA and SPEA2 tuning
  spaces no longer offer `pcx`, `undx` and `simplex`.
- For real encoding, NSGA-III, SMS-EMOA, SPEA2 and MOEA/D reject crossovers and
  mutations of other encodings, as NSGA-II already did. Configurations such as
  `hux` + `pm` or `sbx` + `bitflip` ran on real variables (bit-flip mapped `x`
  to `1 − x` and ignored the configured probability), and others, such as
  `sbx` + `swap`, failed with internal errors. The initializers `lhs`, `scatter`,
  `sobol`, `halton` and `obl` raise `ValueError` for non-real encodings; on
  binary problems such as ZDT5 they produced fractional "bits" without any
  error. Valid configurations give the same results.
- The binary crossover `spx` and the permutation crossover `oxd`, which
  `vamos.algorithms.available_crossover_methods()` lists, run with NSGA-III,
  SMS-EMOA, SPEA2 and (for `oxd`) MOEA/D, as they already did with the other
  algorithms. VAMOS 1.1.0 rejected them there with `ValueError: Unsupported ...
  crossover`.
- AGE-MOEA and RVEA resolve the probability expression `"1/n"` in every
  encoding; they failed with `ValueError: could not convert string to float:
  '1/n'` after evaluating the initial population. Operator probabilities are
  parsed by one interpreter for every algorithm: a number in [0, 1] or `"k/n"`
  (`k / n_var`, capped at 1), ignoring case and spaces, and `ValueError` for
  anything else. NSGA-II, IBEA and SMPSO rejected `"1/N"` and `" 1 / n "` and
  did not check the range; MOEA/D and the binary, integer and permutation
  builders silently replaced an unreadable value such as `"0,5"` or `"abc"`
  with `1/n` and clipped numbers such as 1.7 or -0.2 to [0, 1]. Those values
  now fail when the operators are built.
- A mutation `prob_factor` is consumed before the operator is built:
  `NSGAIIConfig` with `.mutation("pm", prob_factor=2.0)` failed with `Failed to
  initialize mutation 'pm' ...`. The factor must be positive, and the resulting
  probability is capped at 1, where `prob_factor=2.0` with one variable gave
  2.0.
- An operator override that names the algorithm's default operator keeps the
  default parameters it does not set. With `crossover: {method: sbx, eta: 5}`
  (CLI algorithm block or `build_algorithm(variations=...)`), the probability
  dropped from the default to the operator's own 0.9 without notice, and the
  saved configuration recorded only `eta`. `"sbx"` alone, or `{method: sbx}`,
  means the default SBX, and MOEA/D's `{method: de, f: 0.7}` keeps `cr = 1.0`.
  An override that names another operator still replaces the default whole.
- `normalize_operator_tuple`, `normalize_variation_config` and
  `ensure_operator_tuple` raise `ValueError` for specs they cannot read,
  instead of turning them into something else: `("sbx", 0.9)` gave `("sbx",
  {})`, a mapping without `method` vanished, and a read-only mapping gave
  `None`. Any `Mapping` is accepted, and a parameter set to `None` takes its
  default in every form.

#### Problems and reference fronts

- `cec2009_uf3` searches the CEC2009 domain `[0, 1]^n`. It searched `[-1, 1]`
  for variables 2 to n, a search space twice as wide per variable as the
  CEC2009 instance, with the same Pareto front. UF3 runs of earlier versions
  are not comparable with CEC2009 results; rerun them before comparing.
- `cec2009_uf9` adds the squared distance `y_j^2` of the CEC2009 technical
  report, as jMetal, PlatEMO and `cec2009_uf8` do. It used the multimodal
  distance `2 y_j^2 - cos(4 pi y_j) + 1` of UF5, which leaves the Pareto front
  unchanged but changes the landscape: off the Pareto set, objective values
  differed from jMetal by up to 21.5. With the default NSGA-II, 30 000
  evaluations and seeds 1 to 3, the mean IGD to the previously packaged front
  goes from 0.517 to 0.252. UF9 runs of earlier versions are not comparable
  with CEC2009 results; rerun them before comparing.
- `lsmop1`–`lsmop9` follow the definition of Cheng, Jin and Olhofer (2017), as
  PlatEMO implements it. LSMOP1–4 use the linear linkage `(1 + i/n) x_i - 10
  x_1`, and LSMOP9 divides each group's distance by `sublen * nk`, as LSMOP1–8
  do. VAMOS had copied two deviations from jMetal: the nonlinear linkage `(1 +
  cos(0.5 pi i/n)) x_i - 10 x_1` in all nine problems, and a division by `nk`
  only in LSMOP9. The Pareto fronts do not change, but the Pareto set of
  LSMOP1–4 and the landscape of LSMOP1–4 and LSMOP9 do: with the default
  NSGA-II, 30 000 evaluations, M = 2 and seeds 1 to 3, the mean IGD to the
  packaged front goes from 3.24 to 1.59 on LSMOP1 and from 42.85 to 1.69 on
  LSMOP9. LSMOP1–4 and LSMOP9 results of earlier versions are not comparable
  with results on the original problems; rerun them before comparing.
- `rwa7` and `rwa8` search `[0, 1]^4`, the normalized domain in which Goel et
  al. (2007) fitted the rocket-injector response surfaces, as `re37`, Tanabe
  and Ishibuchi's RE37 and jMetal's RWA7 and RWA8 do. Their last TTmax term is
  `-0.281 * dHA * alpha * dOA`, the sign of `re37`; `rwa7` is now `re37` with
  its objectives in another order, and both problems match jMetal exactly. They
  evaluated the polynomials on the physical design ranges, far outside the
  fitted region, with the opposite sign in that term: TTmax ranged from about
  -2242 to 17093 instead of -0.43 to 1.10, and at the middle of the domain RWA7
  gave `(13.70, 46.78, 1624.26)` instead of `(0.464, 0.482, 0.693)`. The
  packaged `rwa7.csv` (3000 points) and `rwa8.csv` (3999 points) are
  regenerated for the normalized formulation; no point is dominated by jMetal's
  RWA7 and RWA8 fronts or by Tanabe and Ishibuchi's RE37 points. RWA7 and RWA8
  values, fronts and indicators of earlier versions are not comparable with the
  published problems; rerun them before comparing.
- `re24` and `re25` use the domains of Tanabe and Ishibuchi's reference
  implementation (`reproblem.py`), which jMetal shares. RE24 searches `x2` in
  `[0.5, 50]` instead of `[4, 50]`; the cut domain removed every design with f₁
  below 480.5, while the reference front starts at f₁ = 60.5. RE25's wire
  diameter is one of the 19 values from 0.092 to 0.5: the reference lists 42
  diameters but bounds the variable to `[0.09, 0.5]` and rounds it to the
  nearest listed value, so the 23 diameters below 0.092 are never used; VAMOS
  allowed all 42. The packaged `re24.csv` is now the authors' approximated
  front (1000 points), redistributed under the MIT license of their repository
  with its notice (`LICENSE-reproblems.txt`); the previous sample only covered
  f₁ from 480.5 to 481.6. With the default NSGA-II, 20 000 evaluations and
  seeds 1 to 3, the mean normalized IGD to the authors' fronts goes from 0.789
  to 0.012 on RE24 and from 0.0093 to 0.0021 on RE25. RE24 and RE25 results of
  earlier versions are not comparable with results on the reference problems;
  rerun them before comparing. RE22's reading of the reference's "3,10" as the
  value 3.10, and its categorical encoding of the reinforcement area, are now
  documented.
- The inaccurate packaged reference fronts are regenerated. Every packaged
  sample is now mutually non-dominated, and the analytic ones lie on their
  fronts within 1e-9. ZDT and DTLZ points were up to 0.015 and 0.108 off their
  fronts, and DTLZ6 had 103 points; WFG1 was shifted by +0.0695 in both
  objectives, and WFG2 had 42 points; LSMOP9 had 52 % dominated points, and
  C1-DTLZ3 a spurious point at (9, 0); UF6 lacked its isolated point (0, 1);
  `cec2009_uf9.csv` was a copy of UF8's spherical front, while UF9's front is
  two pieces of the plane `sum(f) = 1`; DC2-DTLZ1 and DC2-DTLZ3 lay on an
  infeasible layer at `g ≈ 62`.
- The packaged fronts of `re21`, `re31`–`re34`, `re37`, `re41`, `re42` and
  `re61` are now Tanabe and Ishibuchi's approximated fronts, redistributed
  under the MIT license of their repository (`LICENSE-reproblems.txt`); between
  12 % and 52 % of the previous samples' points were dominated by those fronts.
  `re91`, `rwa3`, `rwa9` and `rwa10` are regenerated by dense sampling and
  polishing: a random sample of 100 000 designs dominated between 3 % and 26 %
  of their previous points, and dominates none of the new ones.
  `docs/reference/problems.md` lists the origin of every packaged front, and
  `experiments/scripts/generate_reference_fronts.py` regenerates these files
  without pymoo.
- IGD, IGD+, epsilon and normalized HV values computed with the replaced front
  files change. With the default NSGA-II and seeds 1 to 3, the mean IGD goes
  from 1.263 to 1.453 on WFG1 (M = 2), from 0.252 to 0.318 on UF9 and from
  1.695 to 1.971 on LSMOP9 (M = 2); ZDT1 and DTLZ2 barely change. Indicator
  values of earlier versions that used these files are not comparable;
  recompute them.
- The automatic reference front matches the problem's number of objectives, or
  is absent. C-DTLZ and DC-DTLZ at their default three objectives, WFG and
  LSMOP at three, and DTLZ at two received a front with another number of
  objectives.
- `get_zdt_reference_front("zdt6")` returns ZDT6's Pareto front `f2 = 1 -
  f1^2`, and the ZDT3 front keeps only the non-dominated part of the curve.
  `compute_normalized_hv` changes for ZDT6 and ZDT3; runs and studies use the
  packaged fronts and do not change.
- The documentation cites DC-DTLZ's primary definition, the authors'
  implementation, which VAMOS follows, and notes that jMetal's and PlatEMO's
  DC1 and DC3 differ.
- Problems with a fixed number of variables reject an `n_var` override they
  would ignore: the RE and RWA problems, the TSPLIB `kro*100` instances, and
  `welded_beam`, `fs_real` and `ml_tuning`. The MW problems other than MW4, MW8
  and MW14 reject an `n_obj` override in the same way.
  `make_problem_selection`, `optimize`, studies and the CLI (`--n-var`,
  `--n-obj`) raise `ValueError`, or a usage error in the CLI. The selection
  reported the requested dimension while the problem used its own:
  `make_problem_selection("re21", n_var=10)` reported 10 variables for a
  four-variable problem. The `re24` selection now reports its two variables,
  not four, and the RE registry labels name the formulations they implement.
- Impossible registry dimensions are rejected with `ValueError` when the
  problem is built: WFG with no distance variable (`n_var < k + 1`, including
  `wfg*` with `n_obj >= 13` and the default `n_var`), DTLZ with `n_var <
  n_obj`, LZ09 with `n_var < 3`, and non-integer (float or boolean) `n_var` or
  `n_obj` in named selection. VAMOS 1.1.0 produced NaN objectives, `IndexError`
  or `ZeroDivisionError`, or accepted `n_var=2.5`.
- A problem's objective or constraint function that returns a flat 1-D array of
  length `N * n_obj` (for example `np.hstack([f1, f2])`) raises `ValueError`
  pointing to `np.column_stack`, in `make_problem` (scalar and vectorized) and
  in `Problem.objectives()` and `constraints()`. VAMOS 1.1.0 reshaped it to
  `(N, n_obj)`, silently mixing the objectives of different solutions. A 1-D
  result is still accepted when unambiguous (one objective or constraint, or
  one solution).
- `optimize` checks the problem object before the first evaluation and raises
  `ConfigurationError` naming the faulty attribute: non-integer or non-positive
  `n_var` or `n_obj`, negative `n_constraints`, unknown `encoding`, missing or
  non-callable `evaluate`, and `xl` and `xu` that are not finite scalars or
  `(n_var,)` arrays with `xl <= xu`. VAMOS 1.1.0 failed later with
  `AttributeError` or NumPy errors (`problem=None`, `xl > xu`, wrong lengths,
  infinite bounds), or assumed `n_var=10` and `n_obj=2` when they were missing.
- A problem that writes constraint values without declaring `n_constraints` (an
  `out["G"]`, as with pymoo's `n_ieq_constr`, or an overridden
  `Problem.constraints()`) raises `EvaluationError` at the first evaluation; it
  was optimized as unconstrained without warning. The error for non-finite
  objective or constraint values says how to penalize failed evaluations.
- `make_problem` accepts NumPy integers for `n_var`, `n_obj` and
  `n_constraints`, rejects NaN or infinite bounds with `BoundsError` and a
  negative or non-integer `n_constraints` with `ProblemDimensionError`, and its
  documentation names the errors it raises (`ConfigurationError`,
  `ProblemDimensionError`, `BoundsError`). VAMOS 1.1.0 documented `TypeError`
  and `ValueError`, rejected `numpy.int64` dimensions and accepted those bounds
  and counts.
- An unknown problem name raises `InvalidProblemError`, which is also a
  `KeyError`, with the close matches first and the full list in
  `details["available"]`. VAMOS 1.1.0 raised a bare `KeyError` that listed
  every registered name.
- `fs_real` and `ml_tuning` (`FeatureSelectionProblem`,
  `HyperparameterTuningProblem`) no longer turn every training exception into
  validation error 1.0: data failures warn with their count, an evaluation
  where every model fails raises `EvaluationError`, and other exceptions
  propagate. Only runs where model training fails change.

#### Results, indicators and archives

- `OptimizationResult.front()`, `best()`, `top_k()` and `top_k_report()` rank
  feasible solutions first. When the selected rows carry constraint values `G`,
  only the feasible rows are Pareto-filtered or, when none is feasible, the
  least violated ones: the rule the algorithms use for their non-dominated
  results. The helpers used to filter by `F` alone, so with `optimize(problem,
  algorithm="nsgaii")`, which returns the final population, `best()` could
  return an infeasible solution while feasible ones existed: with one
  constraint, `pop_size=20`, `max_evaluations=20` and `seed=7`, 12 of the 20
  rows were feasible, `front()` had 4 rows, 3 of them infeasible, and `best()`
  was infeasible. Results without `G`, which includes unconstrained problems
  and runs with `constraint_mode="none"`, and results whose rows are all
  feasible are unaffected.
- `top_k()` and `top_k_report()` default to `source="auto"`: the external
  archive when the result has one and the top-level result otherwise. Their
  former default, `source="archive"`, raised `ResultSelectionError` on a result
  without an archive, which is the default for `optimize`. A call that worked
  with the default still ranks the archive.
- `OptimizationResult.top_k(method="kmeans")` gives the same selection on every
  call; it drew an unseeded generator for its k-means initialization.
- Results carry the constraint values of their rows when they come from an
  external archive or SMPSO's leaders, and in their population and archive
  entries. Before, selection helpers, `vamos bench` and Studio counted those
  rows as feasible.
- Every indicator rejects NaN and infinite values in fronts, reference fronts
  and reference points with `ValueError`. MooCore used to drop such rows
  silently: `HVIndicator(ref).compute([[nan, 0.1], [0.5, 0.5]])` returned 0.25.
  `pareto_filter` rejects rows with NaN, which it used to keep as
  non-dominated, and non-empty inputs that are not two-dimensional, which it
  used to return unchanged.
- Without MooCore, the hypervolume contributions for three or more objectives
  follow MooCore's semantics: dominated and duplicated points contribute 0 and
  no longer reduce the others, and the values are never negative. With
  dominated points in the set, the two environments used to differ by up to
  0.039.
- Without `moocore`, hypervolume archive pruning (`pruning: hv` and `mc_hv`)
  and the greedy hypervolume subset selection (`select_top_k_hv_greedy`, used
  by `top_k(method="hv_greedy")`) ranked points by crowding distance instead of
  hypervolume contribution, with a single warning per process. Both now use
  exact hypervolume contributions with or without `moocore`, with the `moocore`
  semantics: points that do not dominate the reference point, dominated points
  and duplicates contribute 0. Hypervolume archive pruning with three or more
  objectives requires `moocore` and raises `DependencyError` when the archive
  is built. Only installations without `moocore` (the `compute` extra) are
  affected: with two objectives, HV-pruned archives and greedy HV selections
  change to the ones `moocore` gives (on a 21-point ZDT1 front with k = 5 and
  reference (1.1, 1.1), the greedy selection's HV goes from 0.7423 to 0.7663);
  with three or more objectives, runs that request HV archive pruning fail
  instead of pruning by crowding distance. Install
  `vamos-optimization[compute]`.
- The `knn` pruning of external archives uses the SPEA2 truncation operator. It
  ranked points by the distance to their second-nearest neighbour alone and
  broke ties by row order, which could remove an extreme point.
- The `archive.external` hook offers its archive only the rows that the
  previous notification did not contain. Steady-state algorithms such as MOEA/D
  notify after every evaluation, and VAMOS 1.1.0 re-inserted the whole
  population each time: on ZDT1 with population 20 and 2000 evaluations, 34 637
  rows in 1981 updates. A bounded hook archive no longer receives rows it has
  already pruned. The hook's archive trace separates `dominated_or_duplicate`
  rows from `pruned_by_capacity`, with run totals, `prune_reason` names the
  pruning policy only when the capacity removed rows, and `inserted` counts the
  rows offered instead of the size of every notification.
- `knee_point_scores` picks the knee on the ideal side of the chord between the
  extreme points of the non-dominated set (Das, 1999; Deb and Gupta, 2011). It
  used the unsigned distance to the chord between the extreme points in f1,
  over every point, so on a concave front it picked the point farthest from the
  ideal, and a dominated point above the chord could win. Dominated points now
  score `inf`. A concave or linear front has no knee: the helper emits a
  `UserWarning` and returns the extreme point with the smallest f1, and
  Studio's "Knee point" ranking shows that warning on the Results page. The
  knee changes on concave or linear fronts and on fronts with dominated points:
  on the concave front `f2 = 1 − f1²` it was (0.5, 0.75), the point farthest
  from the ideal.
- `explore_result_front` shows each point with its own decision variables;
  after a dominated row, every hover text showed another solution's X. It and
  `plot_result_front` plot the feasible non-dominated solutions, as
  `OptimizationResult.front()` selects them.

#### Termination and evaluation

- The hypervolume stopping criterion `termination=("hv", {...})` measures the
  target against the user's fixed `reference_point` and drops the rows that
  fall outside it, as moocore, pymoo and jMetal do; a population with no row
  inside the reference box has HV 0. It affected NSGA-II, NSGA-III, MOEA/D,
  SMS-EMOA, SPEA2, IBEA and SMPSO. With `allow_ref_expand=True`, each check
  widened the reference to the current population, so the target was met at
  once: on ZDT1 (30 variables, population 100, reference [1.1, 1.1], target
  0.6, cap 30 000) every algorithm stopped after 100 or 200 evaluations, four
  of them reporting `hv_reached=True`, with a real HV of 0. They now run until
  the target is met: NSGA-II stops after 3800 evaluations with HV 0.605, and
  MOEA/D after 17 282 with HV 0.600. Without that key, the run aborted with
  `ValueError: reference_point must dominate all points.` as soon as one row
  left the reference box, which on ZDT happens in the initial population; the
  same applied to `--hv-threshold` on the CLI. HV-stopped runs of earlier
  versions that used `allow_ref_expand` ended after the initial population;
  rerun them.
- `--hv-threshold` measures its target against the reference front's nadir plus
  10 % of its range in each objective (1.1 after normalization, as pymoo and
  jMetal do). It added at least 5 to every objective, so most of the
  hypervolume came from the margin, and a target of 0.5 was met by 100 random
  ZDT1 designs (0.558). On ZDT1 the reference goes from (6, 6) to (1.1, 1.1).
- `hv_convergence` with `ref_point: "auto"` fixes its reference at the first
  sample and records it as `reference_point`. It recomputed the reference from
  every sample, so the series was not comparable and convergence was declared
  too early. This stop and `--hv-threshold` now stop later, at a genuinely
  higher hypervolume; thresholds and traces of earlier versions are not
  comparable.
- `MultiprocessingEvalBackend` keeps one pool of worker processes from its
  first evaluation until `close()` (it is also a context manager), instead of
  starting and stopping one in every generation. `optimize()`, the CLI and
  Studio close the pools they create from a strategy name; an instance you pass
  stays open for you to close.
- `MultiprocessingEvalBackend(timeout=...)` bounds each evaluation: when it
  expires, the pending work is cancelled, the workers are stopped and
  `EvaluationError` is raised; the timeout never fired. `n_workers` and
  `timeout` values that are not positive are rejected (`n_workers=0` meant
  every CPU and a negative count one worker), and the default is the CPU count,
  at most 8.
- `DaskEvalBackend` lets the problem's `EvaluationError` through instead of
  reporting a failed Dask evaluation, and its opt-in serial fallback no longer
  evaluates the population a second time.

#### Run artifacts and studies

- Run provenance classifies the installed VAMOS sources with one rule when
  saving and when verifying. A wheel installed in a virtual environment inside
  the user's own Git project was recorded as a VAMOS checkout, with the
  project's revision and dirty state: a later commit to the project made
  `verify_run` report `compatible`, so `reproduce` refused the run and an
  interrupted study could not be resumed, and VAMOS ran `git status` and `git
  diff` on that project at every save. A Git repository now counts as a
  checkout only when its `src/vamos` is the imported package, or when VAMOS is
  installed in editable mode. Runs of an editable install without Git, which
  were saved as wheels and verified as checkouts, now verify as `exact` while
  the sources are unchanged. Checkouts and editable installs also record a
  fingerprint of the packaged data under `vamos/resources`, so a TSPLIB
  instance or weight file changed after saving a run blocks its exact replay
  instead of failing at replay. Runs saved by earlier versions verify as
  before; runs and studies that VAMOS 1.1.0 saved in the wheel-in-project
  layout keep the project's revision as their source, so rerun them.
- A run of a `vamos.make_problem` problem, or of any other user problem, is
  recorded with provider `custom_python`, a component ID from the module and
  qualified name of its function or class, and `manual` replayability. Only a
  problem requested by registry name, or an instance of the exact class the
  registry builds for its name, is a built-in component. User code was recorded
  as a built-in component named after its `name` with `exact` replayability: a
  function named `zdt1` then shared the registered ZDT1's task ID, and
  `vamos.reproduce` ran the registered ZDT1 instead.
- Saving a result returned by `load_result` again records a manual copy, with
  the entry point `vamos.save_result` and the time of the new save. It was
  saved as an `exact` run of `vamos.optimize` with the original timestamps,
  even when its arrays had been changed after loading.
- Loading or verifying a run or a study whose JSON is nested too deeply for
  Python's parser, or holds an integer longer than
  `sys.get_int_max_str_digits()`, raises `ArtifactResourceLimitError` (limits
  `max_json_depth` and `int_max_str_digits`) instead of an untyped
  `RecursionError` or `ValueError`. `vamos results verify` exits with 3 instead
  of 1, and the `vamos study` commands report `RESOURCE_LIMIT` with exit 3
  instead of an infrastructure failure with exit 7.
- `load_result` and `verify_run` reject a result bundle whose `F`,
  `population/F`, `archive/F` or `reference_directions` columns differ from the
  resolved problem's `n_obj`, or whose `X`, `population/X` or `archive/X`
  columns differ from its `n_var`, with `MalformedResultBundleError`.
  `save_result` refuses such a result before writing anything.
- A save lock left by a process that died while saving is identified: the lock
  records its process, host and start time, and the `OutputCollisionError` of a
  later save names the lock and the unfinished staging directories and says
  they can be deleted. Before, the error said only that another save owned the
  destination.
- On Windows without long paths enabled, saving a run (`save_result`,
  `reproduce`, `vamos quickstart`) or creating a study failed with a bare
  `FileNotFoundError` when the staging paths reached 260 characters (248 for a
  directory), even for a destination well under that limit; a study reported
  only `actual='FileNotFoundError'`. The error now gives the path length, the
  limit and the ways out: `LongPathsEnabled`, a shorter destination or the
  `\\?\` prefix. It keeps its type, and a study keeps its
  `ATOMIC_PUBLICATION_FAILED` reason, with the original message now in
  `actual`. Run staging directories use a 22-character token instead of 32 hex
  digits. The troubleshooting guide documents the limit and `NUMBA_CACHE_DIR`
  for the Numba cache.
- `plan_study`, `create_study` and `vamos study plan` and `create` reject study
  plans whose tasks `Study.run()` cannot execute, before anything is published.
  `eval_strategy="multiprocessing"` or `"dask"` fails with
  `UNSUPPORTED_EVALUATION_BACKEND`: studies execute tasks with serial
  evaluation only, so those plans were reported `ready` and then every task
  failed with a non-retryable reconstruction error. A negative seed fails with
  `INVALID_STUDY_SPEC`: the plan was `ready` and the task then failed at run
  time as a retryable error, so `retry` repeated it. `StudySpec` still accepts
  these values, and studies published by 1.1.0 with them still load. Use
  `vamos.optimize(..., eval_strategy=...)` for parallel evaluation of a single
  run.
- Durable studies survive a writer that dies mid-write and readers in other
  processes. `load_study` ignores the `.tmp-` files that the atomic writer
  leaves when a process is killed between writing a temporary and replacing its
  destination, and `resume` deletes them; such a study was rejected
  (`EVENT_HASH_CHAIN_BROKEN` or `UNEXPECTED_LAYOUT`) until the file was removed
  by hand. A reader that lists a directory while the study is being written no
  longer fails on the writer's in-flight temporary. On Windows, the running
  process retries a replace for about 1.5 s when another process (`inspect`,
  `vamos study inspect`, Studio) has the document open, instead of stopping
  with `CHECKPOINT_WRITE_FAILED`. Loading reads each study document once, so
  the parsed content and the canonical-bytes check always come from the same
  version of the file.
- A failed study task records its cause. For a VAMOS error, the task, the
  attempt and the failed run keep its type and its text on one line, up to 500
  characters, with absolute paths (spaces included) replaced by `<path>`. For
  other exceptions, and for a VAMOS error raised from one, only the types are
  kept. The process running the study logs the same message and where each
  exception was raised, never the exception text. The record used to hold a
  fixed text that pointed to an exception nobody could see. Failures that the
  same task would repeat are no longer retryable: configuration and problem
  errors, and evaluations that write non-finite or misshaped values; `retry`
  used to repeat them until `max_attempts_per_task`.

#### Command line

- The `vamos` optimization command runs registry problems that do not declare
  `n_constraints`. Its problem check read the attribute directly, so the 48 RE,
  RWA, ZCAT, `fs_real` and `ml_tuning` problems stopped with an
  `AttributeError` before optimizing, and so did the `bio_feature_selection`
  and `chem_hyperparam_tuning` templates of `vamos quickstart`. A missing
  attribute now means no constraints, as in `vamos.optimize`.
- RVEA runs from the `vamos` command with the default population. The CLI
  builder always used 12 reference-vector partitions, which match only 13
  individuals on two objectives or 91 on three, so `--algorithm rvea` failed
  with the default population of 100. The partitions now follow
  `--population-size` (99 for 100 individuals on two objectives); a population
  with no exact lattice is rejected before optimizing with the nearest valid
  sizes. Combinations that already ran (13 individuals on two objectives, 91 on
  three) keep 12 partitions and give the same results.
- A failed run no longer stops `vamos --algorithm both` or `--problem-set`: the
  error is logged, the other runs and the summary complete, and the command
  exits with a nonzero status that lists the failed runs.
- The `--nsga3-*` options of the `vamos` command and a `defaults.nsgaiii` block
  of a config file reach NSGA-III. The options were stored under `nsga3_*`
  names but read as `nsgaiii_*`, so NSGA-III silently ran with SBX (probability
  1.0, eta 20) and polynomial mutation (1/n, eta 20) and their probabilities
  were not range-checked. NSGA-III runs from the CLI that passed `--nsga3-*`
  options or a `defaults.nsgaiii` block now use the requested operators, so
  their results change.
- An operator parameter given without its operator, such as
  `--nsgaii-crossover-prob 0.5` without `--nsgaii-crossover`, is rejected with
  a usage error (exit status 2) instead of a `ValueError` traceback. The CLI
  also rejects before running, with a usage error, a negative `--seed`, a
  `--n-var` or `--n-obj` below 1, `--n-obj` on a problem with a fixed number of
  objectives, and a `--selection-pressure` larger than the population a problem
  runs with.
- The warning about algorithm-specific options uses an explicit option map: it
  covers the `--nsga3-*` options and `--external-archive-size` (read by
  NSGA-II, SPEA2 and SMPSO; its help said NSGA-II only), names the flag as
  typed, and no longer reports the blocks for other algorithms in a config
  file.
- Every field of a CLI config's algorithm block reaches the algorithm, in
  `defaults` and in `problems.<name>`; the CLI dropped most of them without
  warning (`moead: {neighbor_size: 7}` ran with 20), and a `defaults` block
  lost its `selection` and `repair` for six algorithms. An out-of-domain value
  in any problem stops the command before any run. CLI runs that set these
  fields now use them, so their results and `task_id` change.
- The CLI config accepts operators in list form (`[sbx, {prob: 0.8}]`), as
  ablations do; it ignored them, or raised `TypeError` in a problem block.
- In a CLI config file, the `stopping` and `archive` blocks merge key by key,
  nested mappings included: `defaults`, then the top level, then
  `problems.<name>`. A partial problem override such as
  `stopping.hv_convergence: {patience: 10}` used to replace the whole inherited
  block, which turned the HV stop off, and a top-level block replaced the
  `defaults` one. `problems` keys are case-insensitive, like the problem
  registry; in 1.1.0 `problems.ZDT1` passed validation, but none of its
  overrides reached the run. Two keys that differ only in case are rejected,
  and a key repeated in a YAML or JSON config mapping is an error instead of
  silently keeping the last value.
- The values of `stopping.hv_convergence` and `archive.external` are checked
  when the spec is validated, before any run: integer fields, `window >= 2`,
  the `statistic` and `epsilon_mode` literals, `confidence` in `(0, 1)`, a
  `ref_point` of `auto` or two finite numbers, and positive integer capacities.
  Misspelled or out-of-range values ran as something else (`statistic: medain`
  as the median, `confidence: 95` as 1), and a `ref_point` of the wrong length
  made the stop never trigger. `archive.external.deduplicate_in: decision` or
  `both` is rejected for the hook archive, which stores objective vectors only;
  the run used to abort with an `IndexError`. `capacity: null` is accepted and
  keeps an unbounded hook archive. `ExternalArchiveConfig` rejects boolean or
  fractional `capacity` and `truncate_size`, non-finite tolerances and invalid
  `hv_ref_point` values, and warns about a field that the chosen pruning
  ignores.
- A `stopping.hv_convergence` or `archive.external` block that cannot be built
  (for example `window: 1` or `pruning: hypervolume`) stops `vamos` with exit
  code 2 before any run. It was disabled with a warning, and the run used its
  full evaluation budget without the hooks the config asked for.
- `vamos --config FILE --validate-config` runs the checks that precede a run:
  option values, each selected problem's effective settings and algorithm
  fields, and its stopping and archive hooks. It printed `Config OK.` after
  checking only the keys, so a config with `max_evaluations: -48` passed and
  then failed when run. Errors about an option set in the config file name its
  key, such as `defaults.nsgaii.crossover.prob`.
- The `vamos` run command checks every run directory before the first
  optimization. A taken one, or one that two runs of the command share, exits
  with code 8, the output-collision code of `vamos results` and `vamos
  reproduce`; the command used to optimize first and then stop with a traceback
  and exit code 1, after writing the earlier runs of a series. `vamos
  quickstart` without `--seed` takes the first free seed, and its `Re-run:`
  command works; it always used seed 42, so a second run failed after
  optimizing.
- `vamos --eval-strategy dask --dask-address ADDRESS` evaluates on that
  scheduler. The address never reached the evaluation backend, so Dask from the
  CLI always failed with "DaskEvalBackend is not connected to a Dask
  scheduler". A `problems.<name>` block of a config file can also set
  `dask_address`. The run artifact records the `dask` strategy, not the
  address, and the client the run opens is closed when the run ends. Dask
  connection errors suggest starting or reaching a scheduler, and an
  unreachable `--dask-address` raises a `ConfigurationError` that names the
  address instead of an `OSError` traceback.
- `--hv-threshold` and `hv_threshold` log a warning when the selected algorithm
  is not NSGA-II, the only one that stops on the hypervolume target; the run
  still uses its full evaluation budget, as before. The CLI guide's config
  example no longer pairs `hv_threshold` with MOEA/D.
- CLI run artifacts record only `n_var` and `n_obj` from a `problems.<name>`
  block as problem configuration; the whole block made `vamos reproduce` reject
  the run. Their `requested_spec` is the effective request, with the
  command-line overrides, and keeps the config file under `config_file`. `vamos
  results verify` proves the resolved spec with the reconstruction `reproduce`
  uses, so it no longer reports `exact` for a run `reproduce` rejects. F and X
  do not change; CLI runs with a problem block or `--config` get a different
  `requested_spec` and `task_id`.
- `vamos create-problem` no longer overwrites an existing file without
  `--force`, rejects `--n-var`, `--n-obj` and `--budget` below 1, escapes the
  problem name so any name gives a valid script, avoids default file names that
  hide a module such as `numpy.py`, and its script reports `result.front()` as
  the non-dominated solutions instead of the whole population.
- `vamos study summarize` without `--json` prints its task table, and its next
  step is the command `vamos study inspect STUDY_DIR`; it printed only the
  study header and a bare `inspect`. The human output of `vamos study` errors
  shows the field and the value that the JSON output carries.
- `vamos profile` runs each engine once untimed, then `--repeats` timed runs
  (default 3), and reports their median and range. It timed a single cold run,
  so the first Numba run measured mostly JIT compilation and the speedups
  depended on the order of `--engines`. Unknown engines and problems exit with
  code 2 before any run. A failed engine is reported with its error and makes
  the command exit with 1; it was recorded with time 0 and exit code 0. The
  unused `ProfileResult.memory_mb` field is removed, and the CSV gains the
  `time_min`, `time_max`, `repeats` and `error` columns.
- The hypervolume of `vamos profile` uses the nadir of the problem's reference
  front plus 10 % of its range, or `--hv-reference`. It works with any number
  of objectives and leaves out the points outside the reference; without a
  reference front it is not computed, a warning says so and the CSV cell is
  empty. It used `[1.1, ..., 1.1]` whatever the problem's scale, failed with
  three or more objectives, and stored 0.0 where the table said N/A. Profile
  hypervolumes of earlier versions are not comparable.
- `optimize`'s "Auto-selected defaults" log lists only the values it chose,
  with the population size the run uses, and `verbose=True` restores the
  "vamos" logger when the run ends. VAMOS 1.1.0 also listed `pop_size` and
  `max_evaluations` set by `algorithm_config` or `termination`, and left its
  handler on the logger for the rest of the process.
- Install hints name the PyPI distribution, `pip install
  "vamos-optimization[<extra>]"`, instead of `vamos[<extra>]` (another project)
  or a checkout-only `pip install -e`, and `vamos studio` without the `studio`
  extra exits with code 2 and that hint instead of a traceback.
  `InvalidAlgorithmError` lists all nine built-in algorithms.

#### Benchmarks, ablations and analysis

- `vamos bench` and `vamos ablation` compute indicator references per problem
  and number of objectives. They built one HV reference point and one IGD,
  IGD+, epsilon and Hausdorff reference front from the runs of every problem in
  the suite, so one problem's values depended on the other problems and `lab/`
  reported significant differences that do not exist; suites that mix objective
  counts failed. `hv_source` is now `problem`. HV, IGD, IGD+, epsilon and
  Hausdorff in `metrics.csv`, the LaTeX tables, `lab/` and
  `AblationResult.summary_rows()` change for every suite or ablation with more
  than one problem: in `ZDT_small` (NSGA-II and SPEA2, 5 seeds) the median ZDT1
  HV of NSGA-II goes from 13.568 to 0.887 and its IGD+ from 0.2510 to 0.0016,
  and `lab/wilcoxon` goes from `+++--` (HV) and `--+-+` (IGD+) to `--+--` for
  both. Rerun benchmarks of earlier versions before comparing them.
<!-- REVIEW(#126): pull request #126 merged into #125's branch, not into main, and has no "Changelog entry" section; this entry summarizes its description. Keep or drop. -->
- `vamos bench` and `vamos ablation` compute indicators and their references
  from feasible solutions only (every stored constraint `G <= 0`). A run
  without feasible solutions gets HV 0 and +∞ for IGD, IGD+, epsilon and
  Hausdorff, and stays in the rank tests; it used to get a positive HV (0.6312
  for MOEA/D on MW2, seed 2), and infeasible points shaped the references of
  the other runs. `metrics.csv` gains `n_feasible` and `feasibility_known`,
  which is `false` when a run stores no constraint values, such as
  unconstrained problems and SMPSO with `result_mode: non_dominated`; such runs
  count every solution.
<!-- /REVIEW -->
- `pairwise_wilcoxon`, and the per-problem markers of `vamos bench`, count two
  equal infinite scores, such as the IGD+ of two algorithms with no feasible
  solution on the same problem, as a tie, and give p = 1 to two algorithms
  equal on every problem. SciPy turned the pair into `inf - inf = NaN` and
  dropped it with a RuntimeWarning, which changed the statistic and the
  p-value; with SciPy 1.12 or earlier an all-equal pair raised `ValueError`,
  and later versions warned about a division by zero.
- The statistics of `vamos bench` use the right tests and one indicator
  orientation. The LaTeX tables mark an algorithm that differs from the best of
  its problem with a two-sided rank-sum (Mann-Whitney U) test on the runs, as
  `lab/` does; they paired runs by seed in a signed-rank test, but runs with
  the same seed are independent, and with five or fewer seeds that test cannot
  reach p < 0.05, so the tables carried no marks. When the number of runs
  cannot reach `alpha` (three runs per algorithm give at best p = 0.1), the
  pair is not tested, the table caption says so, and a warning is logged. Every
  indicator except the hypervolume family is minimized under all its aliases;
  with `avg_hausdorff`, `hausdorff`, `eps` or `epsilon_add` the tables put the
  worst value in bold. `statistics.json` gains a `pairwise` list per indicator:
  the signed-rank tests on per-problem means for every pair of algorithms, now
  also with two algorithms, with Holm-adjusted p-values. A mean of +inf from a
  run without feasible solutions shows as `inf` instead of `inf +/- nan`. Marks
  and bold cells of the LaTeX tables change: on ZDT_small (NSGA-II and SPEA2,
  five seeds) the ZDT3 rows of the HV, IGD+ and average Hausdorff tables now
  mark SPEA2 as significantly worse (p = 0.0079; the paired test gave 0.0625),
  and the average Hausdorff table moves the bold cell from SPEA2 to NSGA-II,
  the smaller value, on all five problems. Rerun `vamos bench ...
  --only-report` to regenerate the tables of an earlier benchmark.
- `vamos bench` (experimental) checks its inputs before any run. `--config`
  accepts `population_size`, `offspring_population_size`, `max_evaluations`,
  `seeds`, `engine` and `eval_strategy`, and `max_evaluations` and `seeds` now
  replace the budget and seeds of every experiment; any other key, or an
  invalid value, exits with code 2, where VAMOS 1.1.0 ignored them. An unknown
  suite exits with code 2 instead of a `KeyError` traceback. Every study is
  planned before the first run, and an occupied `study-NNNN` path exits with
  code 5 and runs nothing; VAMOS 1.1.0 ran the earlier studies, then raised
  `StudyOutputCollisionError`.
- `vamos bench --only-report` rewrites `summary/metrics.csv` with the requested
  `--metrics`; it used to leave the file stale. It exits with code 2 when the
  output holds no study of the suite. The `vamos bench` report draws with
  `matplotlib.figure.Figure`: it no longer switches the process-wide backend to
  `Agg` or reuses pyplot figure 1, and its boxplot labels work with matplotlib
  3.7 and 3.8. The `WFG_heavy` suite description no longer claims that it
  requires pymoo.
- Ablation variants run what they declare. `run_ablation_plan` and `vamos
  ablation` read only `population_size`, `offspring_population_size` and
  `eval_strategy` from `base_config` and `config_overrides`, and dropped every
  other key without a warning, so a variant that set `pop_size`, `crossover` or
  any other field of the algorithm configuration ran the base configuration and
  the table showed no effect. Those fields now reach the study's
  `algorithm_configs`. A key that is neither an ablation setting nor a field of
  the algorithm configuration fails with a `ConfigurationError` that suggests
  the closest field. `pop_size` and `offspring_size` set the same sizes as
  `population_size` and `offspring_population_size`, and a variant's value wins
  over `base_config` whichever name each one uses. On ZDT1 (NSGA-II, 400
  evaluations, seeds 1 and 2), a variant with `pop_size: 20` gave the baseline
  HV bit for bit (3.9377 and 3.9947); it now gives 5.6356 and 5.1674.
- Operators in the mapping form `{method: sbx, prob: 0.5}`, which the ablation
  schema accepts and the CLI configuration documents, work in ablation
  `variations` and `config_overrides`; they failed with `UNAVAILABLE_COMPONENT`
  when that variant was planned, after the earlier variants had run, and no
  summary was written. Every variant of an ablation is planned before the first
  one runs, so an invalid variant stops the ablation before any run.
- `vamos ablation` writes each variant's studies to
  `<output_root>/<variant>/study-NNNN`, as `--dry-run` shows, and
  `output_root_by_variant`, `per_variant_output_root` and `--flat-output` take
  effect; every study used to go to `<output_root>/study-NNNN`. With the Python
  API, an `output_root` key in a variant's configuration moves its studies.
- `plot_critical_distance` draws the Nemenyi critical difference with the
  critical value q_alpha for the number of algorithms k (Demšar, 2006, Table 5,
  and otherwise the studentized range quantile with infinite degrees of freedom
  divided by √2). It used the k = 4, alpha = 0.05 value for every k, and the k
  = 3 value for alpha = 0.10; other alpha values fell back to 0.05 with a
  warning. Any alpha strictly between 0 and 1 is now accepted, and other values
  raise `ValueError`. Studio's comparison and the plot share one
  implementation, `vamos.ux.analysis.stats.nemenyi_cd`. The CD bar of
  `plot_critical_distance` and of the `cd_<metric>.pdf` plots of `vamos bench`
  changes whenever k ≠ 4 or alpha ≠ 0.05: with ten problems and alpha = 0.05,
  the CD of ten algorithms goes from 3.48 to 4.28 (it was 19 % too small and
  showed differences that are not significant), and that of two algorithms from
  0.81 to 0.62. Redraw critical-difference diagrams of earlier versions before
  relying on them.
- `plot_pareto_front_2d` and `plot_pareto_front_3d` no longer invert their
  axes. With minimization, the inverted axes put the ideal corner at the upper
  right and made a convex front look concave. Pass `invert_axes=True` for the
  previous view.
- `plot_reduced_front` labels its axes with the objectives it kept. It used the
  first entries of the original labels (or `f0`, `f1`), which named objectives
  it did not draw. Its module now imports Matplotlib and pandas only when a
  helper needs them.
- The `--live-viz` window (`LiveParetoPlot`) rescales its axes at every update;
  it kept the first generation's limits, so a converging population left the
  view. It also labels the axes with the plotted objectives, is titled as a
  population plot, and subsamples large populations with a seeded NumPy
  generator (`rng_seed`) instead of the global `random` state.
- The archive guide described `mc_hv` as a Monte Carlo estimate (it is the
  exact `hv` pruning) and `hv_ref_point` as required (it is optional).

#### Tuning

- `RacingTuner` returns a configuration that is still alive when the race ends,
  chosen by its aggregate over the problem/seed blocks that every survivor was
  evaluated on. Multi-fidelity racing compares the survivors on the last
  fidelity level they reached. The final choice used to include eliminated
  configurations and compared each one on its own blocks, so a configuration
  eliminated early on easy instances, or scored only at a cheap fidelity, could
  be returned. With two instances with score offsets 100 and 0, 12
  configurations of quality `q` from 0.05 to 0.95 and 10 seeds, rank mode with
  tuner seeds 0, 1 and 2 returned `q = 0.214`, eliminated after 3 blocks, and
  now returns 0.95; with statistical tests and noise 0.5, tuner seed 2 returned
  `q = 0.05`, the worst of the 12, and now returns 0.786. This affects
  `RacingTuner`, its multi-fidelity mode, the racing phase of `TwoPhaseTuner`,
  and `vamos tune --backend racing`; the trial history is unchanged.
- Racing with elitist restarts (`Scenario(use_elitist_restarts=True)`) works
  with conditional search spaces. `make_neighbor_config` drew the parameters
  that an elite left out from a fresh `ParamSpace.sample`, which omits inactive
  parameters, so a restart failed with a `KeyError` on every built-in space.
  Neighbours now keep the active assignment of the space: a parameter that the
  perturbation activates is drawn from its own range, and one it deactivates is
  dropped with its descendants.
- Configurations spawned by an elitist restart are evaluated on the blocks the
  race has already played before they enter the elimination tests, as in
  elitist irace, and a restart spawns only as many as `max_experiments` leaves
  room to catch up. They used to enter with no blocks, and statistical
  elimination waits until every alive configuration has the same number of
  blocks, so no configuration was eliminated after the first refill: with one
  instance, 40 seeds, 12 configurations, `max_experiments=600` and statistical
  tests, the race eliminated configurations at 1 of 40 stages and now at 11 of
  29, exploring 37 configurations instead of 14. This affects `RacingTuner`
  with `use_elitist_restarts=True`, including the racing phase of
  `TwoPhaseTuner` with `phase2_use_elitist_restarts=True`; `vamos tune` does
  not enable restarts.
- Racing's `ModelBasedSampler` and `filter_active_config`, which writes the
  trial history and `best_config_active.json`, drop descendants of an inactive
  parameter. In a user space with conditions on the direct parent only (`use_x
  -> x_kind -> x_eta`), about 12 % of the sampler's draws kept `x_eta` with
  `use_x` false, and the filter kept it in the saved configuration. The
  built-in spaces were not affected.
- Without fidelity levels, `smac3` and native `bohb` no longer run at 2
  evaluations per run. With `budget_per_run=5000`, SMAC3 ran all its trials at
  2 evaluations, fewer than any population, so every run failed and the
  returned configuration was arbitrary; they now run at 5000.
- Native `bohb` honours `max_trials`: it samples exactly that many
  configurations, following HpBandSter's brackets with the real smallest budget
  and shrinking the last one. With `max_trials=20` it sampled 2187
  configurations (3280 runs) without levels, and 3 with levels `[1000, 3000,
  5000]`.
- `smac3` and native `bohb` return the best configuration evaluated at the full
  budget, as HpBandSter's `Result.get_incumbent_id()` does, and fail with an
  explicit error when no configuration reached it. They returned the best score
  at any budget, so a cheap or partial evaluation could win.
- With fidelity levels, the `optuna` median pruner and the `bohb_optuna`
  Hyperband pruner can stop a trial after its first level. The median pruner
  could only prune after the last level had been evaluated, and the Hyperband
  pruner never after the first. Pruned trials stay in the history with state
  `PRUNED` and are never returned. With levels `[300, 600, 900]` and 60 trials,
  `optuna` now prunes 18 trials after the first level and runs 80 instead of
  120 evaluations at 900.
- Multi-fidelity racing checks before any run, and before writing any output,
  that its first level (every initial configuration on every training
  instance/seed block) fits in `--tune-budget`, and fails with a message that
  names the options to change.
- With warm starts, multi-fidelity racing gives each instance/seed block only
  its own checkpoint and previous budget. When a block's run had failed, the
  next level received the checkpoint of the last block that succeeded, from
  another instance or seed; it now starts that block from scratch.
- Tuning treats an evaluation that raises or returns a non-finite score as a
  failed evaluation everywhere: it scores the task's failure value
  (`TuningTask.failure_score`, `--failure-score` in `vamos tune`, or -1e30/1e30
  by default), is counted, and can no longer stall a race. One NaN on one block
  used to stop every elimination of `RacingTuner`: with 11 configurations all
  11 survived 40 stages. `RandomSearchTuner` no longer aborts at the first
  exception nor returns a NaN trial as the best, and the model-based backends
  and irace receive the failure value instead of NaN. `vamos tune` marks a
  failed run instead of scoring it like a valid empty front, so the
  validation's `fail_rate` counts it (an impossible `pop_size=150` with a
  budget of 100 reported `fail_rate=0.0`). Racing, random-search and irace
  histories record `failed_evaluations`, and `tuning_summary.json` records
  `evaluations` and `failed_evaluations`.
- Tuning MOEA/D on problems with three or more objectives: the builders of the
  MOEA/D spaces read `n_obj` from the assignment and run the largest
  population, at most the sampled `pop_size`, that has weight vectors (a
  complete Das-Dennis lattice or a packaged jMetal file), instead of failing on
  most sampled populations. `vamos tune` passes `n_obj` to MOEA/D as it already
  did to RVEA; code that calls `config_from_assignment` must add it too.
- `bohb_optuna` is reproducible for a seed with one worker: every Optuna study
  now gets a name, also without `--optuna-storage`, because the Hyperband
  pruner assigns a trial's bracket from a hash of the study name and Optuna's
  default name is random.
- A resumed Optuna study (`--optuna-storage` with `--optuna-load-if-exists`)
  runs only the trials `--tune-budget` still allows, counting its finished
  trials; rerunning a command used to add `--tune-budget` trials each time (5,
  10, then 15 trials for a budget of 5). `best_score` in `tuning_summary.json`
  is the score of the returned configuration (racing's final choice, the
  statistical finisher's winner or irace's elite), and the new
  `best_history_score` is the highest history score, which 1.1.0 wrote as
  `best_score`. The tuning guide documents `champion_fast` (best mean
  validation score, not runtime), `start_instances` and the tie-break by
  position.
- `vamos tune --smoke` (experimental) caps the sampled populations at half its
  30-evaluation run budget, and RVEA's `n_partitions` so that its reference
  directions fit, so every sampled configuration runs. VAMOS 1.1.0 kept
  sampling populations of 20 to 200, and `optimize` rejected most smoke
  configurations, which received `--failure-score` (18 of the 24 evaluations of
  the guide's smoke command). `vamos tune` warns when more than half, but not
  all, of the evaluations of a run fail, and names its auto-generated output
  folder after `--instances` instead of `--problem`.

#### Studio and Assist

- Studio's Statistical Comparison counts a run without feasible solutions as
  the worst IGD+ (∞), as HV already counted it as 0. Such runs used to get IGD+
  = NaN and were dropped in silence, so an algorithm's IGD+ came only from its
  feasible runs, and a problem left the Friedman/Nemenyi ranking when an
  algorithm was never feasible on it. Those runs now enter the Mann-Whitney
  tests, the A12 effect sizes, the medians and the ranks. Every table cell, on
  the page and in LaTeX, shows the feasible runs out of all runs; the Verdict
  and Ranking tabs warn when a run had none; the SAES results file writes their
  IGD+ as `inf`; Distributions lists them instead of drawing them, and Fronts
  says when an algorithm's median run has no feasible solution. IGD+ medians,
  verdicts, A12, ranks and exports change for studies with infeasible runs: on
  ZDT1 with 10 seeds, where MOEA/D finds a better front than NSGA-II but is
  feasible in only 2 runs, MOEA/D's IGD+ goes from n = 2, median 0.0170, A12 =
  1.00, `+` (p = 0.030) to n = 10, median ∞, A12 = 0.20, `-` (p = 0.021), which
  now agrees with HV.
- Studio fronts merged from several seeds keep only the feasible solutions no
  other seed dominates, each with its own decision variables; they stacked
  every row, and X could be shorter than F when a run lacked it. Explore
  Results ranks only feasible solutions by default (**Feasible solutions only**
  is on), and the ranking and its CSV gain a `Feasible` column. Merged fronts,
  rankings and ranking CSVs change for runs with dominated or infeasible rows.
- Batch Comparison in Studio runs every algorithm it offers for an integer
  problem. The tasks of MOEA/D, SMS-EMOA, NSGA-III and SPEA2 failed although
  their plan was valid, because a resolved spec names polynomial mutation by
  its registry name `pm`, which the shared integer operator map did not accept;
  for the same reason `vamos reproduce` of their integer runs failed with
  `ReplayExecutionError`.
- Two Studio sessions that start the same study at once start one background
  runner; the other gets `JobConflictError`. Studio also reads only the
  heartbeat and outcome of the runner it started last, so a late outcome of an
  earlier runner no longer marks the new one as finished.
- The Studio Problem Builder's generated script imports `math` and `numpy as
  np`, the names Studio gives the objective code, uses the form's seed instead
  of 42, always rebuilds the configuration Studio runs with `from_dict`, and
  reports "solutions" instead of "non-dominated solutions". VAMOS 1.1.0 scripts
  failed with `NameError: name 'np' is not defined` for code that ran in
  Studio.
- Studio's LLM drafts go through one parser for every provider, which strips a
  Markdown code fence, requires a JSON object and checks the type and range of
  every field before the form changes; an unknown provider is refused before
  any request, and the request runs off the page's event loop and is abandoned
  after 120 s. VAMOS 1.1.0 rejected a fenced OpenAI answer, could fill part of
  the form and then fail, and sent an unknown provider name to OpenAI.
- `vamos assist apply` and `vamos assist run --overwrite` replace only a folder
  assist created (marked with `.vamos-assist`), remove only the files assist
  wrote there, and refuse an `--out` that is the plan folder or contains it;
  they deleted any existing output folder, the plan included. Folders created
  by earlier versions have no marker, so `--overwrite` refuses them. `vamos
  assist run` reports an in-process failure instead of running the
  configuration again in a subprocess, and writes `run_report.json` when the
  run exits.

<!-- PENDING(user-extras, build/user-oriented-extras): drop this block if that pull request IS merged, because it removes these baselines. -->
- The external NSGA-II baselines of the CLI (`--algorithm pymoo_nsga2`,
  `jmetalpy_nsga2` and `--include-external`) run as their libraries intend.
  `pymoo_nsga2` mutates each variable with probability 1/n; it passed 1/n as
  pymoo's per-individual probability, so each variable mutated with probability
  1/n². `jmetalpy_nsga2 --external-problem-source vamos` runs: its problem
  wrapper did not implement the abstract methods of the jMetalPy (1.6 or later)
  `Problem` class and failed with `TypeError`, and its initial population was a
  single point. The jMetalPy baselines restore Python's and NumPy's global
  random streams after seeding them, and seeds wider than 32 bits, which the
  CLI draws when `--seed` is omitted, also seed NumPy's stream. The
  `pymoo_nsga2` baseline reports a better front: on ZDT1 (population 100,
  10,000 evaluations, seeds 0-4, HV reference (1.1, 1.1)) its median HV goes
  from 0.634 to 0.849, against 0.843 for `jmetalpy_nsga2` and 0.850 for VAMOS
  NSGA-II. Comparisons with `--include-external` from earlier versions
  understated pymoo.
- The pygmo NSGA-II baseline runs with `--external-problem-source native`. It
  passed `dim=` to `pygmo.zdt`, whose argument is `param`, and it passed 64-bit
  seeds to constructors that take unsigned 32-bit seeds; wider seeds are now
  folded into 32 bits.
<!-- /PENDING -->

<!-- PENDING(PRB-23, fix/problem-output-contract): drop this block if that pull request is not merged. -->
- Every built-in problem's `evaluate(X, out)` creates `out["F"]` when the
  caller has not allocated it. ZDT1–ZDT6, the TSP, binary, integer and mixed
  examples, `welded_beam`, `ml_tuning`, `fs_real`, the RE and the RWA problems
  raised `KeyError: 'F'` for `problem.evaluate(X, {})`; runs through the
  algorithms, which allocate it, are unchanged.
<!-- /PENDING -->

<!-- PENDING(PRB-22, fix/example-problems): drop this block if that pull request is not merged. -->
- `mixed_design` clips a categorical value outside `0..cardinality - 1` to the
  nearest category, as it does with integer values. It used to wrap the value
  modulo the cardinality, so a value rounded to the cardinality counted as
  category 0. Values within the bounds evaluate as before, so runs of the
  built-in algorithms are unchanged.
<!-- /PENDING -->

<!-- PENDING(EXP-28, fix/cli-quality-details): drop this block if that pull request is not merged. -->
- `vamos zoo info` with an unknown problem exits with code 1, `vamos zoo list`
  names real-valued problems `real` (27 said `continuous`), and `vamos zoo run`
  no longer prints `HV: None`.
- `vamos check` prints one line per check, the binary and mixed checks
  included, without the summaries of its runs, and a failed NumPy check exits
  with code 1 and a message instead of a traceback.
- Derived study tables warn when indicators other than HV are skipped because
  moocore is not installed; VAMOS 1.1.0 dropped those columns silently.
<!-- /PENDING -->

<!-- PENDING(UX-25, fix/studio-small-fixes): drop this block if that pull request is not merged. -->
- Studio's Run Experiment opens a stored run whose seed is 2^31 or more and
  accepts any seed from 0 to 2^64 − 1, the range that `optimize` and the CLI
  draw from. It shows the seed as text so that every digit is kept, and it says
  when it lowers a run's budget to its maximum of 5,000,000 evaluations.
- Studio's Statistical Comparison computes the Ranking tab's Nemenyi critical
  difference at the chosen significance level instead of always at 0.05, and
  the caption of its LaTeX ranking table now gives that critical difference and
  level. The warning that no difference can be significant uses the smallest
  p-value SciPy's test gives for those run counts, which is larger above eight
  runs per sample, and the LaTeX comparison caption escapes the pivot's name.
- Studio searches for studies in the background, at most four folders below the
  chosen one, and reads at most the 50 most recently changed studies, so a
  large folder no longer blocks every session.
- `plot_result_front(show=False)` no longer switches Matplotlib's process-wide
  backend to Agg; it draws on a figure that pyplot does not track.
- `reduce_objectives` with `target_dim` (correlation method) chooses the
  objectives to keep by their variance relative to their range, so the choice
  no longer depends on units.
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
