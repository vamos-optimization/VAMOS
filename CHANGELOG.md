# Changelog

All notable public changes to VAMOS are documented here.

## [Unreleased]

## [1.5.0] - 2026-11-19

VAMOS 1.5.0 is a minor release. It adds AGE-MOEA-II as a variant of AGE-MOEA,
the authors' reading of RVEA's survival as a variant, one source of reference
vectors for MOEA/D, NSGA-III and RVEA (files, directories and two-layer
lattices), the CTP1–CTP7, IDTLZ1 and IDTLZ2 problems, DTLZ reference fronts with
4, 6 and 8 objectives, and the approximate hypervolume in the experimental
`vamos.ux.api`. It corrects NSGA-III's parent selection, which changes the
results of every NSGA-III run, and the packaged MOEA/D weight files, which
changes MOEA/D's results with four objectives and 100 to 119 individuals. The
unbounded external archive merges its rows when it is read, with the same
results. Seeded tuning runs of the NSGA-III and AGE-MOEA spaces draw different
trials. See the [1.5.0 release notes](docs/project/release-notes-1.5.0.md).

### Added

- Problems: CTP1–CTP7 (Deb, Pratap and Meyarivan, 2001), registered as
  `ctp1`–`ctp7`, with the paper's definition and parameters.
  - CTP1's J = 2 constraint coefficients come from the paper's procedure, and
    CTP2–CTP7 from its constraint generator.
  - g is ZDT4's Rastrigin function, with five variables by default.
  - `vamos.problems.CTP(variant, n_var=..., n_constraints=..., theta=..., a=...,
    b=..., c=..., d=..., e=...)` builds other instances.
  - Two-objective reference fronts `ctp1.csv`–`ctp7.csv` are generated from the
    definition.
  - The constraint of CTP2–CTP7 is the paper's difference, not the NSGA-II
    code's ratio that pymoo uses (#357).
- AGE-MOEA-II (Panichella, GECCO 2022) as a variant of AGE-MOEA:
  - `AGEMOEAConfig.builder().variant("agemoea_ii")`, and `replacement:
    agemoea_ii` in every AGE-MOEA tuning space (Evolver's `replacement:
    agemoeaII`);
  - its survival estimates the front's exponent by Newton-Raphson from the
    point nearest to the ideal point, and measures diversity with geodesic
    distances on the estimated manifold;
  - its exponents, distances and survival scores are checked against jMetal's
    `AGEMOEA2EnvironmentalSelection`;
  - the default, `"agemoea"`, is unchanged; seeded tuning runs of the AGE-MOEA
    spaces draw different trials (#359).
- Reference fronts: DTLZ1–DTLZ4 and DTLZ7 with 4, 6 and 8 objectives
  (`dtlz1.<m>d.csv`, `dtlz2.<m>d.csv` and `dtlz7.<m>d.csv`), which
  `resolve_reference_front_path` finds by `n_obj`; DTLZ3 and DTLZ4 resolve to
  DTLZ2's file. DTLZ1–DTLZ4 use the Das-Dennis lattice of PlatEMO's reference
  fronts, the points of jMetal's DTLZ2–DTLZ4 files with those numbers of
  objectives (#362).
- Indicators: the experimental `vamos.ux.api` exports `HVIndicator`,
  `HVApproxIndicator`, `HV_APPROX_METHODS` and `IndicatorResult`, documented in
  Analysis and Visualization with a note on the exact hypervolume's cost from
  about six objectives. `HVApproxIndicator.compute` passes `seed` and `method`
  to `moocore.hv_approx` (an unknown method raises `ValueError`; `None` keeps
  MooCore's default) and records `samples`, `seed` and the method used in
  `details` (#363).
- MOEA/D, NSGA-III and RVEA take their vectors from one kind of option, as
  jMetal and Evolver configure them: MOEA/D's `weight_vectors`, NSGA-III's
  `reference_directions` and RVEA's new `reference_vectors` accept a file
  (whitespace- or comma-separated) or a directory with jMetal's `W<m>D_<n>.dat`
  names, the lattice of `divisions`, or, with the new `inner_divisions`, Deb and
  Jain's two-layer lattice (156 vectors with eight objectives, 3 and 2
  divisions). The vectors are the population, so RVEA can run 100 individuals
  with three objectives from `W3D_100.dat`, and the three algorithms can start
  from the same vectors. A path that does not exist raises
  `ConfigurationError`. The resolved spec records a file for NSGA-III and RVEA
  and the two-layer lattice for MOEA/D.
  `vamos.algorithms.moead_populations(n_obj)` lists the populations MOEA/D runs
  without weight options (#365).
- Problems: IDTLZ1 and IDTLZ2, the inverted DTLZ1 and DTLZ2 of Jain and Deb
  (2014) as PlatEMO defines them, registered as `idtlz1` and `idtlz2` with a
  configurable number of objectives (`n_var = n_obj + 4` and `n_obj + 9` by
  default), with reference fronts for two and three objectives
  (`idtlz1.2d.csv` to `idtlz2.3d.csv`) (#366).
- RVEA's `variant`: `"platemo"` runs the authors' reading of the survival, that
  of PlatEMO's `RVEA/EnvironmentalSelection.m` and of jMetal (M equal to the
  number of objectives in the angle penalty also with two objectives,
  translation by each generation's minimum, vector adaptation to the
  population's range); the default `"pymoo"` keeps VAMOS's results (#367).

### Changed

- An unbounded external archive keeps the rows each generation offers and
  merges them (with moocore's non-dominated filter when moocore is installed)
  only when the archive is read or the kept rows outnumber twice the archive,
  instead of refiltering itself every generation.
  - The rows, their order and every result are the same.
  - A run that reads its archive only at the end filters it once: over 200 000
    evaluations, 0.28 s instead of 85 s on a 112 000-row archive.
  - `UnboundedArchive(buffered=False)` keeps the eager updates (#360).
- Two-layer lattices whose layers repeat a direction raise
  `ConfigurationError`. A MOEA/D `weight_vectors` path that does not exist
  raises `ConfigurationError` instead of being created as a directory for the
  generated lattice, and an unknown key in `weight_vectors` or
  `reference_directions` raises instead of being ignored (#365).

### Fixed

- MOEA/D with four objectives and 100 to 119 individuals runs the 100 weight
  vectors of the packaged `W4D_100.dat`, which held them twice (as the jMetalPy
  file it comes from) and was skipped, so VAMOS ran 84 individuals. The
  packaged `W4D_165.dat` is space-separated, and weight files may separate
  their components with whitespace or commas and depart from the simplex by
  jMetal's rounding (`1e-5`), so all 31 packaged files load, also from the
  command line, where 14 of them failed. NSGA-III reads
  `reference_directions(path=...)` with the same reader, so jMetal's
  whitespace-separated files such as `W3D_100.dat` work, and a path that does
  not exist raises `ConfigurationError` instead of generating a lattice and
  saving it there. Results change for MOEA/D with four objectives and 100 to
  119 individuals; NSGA-III runs whose path was missing now fail (#361).
- NSGA-III selects its parents as configured, and never by crowding distance,
  which is not part of NSGA-III. It ran a binary tournament on rank and
  crowding distance, as NSGA-II, whatever its selection, so it was not the
  published algorithm. `selection("random")`, now the default of
  `NSGAIIIConfig.default()`, the command line and the tuning spaces, draws
  parents at random, as Deb and Jain (2014); with constraints and feasibility
  first, each parent wins a binary tournament on the constraint violation,
  decided at random between feasible solutions, as Jain and Deb's constrained
  NSGA-III and pymoo. `selection("tournament", size=k)` is jMetal's NSGA-III
  tournament on rank, with ties drawn at random. Other selection methods and
  options raise `ConfigurationError`. The NSGA-III tuning spaces choose
  `selection`, `random` or `tournament` with `tournament_size`, as Evolver's,
  and assignments of earlier spaces still build their tournament; seeded tuning
  runs of these spaces draw different trials. `--selection-pressure` no longer
  applies to NSGA-III. NSGA-III's results change, by default and with a
  tournament, so experiments run with NSGA-III should be rerun (#364).
- A run whose MOEA/D `weight_vectors`, NSGA-III `reference_directions` or RVEA
  `reference_vectors` names a file records the file it read and the `sha256` of
  its vectors in its resolved configuration. Replay refuses the file if its
  vectors have changed, where VAMOS 1.4.0 replayed whatever the file then held.
  The run's record of the file is rebuilt from the stored configuration, so the
  run verifies without the file (#365).

## [1.4.0] - 2026-11-17

VAMOS 1.4.0 is a minor release. The built-in tuning spaces become YAML grammars
with a default for every parameter (schema `vamos.parameter-space` 1.0.0), which
the tuning backends and Studio's algorithm forms read; seeded tuning runs draw
different trials, and persistent Optuna studies of 1.3.0 are not resumed. MOEA/D
and SMS-EMOA come closer to Evolver: MOEA/D gains Evolver's order of the
subproblems, DE schemes and base vector, and aggregation functions with the
normalization of the objectives, which its tuning spaces offer too, and the
spaces offer Evolver's operators and opposition-based and Halton initialization.
As announced with 1.3.0, VAMOS Assist, Studio's Problem Builder and the LLM
extras are removed. PyYAML becomes a dependency, and the Numba engine keeps its
compiled variation with NumPy 2.4. The default configurations of the algorithms
give the results of 1.3.0. See the [1.4.0 release
notes](docs/project/release-notes-1.4.0.md).

### Added

- MOEA/D: `subproblem_order="cyclic"` visits the subproblems in index order, as
  Evolver's `cyclicIntegerSequence`; the default, `"random_permutation"`, is the
  order of earlier versions (#337).
- MOEA/D: the DE crossover takes `scheme="rand_1_exp"`, `"rand_2_bin"` or
  `"rand_2_exp"`, jMetal's `DifferentialEvolutionCrossover` variants; the
  default is `"rand_1_bin"`. rand/2 needs `neighbor_size >= 4` (#337).
- MOEA/D: the `augmented_tchebycheff` (`rho`, default `1e-4`) and `inverted_pbi`
  (`theta`, default `0.1`; Sato, 2014) aggregations, and `normalize=True` with
  `epsilon` (default `1e-6`) in any aggregation, which normalizes the objectives
  with the ideal point and jMetal's estimate of the nadir point. Off by default
  (#337).
- `vamos.engine.tuning.grammar` (Experimental) reads tuning parameter spaces
  written as strict grammars with defaults, under the schema
  `vamos.parameter-space` 1.0.0, from YAML or JSON, and rejects a space that
  breaks one of the rules G1–G6 with an error that names the rule and the place:
  unique snake_case names, one parent per parameter and no condition
  expressions, a default inside every domain, ordered ranges, non-empty values
  without duplicates or values written alike, and string values where one has a
  production, and a known schema, algorithm and encoding. The schema has no
  forbidden clauses and one range per parameter.
  `check_default_configuration(space, problem)` checks rule G7 by running the
  space's defaults and the algorithm's `default()` configuration with the same
  seed. A configuration of a space has three forms that read back into it: the
  document `vamos.parameter-configuration` 1.0.0, with the space's name, version
  and SHA-256; the flat mapping that `config_from_assignment` reads; and the
  command line of Evolver's `VAR_CONF.txt`, with VAMOS's names or Evolver's
  (#333).
- MOEA/D: the `inverse_weight_tchebycheff` aggregation, jMetal's
  `ModifiedTschebyscheff` (Evolver's `modifiedTschebyscheff`), `max_i |f_i -
  z*_i| / w_i`: the Tchebycheff function with Qi et al.'s WS-transformed weights
  (#349).
- MOEA/D: the DE crossover takes `base="random"`, which takes the mutant's base
  vector from one more random parent, as Evolver's MOEA/D; the default,
  `"current"`, keeps the current solution (#349).
- The built-in tuning spaces are YAML grammars (`vamos.parameter-space` 1.0.0)
  packaged in `vamos/resources/parameter_spaces`, one per space and maintained
  as YAML, which VAMOS reads at run time.
  `vamos.engine.tuning.grammar.builtin_parameter_space(name)` reads one and
  `to_algorithm_config_space` gives the flat view that
  `build_<space>_config_space()` returns. `python
  tools/update_parameter_spaces.py` checks the files and writes the test
  snapshot of what the bridges build from them (#334).

### Changed

- PyYAML is a dependency of VAMOS (`pyyaml>=6.0`), which reads its YAML tuning
  spaces at run time; it was in the `dev` extra (#333).
- The CLI's `--moead-aggregation` help lists every aggregation it accepts
  (#349).
- The built-in tuning spaces no longer tune `pop_size`: it is a run setting that
  the caller adds to an assignment, as `n_obj` (100 without it), and `vamos
  tune` runs every algorithm, RVEA included, with `--pop-size`; MOEA/D, NSGA-III
  and RVEA derive their populations from it as from the sampled one. Every
  parameter has a default, and a space's defaults run as its algorithm's
  `default()` configuration. `initial_solutions` and `repair` offer `auto`,
  their default. The external archive is one choice, `external_archive` (`none`,
  `unbounded` or `bounded`); polynomial and linked polynomial mutation have an
  index each; rates hang from their operators; CCMO and CMOEA-MS choose a
  `variation`; the integer spaces name polynomial mutation `polynomial`; and the
  mutation probability is `mutation_probability_factor` in every space but the
  permutation spaces of the algorithms other than NSGA-II. Every seeded tuning
  run draws different trials. The `effective` record of `tuning_history.json`
  and `best_config_effective` of `tuning_summary.json` give the population
  MOEA/D runs instead of `--pop-size` (136 for 150 with three objectives).
  Assignments saved from earlier spaces build the configurations they built
  (#334).
- The segments of NSGA-II's mixed tuning space sample their mutation
  probabilities in [0.01, 0.5], as the other mixed spaces; they sampled [0.25,
  3.0], so above 1 every mutation applied (#334).
- The tuning backends read a built-in space's grammar instead of condition
  expressions. ConfigSpace (SMAC3, BOHB) gets hyperparameters with the grammar's
  defaults and its tree as conditions. Optuna walks the tree and starts a new
  study from the default configuration, and irace and racing also race the
  default configuration first. Racing and random search draw only active
  parameters. Every seeded tuning run draws different trials (#335).
- `vamos tune` names persistent Optuna studies after the space it searches:
  `__vamos_cli_final_population_hv_<hash>` (SMPSO:
  `__vamos_cli_leaders_hv_<hash>`), with 16 hexadecimal digits of the space's
  SHA-256, instead of a version suffix. A changed space is a new study; studies
  of VAMOS 1.3.0 and earlier are not resumed (#335).
- Studio's algorithm forms are the grammar of the tuning space. The controls
  follow its tree, start from its defaults (the VAMOS default configuration),
  and edit each parameter within the range that tuning uses. `initial_solutions`
  and `repair` start at `auto`. The forms no longer offer a "VAMOS default"
  choice or widen a range, and Studio no longer warns about default operators
  that a space did not offer (#336).
- The real-coded MOEA/D and SMS-EMOA tuning spaces offer what Evolver gives
  these algorithms and VAMOS runs:
  - Sobol initialization;
  - the Lévy flight and power-law mutations;
  - the Laplace and fuzzy crossovers, plus BLX-αβ and whole arithmetic in
    SMS-EMOA;
  - in MOEA/D, a `variation` choice between differential evolution, the default,
    and crossover and mutation with SBX, BLX-α, BLX-αβ, whole arithmetic,
    arithmetic, Laplace or fuzzy recombination;
  - SBX and polynomial distribution indices in [5, 400].
  Assignments saved with `crossover: de` or `sbx` build what they built (#350).
- The mixed tuning spaces choose between the default mixed operators (`mixed`,
  the default, with a crossover probability and a mutation factor) and an
  operator per segment (`per_segment`). Their defaults now run as the
  algorithm's default configuration (#350).
- Seeded tuning runs of the real-coded MOEA/D and SMS-EMOA spaces and of the
  mixed spaces draw different trials (#350).
- The MOEA/D tuning spaces offer Evolver's MOEA/D options:
  - the order of the subproblems (`subproblem_order`, random permutation or
    cyclic);
  - the augmented Tchebycheff (`atch_rho`), the inverted PBI (`ipbi_theta`) and
    the inverse-weight Tchebycheff aggregations;
  - the normalization of the objectives (`normalize_objectives`, with
    `normalization_epsilon`);
  - in the real-coded spaces, the DE scheme (`de_scheme`: rand/1 or rand/2,
    binomial or exponential) and the base of the mutant (`de_base`).
  All are off by default, so the defaults build the configuration of VAMOS
  1.3.0. Seeded MOEA/D tuning runs draw different trials (#355).
- The real-coded MOEA/D and SMS-EMOA tuning spaces, constrained ones included,
  also offer opposition-based (`obl`, Evolver's `oppositionBased`) and Halton
  initialization. Seeded SMS-EMOA tuning runs draw different trials (#355).
- The Evolver name table reads Evolver's inverted PBI as `inverted_pbi` with
  normalization, and its DE as `de_base: random`, as Evolver runs them.
  `to_evolver` refuses the combinations that Evolver cannot express (#355).

### Removed

- `vamos.assist` and the Experimental `vamos assist` command, which planned and
  ran experiments from templates or with an OpenAI planner. VAMOS calls no LLM
  provider. `vamos assist ...` now fails as unrecognized arguments (exit status
  2) (#332).
- Studio's Experimental Problem Builder page, which ran Python objective code
  typed into Studio or drafted by Gemini, OpenAI or Anthropic, and exported
  scripts. Studio runs registered problems only; define a problem with
  `make_problem` or `vamos create-problem` (#332).
- The `openai`, `anthropic` and `gemini` extras, also from `all`. No extra
  installs an LLM client (#332).

### Fixed

- The tuning name table read Evolver's `modifiedTschebyscheff` as VAMOS's
  `modified_tchebycheff`, a different function, so a configuration from Evolver
  ran another aggregation. It now reads it as `inverse_weight_tchebycheff`, and
  `augmentedTschebyscheff` as `augmented_tchebycheff`; `to_evolver` raises
  `ValueError` for `modified_tchebycheff`, which Evolver does not have, instead
  of writing `modifiedTschebyscheff` (#349).
- The mixed variation of MOEA/D, NSGA-III, SMS-EMOA and SPEA2 runs the segment
  settings of its configuration (`perm_crossover`, `real_mutation_prob`,
  `int_mutation_step`, ...). It ran the default mixed operators whatever they
  said, so every point of these algorithms' mixed tuning spaces ran as the
  default. Default configurations, which set no segment, do not change (#334).
- The default `epsilon_decay_fraction` of the constrained tuning spaces is 0.8,
  Takahama and Sakai's Tc = 0.8 Tmax; it was 0.2 (#350).
- `engine="numba"` keeps its compiled SBX, polynomial mutation and steady-state
  steps with NumPy 2.4. NumPy 2.4 computes `np.power(x, 0.5)` as `sqrt(x)` and
  `np.power(x, -1)` as `1 / x`, so the check that compiled `pow` matches
  `np.power` failed, and the Numba engine fell back to the NumPy operators (same
  results, slower). The compiled power now takes the same shortcuts (#353).
- Studio's MOEA/D form applies the parameters of the modified Tchebycheff
  aggregation (`mtch_rho`), which it ignored (#355).

## [1.3.0] - 2026-11-16

VAMOS 1.3.0 is a minor release. It requires Python 3.11 or newer, as announced
with 1.2.0. It corrects operators and settings that departed from their
published definitions, which changes results: the non-uniform, Gaussian and
Cauchy mutations, three crossovers, `mutation_prob_factor` in six algorithms and
command-line NSGA-III's SBX index. It rebuilds the built-in tuning spaces and
racing's statistics, gives every algorithm one live-callback contract, makes
studies robust to interruptions, and adds a wall-clock time limit, problem
options by name, jMetal's reference fronts and MOEA/D configurations as options,
a final archive subset, and an access token for VAMOS Studio. See the [1.3.0
release notes](docs/project/release-notes-1.3.0.md).

### Added

- Two-objective reference fronts for DTLZ1–DTLZ7 (`dtlz1.2d.csv` to
  `dtlz7.2d.csv`, 1000 points of each closed-form front), so a two-objective
  DTLZ run has an automatic reference (#285).
- `SMSEMOAConfig.builder().reference_point(rule="jmetal")`: SMS-EMOA measures
  the hypervolume contributions from jMetal's and Evolver's reference point,
  each coordinate the worst value of the population and the new child times 1.1
  (divided by 1.1 when negative), so that results can be compared with theirs.
  The default rule, `nadir + offset * (nadir - ideal)`, does not change (#290).
- `optimize(name, problem_kwargs=...)` and a study's `problem_kwargs` accept the
  options a registered problem's factory declares besides `n_var` and `n_obj`:
  `capacity_ratio` for `bin_knapsack`, and `complicated_pareto_set`, `level`,
  `bias` and `imbalance` for `zcat1`–`zcat20`. They are recorded in
  `resolved_spec.problem.config`, and `verify_run` and `reproduce` rebuild the
  same problem. `make_problem_selection(name).with_options({...})` builds such
  an instance, and `ProblemSpec.option_defaults()` lists the options. An
  undeclared option still raises `ConfigurationError`, which now lists the
  accepted ones, and so does a number that the run artifact cannot record
  exactly (#286).
- `optimize(termination=("max_time", seconds))` stops a run of any built-in
  algorithm once `seconds` of wall-clock time have passed, checked after each
  generation; `("max_time", {"seconds": s, "max_evaluations": n})` also caps the
  evaluations, and `result.meta["time_limit_reached"]` says which stopped it.
  Runs limited by time are not reproducible across machines or even on the same
  one: their resolved spec records `vamos.termination:max_time@1` and
  `determinism.declared: false`, the manifest gives the replayability reason
  `wall_clock_termination`, and `verify_run` and `reproduce` refuse an exact
  replay (#289).
- `live_callback_mode` (`"population"`, `"nd_only"`, `"population_archive"`) in
  the configurations of AGE-MOEA, CCMO, CMOEA-MS, C-TAEA, IBEA, MOEA/D,
  NSGA-III, RVEA, SMPSO, SMS-EMOA, SPEA2 and ToP, as in NSGA-II. Their default,
  `"population"`, keeps today's payload; `"population_archive"` adds the
  external archive in `stats["archive"]`, so a live view of a run with an
  archive ends with what the run returns (#288).
- `load_study(path, verify="metadata")` loads a study whose runs are damaged or
  missing, so that `inspect()` and `summarize()` report them as issues instead
  of raising; `run()`, `resume()` and `retry()` still verify every run (#303).
- `load_study` and `create_study` accept `run_limits=`, a `LoadLimits` for the
  study's runs, and the returned `Study` keeps it and its `StudyLoadLimits` as
  `Study.run_limits` and `Study.limits` (#303).
- Stable functions may gain optional keyword-only arguments during 1.x; the
  compatibility check accepts them and still rejects removed, renamed, reordered
  or required parameters (#303).
- `select_top_k_farthest(..., normalize=True)` scales each objective to [0, 1]
  by its range before measuring distances, so the farthest-point subset no
  longer depends on the objectives' units or offsets. The default keeps raw
  distances (#320).
- jMetal's reference fronts of ZDT1–ZDT4, ZDT6 and DTLZ1–DTLZ7 with two and
  three objectives (commit `1e4b83cae6`, MIT), packaged as an optional source
  next to VAMOS's samples, which stay the default:
  `resolve_reference_front_path(..., source="jmetal")`, and `vamos
  --hv-reference-front jmetal`, which records `reference_front_source` in the
  run. They make IGD-like indicators comparable with jMetal's and Evolver's
  (#322).
- `MOEADConfig.default(variant=...)`: `"de"` gives jMetal's MOEA/D-DE (DE
  rand/1/bin with CR = 1.0 and F = 0.5, Tchebycheff aggregation; Evolver's
  `MOEADDEDefault`), and `"sbx"` jMetal's MOEA/D with SBX (p = 0.9, eta = 20,
  PBI with theta = 5; Evolver's `MOEADDoubleDefault`). Without `variant`, the
  default is unchanged (DE with PBI); VAMOS 2.0 will make `"de"` the default
  (#326).
- `ExternalArchiveConfig.final_selection` (`"farthest"`, `"crowding"`,
  `"hv_greedy"` or `"kmeans"`) and `final_size` (the population size by
  default), also through `external_archive(...)`: the result reports that subset
  of the external archive when the run ends, as jMetal's and Evolver's results
  do, while `result.data["archive"]` and the saved run keep the whole archive
  and `result.data["archive_subset"]` records the selection. The default,
  `"none"`, reports the whole archive as before (#323).
- The resolved spec of a MOEA/D run records where its weight vectors come from,
  in `algorithm.resolution.weight_vectors`: a Das-Dennis lattice with its
  divisions, a packaged jMetal file such as `W3D_100.dat`, a user's file, or
  uniform weights with two objectives, and their number. `vamos tune` records
  the population a MOEA/D trial ran when the bridge lowered the sampled one
  (`details.effective` in `tuning_history.json`, `best_config_effective` in
  `tuning_summary.json`), and `assignment_adjustments` returns it for code that
  calls `config_from_assignment`. MOEA/D runs stored by earlier versions are not
  verified as exact (#321).

### Changed

- The built-in tuning spaces tune only what reaches the algorithm. The
  real-coded spaces no longer offer `repair="round"` or `blx_repair="round"`,
  which put the whole population on the vertices of the bounding box. The SMPSO
  spaces drop `repair`, which never reached SMPSO, and the SPEA2 spaces drop
  `selection_pressure`, since SPEA2 always mates with its own binary tournament.
  The RVEA spaces sample `pop_size` in [20, 200] instead of `n_partitions` in
  [4, 12], and run the largest Das-Dennis lattice with at most `pop_size`
  directions for the problem's number of objectives. The fixed range of
  partitions gave 5 to 13 individuals with two objectives and up to 293,930 with
  ten. Assignments saved from earlier spaces still build. Seeded tuning runs
  draw different trials. Persistent Optuna studies move to
  `__vamos_cli_final_population_hv_v3` and, for SMPSO,
  `__vamos_cli_leaders_hv_v2`, so studies stored by VAMOS 1.2.1 or earlier are
  not resumed (#291).
- Stable configuration fields that have no effect emit a `FutureWarning` when
  they are set and are planned for removal in VAMOS 2.0. They are
  `NSGAIIIConfig.pop_size_auto` and `enforce_ref_dirs` (NSGA-III runs the
  largest lattice that fits in `pop_size`, with at least one individual per
  direction), `track_genealogy` of `AGEMOEAConfig` and `RVEAConfig`, and a
  `SPEA2Config.selection` other than a binary tournament.
  `NSGAIIIConfig.default()` does not warn (#292).
- The SMS-EMOA tuning spaces choose the mating selection, `random` or
  `tournament` (with `selection_pressure` only for a tournament), and
  `reference_point_offset` in [0.1, 2.0], so they include `SMSEMOAConfig`'s
  defaults. They always used a tournament and an offset of 0.1, which an
  assignment without the new parameters still builds. Seeded SMS-EMOA tuning
  runs draw different trials (#293).
- VAMOS Studio admits only browsers that open it with the access token of its
  launch: `vamos studio` prints a link with the token and opens it through a
  private file, the browser keeps a cookie, and session websockets without it
  are refused. Before, any user of the same computer could open a session and
  run code with the owner's account and LLM keys. No option turns the token off
  (#302).
- With a non-loopback address, Studio no longer offers Problem Builder or Batch
  Comparison, which run code with the account that serves it, unless
  `--allow-remote-code-execution` is given. `python -m vamos.ux.panel.app` now
  refuses a non-loopback address without `--allow-remote-binding`, like `vamos
  studio`, and `vamos studio` rejects unknown options instead of passing them to
  the server (#302).
- Live callbacks (`live_viz`) receive one documented payload from every built-in
  algorithm: `on_generation` gets the objective and decision vectors of the
  whole current population, row by row, and `stats["evals"]`, starting at
  generation 0. MOEA/D reports one generation per `pop_size` evaluations instead
  of one per evaluation, and a run resumed from a checkpoint continues the
  numbering. NSGA-II keeps sending its first front with its default
  `live_callback_mode="nd_only"`. Only NSGA-II passed `X`; MOEA/D and SMPSO sent
  the first front; and some `ask`/`tell` paths left out `stats`. With MOEA/D,
  the CLI's convergence hooks sample once per `pop_size` evaluations (#287).
- The MOEA/D tuning spaces take Evolver's ranges, `neighbor_size` in [5, 50],
  `delta` in [0, 1] and `pbi_theta` up to 200, and offer the
  `modified_tchebycheff` aggregation; the real-coded ones add
  `linked_polynomial` mutation, `initializer` and `repair`. The real-coded
  MOEA/D and SMS-EMOA spaces tune `mutation_prob_factor` (probability = factor /
  n, as in NSGA-II's space) instead of an absolute `mutation_prob`. Assignments
  from earlier spaces build the same configurations; seeded tuning runs draw
  different trials (#294).
- When a run ends inside a generation (the budget, the HV target or a stop),
  MOEA/D and SMS-EMOA send the evaluations after their last notification as a
  final, shorter generation, as the generational algorithms do, so a live view
  ends with the run's result. SMS-EMOA stopped notifying at the last multiple of
  `pop_size`: with 67 evaluations and `pop_size=12`, a callback got generations
  up to 4 (60 evaluations) and now also 5 (67) (#288).
- The random streams of an external archive's `ref_dirs` pruning and of the
  hypervolume convergence monitor's bootstrap follow the run seed.
  `ExternalArchiveConfig.rng_seed` and `HVConvergenceConfig.rng_seed` default to
  `None`, which derives each stream from the run seed as a child of its
  `SeedSequence`, so the algorithm's own stream does not change. An integer
  still fixes the stream, and `optimize` records the effective archive seed in
  the run's resolved configuration while its requested specification keeps
  `None`. The monitor bootstraps with a `numpy.random.Generator` instead of
  Python's `random.Random`. Both fields defaulted to 0, so every seed of an
  experiment shared those streams. Runs that use `ref_dirs` pruning without
  `rng_seed`, or a convergence monitor with `confidence`, give different results
  (#304).
- Racing's statistical race eliminates configurations with F-race, as irace
  does. A Friedman test on the ranks of each instance/seed block is followed by
  Conover's post-hoc test on the rank sums, and a paired Wilcoxon signed-rank
  test decides when two configurations are alive. The first test waits for
  `Scenario.first_test` blocks (5, irace's `firstTest`) or
  `min_blocks_before_elimination`, whichever is larger. Before, paired t-tests
  on the raw scores against the configuration of best mean eliminated nothing
  when instances had scores on different scales: 0 configurations of 8 on 10
  blocks with half the instances 100 times larger, against about 3 for F-race.
  Seeded racing runs eliminate different configurations (#305).
- The tuning spaces name their parameters after Evolver's, in snake_case
  (`sbx_distribution_index`, `neighborhood_size`, `de_crossover_rate`,
  `tournament_size`, …), and `undx` becomes `undc`; the real-coded spaces say
  `polynomial` instead of `pm`. `config_from_assignment` accepts both the new
  and the earlier names, so saved assignments build the same configurations.
  `vamos.engine.tuning.racing.parameter_names` holds the table and converts
  assignments to and from Evolver's camelCase names (`to_evolver`,
  `from_evolver`). Seeded tuning runs draw different trials. Configuration
  fields do not change (#295).
- `gaussian` and `cauchy` read `sigma` and `gamma` as fractions of each
  variable's range, as pymoo does, and `gaussian` defaults to `sigma = 0.1`.
  They were in the variables' units, so the same value was negligible or huge
  depending on the range, and `gaussian` needed an explicit `sigma`. Runs on
  variables with a range of 1 do not change (#307).
- The tutorials and the recorded output of the algorithm guides are regenerated
  with VAMOS 1.2.1. Their requirement files install `vamos-optimization==1.2.1`
  instead of 1.0.1, and the outputs that the 1.1.0 and 1.2.0 corrections changed
  are updated (T01, T03, SMS-EMOA, SPEA2, IBEA, SMPSO, RVEA, the
  model-hyperparameter and tuning lessons, and the recorded study report). The
  NSGA-III guide no longer says that its survival ignores constraints, and the
  SMPSO guide explains that `c1` and `c2` are the lower bounds of the
  coefficients drawn at every step (#318).
- VAMOS requires Python 3.11 or newer, as announced with 1.2.0: Python 3.10
  reached its end of life in October 2026, and pip no longer installs VAMOS
  1.3.0 on it. CI and the release checks test Python 3.11 to 3.14 on Linux and
  3.12 to 3.14 on Windows and macOS (#329).
- The declared minimum of NumPy is 1.23.2, the first NumPy release for Python
  3.11 (#329).
- MOEA/D with an external archive offers the archive only the rows that each
  step changes, skips the updates that would repeat the last one, and the
  archive inserts a few rows instead of sorting itself; results are
  bit-identical. A 25,000-evaluation MOEA/D run on WFG with an external archive
  is 1.6 to 14 times faster (1.1 times with `ref_dirs` pruning, whose
  truncations draw random numbers) (#284).

### Deprecated

- VAMOS 1.4.0 removes three Experimental features that depend on LLM providers:
  VAMOS Assist with the `vamos assist` command, Studio's Problem Builder, and
  the `openai`, `anthropic` and `gemini` extras. Define a problem with
  `make_problem` or `vamos create-problem` instead.

### Removed

- Support for Python 3.10 (#329).
- The typing-extensions dependency, which only Python 3.10 needed (#329).

### Fixed

- `mutation_prob_factor` now sets the mutation probability to `factor / n_var`
  (at most 1) in MOEA/D, NSGA-III, SMS-EMOA, SPEA2, AGE-MOEA and RVEA, as it did
  in NSGA-II and IBEA. Those six ignored it, so a run that sets it now changes
  (#292).
- A Ctrl+C in `vamos study run`, `resume` or `retry`, or a Stop in VAMOS Studio,
  that arrived just after a study wrote a journal event and before it recorded
  that event in its checkpoints left a study that `load_study` rejected with
  `CHECKPOINT_JOURNAL_INCONSISTENCY`; the CLI then failed with
  `CANCELLATION_PUBLICATION_FAILED`. The study layer now holds those
  interruptions until the event and its checkpoints are written (#301).
- A study whose process was killed after its first journal event
  (`execution_started`, or the `study_cancelled`/`study_completed` of an idle or
  empty study) and before its root checkpoint recorded it could not be loaded
  (`CHECKPOINT_JOURNAL_INCONSISTENCY`), although a checkpoint behind any later
  event loaded. It now loads with the journal's view, without writing, and
  resumes (#301).
- A `Study` handle loaded or created with larger `StudyLoadLimits` reloaded with
  the defaults in `run()`, `resume()`, `retry()`, `cancel()`, `inspect()` and
  `summarize()`, so a study that needed them could be created but not run or
  inspected; the study's runs were always read with the default `LoadLimits`
  (#303).
- `vamos summarize`, `vamos open-results` and `discover_runs` no longer count a
  replay written by `vamos reproduce` (by default under
  `<source-parent>/replays/`) as one more run of its seed;
  `discover_runs(include_replays=True)` lists replays, marked by
  `RunInfo.replay`. The run-artifacts guide explains how to tell a mismatching
  replay, stored as a completed run, from an exact one (#308).
- `non_uniform` is Michalewicz's non-uniform mutation, as in jMetal: each
  mutated variable moves towards a bound chosen at random, by a fraction of its
  distance to it, and never leaves its bounds. The step was the whole range with
  a random sign, so 25% to 41% of the mutated genes left their bounds. Without
  the run's progress, `perturbation` has no effect, and the tuning spaces no
  longer offer `nonuniform_perturbation` (#307).
- MOEA/D, SPEA2 and SMPSO notify a live callback that subclasses
  `NoOpLiveVisualization` every generation, as the other algorithms do. They
  skipped every instance of the no-op class, so such a callback received
  generation 0 only (#310).
- `vamos` warns before a run that Numba or MooCore is not installed when the
  config spells the engine in another case, such as `engine: NUMBA`; the config
  check and the run already accepted any case, but the run started without the
  warning (#317).
- Command-line NSGA-III (`vamos --algorithm nsgaiii`, experiment configuration
  files, `vamos zoo`) uses SBX with distribution index 30, as Deb and Jain
  (2014), pymoo, the MOEA Framework, `NSGAIIIConfig.default()` and
  `vamos.optimize` do; it used VAMOS's general index 20. Command-line NSGA-III
  runs on real-coded problems follow a different trajectory;
  `--nsga3-crossover-eta 20` reproduces the earlier setting (#312).
- Three crossovers no longer bias the search or collapse its diversity:
  - The integer SBX rounds each child to the nearest integer instead of
    truncating it towards zero, as jMetal and jMetalPy do: parents 3 and 4 gave
    children with mean 3.25 instead of 3.5.
  - The integer arithmetic crossover (`arithmetic`, `blend`) and the real part
    of the mixed crossover, which is the default for mixed problems, give
    complementary children with a random weight per pair, where both children
    were the parents' mean. `real_crossover="mean"` keeps the mean.
  - `blx_alpha_beta` defaults to `alpha = beta = 0.5`, jMetal's values. With
    0.75 and 0.25, crossover alone drove a population from a mean of 0.5 to
    0.006 in 30 generations (#324).
  Runs on mixed problems with the default crossover, integer runs with
  SBX or the arithmetic crossover, and `blx_alpha_beta` without explicit
  parameters give different results (#324).
- After Ctrl+C, `vamos study run`, `resume` and `retry` no longer tell you to
  run `vamos study resume`: the interruption cancels the study, which cannot be
  resumed, and `inspect()` no longer suggests `resume_to_reconcile` for a
  cancelled or failed study. The studies guide now says so, and documents that a
  second process running the same study is not detected and that Windows writes
  skip the directory fsync (#325).
- A saved SPEA2 run keeps its external archive in `result.npz`
  (`external_archive/F`, `X`, `G` and `CV`) when the result's `X`/`F` do not
  hold it, as with `result_mode="population"`: it was lost before. A VAMOS 1.2.1
  reader rejects these bundles; every other bundle is unchanged (#323).

## [1.2.1] - 2026-11-15

VAMOS 1.2.1 is a patch release. It corrects C2-DTLZ2's default radius with two
objectives, which changes results of affected experiments, and the constraint
count of eight ZXH-CF problems, keeps persistent Optuna studies of the tuning
CLI from mixing with those of earlier versions, and corrects two install
instructions in the documentation. See the [1.2.1 release
notes](docs/project/release-notes-1.2.1.md).

### Fixed

- C2-DTLZ2's default radius with two objectives is 0.5, as in Jain and Deb
  (2014) and jMetal, instead of 0.2: the whole quarter circle of its front is
  feasible, and its packaged two-objective front is regenerated. This changes
  the results of two-objective C2-DTLZ2 runs; `r=0.2` gives the former problem
  (#280).
- ZXH-CF5–8 and ZXH-CF13–16 have one constraint per constrained angle, the
  k_theta constraints of Zhou, Xiang and He (2021), as ZXH-CF9–12 do: `k + 1`
  or `k + 2` constraints instead of `2k + 1` or `2k + 2`. Feasibility and the
  overall constraint violation do not change; techniques that use the
  individual constraints, such as `violated_constraints`, can rank differently
  (#281).
- `vamos tune` stores persistent Optuna studies under
  `__vamos_cli_final_population_hv_v2`. VAMOS 1.2.0 changed the built-in search
  spaces and scores but kept `_v1`, so resuming an earlier study mixed trials of
  two contracts or stopped with Optuna's `ValueError` ("CategoricalDistribution
  does not support dynamic value space"); such a study now starts afresh
  (#282).
- Documentation: the troubleshooting guide no longer lists the removed
  `research` extra and gives notebooks' `[analysis,examples]` install, and the
  algorithms reference describes the removed baselines with the 1.1.0 install
  command.

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
- CCMO (`ccmo`), the coevolutionary framework for constrained problems of Tian
  et al. (2021), following jMetal's port: its fitness and selection match
  jMetal's on fixed populations, and its IGD on MW1, MW3 and C1-DTLZ1 matches
  the published values. `CCMOConfig` and the `ccmo` tuning space configure it,
  and its budget is exact. It stays outside the stable API, which the 1.0
  compatibility check now lets grow; `VariationPipeline` gains `recombine`
  and `mutate` (#272).
- C-TAEA (`ctaea`), the two-archive algorithm for constrained problems of Li et
  al. (2019), following jMetal's port: its weight vectors, archive updates and
  mating selection match jMetal's on fixed inputs, and its IGD+ and IGD on
  C1-DTLZ1, C3-DTLZ4, MW1, MW3 and LIR-CMOP7 agree with jMetal's runs and the
  published values. `CTAEAConfig` and the `ctaea` tuning space configure it.
  Its population is the number of weight vectors that `pop_size` gives, its
  budget is exact, and it stays outside the stable API (#275).
- ToP (`top`), the two-phase framework for constrained problems of Liu and
  Wang (2019) with NSGA-II in its second phase, following jMetal's port and
  PlatEMO's defaults: its sorting, crowding distance, differential evolution,
  mutation and phase-1 rules match jMetal's on fixed inputs, and its IGD on
  MW1, MW3, MW11, LIR-CMOP2, LIR-CMOP7, LIR-CMOP10 and C1-DTLZ1 matches
  jMetal's runs. `ToPConfig` and the `top` tuning space configure it, its
  budget is exact, and it stays outside the stable API (#277).
- CMOEA-MS (`cmoeams`), the multi-stage algorithm for constrained problems of
  Tian et al. (2022), following jMetal's port: SPEA2's fitness on density and
  normalized violation in stage A, violation first in stage B, with the
  paper's defaults and a DE variation with jMetal's binomial and exponential
  crossovers. Its violation, density, fitness, selections, stage rule and
  operators match jMetal's on fixed inputs, and its IGD on MW1, MW3, MW7,
  LIR-CMOP2, LIR-CMOP7, LIR-CMOP8, LIR-CMOP10 and C1-DTLZ1 matches jMetal's
  runs. `CMOEAMSConfig` and the `cmoeams` tuning space configure it, its
  budget is exact, and it stays outside the stable API (#278).

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

- `plot_critical_distance` no longer accepts `higher_is_better`, which had no
  effect (#276).

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
- Studio opens runs with 64-bit seeds, uses the chosen significance level for
  the Nemenyi difference, searches studies in the background, and keeps
  Matplotlib's backend; `reduce_objectives` no longer depends on units (#276).

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

[Unreleased]: https://github.com/vamos-optimization/VAMOS/compare/v1.5.0...HEAD
[1.5.0]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.5.0
[1.4.0]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.4.0
[1.3.0]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.3.0
[1.2.1]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.2.1
[1.2.0]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.2.0
[1.1.0]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.1.0
[1.0.2]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.0.2
[1.0.1]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.0.1
[1.0.0]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.0.0
