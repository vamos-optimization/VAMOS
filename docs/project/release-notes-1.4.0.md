# VAMOS 1.4.0 release notes

Released 2026-11-17.

VAMOS 1.4.0 is a minor release of [VAMOS 1.3.0](release-notes-1.3.0.md). Its
built-in tuning spaces become grammars: YAML files with a default for every
parameter, which the tuning backends and Studio's algorithm forms read. MOEA/D
and SMS-EMOA gain the choices that Evolver offers for them, so that a
configuration tuned or published with Evolver can be expressed in VAMOS, as
VAMOS-Studio requested. As VAMOS 1.3.0 announced, VAMOS Assist, Studio's
Problem Builder and the LLM extras are removed. The default configurations of
the algorithms give the results of 1.3.0, but seeded tuning runs draw different
trials and persistent Optuna studies of 1.3.0 are not resumed: read
[which results change](#which-results-change). The stable 1.x API, the
omitted-engine NumPy default, and the 1.0.0 run and study schemas remain the
compatibility baseline.

## Tuning spaces as grammars

The [VAMOS-CONFIG design](../dev/vamos-config-design.md) turns each built-in
tuning space into a grammar under the schema `vamos.parameter-space` 1.0.0.
Tuning is Experimental, so these changes ship in a minor release.

- **Schema and validator.** `vamos.engine.tuning.grammar` reads a space from
  YAML or JSON and rejects one that breaks a rule of the schema, naming the rule
  and the place. `check_default_configuration` checks that a space's defaults
  run as its algorithm's `default()` configuration. A configuration of a space
  has three forms that read back into it: the document
  `vamos.parameter-configuration` 1.0.0, the flat assignment that
  `config_from_assignment` reads, and the command line of Evolver's
  `VAR_CONF.txt`, with VAMOS's names or Evolver's.
- **Shipped grammars.** The 53 built-in spaces are YAML files packaged in
  `vamos/resources/parameter_spaces`. Every parameter has a default; the spaces
  no longer tune `pop_size`, which is a run setting as in Evolver; the external
  archive is one choice, `external_archive`; operator rates hang from their
  operators; and the mutation probability is a factor of `1/n` wherever its
  default is `1/n`. Assignments saved from the spaces of VAMOS 1.3.0 still build
  the configurations they built.
- **Backends.** ConfigSpace (SMAC3, BOHB), Optuna, irace, racing and random
  search read the grammar: they sample only active parameters, and Optuna,
  irace and racing try the default configuration first.
- **Studio.** The algorithm forms are the grammar of the tuning space: they
  follow its tree, start from its defaults and edit each parameter within the
  range that tuning uses.

## Evolver's choices for MOEA/D and SMS-EMOA

New options of MOEA/D, each off by default, with the definitions of jMetal 7.8,
which Evolver uses (see the [MOEA/D configuration](../reference/api/algorithms/moead.md)):

- `subproblem_order="cyclic"` visits the subproblems in index order (Evolver's
  `cyclicIntegerSequence`); the default stays the random permutation.
- The DE crossover takes a `scheme`, `rand_1_bin` (the default), `rand_1_exp`,
  `rand_2_bin` or `rand_2_exp`, and a `base`: the current solution (the
  default, as MOEA/D-DE) or one more random parent, as Evolver's MOEA/D.
- The aggregations `augmented_tchebycheff`, `inverted_pbi` (Sato, 2014) and
  `inverse_weight_tchebycheff`, which is jMetal's and Evolver's modified
  Tchebycheff, and `normalize=True` with `epsilon` for any aggregation, which
  normalizes the objectives with the ideal point and jMetal's estimate of the
  nadir point.

The real-coded MOEA/D and SMS-EMOA tuning spaces offer what Evolver gives these
algorithms and VAMOS runs: Sobol initialization, the Lévy flight and power-law
mutations, the Laplace and fuzzy crossovers (with BLX-αβ and whole arithmetic
in SMS-EMOA), MOEA/D's choice between differential evolution and crossover and
mutation, and distribution indices in [5, 400]. The tuning name table now reads
Evolver's `modifiedTschebyscheff` as `inverse_weight_tchebycheff`: it read it
as VAMOS's `modified_tchebycheff`, a different function.

## Which results change

The [changelog](https://github.com/vamos-optimization/VAMOS/blob/main/CHANGELOG.md)
lists each change with its pull request. Rerun or redo the following before
comparing them with results of earlier versions.

| Area | Change | Rerun |
| --- | --- | --- |
| Tuning | Every built-in space is a grammar: `pop_size` is no longer tuned, every parameter has a default, the backends draw only active parameters and try the default configuration first, and the real-coded MOEA/D and SMS-EMOA spaces and the mixed spaces offer more choices. | Every seeded tuning run draws different trials. Stored assignments build the configurations they built. |
| Persistent Optuna studies | `vamos tune` names them after the hash of the space it searches (`__vamos_cli_final_population_hv_<hash>`, SMPSO: `__vamos_cli_leaders_hv_<hash>`). | A study of VAMOS 1.3.0 or earlier is not resumed; a changed space starts a new study. |
| Mixed variation | MOEA/D, NSGA-III, SMS-EMOA and SPEA2 run the segment settings of their configuration (`perm_crossover`, `real_mutation_prob`, `int_mutation_step`, ...); they ran the default mixed operators whatever these said. | Mixed-encoding runs of those algorithms that set a segment. Default configurations do not change. |
| Constrained tuning spaces | The default `epsilon_decay_fraction` is 0.8, Takahama and Sakai's; it was 0.2. | Tuning runs that rely on that default. |
| Configurations exchanged with Evolver | Evolver's `modifiedTschebyscheff` reads as `inverse_weight_tchebycheff`, and `to_evolver` refuses `modified_tchebycheff`, which Evolver does not have. | MOEA/D configurations read from Evolver with that aggregation. |

The default configurations of the algorithms, runs that set none of these, and
replays of stored runs give the same results as before. MOEA/D's new options
leave a run that does not set them unchanged, bit for bit, as tests pin.

## Removed

As announced with VAMOS 1.3.0, the Experimental features that depended on LLM
providers are removed:

- `vamos.assist` and the `vamos assist` command, which now fails as
  unrecognized arguments (exit status 2);
- Studio's Problem Builder page, which ran Python typed into Studio or drafted by
  an LLM provider; Studio runs registered problems only;
- the `openai`, `anthropic` and `gemini` extras, also from `all`.

Define a problem with `make_problem` or `vamos create-problem`, and run it from
Python, the command line or Studio.

## Dependencies

- PyYAML (`pyyaml>=6.0`) is a dependency of VAMOS, which reads its YAML tuning
  spaces at run time; it was in the `dev` extra.
- With NumPy 2.4, the Numba engine keeps its compiled SBX, polynomial mutation
  and steady-state steps: NumPy 2.4 computes `np.power(x, 0.5)` as `sqrt(x)` and
  `np.power(x, -1)` as `1 / x`, and the compiled power now takes the same
  shortcuts. VAMOS 1.3.0 fell back to the NumPy operators there, with the same
  results but slower. A NumPy build whose `np.power` uses vendor SIMD code
  still keeps the NumPy operators (see the [known limitations](known-limitations.md)).
- The release CI tests NumPy 2.4.6 and SciPy 1.17.1 on every Python, 3.11 to
  3.14. The declared minimums do not change.

## Known limitations

See [known limitations](known-limitations.md).
