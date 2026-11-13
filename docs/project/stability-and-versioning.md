# Stability and versioning

VAMOS 1.0.0 is the first official public release and the first compatibility
baseline. Earlier version strings, Git tags, artifacts, and undocumented APIs
were internal development markers and are not public compatibility contracts.

This document defines the surfaces covered by semantic versioning. The
machine-readable baseline under `tests/compatibility/v1_0_0/` is the
enforcement companion to this policy.

## Stability labels

Every supported surface has one of three labels:

- **Stable** means that the compatibility rules in this document apply for the
  complete 1.x series.
- **Experimental** means that the feature is supported for evaluation but may
  change incompatibly in a minor release. Experimental status is stated in
  its documentation and help.
- **Internal** means that the implementation is not a public interface and has
  no compatibility guarantee.

An importable name is not automatically stable. A surface is stable only when
it is listed here and represented in the VAMOS 1.0 compatibility baseline.

## Stable Python API

The stable Python surface is exposed through `vamos`, `vamos.api`,
`vamos.algorithms`, `vamos.problems`, `vamos.run_artifacts`, and
`vamos.study_artifacts`. `vamos.api` exports the same stable names as `vamos`;
examples import from `vamos`. `vamos.ux.api` is an experimental facade.

These facades also export a few experimental names, which this policy does not
cover even though a stable facade exports them: `select_survivors` and
`EnvironmentalSelectionResult` (direct environmental selection),
`configure_logging` and `run_self_check` (development diagnostics) from `vamos`
and `vamos.api`, and `GenericAlgorithmConfig`, `discover_algorithm_plugins`,
and `resolve_algorithm` (custom and plugin algorithms) from `vamos.algorithms`.
Their docstrings say so.

### Core optimization

The stable core consists of:

- `Problem`, `make_problem`, `make_problem_selection`, and
  `available_problem_names`;
- `optimize`;
- `OptimizationResult` and `StudyResult`;
- the documented problem classes exported by `vamos.problems`;
- the built-in algorithm configuration classes exported by
  `vamos.algorithms`;
- `available_algorithms`, `available_crossover_methods`, and
  `available_mutation_methods`;
- the `__version__` runtime version.

The stable built-in algorithm identifiers are `agemoea`, `ibea`, `moead`,
`nsgaii`, `nsgaiii`, `rvea`, `smpso`, `smsemoa`, and `spea2`. Their public
configuration fields are frozen in
`tests/compatibility/v1_0_0/stable_algorithm_configs.json`.

**Development after 1.1.0:** built-in algorithms may be added during 1.x; the
compatibility check accepts identifiers added after the frozen ones and still
fails when a 1.0 identifier disappears. `ccmo` and its `CCMOConfig` are the
first such addition, and `ctaea` with its `CTAEAConfig` the second. Until a
release lists them here, they are not part of the stable surface, although
`vamos.algorithms` exports `CCMOConfig` and `CTAEAConfig`.

For these entry points:

- `max_evaluations` is a hard evaluation budget unless the selected algorithm
  documents a population-cardinality requirement that makes the requested
  value invalid;
- a supplied integer seed owns the stochastic path and produces deterministic
  same-environment behavior for the documented built-in implementation;
- an omitted seed is the fixed default seed 42, so repeating a call without a
  seed repeats the same run; `seed=None` draws a new seed before execution, and
  a canonical run persists the resolved integer;
- an explicitly requested unavailable backend fails rather than silently
  selecting another backend;
- `numpy` is the deterministic reference backend. Cross-backend bitwise
  equality is not promised.

Omitting `engine` continues to select NumPy throughout 1.x. Explicit `auto`
opts into an environment-dependent acceleration heuristic; its coverage may
expand as built-in backends gain support. Persisted runs and created studies
retain their resolved engine and never reapply that heuristic for exact replay
or execution of an existing study plan. See the
[automatic engine policy](../reference/algorithms.md#automatic-engine-selection).

### Canonical run lifecycle

The stable run lifecycle consists of:

- `save_result`;
- `load_run`;
- `load_result`;
- `verify_run`;
- `reproduce`;
- `StoredRun`, `RunManifest`, `LoadLimits`,
  `IncompleteRunMetadataError`, `CompatibilityReport`,
  `VerificationReport`, and `ReplayReport`.

Loading and verification are inert, data-only operations. `reproduce` is the
separate executable operation and supports exact replay only for reconstructable
built-in components in the same materially relevant environment and backend.

The successful run layout contains `manifest.json`, `result.npz`, and
`environment.json`. A failed canonical run contains `manifest.json` and
`environment.json` and does not pretend to have numerical results. Relative,
root-confined artifact references and their integrity evidence are part of the
stable contract.

### Canonical single-owner study lifecycle

The stable study lifecycle consists of:

- `StudySpec`, `StudyLoadLimits`, and `StudyPlanReport`;
- `plan_study`, `create_study`, and `load_study`;
- `Study` and its documented `run`, `inspect`, `summarize`, `cancel`, `resume`,
  and `retry` methods;
- `StudyReport` and `StudySummary`.

Study mutation is single-owner in VAMOS 1.1.0. Do not run concurrent
`run`, `resume`, or `retry` operations against the same study. There is no
cross-process cancellation command. Inspection and summary are immutable,
data-only projections.

### Exceptions

**Development after 1.1.0:** the stable API raises errors of one hierarchy,
exported by `vamos.exceptions`. Catch them by class; their messages may change.

- `VAMOSError` is the base of every error below.
- `ConfigurationError` (also a `ValueError`) and its subclass
  `InvalidAlgorithmError` report invalid arguments and configurations.
- `ProblemError` and its subclasses `InvalidProblemError` (also a `KeyError`,
  the type an unknown problem name raised in VAMOS 1.1.0),
  `ProblemDimensionError`, and `BoundsError` report invalid problems.
- `OptimizationError` and its subclass `EvaluationError` report failures while
  a run executes.
- `ResultSelectionError` (also a `ValueError`) and its subclass
  `NoSolutionsError` report selections that a result cannot satisfy.
- `DependencyError` reports a missing optional dependency.
- `RunArtifactError` is the base of the errors of the run lifecycle, and
  `StudyError` the base of the errors of the study lifecycle. In VAMOS 1.1.0
  they derived only from `Exception`.

`InvalidEngineError`, `InvalidOperatorError`, `MissingConfigError`,
`ConvergenceError`, `ConstraintViolationError`, `ResultsNotFoundError`, and
`BackendNotAvailableError` are exported, but VAMOS never raises them. They stay
importable during 1.x and are planned for removal in 2.0.

## Stable CLI

The stable command surface is:

- the documented top-level optimization invocation;
- `vamos results inspect`;
- `vamos results verify`;
- `vamos reproduce`;
- `vamos study plan`;
- `vamos study create`;
- `vamos study run`;
- `vamos study inspect`;
- `vamos study resume`;
- `vamos study retry`;
- `vamos study summarize`.

The command tree, documented argument names, JSON output shapes, and exit-code
semantics are frozen in `tests/compatibility/v1_0_0/`. During 1.x, a stable
command may gain an optional argument or a JSON object may gain an additive
field, but existing valid invocations and fields retain their meaning. JSON
mode writes exactly one JSON document to stdout; diagnostics and warnings use
stderr.

Other commands, including `studio`, `assist`, `tune`, `ablation`, `profile`,
`bench`, `zoo`, `quickstart`, `create-problem`, `summarize`, `open-results`,
and development diagnostics, are experimental unless a later stability policy
explicitly promotes them.

## Stable artifact schemas and envelopes

The first public artifact baseline uses:

- `vamos.run-manifest` schema `1.0.0`;
- `vamos.environment` schema `1.0.0`;
- the canonical study documents `vamos.study-spec`,
  `vamos.resolved-study-plan`, `vamos.study-manifest`, `vamos.study-task`,
  `vamos.study-attempt`, and `vamos.study-event`, all schema `1.0.0`;
- `vamos.study-report` and `vamos.study-summary`, schema `1.0.0`;
- `vamos.study-command-result`, schema `1.0.0`;
- run inspection, verification, replay, and command-error envelopes at
  envelope version `1`.

Future VAMOS 1.x releases must continue to load, verify where applicable, and
inspect valid public 1.0.0 run and study artifacts. A breaking artifact change
requires a new schema identity or version and a supported transition for
publicly released artifacts. Integrity hashes are tamper-detection evidence,
not authenticity signatures.

## Stable configuration

Documented fields of stable algorithm configuration classes and `StudySpec`
retain their names and meanings during 1.x. New optional fields and stricter
rejection of previously invalid input are compatible changes. Removing a
field, changing a default incompatibly, changing an identifier's meaning, or
accepting a different backend silently is a breaking change.

Undocumented provider, plugin, tuning, Studio, and internal configuration is
not covered by this promise.

## Experimental surface

The following remain experimental in VAMOS 1.1.0:

- Studio and all generated-code execution;
- LLM-provider and provider-specific assist integrations;
- plugin discovery and custom component interfaces without a frozen
  descriptor contract;
- tuning and racing APIs outside the stable facade;
- `vamos.ux.api`, statistical analysis, visualization, and MCDM helpers;
- direct environmental-selection helpers;
- optional third-party integrations without a compatibility contract;
- profiling, research, development, and the non-stable CLI commands listed
  above;
- multiprocess ownership, worker pools, and distributed study execution.

Studio executes reviewed Python only as explicitly trusted local code. Its AST
validation is input validation, not a security sandbox.

## Internal surface

Deep modules not exported through a documented stable facade are internal.
This includes filesystem transaction helpers, journal implementation,
checkpoint projection, schema decoders, internal registries, numerical helpers
not listed in the stable API, and test hooks. Documentation or source access to
an internal module does not create a compatibility promise.

## Semantic-versioning policy

VAMOS follows semantic versioning from 1.0.0 onward:

- patch releases fix defects without intentionally changing stable behavior;
- minor releases may add stable functionality and may change experimental
  functionality;
- incompatible changes to a stable Python API, CLI contract, configuration,
  or public artifact schema require a new major version, except for the
  scientific corrections and the security exception below.

## Scientific corrections

**Development after 1.1.0:** a built-in algorithm, operator, problem, or
indicator that departs from its published definition is a defect. Its fix
ships in a patch or minor release even when it changes numerical results, or
the default value of a stable signature, because results of the faulty
definition are not results of the published method. The changelog lists such a
fix as Fixed with a note on which results change and how, so that affected
experiments can be rerun. Two fixes after 1.1.0 are examples:

- CEC2009 UF3 searches its published domain `[0, 1]^n` (pull request #104).
  VAMOS 1.1.0 searched `[-1, 1]` for every variable after the first, a
  different problem.
- IBEA defaults to `kappa=0.05`, the value of Zitzler and Künzli (2004),
  instead of 1.0, and normalizes the objectives before the epsilon indicator,
  as adaptive IBEA does (pull request #115).

Choices that the literature leaves open, such as a default population size,
are not corrections; changing them follows the rules for stable defaults.

## Deprecation policy

A stable API is documented as deprecated before removal and remains available
through the rest of the 1.x series. Removal normally occurs only in a later
major release. Deprecation warnings identify the replacement and planned
major-version boundary. Compatibility aliases are not added for pre-1.0
internal prototypes.

## Security exception

A critical security fix may restrict or remove stable behavior before the next
major release when preserving it would expose users to material harm. The
release notes and security advisory must identify the exception, impact,
replacement where available, and affected versions.

## Python and operating-system support

VAMOS 1.1.0 supports only the Python versions and operating systems exercised
by the final hosted release matrix and declared in package metadata. Dropping
a supported Python version during 1.x requires advance documentation and a
minor release. Optional dependencies may have narrower platform support and
must fail clearly when unavailable.

**Development after 1.1.0:** CI tests Python 3.10 to 3.14 on Linux and
Python 3.12 to 3.14 on Windows and macOS, and the package metadata declares
Python 3.10 to 3.14. The metadata also declares the development status
"Production/Stable", which matches the commitments of this policy. VAMOS 1.1.0
declared Python 3.10 to 3.12 and the status "Beta".

**Development after 1.1.0:** Python 3.10 reached its end of life in October
2026. VAMOS 1.2.0 is the last minor release that supports it: VAMOS 1.3.0 will
require Python 3.11 or newer, and the 1.2 patch releases keep Python 3.10.

## Support window

The newest 1.x release is the primary supported release. Valid public 1.0.0
artifacts remain loadable and inspectable throughout 1.x even when users are
asked to update to the newest patch or minor release. Security backports to an
older installer are made only when explicitly announced.

## Pre-1.0 development policy

Pre-1.0 development artifacts, internal version tags, undocumented APIs, and
prototype formats are unsupported. They do not receive readers, migrations,
aliases, or deprecation cycles. Git history and the external pre-public tag
archive preserve that development history without turning it into a runtime
compatibility burden.
