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
`vamos.study_artifacts`.

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

For these entry points:

- `max_evaluations` is a hard evaluation budget unless the selected algorithm
  documents a population-cardinality requirement that makes the requested
  value invalid;
- a supplied integer seed owns the stochastic path and produces deterministic
  same-environment behavior for the documented built-in implementation;
- an omitted seed is resolved before execution and the resolved integer is
  persisted in a canonical run;
- an explicitly requested unavailable backend fails rather than silently
  selecting another backend;
- `numpy` is the deterministic reference backend. Cross-backend bitwise
  equality is not promised.

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

Study mutation is single-owner in VAMOS 1.0.1. Do not run concurrent
`run`, `resume`, or `retry` operations against the same study. There is no
cross-process cancellation command. Inspection and summary are immutable,
data-only projections.

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

The following remain experimental in VAMOS 1.0.1:

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
  security exception below.

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

VAMOS 1.0.1 supports only the Python versions and operating systems exercised
by the final hosted release matrix and declared in package metadata. Dropping
a supported Python version during 1.x requires advance documentation and a
minor release. Optional dependencies may have narrower platform support and
must fail clearly when unavailable.

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
