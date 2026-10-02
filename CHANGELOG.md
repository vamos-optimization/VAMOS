# Changelog

All notable public changes to VAMOS are documented here.

## [Unreleased]

### Added

- Experimental `vamos gui` command and `gui` extra: a local NiceGUI interface that
  lists canonical runs, explores their fronts interactively (projections,
  selection, CSV export, integrity verification), and launches built-in runs in
  worker processes with live front and progress-hypervolume updates.

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

[Unreleased]: https://github.com/vamos-optimization/VAMOS/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.0.1
[1.0.0]: https://github.com/vamos-optimization/VAMOS/releases/tag/v1.0.0
