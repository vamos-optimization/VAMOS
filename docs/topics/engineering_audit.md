# VAMOS Engineering Audit

Last updated: 2026-05-02

This page records the current publication-readiness audit for VAMOS. It is intentionally blunt: the goal is to keep the public docs aligned with what the codebase actually does today.

## Follow-up Status

Additional follow-up on 2026-05-02 addressed the highest-risk engineering
findings outside release metadata:

- archive config validation now accepts the maintained bounded-archive schema
- population evaluation rejects missing, wrong-shaped, or non-finite objective/constraint outputs
- build smoke checks clean stale build artifacts and verify wheel modules against `src/vamos`
- algorithm package exports are lazy, and import-cycle checks now cover algorithm modules
- AGE-MOEA and RVEA config schemas are included in experiment-spec validation
- algorithm plugins can be discovered through the `vamos.algorithms` entry point group
- operator validation derives known names from runtime registries instead of a stale hardcoded list
- bounded archive nondominated/crowding paths reuse vectorized kernel helpers, and 2D HV contribution pruning is linear in archive size
- Dask evaluation fallback is explicit via `fallback_to_serial=True`
- CI mypy coverage now includes registry, config-spec, evaluation, and optimization-result layers

Release metadata/classifier status is tracked separately from this page.

The highest-friction CLI issues identified in this audit were fixed in a follow-up patch on 2026-03-31:

- standard `vamos --problem ...` execution now has subprocess smoke coverage
- `vamos quickstart --yes ...` execution now has subprocess smoke coverage
- `vamos create-problem --yes ...` execution now has subprocess smoke coverage
- CLI `--engine auto` is now accepted and covered by smoke tests
- config-driven `vamos --config ...` execution and `--validate-config` now have subprocess smoke coverage
- `vamos bench ... --smoke` now provides a cheap, real orchestration path for benchmark CLI smoke coverage
- `vamos tune ... --backend random --smoke` now provides a cheap, real orchestration path for tuning CLI smoke coverage
- `vamos profile` and the common `vamos zoo` commands now have real execution smoke coverage
- CLI guide command examples now have dedicated command-level smoke coverage in the docs test suite
- README command examples now have dedicated command-level smoke coverage for the published onboarding path
- The canonical tuning guide now has command-level smoke coverage for the maintained tune smoke path
- SPEA2 raw-fitness and density computation now run through a vectorized path rather than nested Python loops
- IBEA hypervolume indicators now use an exact vectorized matrix formulation instead of repeated pairwise hypervolume calls
- the NumPy kernel now uses chunked tournament sampling and a memory-safe blocked non-dominated sort path for large arrays
- genealogy tolerance matching now uses a shared blocked matcher instead of row-by-row survivor scans
- `optimize(..., seed=[...])` now returns `StudyResult` with built-in aggregation helpers
- algorithm config builders now share internal mixins instead of repeating the same fluent method bodies across every algorithm
- source docstrings are now normalized away from Google-style `Args:` sections, and tests now guard against reintroducing `np.random.rand(` in the test suite
- a pinned publication environment was recorded during this audit; it is now maintained with the manuscript outside the public repository

The remaining sections preserve the audit rationale, especially around vectorization semantics, reproducibility expectations, and test-surface gaps outside the common path.

## Executive Summary

VAMOS has a credible technical core and is in materially better shape than it was at the time of the original audit. The programmatic `optimize(...)` API now has a stronger multi-run surface through `StudyResult`, the standard CLI onboarding path is smoke-tested, and the published README/CLI/tuning command surface has real command-level coverage instead of relying on help text or trust. The main technical blockers around SPEA2, IBEA, large-tournament fallback behavior, and large-population non-dominated sorting were also reduced with exact vectorized or blocked implementations. The remaining publication risks are narrower: some performance-sensitive utilities outside the main kernels still deserve more benchmarking, the heaviest optional tuning and benchmark matrices still have thinner verification than a publication-grade framework should target, and packaging metadata is not yet at a final 1.0-style release gate. Against pymoo, VAMOS is now materially closer on user-facing polish and evaluation-surface clarity, but it still trails on ecosystem maturity and benchmark ergonomics.

Current grades:

- Architecture and design: B
- API design and user-friendliness: B-
- Code quality and maintainability: B
- Testing and reliability: B+
- Performance and vectorization: B-
- Packaging and distribution: B
- Documentation and onboarding: B+
- Comparative best practices vs pymoo: B

## Critical Issues

### 1. Run-oriented CLI onboarding was broken at audit time

Verified behavior:

- `vamos --problem zdt1 --algorithm nsgaii ...` fails with `Invalid operator spec for 'crossover'`.
- `vamos quickstart --yes --no-plot ...` fails with the same underlying error.

Root cause:

- `src/vamos/experiment/cli/common.py` emits variation override dictionaries even when the operator `method` is `None`.
- `src/vamos/engine/algorithm/_builder_common.py` forwards those dictionaries into variation resolution.
- `src/vamos/engine/config/variation.py` correctly rejects them as invalid operator specs.

Status:

- Fixed in follow-up. Base CLI runs and quickstart execution now have smoke coverage for the common path.

### 2. CLI help and runtime disagreed on `--engine auto`

Verified behavior:

- `src/vamos/experiment/runtime/catalog.py` supports programmatic `engine="auto"`.
- `src/vamos/experiment/cli/args_core.py` restricts CLI `--engine` choices to `numpy`, `numba`, and `moocore`.
- The CLI help text still tells users to pass `--engine auto`.

Status:

- Fixed in follow-up. The CLI parser now accepts `--engine auto`, and the common path is smoke-tested.

### 3. `make_problem(..., vectorized=False)` is elementwise adaptation, not auto-vectorization

Verified behavior:

- In `src/vamos/foundation/problem/builder.py`, scalar problems are evaluated with a Python loop over rows when `vectorized=False`.
- That is compatibility-focused elementwise adaptation, not real vectorized execution.

Comparison to pymoo:

- pymoo explicitly distinguishes batched `Problem` and elementwise `ElementwiseProblem`.
- VAMOS should be equally explicit instead of describing the elementwise adapter as "auto-vectorization".

Documentation consequence:

- Tell users that scalar callables are supported for convenience.
- Tell performance-sensitive users to pass `vectorized=True` and implement a true batched function.

## Important Improvements

### Extend smoke coverage beyond the common onboarding path

The current test suite exercises the Python API and many core internals. The common CLI onboarding path, config-driven CLI execution, benchmark smoke mode, tune smoke mode, create-problem, profile, zoo, and the published README/CLI-guide/tuning-guide commands now have subprocess smoke coverage. The next gap is breadth rather than total absence. VAMOS should continue by adding:

- broader benchmark/tuning smoke coverage beyond the current `vamos bench ... --smoke`, `vamos tune ... --smoke`, and the existing tune split/fallback slices
- broader docs smoke coverage for additional CLI subcommands beyond the current guide examples and common paths

### Stop publishing stale CLI and tooling commands

The docs had drift such as:

- `--nsgaii-replacement-size` documented after it was removed from the parser
- `python tools/health.py --mypy-full` documented even though the tool does not implement that flag

Every CLI or tooling command that appears in public docs should either be smoke-tested or clearly labeled as reference syntax.

### Tighten public typing and exception contracts

Internally, the repo already uses mypy and a custom exception hierarchy. Public surfaces still expose some weak contracts, including `object` and `Any` in optimize-layer overloads and inconsistent exception documentation around `make_problem()` and `Problem.evaluate()`. This is fixable, but the docs should not overclaim a stricter public contract than the code currently provides.

### Publish a pinned paper environment

The `research` extra intentionally keeps wide version ranges for benchmarking libraries. That is acceptable for installation flexibility, but it is not enough for paper-grade reproducibility. Publication artifacts should include a lockfile or fully pinned environment spec.

## Comparative Notes vs pymoo

Where VAMOS currently falls short of reviewer expectations set by pymoo:

- pymoo is clearer about evaluation mode. Its docs separate elementwise and batched problems instead of implying that one becomes the other automatically.
- pymoo's getting-started surface is more internally consistent: advertised commands and documented options generally work as shown.
- VAMOS has stronger architecture-health guardrails than a typical research repo, but reviewers will notice onboarding breakage before they notice internal discipline.

## Strengths

The audit is not uniformly negative. Specific strengths worth preserving:

- The programmatic `optimize(...)` path is viable and should remain the primary public facade.
- The registry/config architecture provides a real extension story for algorithms, operators, and problems.
- The repo already ships typed package metadata (`py.typed`) and runs architecture-focused tests, not only numerical correctness tests.
- Optional backends and extras are cleanly separated enough that the core package is still usable without the full research stack.

## Maintainer Guidance

When updating docs, examples, or agent playbooks:

- Prefer `optimize(...)` for the shortest first script, but it is now accurate to document the standard CLI path as smoke-tested for NSGA-II/ZDT1.
- Do not claim that scalar custom problems are vectorized unless the user explicitly passes `vectorized=True`.
- CLI `--engine auto` is now supported; keep parser/help/runtime behavior aligned and covered by tests.
- Treat benchmark-facing environments as pinned artifacts, not just extras declarations.
