# VAMOS Software Engineering Audit

This is a historical audit. References to manuscript files and publication
commands describe that snapshot; those materials are now maintained separately
from the public repository.

## 1. Executive Summary

VAMOS has a credible research-software core: it has a clear `src` layout, public facades, typed package metadata, a large test suite, architecture guard tests, optional backend separation, and a usable `vamos` console command. It is not yet publication-grade for an IEEE TEVC software contribution without remediation, mainly because release identity is inconsistent, the current source type check fails, formatting debt is explicitly budgeted, and some publication-facing workflows have weak coverage. The "Vectorized Architecture" claim is mostly supported in core kernels and batched problem protocols, but the default functional-problem adapter is intentionally elementwise and needs clear positioning. Overall readiness: **Promising but not submission-ready**.

| Dimension | Grade | Rationale |
|---|---:|---|
| Architecture & Design | B | Layering is explicit and guarded by tests, facades exist, and extension registries are present; archive typing/private backend coupling still leaks across internal boundaries. |
| API Design & User-Friendliness | B | `optimize`, `make_problem`, `vamos.algorithms`, and CLI subcommands give a coherent path; `python -m vamos` and scalar/vectorized semantics remain onboarding friction. |
| Code Quality & Maintainability | C+ | Ruff lint passes for `src/vamos tests`, but full mypy fails and formatting debt is accepted by a budget. |
| Testing & Reliability | B- | 1151 tests pass and architecture tests are strong; total coverage is 69% with 0-24% coverage in several publication-facing modules. |
| Performance & Vectorization | B- | Core NumPy kernels use vectorized/blocked paths; scalar adapters and generic HV contribution fallback create credible performance caveats. |
| Packaging & Distribution | C+ | Metadata, extras, entry point, package data, and `py.typed` exist; package/import/citation versions disagree. |
| Documentation & Onboarding | B | README and guide are detailed and command-oriented; some claims need sharper wording around vectorization and release identity. |
| Comparative Best Practices | B- | VAMOS has framework-style facades and guardrails; without inspecting pymoo source/docs in this audit, comparisons are limited to general framework expectations. |
| TEVC Readiness | C | Strong foundation, but release metadata, type-check cleanliness, reproducibility packaging, and coverage gaps must be addressed before submission. |

## 2. Critical Issues

### F-001 - [Packaging & Distribution] Release identity is internally inconsistent

- Severity: Critical
- Effort: S
- Location: `pyproject.toml:L5-L8`, `src/vamos/foundation/version.py:L7-L12`, `CITATION.cff:L20-L21`, `README.md:L341-L348`
- Evidence:

```toml
name = "vamos-optimization"
version = "1.0.0"
```

```python
__version__ = "0.1.0"
```

The installed distribution reports `vamos-optimization 1.0.0`, while `import vamos; vamos.__version__`, `CITATION.cff`, and the README citation report `0.1.0`. A TEVC reviewer or artifact evaluator will treat this as a reproducibility and release-management defect, not a cosmetic issue.

Recommendation:
Align `pyproject.toml`, `src/vamos/foundation/version.py`, `CITATION.cff`, README citation metadata, generated package metadata, and release documentation to one version. Add `tests/architecture/test_version_metadata_consistency.py` that compares `importlib.metadata.version("vamos-optimization")`, `vamos.__version__`, `CITATION.cff`, and README citation text.

Backlog details:
- Files likely to change: `pyproject.toml`, `src/vamos/foundation/version.py`, `CITATION.cff`, `README.md`, possibly generated release notes.
- Tests to add/update: `tests/architecture/test_version_metadata_consistency.py`, wheel smoke in `.github/workflows/ci.yml`.
- Risk: Low.
- Fix type: Mechanical.
- TEVC timing: Before submission.

### F-002 - [Code Quality & Maintainability] Source mypy fails in archive typing and private optional-backend coupling

- Severity: Critical
- Effort: S
- Location: `src/vamos/engine/algorithm/components/archive.py:L7-L22`, `src/vamos/engine/archive/factory.py:L21-L27`, `src/vamos/engine/hooks/hv_archive_hooks.py:L125-L148`
- Evidence:

```python
_moocore = _subset_selection._moocore
```

```python
else:
    ArchiveManager = Any
    ResultArchiveManager = Any
```

`python -m mypy src` and the CI-style scoped mypy command both failed with 8 errors. Four errors come from `archive.py` reaching into `_subset_selection._moocore`, and four come from runtime variables used as types in `factory.py` and `hv_archive_hooks.py`.

Recommendation:
Replace the runtime `Any` alias pattern with `typing.TypeAlias` definitions guarded cleanly under `TYPE_CHECKING`, or introduce a small archive `Protocol` covering the methods actually used. Move the MooCore test hook behind an explicit helper API instead of mutating another module's private `_moocore` global.

Backlog details:
- Files likely to change: `src/vamos/engine/algorithm/components/archive.py`, `src/vamos/engine/algorithm/components/subset_selection.py`, `src/vamos/engine/archive/factory.py`, `src/vamos/engine/hooks/hv_archive_hooks.py`.
- Tests to add/update: `tests/architecture/test_mypy_error_budget.py`; add a focused archive factory typing/import smoke if needed.
- Risk: Medium, because archive hooks are shared by algorithms and result modes.
- Fix type: Design-sensitive but small.
- TEVC timing: Before submission.

## 3. Important Improvements

### F-003 - [Code Quality & Maintainability] Formatting debt is budgeted, and repo-wide Ruff fails outside the CI scope

- Severity: Important
- Effort: M
- Location: `tests/architecture/ruff_format_budget.json:L1-L4`, `.github/workflows/ci.yml:L46-L57`, `tests/architecture/mypy_error_budget.json:L1-L5`
- Evidence:

```json
"max_files_to_reformat": 100
```

```yaml
run: ruff check src/vamos tests
```

`python -m ruff check src/vamos tests` passes, but `python -m ruff format --check src/vamos tests` reports 56 files that would be reformatted. `python -m ruff check .` reports 169 issues, mostly in examples, experiments, paper scripts, tools, and website compatibility code. The current gates prevent regression but do not represent a clean publication artifact.

Recommendation:
Before TEVC artifact freeze, either clean formatting debt in `src/vamos tests` and lower the budget to zero, or document the budget as transitional and exclude non-library research scripts from release validation. For examples and paper scripts that are part of the artifact, either lint them or move them under an explicitly archival path.

Backlog details:
- Files likely to change: `src/vamos/**`, `tests/**`, `tests/architecture/ruff_format_budget.json`, possibly `pyproject.toml`.
- Tests to add/update: existing `tests/architecture/test_ruff_format_gate.py` should enforce a lower budget; CI should run the same command.
- Risk: Low.
- Fix type: Mechanical.
- TEVC timing: Before submission for `src/vamos tests`; after submission is acceptable for archival scripts not shipped as artifact.

### F-004 - [Testing & Reliability] Publication-facing modules have weak or zero coverage despite a large passing suite

- Severity: Important
- Effort: L
- Location: `src/vamos/experiment/benchmark/cli.py:L72-L120`, `src/vamos/experiment/profiler/runner.py:L84-L151`, `src/vamos/ux/panel/pages/problem_builder.py:L23-L80`, `src/vamos_contrib/interop/pymoo.py:L26-L53`
- Evidence:

```python
def main(argv: list[str] | None = None) -> int:
```

```python
def run_profile(... ) -> ProfileReport:
```

Coverage passed at 69% total, but the term report showed 0% coverage for `experiment/benchmark/cli.py`, `experiment/profiler/cli.py`, `experiment/profiler/runner.py`, `experiment/zoo/cli.py`, Panel UI modules, and `vamos_contrib/interop/pymoo.py`. These are external-researcher surfaces: benchmark execution, profiling, studio onboarding, and interop.

Recommendation:
Add focused smoke and behavior tests around the publication-facing entry points rather than trying to maximize global coverage. High-value tests:
- `tests/experiment/test_benchmark_cli_execution_smoke.py::test_bench_smoke_writes_report_index`
- `tests/experiment/test_profiler_runner.py::test_run_profile_records_failed_backend_without_crashing`
- `tests/ux/test_panel_problem_builder_smoke.py::test_problem_builder_state_defaults_match_supported_algorithms`
- `tests/integration/test_pymoo_adapter.py::test_pymoo_adapter_delegates_evaluate_shape`

Backlog details:
- Files likely to change: tests under `tests/experiment`, `tests/ux`, `tests/integration`; possibly small fixes in covered modules.
- Tests to add/update: listed above.
- Risk: Medium, because testing UI/optional dependencies can become flaky if not isolated.
- Fix type: Design-sensitive test design.
- TEVC timing: Before submission for benchmark/profiler/interop; UI breadth can be after submission if Studio is not a core claim.

### F-005 - [API Design & User-Friendliness] `python -m vamos` does not work although the console script does

- Severity: Important
- Effort: S
- Location: `pyproject.toml:L165-L166`
- Evidence:

```toml
[project.scripts]
vamos = "vamos.experiment.cli.main:main"
```

The console command `vamos --help` succeeds, but `python -m vamos --help` fails with `No module named vamos.__main__`. This is not a defect if only the console script is documented, but the audit prompt explicitly required checking `python -m vamos --help or equivalent`, and module execution is a common Python packaging expectation.

Recommendation:
Add `src/vamos/__main__.py` that imports and calls `vamos.experiment.cli.main:main`, or document that `vamos` is the only supported CLI invocation. Prefer adding `__main__.py` because it is low-risk and aligns Python packaging expectations.

Backlog details:
- Files likely to change: `src/vamos/__main__.py`, `tests/e2e/test_cli_smoke.py`, docs if invocation guidance changes.
- Tests to add/update: `tests/e2e/test_cli_smoke.py::test_python_m_vamos_help`.
- Risk: Low.
- Fix type: Mechanical.
- TEVC timing: Before submission.

### F-006 - [Performance & Vectorization] Default functional custom problems are elementwise, not actually vectorized

- Severity: Important
- Effort: M
- Location: `src/vamos/foundation/problem/builder.py:L72-L77`, `docs/guide/getting-started.md:L64-L80`, `README.md:L185-L209`
- Evidence:

```python
results = [self._fn(X[i]) for i in range(X.shape[0])]
```

The code and current docs correctly say `vectorized=False` evaluates one solution at a time. The risk is branding: users can create a problem through the friendliest API and unknowingly run through a Python loop, which weakens the "Vectorized Architecture" claim if not framed as a compatibility adapter.

Recommendation:
Keep scalar support, but make docs and result metadata explicit: scalar callables are convenience mode; publication benchmarks and performance claims require `vectorized=True` or class-based batched `Problem.objectives`. Add a test that counts scalar calls vs. vectorized calls and asserts metadata records the mode.

Backlog details:
- Files likely to change: `src/vamos/foundation/problem/builder.py`, `README.md`, `docs/guide/getting-started.md`, `docs/dev/add_problem.md`.
- Tests to add/update: `tests/foundation/test_make_problem.py::test_make_problem_records_vectorization_mode`.
- Risk: Low.
- Fix type: Design-sensitive documentation/API metadata.
- TEVC timing: Before submission.

### F-007 - [Performance & Vectorization] Generic hypervolume contribution fallback is computationally expensive

- Severity: Important
- Effort: M
- Location: `src/vamos/foundation/quality_indicators/hypervolume.py:L285-L291`, `src/vamos/engine/archive/bounded_archive.py:L295-L318`
- Evidence:

```python
for i in range(points.shape[0]):
    without_i = np.delete(points, i, axis=0)
```

When MooCore is unavailable and objectives exceed the optimized 2D path, contribution computation repeatedly recomputes hypervolume after deleting each point. This is plausible O(n) hypervolume calls per pruning operation, and recursive HV can compound the cost in many objectives.

Recommendation:
Route many-objective HV contribution pruning to MooCore when available and otherwise require explicit opt-in to approximate pruning (`mc_hv`) for archive sizes above a threshold. Add a microbenchmark comparing `hv`, `mc_hv`, `crowding`, and `maxmin` for `n=100, 500` and `m=3, 5`.

Backlog details:
- Files likely to change: `src/vamos/foundation/quality_indicators/hypervolume.py`, `src/vamos/engine/archive/bounded_archive.py`, `tests/performance/test_kernel_perf_smoke.py`.
- Tests to add/update: `tests/foundation/test_hypervolume_metrics.py::test_many_objective_hv_contrib_requires_backend_or_warns`.
- Risk: Medium.
- Fix type: Design-sensitive numerical behavior.
- TEVC timing: Before submission if HV archive pruning is part of claims; otherwise medium-term.

### F-008 - [Scientific-Software Readiness] Reproducibility story is present but not yet a single standardized artifact path

- Severity: Important
- Effort: M
- Location: `README.md:L63-L68`, `pyproject.toml:L47-L163`, `docs/topics/engineering_audit.md:L134-L137`
- Evidence:

```bash
pip install -r paper/requirements-publication.txt
```

The project has a publication environment pointer and optional extras, but the audit did not verify a lockfile, manifest schema, or one-command artifact reproduction path. The broad `research` extra is reasonable for installation flexibility, but paper-grade results need a narrower, frozen environment and standardized output manifest.

Recommendation:
Publish a `paper/requirements-publication.txt` or lockfile as the canonical TEVC artifact, add a `vamos check --publication` or documented equivalent, and ensure benchmark reports include package version, git commit, Python version, backend availability, and seed matrix.

Backlog details:
- Files likely to change: `paper/requirements-publication.txt`, `docs/guide/getting-started.md`, benchmark report writers under `src/vamos/experiment/benchmark`.
- Tests to add/update: `tests/reference/test_publication_environment_metadata.py`.
- Risk: Medium.
- Fix type: Design-sensitive reproducibility workflow.
- TEVC timing: Before submission.

## 4. Minor Suggestions

### F-009 - [Documentation & Onboarding] README citation points to maintained metadata that currently disagrees

- Severity: Minor
- Effort: S
- Location: `README.md:L341-L351`, `CITATION.cff:L20-L21`
- Evidence:

```bibtex
version = {0.1.0},
```

This is covered by F-001, but the documentation impact is separate: the README tells users the maintained citation metadata lives in `CITATION.cff`, while both show the old version. Once F-001 is fixed, add a doc smoke test for citation/version consistency.

Recommendation:
Update README citation and `CITATION.cff` together and test them.

Backlog details:
- Files likely to change: `README.md`, `CITATION.cff`.
- Tests to add/update: included in `test_version_metadata_consistency`.
- Risk: Low.
- Fix type: Mechanical.
- TEVC timing: Before submission.

### F-010 - [Comparative Best Practices] Pymoo interop exists but is not covered by tests in the current coverage run

- Severity: Minor
- Effort: S
- Location: `src/vamos_contrib/interop/pymoo.py:L26-L53`
- Evidence:

```python
class PymooProblemAdapter(ProblemProtocol):
```

VAMOS includes a local pymoo adapter, but coverage reported `src/vamos_contrib/interop/pymoo.py` at 0%. This audit did not inspect pymoo source/docs, so no concrete pymoo API comparison is made here. The local adapter still deserves smoke coverage because comparative claims and external baselines are part of the project positioning.

Recommendation:
Add an adapter test using a tiny fake pymoo-like object. If pymoo is installed in CI with the `research` extra, add one optional integration test against `pymoo.problems.get_problem("zdt1")`.

Backlog details:
- Files likely to change: `tests/integration/test_pymoo_adapter.py`.
- Tests to add/update: fake-object unit test plus optional `research` integration test.
- Risk: Low.
- Fix type: Mechanical.
- TEVC timing: Medium-term unless pymoo interop is highlighted in the submission.

## 5. Strengths

- Public API facades are intentionally small. `src/vamos/__init__.py:L15-L41` exports `optimize`, `Problem`, problem helpers, diagnostics, and lazy `problems`, while `src/vamos/api.py:L1-L10` states the public API policy.
- Algorithm extension is explicit. `src/vamos/engine/algorithm/registry.py:L108-L183` registers built-ins and loads plugins from the `vamos.algorithms` entry point group.
- Problem definition is accessible. `src/vamos/foundation/problem/base.py:L88-L143` validates batched objective shape, and `src/vamos/foundation/problem/builder.py:L124-L137` provides a function-based custom problem path.
- Reproducibility metadata is recorded in results. `src/vamos/experiment/unified.py:L381-L405` writes resolved problem, algorithm, engine, seed, dimensions, encoding, and default sources into result metadata.
- Core performance code is not naive throughout. `src/vamos/foundation/kernel/numpy_backend.py:L34-L122` implements dense and blocked nondominated sorting, and `src/vamos/foundation/kernel/numpy_backend.py:L196-L217` chunks large tournament sampling.
- Tests are unusually broad for a research framework. The executed suite passed `1151` tests with architecture, docs, engine, foundation, experiment, integration, performance, reference, and UX test groups.
- Architecture guardrails are real. `tests/architecture/test_layer_boundaries.py:L12-L20` defines allowed layer imports, and `tests/architecture/test_ruff_gate.py:L10-L18` executes Ruff against `src/vamos tests`.

## 6. TEVC Risk Assessment

A skeptical IEEE TEVC reviewer would likely challenge four areas:

1. Release credibility: the distribution is versioned as 1.0.0, while runtime and citation metadata report 0.1.0.
2. Static quality: a package claiming typed research-software maturity ships with failing mypy on `src` and a formatting budget.
3. Vectorization claim: the core kernels are vectorized, but the friendliest custom-problem path defaults to scalar row-wise adaptation.
4. Reproducible benchmarking: tests pass and docs mention a publication environment, but several benchmark/profiler/interop paths have weak coverage and the audit did not verify a one-command TEVC reproduction workflow.

Readiness rating: **Promising but not submission-ready**. The architecture and test base are strong enough to justify continued investment, but the critical issues should be fixed before submission. UI polish, optional external-framework breadth, and expanded many-objective performance benchmarking can remain future work if the paper scopes claims carefully.

## 7. Recommended Remediation Roadmap

### Immediate fixes before submission

- F-001: Align package/runtime/citation version metadata. Effort S.
- F-002: Make archive typing and optional MooCore hook mypy-clean. Effort S.
- F-003: Format `src/vamos tests` or reduce the format budget to a publication-acceptable threshold. Effort M.
- F-005: Add `src/vamos/__main__.py` or explicitly document only the console script. Effort S.
- F-006: Tighten docs and metadata around scalar vs. vectorized custom problems. Effort M.
- F-008: Freeze and document the TEVC reproduction environment and benchmark manifest fields. Effort M.

### Medium-term engineering improvements

- F-004: Add focused tests for benchmark CLI, profiler, pymoo adapter, and high-value Studio state. Effort L.
- F-007: Add backend-aware or threshold-aware HV contribution behavior and performance tests. Effort M.
- Add coverage thresholds for publication-facing modules rather than only global coverage. Effort M.
- Add a command-level smoke check for `vamos bench`, `vamos tune`, `vamos profile`, `vamos zoo`, and `python -m vamos` in the same CI job that builds the wheel. Effort M.

### Longer-term research-software improvements

- Build a single `vamos reproduce tevc` or documented equivalent that validates environment, runs a smoke slice, and records manifest metadata. Effort L.
- Add benchmark artifact schemas for result directories, run manifests, and statistical reports. Effort L.
- Expand performance benchmarks across population size, objective count, backend, and archive pruning policy. Effort L.
- Provide a clear comparison matrix against mature MOEA frameworks after inspecting their current source/docs in the same environment. Effort M.

## 8. Evidence Coverage

Commands run:

- Required environment and gate commands: `git status --short`, `find . -maxdepth 3 -type f | sort`, scoped file listing, `python --version`, `python -m pip show vamos`, `python -m pip show vamos-optimization`, `python -m pytest -q`, coverage, Ruff, mypy, pyright, `python -m vamos --help`, and `vamos --help`.
- Targeted repository inspection: `rg --files`, `rg -n`, `nl -ba`/`Get-Content` on API facades, algorithm registry/config, problem abstractions, kernels, archives, benchmark/profiler/UX/contrib modules, docs, CI, and architecture tests.
- Validation commands should be read from `docs/audit/commands_used.md`.

Files inspected:

- Public API and packaging: `pyproject.toml`, `README.md`, `CITATION.cff`, `src/vamos/__init__.py`, `src/vamos/api.py`, `src/vamos/algorithms.py`, `src/vamos/foundation/version.py`.
- Core framework: `src/vamos/experiment/unified.py`, `src/vamos/foundation/problem/base.py`, `src/vamos/foundation/problem/builder.py`, `src/vamos/engine/algorithm/registry.py`, algorithm config files, NSGA-II/MOEA-D helpers, kernel backends, quality indicators, archive components.
- Tooling/tests/docs: `.github/workflows/ci.yml`, `tests/architecture/*`, coverage output, docs getting-started/engineering audit pages, benchmark/profiler/UX/contrib modules.

Files not inspected or lightly inspected:

- Most individual benchmark problem implementations were only sampled through coverage and file listings.
- Paper scripts, generated `site/`, `results/`, and historical report artifacts were not audited deeply.
- Pymoo source/docs were not inspected; comparative notes are high-level unless they refer only to local VAMOS pymoo adapter code.

Unavailable tools:

- `python -m pyright` failed because `pyright` is not installed. This is not counted as a defect because `pyright` is not declared in the project tooling inspected here.

Limitations:

- The audit is static plus test/CLI execution; it does not reproduce full TEVC experiments.
- Coverage output was read from the local run; no coverage XML/HTML artifact was generated for this audit.
- `.coverage` was produced by the requested coverage command and left untracked.
