# VAMOS agent contract

## Scope and precedence

This file applies to the entire repository and is the sole global normative source for coding agents.

Read it before editing. A nested `AGENTS.md` may add rules only for its declared subtree. The closest scoped file wins only for an explicitly identified local rule; it does not replace this contract. Model-specific adapters and file-pattern instructions cannot override architecture, API, testing, Git, security, or artifact rules.

Resolve factual disagreements in this order: the current implementation and tests, accepted ADRs and canonical contracts, CI and packaging configuration, then developer documentation. Fix guidance that disagrees with a higher authority.

## Compatibility policy

Pre-1.0 development artifacts and undocumented APIs are unsupported. Starting with VAMOS 1.0.0, the stable APIs, CLI contracts, configuration fields, and artifact schemas listed in [Stability and versioning](docs/project/stability-and-versioning.md) are compatibility commitments for the 1.x series.

Do not add readers, migrations, aliases, or deprecation cycles for internal pre-1.0 prototypes. For a stable post-1.0 surface, follow the public deprecation and schema-transition policy instead. Maintain one current implementation, vocabulary, and contract per feature; Git history and the external pre-public tag archive preserve the earlier development record.

## Repository map and dependency direction

- `src/vamos/foundation/`: base abstractions, problems, kernels, evaluation, indicators, shared data, and errors.
- `src/vamos/engine/`: algorithms, canonical variation pipelines, operator implementations, archives, hooks, hyperheuristics, and tuning.
- `src/vamos/experiment/`: orchestration, CLI, studies, diagnostics, artifacts, and external integrations.
- `src/vamos/ux/`: public analysis, visualization, and Studio surfaces.
- `src/vamos/assist/`: assisted workflows built on foundation, engine, and experiment.
- `src/vamos/resources/`: packaged data; it imports no VAMOS layer.
- `tests/`: unit, integration, architecture, documentation, and smoke coverage.
- `docs/`, `examples/`, `notebooks/`: maintained learning and user material.
- `tools/`: repository checks and maintenance utilities.
- `experiments/`: research material, not runtime APIs. `paper/` is an optional local manuscript workspace; it is ignored and must not be tracked or published.
- `experiments/scripts/canonical_runs.py`: the shared research collector for canonical `load_run`/`load_result` access.
- `website/`: the separately configured multilingual public site; it uses the official MkDocs i18n plugin.

Permitted dependencies are: foundation -> foundation/resources; engine -> engine/foundation/resources; ux -> ux/foundation/engine/resources; experiment -> experiment/foundation/engine/ux/assist/resources; assist -> assist/foundation/engine/experiment/resources. Do not create reverse imports or import-time initialization to bypass these boundaries.

Read [Architecture Health](docs/dev/architecture_health.md) and the [ADR index](docs/dev/adr/index.md) before adding a module, dependency, or public API.

## Public and internal APIs

User examples and public docs use the curated facades: `vamos`, `vamos.algorithms`, `vamos.problems`, and `vamos.ux.api`. Deep imports are for implementation and contributor tests only. Public exports are guarded by `tests/architecture/test_public_api_snapshot.py`.

The canonical run lifecycle is:

```python
import vamos

result = vamos.optimize(...)
stored = vamos.save_result(result, path)
run = vamos.load_run(path)
loaded = vamos.load_result(path)
verification = vamos.verify_run(path)
replay = vamos.reproduce(path)
```

`load_result` returns the optimization result from a stored run. Loading, verification, and reproduction are different operations: load reads data, verify checks the canonical artifact and replayability, and reproduce executes an exact replay. Replay is available only for supported built-in components in a matching material environment and is reconstructed from the persisted resolved specification and seed.

VAMOS supports one run schema: `vamos.run-manifest` version `1.0.0`. A successful run contains `manifest.json`, `result.npz`, and `environment.json`. Writers and readers live under `src/vamos/experiment/artifacts/`; `src/vamos/run_artifacts.py` and the top-level facade expose the public surface.

The durable study schema is `vamos.study-manifest` version `1.0.0`, governed by `docs/dev/study_manifest_contract.md`, ADR 0008, the complete SA-001 through SA-074 acceptance inventory, and the separate PL-001 through PL-021 planning inventory. Atomic create/data-only load, bounded sequential `Study.run()`, persisted failure policy, single-process graceful cancellation, explicit reconciliation, resume, bounded retry, immutable data-only `Study.inspect()`/`Study.summarize()` projections, the complete single-owner study CLI, and canonical package/research callers are implemented. Local multi-process ownership and coordination remain the next contract Goal. RunManifest remains the sole owner of resolved per-run truth and arrays.

Shared algorithm variation pipelines live only in `src/vamos/engine/variation/`; concrete operator implementations and their registry live under `src/vamos/engine/operators/`. The external archive model is `ExternalArchiveConfig` in `src/vamos/engine/archive/config.py`, and experiment-spec parsing accepts the single `archive.external` block through `build_archive_cfg`.

Research collectors discover `manifest.json` and call the public `load_run`/`load_result` APIs through `experiments/scripts/canonical_runs.py`; they do not infer run data from filenames. The public website uses `website/mkdocs.yml` with `i18n.docs_structure: folder`, Material reconfiguration disabled, and search reconfiguration enabled.

## Environment

VAMOS supports Python 3.10 or newer. Select or create an environment explicitly; a repository-local virtual environment is optional, not assumed.

```bash
python -m venv .venv
python -m pip install -U pip
python -m pip install -e ".[dev,docs]"
```

Canonical typing uses Python 3.12, compiled mypy 2.3.1, typing-extensions 4.16.0, no stub distributions, and the constrained extras documented in [Typing policy](docs/dev/typing.md). Install that environment exactly before running typing or health:

```bash
python -m pip install -c constraints/ci.txt -e ".[dev]"
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux/macOS. An already selected environment with the required extras is equally valid.

Before a subprocess-based check, anchor imports to this checkout and verify them:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -c "import pathlib, vamos; print(pathlib.Path(vamos.__file__).resolve())"
```

```bash
export PYTHONPATH="$(pwd)/src"
python -c 'import pathlib, vamos; print(pathlib.Path(vamos.__file__).resolve())'
```

The printed path must be under the intended worktree. A final wheel smoke must use a clean, non-editable environment.

## Validation tiers

Use the narrowest tier that proves the change while iterating, then run every higher applicable tier before completion.

- Targeted: `python -m pytest -q <nearest-test-files>`, `python tools/typecheck.py --scope strict` for typed production changes, and `python tools/check_agent_docs.py` for agent/docs changes.
- Quick: `python -m pytest -q tests/test_check_agent_docs.py tests/architecture/test_docs_and_workflows.py tests/docs`.
- Full: `python tools/health.py`, `python -m pytest -q`, and `mkdocs build --strict`.
- Release: install the pinned build/release tools, then run the canonical `python tools/release_check.py --version 1.0.0`; it includes the full tier, stable/release typing, informational full-zero typing, frozen distribution inspection, installed-wheel smoke, dependency audits, SBOM, checksums, and provenance described in [Release verification](docs/release_smoke.md).

`tools/health.py` is the canonical local fast-fail architecture/tooling suite. CI has a distinct matrix and coverage scope. Both run exactly `python tools/check_agent_docs.py` for agent-documentation integrity; do not describe their complete suites as identical.

`tools/typecheck.py` is the sole typing entry point. Health and the dedicated CI typing job each run `--scope strict` and `--scope full` with identical arguments. Strict and stable scopes permit zero diagnostics. Full development typing enforces the structured baseline in `typing/mypy-baseline.json`, including exact fingerprints and multiplicities; every changed production file must be clean. Release combines strict zero, stable-facade zero, the exact full-source ratchet, and health. `--scope full-zero` remains the explicit global-zero objective and is informational for VAMOS 1.0.0. Ratchet success is not a claim that full-source mypy is clean.

`tools/check_pre_release_remnants.py` owns the repository semantic no-remnant scan and the shared discarded-token definitions. `tools/check_agent_docs.py` reuses those definitions for active guidance, then adds instruction-specific scope, adapter, link, declaration, duplication, and policy checks; it does not rerun the repository scan. Health and CI invoke each checker once.

For a bounded code change, also run Ruff on changed Python, `ruff format --check` on changed Python, the canonical strict/full typechecks, `python -m compileall` on changed Python/tests, and `git diff --check`. Report every skipped or unavailable gate honestly.

## Change discipline

1. Inspect status, worktrees, governing instructions, implementation, and nearest tests before editing.
2. Make a bounded change; preserve user work and ignore unrelated dirty files.
3. Use an isolated worktree for a long Goal or when the primary checkout must remain untouched.
4. Do not reset, clean, stash, rewrite history, or overwrite generated/user data without explicit authority.
5. Do not change public behavior, dependencies, schemas, or defaults as incidental cleanup.
6. Add or update the nearest tests and canonical guide with the implementation.
7. Do not push unless explicitly requested. Commit only after the requested validation passes.

## Extension routes

- Problems: [Adding a problem](docs/dev/add_problem.md)
- Operators: [Adding an operator](docs/dev/add_operator.md)
- Algorithms: [Adding an algorithm](docs/dev/add_algorithm.md)
- Kernel backends: [Adding a backend](docs/dev/add_backend.md)
- Quality indicators/metrics: [Adding a metric](docs/dev/add_metric.md)
- CLI: [Changing the CLI](docs/dev/cli.md)
- Run artifacts and exact replay: [Run artifacts and replay](docs/dev/run_artifacts_and_replay.md)
- Studies: [Changing studies](docs/dev/studies.md), [StudyManifest v1 contract](docs/dev/study_manifest_contract.md), [persisted-state acceptance](docs/dev/study_manifest_acceptance_tests.md), and [planning acceptance](docs/dev/study_plan_acceptance_tests.md)
- Testing: [Testing and validation](docs/dev/testing.md)

Follow at most one scoped `AGENTS.md` plus the linked canonical guide. Search the implementation when a task falls outside these routes; do not infer a new extension point from an old example.

## Scientific invariants

- Termination honors exact evaluation budgets; do not hide overshoot in orchestration.
- Stochastic production paths receive and own an explicit `numpy.random.Generator`; do not use global random state.
- Problems, operators, and algorithms must agree on real, integer, binary, permutation, or mixed encoding.
- Reference-direction algorithms preserve their population/reference-direction cardinality contract.
- An unavailable backend fails clearly; never substitute another backend silently.
- Exact replay uses the stored resolved configuration, effective backend, and resolved seed, and refuses unsupported reconstruction.

## Documentation and generated material

Documentation is part of the tested interface. Prefer one canonical guide per workflow and link to it instead of copying a path list or command sequence.

- Verify public examples against the current facade and parser.
- Keep Markdown links repository-relative and valid under MkDocs strict mode.
- Put executable learning paths in the docs smoke manifest when they establish supported behavior.
- Keep notebook output bounded and remove transient errors, widget state, and secrets.
- Do not hand-edit generated manifests, API snapshots, distributions, or report output except through their owning update tool.
- Do not commit run directories, large arrays, build output, caches, or local environments.
- Update `mkdocs.yml` when adding a maintained page that readers must discover.
- Keep contributor docs aligned with the validation tiers here; link back instead of redefining them.
- Treat comments, docstrings, fixtures, help text, and prompt files as active guidance during a vocabulary replacement.
- Use current positive terminology; avoid narrating discarded pre-release paths in active how-to material.
- Preserve citations and research attribution when changing algorithm/problem documentation.
- Run the relevant docs smoke and strict build after changing a supported command or example.

## Security and trust boundaries

- Canonical loading and verification are inert, bounded, data-only operations and use no pickle.
- Replay executes only explicitly supported built-in components; manifest text never selects arbitrary importable code.
- Custom/plugin code and artifact input are untrusted by default.
- Studio execution of raw Python is not a sandbox. Do not claim isolation that the implementation does not provide.
- Never place credentials, access tokens, personal paths, or secrets in fixtures, manifests, logs, examples, or commits.

## Completion report

Report the base, branch/worktree, files changed, commands with exit status, limitations, final commit (or explicit no-commit state), worktree cleanliness, push status, and the exact next Goal when one is defined.

```agent-docs
path: docs/dev/architecture_health.md
path: docs/project/stability-and-versioning.md
path: docs/dev/adr/index.md
path: CODING_GUIDELINES.md
path: docs/dev/add_problem.md
path: docs/dev/add_operator.md
path: docs/dev/add_algorithm.md
path: docs/dev/add_backend.md
path: docs/dev/add_metric.md
path: docs/dev/cli.md
path: docs/dev/run_artifacts_and_replay.md
path: docs/dev/studies.md
path: docs/dev/study_manifest_contract.md
path: docs/dev/study_manifest_acceptance_tests.md
path: docs/dev/study_plan_acceptance_tests.md
path: docs/dev/study_manifest_examples/README.md
path: docs/dev/adr/0008-durable-study-manifest-contract.md
path: docs/dev/testing.md
path: docs/dev/typing.md
path: docs/release_smoke.md
path: release/requirements-build.txt
path: release/requirements-tools.txt
path: docs/dev/adr/0007-canonical-typing-gates.md
path: tools/typecheck.py
path: tools/release_check.py
path: tools/release_smoke.py
path: typing/mypy-baseline.json
path: src/vamos/experiment/artifacts
path: src/vamos/run_artifacts.py
path: src/vamos/engine/variation
path: src/vamos/engine/operators/impl/registry.py
path: src/vamos/engine/archive/config.py
path: src/vamos/engine/hooks/config_parse.py
path: experiments/scripts/canonical_runs.py
path: website/mkdocs.yml
path: tests/architecture/test_public_api_snapshot.py
symbol: vamos:optimize
symbol: vamos:save_result
symbol: vamos:load_run
symbol: vamos:load_result
symbol: vamos:verify_run
symbol: vamos:reproduce
symbol: vamos:plan_study
symbol: vamos:VerificationReport
symbol: vamos:ReplayReport
symbol: vamos.engine.variation:VariationPipeline
symbol: vamos.engine.archive.config:ExternalArchiveConfig
symbol: vamos.engine.hooks.config_parse:build_archive_cfg
cli: vamos --help
cli: vamos results inspect --help
cli: vamos results verify --help
cli: vamos reproduce --help
cli: vamos create-problem --help
cli: vamos study plan --help
cli: vamos study create --help
cli: vamos study run --help
cli: vamos study inspect --help
cli: vamos study resume --help
cli: vamos study retry --help
cli: vamos study summarize --help
command: python tools/check_agent_docs.py
command: python -m pytest -q tests/test_check_agent_docs.py tests/architecture/test_docs_and_workflows.py tests/docs
command: python tools/health.py
command: python tools/typecheck.py --scope strict
command: python tools/typecheck.py --scope stable
command: python tools/typecheck.py --scope full
command: python tools/typecheck.py --scope release
command: python tools/typecheck.py --scope full-zero
command: python -m pytest -q
command: mkdocs build --strict
command: python -m build
command: python tools/release_check.py --version 1.0.0
command: python tools/release_smoke.py --version 1.0.0 --mode full
```
