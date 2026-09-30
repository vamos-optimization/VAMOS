# Architecture Health

Purpose: prevent future refactors by enforcing layer boundaries and limiting module bloat.
These rules are guardrails for long-term maintainability in a research-oriented codebase.

## Canonical rules
Read before any architectural change.
- Layering and facades: layers depend only in the directions listed in `AGENTS.md`. Public entry points are the curated facades `vamos`, `vamos.algorithms`, `vamos.problems`, and `vamos.ux.api`, and `vamos/__init__.py` stays a small facade.
- Import-time purity: importing a module runs only imports, constant and type-alias assignments, function and class definitions, decorators, `if TYPE_CHECKING:` blocks, and `if __name__ == "__main__":` blocks. File IO, dynamic imports, environment reads, and initialization live inside functions or CLI entry points.
- Optional dependencies: core dependencies live in `[project].dependencies`, and heavy or optional libraries only in `[project.optional-dependencies]`. Optional imports are lazy or guarded with `try/except ImportError`. `experiment/external` is the integration boundary for benchmark backends, and Panel imports stay in `ux/panel` and the Studio UI.
- No shims or allowlists: do not add alternate wrappers for superseded APIs or allowlists for architecture gates. Fix violations at the source and update call sites.
- Health gates and retention: health and CI run the shared gates below with identical commands. Raw audits, Goal handoffs, and validation logs stay outside the product tree, as described in [Repository hygiene](repository_hygiene.md).
- Run artifacts, typing, and studies follow the [run artifact contract](run_artifact_contract.md), the [typing policy](typing.md), and the [StudyManifest v1 contract](study_manifest_contract.md).

## Health Gates (run locally)
- `python tools/health.py` (local fast-fail suite, including strict and full development typing)
- `python tools/health.py --continue-on-failure` (run the full gate list without fast-fail)
- `python tools/check_repository_hygiene.py` (tracked-file root, output, duplicate, size, archive and notebook policy)
- `python tools/check_repository_identity.py` (canonical metadata, classified mirror references, and publication guards)
- `python tools/check_agent_docs.py` (the same command and arguments used by CI)
- `python tools/typecheck.py --scope strict` (zero diagnostics in the protected scope)
- `python tools/typecheck.py --scope full` (exact structured no-regression baseline and clean changed modules)
- `python tools/typecheck.py --scope stable` (zero diagnostics across the stable public facades)
- `python tools/typecheck.py --scope release` (strict/stable zero plus the full ratchet and health)
- `python tools/typecheck.py --scope full-zero` (informational global-zero objective for VAMOS 1.0.0)
- `pytest -q tests/architecture/test_layer_boundaries.py`
- `pytest -q tests/test_monolith_guard.py`
- `pytest -q tests/test_public_api_guard.py`
- `pytest -q tests/test_import_time_smoke.py`
- `pytest -q tests/architecture/test_no_import_time_side_effects.py`
- `pytest -q tests/architecture/test_public_api_snapshot.py`
- `pytest -q tests/architecture/test_dependency_policy.py`
- `pytest -q tests/architecture/test_no_facade_imports.py`
- `pytest -q tests/architecture/test_experiment_import_cycles.py`
- `pytest -q tests/test_check_repository_hygiene.py`
- `pytest -q tests/test_no_deprecation_shims.py`
- `pytest -q tests/test_no_prints_in_library.py`
- `pytest -q tests/test_optional_deps_policy.py`
- `pytest -q tests/test_logging_policy.py`
- `pytest -q`

## Typing policy

The canonical environment, path inventory, diagnostic fingerprint schema, baseline update procedure, and debt-reduction order live in [Typing policy](typing.md). Health and CI invoke strict and full with identical command arguments. Full development success means the structured ratchet matched exactly; it does not mean full-source typing is clean. The VAMOS 1.0 release gate requires strict and stable zero, the exact full-source ratchet, and health. Full-source zero remains separately visible through `--scope full-zero`.

## Layering Policy (current reality)
- foundation may depend on foundation/resources only.
- engine may depend on engine/foundation/resources.
- experiment may depend on experiment/foundation/engine/ux/assist/resources.
- ux may depend on ux/foundation/engine/resources.
- assist may depend on assist/foundation/engine/experiment/resources.
- resources must not import other VAMOS layers.
- Facades: prefer `vamos.api`, `vamos.algorithms`, `vamos.problems`, `vamos.ux.api`.

## Optional Dependencies Policy
- foundation/** and engine/**: no top-level imports of optional/heavy deps.
- experiment/external/**: integration boundary for optional deps; imports must be lazy or guarded.
- ux/panel/** and ux/studio/**: Panel is optional and confined to UI modules.
- Dynamic import loopholes (`importlib.import_module`, `__import__`) are disallowed at top-level.
- Dependency list is enforced by `tests/architecture/test_dependency_policy.py`.

## Public API Snapshot
- Public facades are frozen via `tests/architecture/test_public_api_snapshot.py`.
- Update the snapshot intentionally with `python tools/update_public_api_snapshot.py`.

## Import-Time Purity
- No executable calls at module import time (top-level `ast.Call` outside `TYPE_CHECKING`/`__main__` blocks).
- Move initialization into functions or CLI entrypoints; use lazy factories for registries.
- Avoid top-level env reads or dynamic import calls; perform them inside runtime functions.

## Repository hygiene and audit evidence

- The canonical contract is [Repository hygiene](repository_hygiene.md).
- Raw audits, Goal handoffs and validation logs live outside the product tree or in CI artifact storage.
- Durable public conclusions are maintained documentation, not copied audit transcripts or root aliases.
- Generated performance reports use ignored `artifacts/performance/` locally and CI artifact upload remotely.
- The machine policy and exact exception manifest live under `release/`.

## No Monoliths Policy
- File size thresholds: core <= 450 LOC, CLI/UI <= 350 LOC.
- Function size <= 250 LOC; class size <= 400 LOC.
- Allowlists are forbidden. Split instead.
- Split pattern: create a package with focused modules and keep orchestration thin.

## Current Guard Limits
- `tests/test_monolith_guard.py` currently classifies only `src/vamos/experiment/cli/` and `src/vamos/ux/studio/` as CLI/UI. If additional UI surfaces such as `src/vamos/ux/panel/` grow, update the categorizer instead of assuming the guard already covers them.
- `tests/architecture/test_experiment_import_cycles.py` currently scans `vamos.experiment` only. It is an important guard, but it is not proof that the whole package is cycle-free.

## Logging/Printing Policy
- No `print()` in library code (allowed only in CLI/UI).
- No `logging.basicConfig()` in library modules.
- CLI logging config happens at invocation only via local handlers.

## Extension guides
- Problems: [add problem](add_problem.md).
- Operators: [add operator](add_operator.md).
- Algorithms: [add algorithm](add_algorithm.md).
- Backends: [add backend](add_backend.md).
- Metrics: [add metric](add_metric.md).
- Testing: [testing](testing.md).
