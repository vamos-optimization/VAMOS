# Architecture Health

Purpose: prevent future refactors by enforcing layer boundaries and limiting module bloat.
These rules are guardrails for long-term maintainability in a research-oriented codebase.

## Canonical rules
Read before any architectural change.
- Layering and facades: layers depend only in the directions listed in `AGENTS.md`. Public entry points are the curated facades `vamos`, `vamos.algorithms`, `vamos.problems`, and `vamos.ux.api`, and `vamos/__init__.py` stays a small facade.
- Import-time purity: importing a module runs only imports, constant and type-alias assignments, function and class definitions, decorators, `if TYPE_CHECKING:` blocks, and `if __name__ == "__main__":` blocks. File IO, dynamic imports, environment reads, and initialization live inside functions or CLI entry points.
- Optional dependencies: core dependencies live in `[project].dependencies`, and heavy or optional libraries only in `[project.optional-dependencies]`. Optional imports are lazy or guarded with `try/except ImportError`. `experiment/external` is the integration boundary for benchmark backends, and Panel imports stay in `ux/panel` and the Studio UI.
- No shims, and recorded debt only shrinks: do not add alternate wrappers for superseded APIs. Fix violations at the source and update call sites. A gate that tolerates existing debt records it in a ratchet, not an allowlist of exempt paths: `tests/architecture/monolith_guard_budget.json`, `tests/architecture/ruff_format_budget.json`, `typing/mypy-baseline.json`, and `release/repository-hygiene-exceptions.json`. Lower a budget or drop a baseline entry when the debt goes; never raise a budget or add a baseline entry to pass a gate. A hygiene exception names its owner, reason, and review condition.
- Health gates and retention: `tools/health.py` runs the gates below, and CI runs the same checkers and test files in its agent-documentation step, architecture step, and typing job. Raw audits, Goal handoffs, and validation logs stay outside the product tree, as described in [Repository hygiene](repository_hygiene.md).
- Run artifacts, typing, and studies follow the [run artifact contract](run_artifact_contract.md), the [typing policy](typing.md), and the [StudyManifest v1 contract](study_manifest_contract.md).

## Health gates
`python tools/health.py` runs these gates in this order and stops at the first failure; `python tools/health.py --continue-on-failure` runs them all. `tests/architecture/test_docs_and_workflows.py` keeps this list equal to `tools/health.py`.

1. `python tools/check_repository_hygiene.py` (tracked-file root, output, duplicate, size, archive and notebook policy)
2. `python tools/check_repository_identity.py` (canonical metadata, classified mirror references, and publication guards)
3. `python -m pytest -q tests/test_check_repository_identity.py`
4. `python -m pytest -q tests/test_check_repository_hygiene.py`
5. `python tools/check_pre_release_remnants.py`
6. `python -m pytest -q tests/test_check_pre_release_remnants.py`
7. `python tools/check_agent_docs.py` (the same command and arguments used by CI)
8. `python tools/check_learning_content.py` (removed APIs and stale commands in the README, docs, examples and notebooks)
9. `python tools/check_notebook_links.py`
10. `python tools/check_notebook_quality.py`
11. `python -m pytest -q tests/architecture/test_layer_boundaries.py`
12. `python -m pytest -q tests/test_monolith_guard.py`
13. `python -m pytest -q tests/test_public_api_guard.py`
14. `python -m pytest -q tests/test_import_time_smoke.py`
15. `python -m pytest -q tests/architecture/test_no_import_time_side_effects.py`
16. `python -m pytest -q tests/architecture/test_public_api_snapshot.py`
17. `python -m pytest -q tests/architecture/test_dependency_policy.py`
18. `python -m pytest -q tests/architecture/test_no_facade_imports.py`
19. `python -m pytest -q tests/architecture/test_experiment_import_cycles.py`
20. `python -m pytest -q tests/test_optional_deps_policy.py`
21. `python -m pytest -q tests/test_logging_policy.py`
22. `python -m pytest -q tests/test_no_prints_in_library.py`
23. `python -m pytest -q tests/test_no_deprecation_shims.py`
24. `python -m pytest -q tests/architecture/test_ruff_gate.py`
25. `python -m pytest -q tests/architecture/test_ruff_format_gate.py`
26. `python tools/typecheck.py --scope strict` (zero diagnostics in the protected scope)
27. `python tools/typecheck.py --scope full` (exact structured no-regression baseline and clean changed modules)
28. `python -m pytest -q tests/architecture/test_build_smoke.py` (builds the sdist and wheel from a copy of the tree, outside the checkout)
29. `python -m pytest -q tests/architecture/test_py_typed_present.py`

## Other gates
Health does not run these:
- `python tools/typecheck.py --scope stable` (zero diagnostics across the stable public facades)
- `python tools/typecheck.py --scope release` (strict/stable zero plus the full ratchet and health)
- `python tools/typecheck.py --scope full-zero` (informational: every diagnostic in the production source, without the baseline)
- `python -m pytest -q` (the full suite)

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
- Split new violations instead of adding them to `tests/architecture/monolith_guard_budget.json`. The budget lists the existing ones, and none may grow.
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
