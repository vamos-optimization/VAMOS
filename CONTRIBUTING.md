# Contributing to VAMOS

Thank you for considering a contribution! This project is organized to make adding new components straightforward.

Open issues and pull requests in the [official repository](https://github.com/vamos-optimization/VAMOS).
See [repository governance](docs/project/repository-governance.md) for source and mirror ownership.

Repository-wide contributor and agent rules, including validation tiers and Git discipline, live in `AGENTS.md`. Nested `AGENTS.md` files add only local subtree rules.

## Adding a new problem
- Put the implementation under `src/vamos/foundation/problem/` (one file per problem family).
- Register it by adding a spec to the correct family module under `src/vamos/foundation/problem/registry/families/`.
- Add a short docstring to the problem class describing the landscape and encoding.
- Add a small smoke test in `tests/` to validate instantiation and `evaluate` shape.
- See `docs/dev/add_problem.md` for a step-by-step template.

## Adding a new algorithm
- Implement the vectorized core under `src/vamos/engine/algorithm/`.
- Create a config dataclass/builder under `src/vamos/engine/algorithm/config/` to keep construction declarative and serializable.
- Register the algorithm in `src/vamos/engine/algorithm/registry.py` so orchestration layers and the CLI can resolve it by name.
- Add a minimal smoke test (tiny population/evaluation budget) to catch wiring issues.
- See `docs/dev/add_algorithm.md` for a template and checklist.

## Adding an operator or metric
- Follow `docs/dev/add_operator.md` for variation/repair components.
- Follow `docs/dev/add_metric.md` for quality indicators and metric ownership.

## Adding a new kernel backend
- Implement the `KernelBackend` interface in `src/vamos/foundation/kernel/` (see `kernel/backend.py` for required methods).
- Register it in `src/vamos/foundation/kernel/registry.py` with a unique engine name.
- Add a backend-marked smoke test (`@pytest.mark.<engine>`) that runs a small NSGA-II job; use `pytest.importorskip` to skip when the dependency is missing.
- See `docs/dev/add_backend.md` for required methods and a smoke-test example.

## Architecture health (mandatory)
- Read the canonical rules in `docs/dev/architecture_health.md` before any architectural change.
- Run the local fast-fail health command: `python tools/health.py`.
- CI has a different platform/version and coverage scope. Both health and CI run `python tools/check_agent_docs.py` with identical arguments.
- Typing has one entry point: `python tools/typecheck.py --scope strict|stable|full|release|full-zero`. Health and CI run strict and full. Release requires strict/stable zero, the exact full-source ratchet, and health; `full-zero` remains the visible global-zero roadmap gate.
- If you change public APIs, update the snapshot: `python tools/update_public_api_snapshot.py`.

## Continuous Integration
- CI runs its configured matrix of lint, targeted mypy, architecture, test, docs, notebook, and build jobs.
- Before opening a PR, run the applicable full tier from `AGENTS.md`:
  - `python tools/health.py`
  - `python -m pytest -q`
  - `mkdocs build --strict`

## Coding style and typing
- The project uses a `src/` layout and prefers type hints on public-facing functions/classes.
- Performance-critical loops (kernels, variation) should remain lightweight; avoid refactors that change behavior without explicit benchmarks.
- Every changed production module must be clean under the canonical typecheck. New diagnostics and increased baseline multiplicity are forbidden.
- Read `docs/dev/typing.md` for the pinned environment, strict/full/release semantics, and baseline-reduction procedure.
- Read `docs/project/stability-and-versioning.md` before changing a stable API, CLI argument, configuration field, or public artifact schema.

## Tuning package layout
- All tuning utilities (parameter spaces, samplers, racing loop, random search) live under `src/vamos/engine/tuning/racing/`.
- Import from `vamos.engine.tuning`. Do not document or depend on a `vamos.tuning` facade; it is not part of the public API.

## Self-check
- After changes, run `vamos check` for a quick sanity check.
- CI-friendly tests live under `tests/`; keep populations/evaluation budgets small for speed.

## Before opening a pull request (human or AI-assisted)
1. Run the health gates: `python tools/health.py`.
2. Run the full suite: `pytest -q`.
3. If you touched tuning, durable studies, or benchmarking:
   - Run the smallest relevant `vamos bench` suite.
4. If you added docs or notebooks:
   - Build docs with `mkdocs build --strict` and run the affected notebook smoke.
5. For AI-assisted work, start at root `AGENTS.md`; use at most one scoped local instruction and one routed developer guide.
