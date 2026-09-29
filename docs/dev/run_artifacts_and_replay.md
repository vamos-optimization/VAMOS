# Modifying run artifacts and exact replay

The normative schema is [the canonical run-artifact contract](run_artifact_contract.md),. VAMOS supports `vamos.run-manifest` version `1.0.0` only.

## Ownership

- `src/vamos/experiment/artifacts/models.py`, `manifest.py`, and `specs.py`: typed/validated schema model.
- `storage.py` and `persistence.py`: atomic, non-destructive publication and public save/load behavior.
- `reader.py`, `bundle.py`, `bundle_safety.py`, and `paths.py`: bounded inert reads, integrity, NumPy safety, and path confinement.
- `verification.py`, `compatibility.py`, and `component_support.py`: inert verification and environment/component classification.
- `reconstruction.py`, `replay.py`, `comparison.py`, and `lineage.py`: supported built-in reconstruction, exact execution/comparison, and provenance.
- `src/vamos/run_artifacts.py` and top-level `vamos`: public facade.
- `src/vamos/experiment/cli/run_artifact_cli.py`: CLI translation only.

Loading is not verification, and verification is not replay. `load_run` reads the stored run, `load_result` returns its numerical result, `verify_run` performs no optimization, and `reproduce` is the only replay operation. Exact replay reconstructs solely from the persisted resolved specification/seed, requires a materially exact environment, executes supported built-ins, compares deterministic arrays bitwise, and writes a new canonical run with lineage.

## Change procedure

1. Read the contract, acceptance matrix, models, and affected public API tests.
2. Identify the single owner for each field or byte. Never add a parallel writer, copied study payload, or consumer-side filename inference.
3. Preserve `allow_pickle=False`, bounded pre-allocation inspection, confined relative paths, hashes, semantic manifest self-hash, collision refusal, and atomic publication.
4. Update writers and every reader/consumer together. Analysis, Studio, studies, Python, and CLI must converge on the same API.
5. If replay reconstruction changes, cover each affected algorithm, encoding, backend, seed, failure boundary, source immutability, lineage, and exact comparison.
6. Update the contract and acceptance matrix in the same change. Keep the current schema statement positive and singular.

Custom/plugin reconstruction, cross-backend execution, environment installation, and best-effort execution are outside exact replay. Do not weaken refusal into silent execution.

## Required validation

```bash
python -m pytest -q tests/experiment/run_artifacts
python -m pytest -q tests/experiment/test_cli_run_artifacts.py tests/experiment/test_cli_results.py
python -m pytest -q tests/ux/test_results_loader.py tests/ux/test_studio_data_dm.py
```

Run architecture/public API tests and the full tier from `/AGENTS.md` for any contract change.

```agent-docs
path: docs/dev/run_artifact_contract.md
path: docs/dev/run_artifact_acceptance_tests.md
path: src/vamos/experiment/artifacts
path: src/vamos/experiment/artifacts/persistence.py
path: src/vamos/experiment/artifacts/verification.py
path: src/vamos/experiment/artifacts/reconstruction.py
path: src/vamos/experiment/artifacts/replay.py
path: src/vamos/run_artifacts.py
path: src/vamos/experiment/cli/run_artifact_cli.py
path: tests/experiment/run_artifacts
path: tests/experiment/test_cli_run_artifacts.py
path: tests/experiment/test_cli_results.py
path: tests/ux/test_results_loader.py
path: tests/ux/test_studio_data_dm.py
symbol: vamos:save_result
symbol: vamos:load_run
symbol: vamos:load_result
symbol: vamos:verify_run
symbol: vamos:reproduce
symbol: vamos:VerificationReport
symbol: vamos:ReplayReport
cli: vamos results inspect --help
cli: vamos results verify --help
cli: vamos reproduce --help
command: python -m pytest -q tests/experiment/run_artifacts
command: python -m pytest -q tests/experiment/test_cli_run_artifacts.py tests/experiment/test_cli_results.py
command: python -m pytest -q tests/ux/test_results_loader.py tests/ux/test_studio_data_dm.py
```
