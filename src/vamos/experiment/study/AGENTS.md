# Scope

Applies only to `src/vamos/experiment/study/**`.

Inherits all repository-wide rules from `/AGENTS.md`. This file contains local deltas only.

## Responsibility and invariants

- `models.py`, `planning.py`, `preflight.py`, `creation.py`, and `loading.py` own the canonical durable model, immutable plan, read-only planning report, atomic creation, and data-only loaded view. `projection.py` is the only current-state interpretation service for immutable `StudyReport` and `StudySummary` values in `report_models.py`.
- `execution.py` owns the no-argument, newly-created-only sequential `Study.run()` path. It executes ascending `task_id`; `plan_index` is presentation metadata. `failure_policy.py` owns persisted fail-fast/continue study outcomes, while `cancellation.py` owns durable single-process cancellation.
- `journal.py` validates and replays immutable events; `checkpoint_projection.py` validates lagging checkpoints and builds the effective immutable view without writes.
- `commits.py`, `run_publication.py`, and `writing.py` own event/checkpoint commits, verified RunManifest linkage, task-failure publication, and atomic file primitives. `reconciliation.py` owns evidence-driven recovery writes; `recovery.py` owns resume/retry selection and operation orchestration.
- A study summary may reference or aggregate verified RunManifest metadata but must not load or create a second copy of canonical per-run arrays or specifications. `Study.summarize()` is in-memory only; later output renderers must consume this same projection.
- RunManifest remains the sole authority for resolved per-run truth, environment, provenance, arrays, replayability, and outcome. Study attempts retain only bounded root-relative manifest references.
- Resume and explicit bounded retry are single-process operations that reconcile before claims and preserve every terminal attempt. Failure policy is fixed in the published spec, and running cancellation is cooperative only within the process that owns the sequential runner.
- Do not implement locks, leases, parallel workers, cross-process cancellation, or format migration before their ordered contract Goals. `Study.inspect()` and `Study.summarize()` are data-only and write-free; the CLI exposes the complete current single-owner lifecycle and writes summaries only to an explicit derived destination.
- Preserve task order, explicit seeds, indicator failure reporting, and optional-dependency behavior.

## Change route

Follow [Changing studies](/docs/dev/studies.md) and the linked StudyManifest contract and acceptance inventory. Check run-artifact ownership before adding any persistence field or output. Implement roadmap slices in order and do not add a second layout, fallback reader, second execution service, or automatic retry.

## Targeted validation

Run `python -m pytest -q tests/experiment/study_manifest tests/experiment/run_artifacts/test_verification_replay.py tests/experiment/test_ablation_study_api.py tests/experiment/test_cli_ablation.py` plus the nearest study/indicator tests.

```agent-docs
path: src/vamos/experiment/ablation.py
path: src/vamos/experiment/benchmark/runner.py
path: src/vamos/experiment/study_analysis.py
path: src/vamos/ux/studio/data.py
path: src/vamos/ux/analysis/tuning_viz.py
path: src/vamos/experiment/study/models.py
path: src/vamos/experiment/study/planning.py
path: src/vamos/experiment/study/preflight.py
path: src/vamos/experiment/study/creation.py
path: src/vamos/experiment/study/loading.py
path: src/vamos/experiment/study/projection.py
path: src/vamos/experiment/study/report_models.py
path: src/vamos/experiment/study/execution.py
path: src/vamos/experiment/study/journal.py
path: src/vamos/experiment/study/journal_loading.py
path: src/vamos/experiment/study/checkpoint_projection.py
path: src/vamos/experiment/study/run_publication.py
path: src/vamos/experiment/study/reconciliation.py
path: src/vamos/experiment/study/recovery.py
path: src/vamos/study_artifacts.py
path: tests/experiment/study_manifest
path: tests/experiment/test_ablation_study_api.py
path: tests/experiment/test_cli_ablation.py
path: docs/dev/studies.md
path: docs/dev/study_manifest_contract.md
path: docs/dev/study_manifest_acceptance_tests.md
path: docs/dev/study_plan_acceptance_tests.md
path: docs/dev/study_manifest_examples/README.md
symbol: vamos.experiment.study.models:Study.run
symbol: vamos.experiment.study.models:Study.resume
symbol: vamos.experiment.study.models:Study.retry
symbol: vamos.experiment.study.models:Study.inspect
symbol: vamos.experiment.study.models:Study.summarize
symbol: vamos.study_artifacts:create_study
symbol: vamos.study_artifacts:load_study
symbol: vamos.study_artifacts:plan_study
cli: vamos study plan --help
cli: vamos study create --help
cli: vamos study run --help
cli: vamos study inspect --help
cli: vamos study resume --help
cli: vamos study retry --help
cli: vamos study summarize --help
command: python -m pytest -q tests/experiment/study_manifest tests/experiment/run_artifacts/test_verification_replay.py tests/experiment/test_ablation_study_api.py tests/experiment/test_cli_ablation.py
```
