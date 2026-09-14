# Experiment artifact contract

Experiment and paper tooling has two canonical inputs with distinct ownership:

- a StudyManifest directory owns a campaign's plan, attempts, state, recovery,
  and task-to-run references;
- a RunManifest directory owns one execution's resolved configuration,
  numerical arrays, environment, and outcome.

Both use only schema version `1.0.0`. A CSV, generated config, directory name,
or launcher index is never campaign state.

Current campaign collectors call `vamos.load_study(...).summarize()` first.
When numerical analysis is explicitly required, they follow the summary's
verified run reference and call `vamos.load_result`. Ordinary single-run and
publication-archive tools may discover `manifest.json` through
`experiments/scripts/canonical_runs.py`; that helper is not a study reader.
No collector infers algorithm, problem, backend, seed, or configuration from
directory names.

## Artifact classes

- `ACTIVE_CURRENT_WORKFLOW`: `examples/tuning/ablation_runner.py`, the maintained
  ablation notebooks, and campaign collectors use `StudySpec`, `plan_study`,
  `create_study`, `Study.run/resume/retry/inspect/summarize`, and StudySummary.
- `DERIVED_REGENERABLE_OUTPUT`: tidy CSV tables, statistics, plots, LaTeX, and
  ordinary run outputs under ignored `artifacts/` or `paper/generated/` paths.
  The tiny files under `experiments/sample_outputs/` are explicit reporting
  fixtures, not general output destinations.
- `SCIENTIFIC_SOURCE_INPUT`: campaign YAML, problem catalogs, frozen reference
  points, and operator/config inventories under `experiments/configs/` and
  `experiments/catalog/`; these are preserved inputs, not a supported study
  serialization.
- `REFERENCE_BENCHMARK_DATA`: only the minimal paper CSVs declared in
  `experiments/REFERENCE_RESULTS.md`. They retain a source commit, command,
  schema, expected use and size budget.
- `PUBLICATION_SOURCE`: TeX, rebuild scripts, author correspondence and source
  figures are maintained separately from the public repository. An optional
  local `paper/` workspace and all publication outputs are ignored.
- `OBSOLETE_PRE_RELEASE_WORKFLOW`: the removed custom campaign launchers that
  generated per-task CLI configs, inferred completion by scanning directories,
  and treated `runs_index.jsonl` as resume state.
- `SEMANTICALLY_UNRELATED`: single-run tutorials, tuning-database examples,
  registry/catalog builders, and cross-framework source experiments that do
  not claim to be VAMOS durable studies.

## Derived tidy tables

`experiments/scripts/collect_campaign_runs.py` is the current durable-campaign
collector. It produces one row per StudySummary task and retains study ID, plan
ID, task ID, attempt ID, run ID, relative RunManifest path, and manifest hash.
`experiments/scripts/canonical_runs.py` remains the shared ordinary-run adapter.
Objective summaries are calculated only after following canonical run evidence
and loading arrays through `load_result`.

Derived tables are written below `artifacts/` or `paper/generated/`. Their
filenames and columns are analysis interfaces only and must never be treated as
a loadable VAMOS run format. New sample fixtures require an explicit fixture
contract and test.

## Validation

Use a tiny canonical study to test StudySummary derivation and table generation,
plus a tiny canonical run to test ordinary-run discovery and array loading.
Publication-scale campaigns are outside routine validation.
