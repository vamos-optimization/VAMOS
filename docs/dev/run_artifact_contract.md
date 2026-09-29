# VAMOS canonical v1 run-artifact contract

Status: implemented pre-release contract
Document identity: `vamos.run-manifest`
Schema version: `1.0.0`

## 1. Scope and authority

This document defines the only supported VAMOS run-artifact format. It governs
the top-level Python persistence API, CLI run output, analysis discovery, and
Studio loading. A run artifact represents one execution attempt. It is not a
study database or a general export format. Replay is an explicit service that
consumes one verified run and publishes another through the same writer.

VAMOS 1.0.0 establishes the first public run-artifact baseline. The package
supports one schema and one reader/writer path. Internal development inputs
outside schema `1.0.0` must be regenerated with the current version.

> Internal development run directories created before the canonical schema
> 1.0.0 implementation are unsupported and should be regenerated.

## 2. Canonical directory

A successful run directory contains exactly these required files:

```text
<run>/
├── manifest.json
├── result.npz
└── environment.json
```

`manifest.json` is the authoritative envelope and artifact index.
`result.npz` is the authoritative numerical bundle. `environment.json` is the
authoritative bounded environment description. Additional files are not
produced by the v1 writer. Optional future roles such as `metrics` or `events`
require a schema-defined, nonduplicated responsibility before a writer may emit
them.

The run directory is relocatable. Manifest artifact paths are normalized,
relative POSIX paths confined to the run directory. Absolute paths, traversal,
empty segments, backslashes, drive prefixes, URI forms, NUL bytes, and symlink
escapes are rejected.

## 3. Writer ownership

There is one persistence implementation. These entry points all converge on
it:

- `vamos.save_result`;
- the CLI `StorageObserver` adapter;
- experiment, benchmark, zoo, quickstart, and ablation executions that use the
  CLI runner.

Study summaries may reference a run directory, but they never copy or rebuild
its files. Analysis and Studio are readers only. Result helper and example code
must use `vamos.save_result`; they do not implement their own persistence.

The writer never overwrites, merges, or repairs a destination. Every existing
path is an `OutputCollisionError`, including an empty directory or a partial
run. It snapshots supported arrays, writes a uniquely owned sibling staging
directory, fsyncs files, commits the terminal manifest last, and publishes with
one directory rename. Failure removes only staging state owned by that writer.

## 4. Public Python API

The authoritative imports are top-level:

```python
import vamos

stored = vamos.save_result(result, "runs/run-1")
run = vamos.load_run("runs/run-1")
result = vamos.load_result("runs/run-1")
```

The signatures are:

```text
save_result(result, path, *, requested_spec=None, resolved_spec=None,
            labels=None, limits=None) -> StoredRun
load_run(path, *, verify="required", limits=None) -> StoredRun
load_result(path, *, verify="required", limits=None) -> OptimizationResult
verify_run(path, *, require_level=None, limits=None) -> VerificationReport
reproduce(path, *, output=None, limits=None) -> ReplayReport
```

There is no persistence export from `vamos.ux.api`.

An `OptimizationResult` returned by `vamos.optimize` carries captured requested
and resolved specs. A manually constructed result must pass both
`requested_spec=` and `resolved_spec=`. Supplying neither or only one raises
`IncompleteRunMetadataError`; VAMOS never invents an algorithm, backend,
termination, provenance, or seed from array shapes.

## 5. Requested and resolved specifications

`requested_spec` preserves user intent as JSON data. Omitted defaults remain
omitted. A requested seed may be `null` when the user explicitly asks VAMOS to
choose one.

`resolved_spec` has identity `vamos.resolved-run-spec/1` and records the actual
execution state:

- problem identity, dimensions, encoding, constraint convention, provider, and
  configuration;
- algorithm identity, provider, complete configuration, and result mode;
- active/inactive operator descriptors and resolved parameters;
- kernel and evaluation backend descriptors;
- termination descriptor and effective hard budget;
- actual integer seed;
- population, offspring, and archive sizes;
- defaults applied and their sources;
- determinism declaration and RNG family.

`resolved_spec.seed` is always an integer. Zero is valid. For `seed=None`, VAMOS
generates an integer before constructing any stochastic execution object,
exposes it in `OptimizationResult.meta["seed"]`, and persists it in the resolved
spec. No sentinel represents an unknown seed.

`task_id` is `sha256:` plus the SHA-256 of canonical JSON bytes for the complete
resolved spec. `run_id` is a unique UUIDv4 for the attempt. Repeated executions
of the same task share a task ID but have distinct run IDs.

## 6. Run manifest

The manifest requires:

- `document_type`, exactly `vamos.run-manifest`;
- `schema_version`, exactly `1.0.0`;
- `run_id` and content-derived `task_id`;
- `status` and RFC 3339 `timestamps`;
- `requested_spec` and `resolved_spec`;
- `provenance` and `replayability`;
- `outcome` for a terminal run;
- `artifacts` descriptors;
- `integrity.manifest_sha256` for a terminal run.

Statuses are `running`, `succeeded`, `failed`, `partial`, and `cancelled`.
`completed_at` is required only for terminal states. A succeeded run requires
both `result_bundle` and `environment` descriptors. A failed run requires a
structured `failure` object and has no implied numerical result.

`outcome` records counters, runtime, termination reason, interruption and
usability flags, result mode, array-derived solution/dimension counts, and
bounded JSON metrics. Cached counts never override result array shapes.

Every artifact descriptor records `role`, relative `path`, `media_type`, exact
`bytes`, lowercase SHA-256, `required_for`, and whether it is canonical. The
result descriptor also records the name, shape, and exact NumPy dtype string of
each array.

The manifest self-hash is computed from canonical JSON with
`integrity.manifest_sha256` omitted. Object keys are sorted, UTF-8 is used,
duplicate keys are rejected, and non-finite JSON numbers are forbidden.
Whitespace-only reformatting does not change the semantic hash.

## 7. Numerical ResultBundle

`result.npz` is an NPZ of independent NPY members loaded with
`allow_pickle=False`. `F` is required for a succeeded run. Supported names are:

- `F`, `X`, `G`, `CV`, and `reference_directions`;
- `population/F`, `population/X`, `population/G`, `population/CV`;
- `archive/F`, `archive/X`, `archive/G`, `archive/CV`.

V1 accepts only fixed-width boolean, signed integer, unsigned integer, and
floating arrays. Object, string, structured, complex, datetime, and
pickle-backed arrays are rejected. Shape relationships and manifest array
contracts are validated before values are exposed. Loaded arrays are defensive
copies and are never executable objects.

## 8. Environment and provenance

`environment.json` has identity `vamos.environment` and version `1.0.0`. It
contains Python, operating-system/architecture, installed distributions,
backend package, BLAS, allowlisted thread controls, locale, and timezone. It
does not store hostnames, account names, arbitrary environment variables,
secrets, or personal paths.

Provenance records VAMOS/distribution identity, Git/source evidence when
available, the environment artifact role, and the same timestamps as the
manifest. Python saves identify `vamos.optimize`; CLI saves identify a
sanitized CLI entry point. A caller-supplied manual context is marked as such.

Replayability levels are `exact`, `compatible`, `best_effort`, `manual`, and
`unavailable`. They are evidence declarations, not an execution API. An exact
claim requires the same resolved spec, implementation, backend, materially
equivalent environment, and a deterministic path. Loading does not test replay
equivalence.

Exact verification compares only material evidence: VAMOS version and
implementation fingerprint, source kind, Python implementation and major/minor
version, operating system and architecture, NumPy/SciPy, selected backend and
backend package, captured capabilities, BLAS, and allowlisted thread controls.
The complete installed-package inventory is not material. Missing material
evidence blocks exact replay. A dirty checkout qualifies only with a matching
reproducible content fingerprint. Current evidence capture uses no Git command,
shell, network, installation, or mutation.

## 9. Loading, verification, and errors

Loading is data-only. It performs no optimization, dynamic import, plugin
resolution, custom-code execution, pickle load, shell command, network request,
or filesystem access outside the run directory.

`verify_run` is also data-only. It verifies every referenced artifact's bytes,
parses known environment/numerical artifacts through bounded safe readers, and
reports artifact integrity, path/NPZ safety, environment compatibility,
component reconstructability, and effective replayability independently.
`vamos results inspect` performs manifest-only inspection without materializing
arrays; `vamos results verify` performs full verification. `--require-level
exact` fails when the effective level is lower than exact.

Verification modes are:

- `manifest`: validate manifest JSON, semantics, task ID, and self-hash;
- `required` (default): additionally verify artifacts required for loading;
- `all`: verify every known referenced artifact.

An accessed artifact is always parsed through its bounded safe reader even in
`manifest` mode. Unknown optional roles remain inert. An unknown role required
for loading is rejected before it is opened.

Errors derive from `RunArtifactError` and include stable fields for operation,
category, reason, expected/actual values, path/role/field, and action. Missing,
modified, malformed, unsafe, oversized, unsupported-schema, collision,
incomplete-run, and incomplete-metadata cases have distinct typed errors.

A directory without a supported manifest, or a manifest with another document
identity/version, is rejected uniformly. The action states that this is a
pre-1.0 development format and the run must be regenerated with the current VAMOS
version. The reader does not inspect filenames to classify an earlier format.

## 10. Defensive limits

Default limits are finite: manifest 8 MiB, environment 16 MiB, one artifact or
array 512 MiB, total uncompressed arrays 1 GiB, 128 descriptors/ZIP members, 64
arrays, 100 million elements, 8 dimensions, 64 KiB NPY headers, JSON depth 64,
and compression ratio 1000:1. ZIP member names, overlaps, compression flags,
headers, sizes, shapes, dtypes, and ratios are checked before materialization.

Trusted callers may pass an explicit `LoadLimits`. VAMOS never silently raises
a limit following rejection.

## 11. Consumers

`discover_runs` finds `manifest.json` and accepts only manifests validated by
the canonical reader. `load_run_data` and Studio load numerical values only
through `load_run`/`load_result`. Aggregation reads resolved identity, seed,
timestamps, outcome counters, runtime, termination, and optional metrics from
the manifest. Directory names are presentation, not data recovery heuristics.

Study CSV summaries are derived reports. They may contain a run path or IDs but
do not duplicate specifications or numerical arrays. A durable StudyManifest,
resume/retry orchestration, and study artifact ownership are separate work.

## 12. Explicit non-goals

This v1 consolidation does not implement:

- any alternate reader, writer, detector, adapter, or field spelling;
- replay of plugins, custom Python, closures, notebook-local code, or arbitrary
  import paths;
- cross-backend or best-effort replay, backend overrides, dependency
  installation, or environment repair;
- a durable StudyManifest or study resume/retry system;
- authentication/signatures (SHA-256 is integrity evidence, not trust);
- a general CSV export API or performance optimization initiative.

## 13. Exact built-in replay

`vamos reproduce RUN_DIR` and `vamos.reproduce(path)` first use the same full
verification service. Execution is permitted only for an effective `exact` run
whose problem, algorithm, operators, evaluation backend, termination, and
kernel use stable schema-1 built-in IDs. No plugin entry point is discovered
and no manifest-provided module name is imported.

The replay plan is reconstructed solely from `resolved_spec`: typed algorithm
configuration, operators, problem dimensions/encoding, population and
reference-direction settings, archive/stopping configuration, termination
budget, backend, and concrete seed. VAMOS regenerates a resolved spec from that
explicit plan and requires canonical semantic equality before optimization.
Current defaults never fill or replace persisted resolved values.

The stored result is the comparison target, not an initial state. Exact
comparison requires `F` and `X` with identical NumPy dtype descriptor, shape,
logical order, and contiguous C-order logical bytes. Every other deterministic
array role present in either run is compared separately. Reports include
per-array hashes, first differing logical index when safe, maximum absolute
difference when meaningful, and mismatch classification. Timestamps, duration,
run IDs, paths, and timing metrics are excluded.

Every executed replay publishes a new schema `1.0.0` run with a new `run_id`,
the same content-derived `task_id`, immediate/root lineage, bounded depth,
source-manifest hash, replay-plan hash, compatibility level, and comparison
evidence. The default destination is `<source-parent>/replays/<new-run-id>`.
Existing destinations collide and the source is never modified. Replay of a
replay retains the root run ID without copying prior manifests.

If execution begins and fails, the atomic writer publishes an inspectable
failed attempt containing `manifest.json` and `environment.json`, sanitized
failure evidence, and replay lineage. Verification/plan refusal publishes
nothing. A completed numerical mismatch is stored with mismatch evidence and is
reported as failure, never exact success.

CLI exit codes are: 0 success, 2 usage, 3 integrity/path/malformed artifact, 4
unsupported or invalid schema, 5 compatibility requirement, 6 unavailable or
untrusted component/replay, 7 execution or exact-comparison failure, and 8
output collision. JSON mode emits one machine-readable document. Verification
is the separate `results verify` command; reproduction always executes.

## 14. Examples and acceptance

Sanitized machine-readable fixtures live in
[`run_artifact_examples/`](run_artifact_examples/README.md). The normative
acceptance matrix is
[`run_artifact_acceptance_tests.md`](run_artifact_acceptance_tests.md).
