# Save and load run artifacts

VAMOS writes one relocatable canonical v1 directory for each run.

```python
import vamos

result = vamos.optimize(
    "zdt1",
    algorithm="nsgaii",
    pop_size=40,
    max_evaluations=400,
    seed=7,
)

stored = vamos.save_result(result, "runs/zdt1-seed-7")
loaded = vamos.load_result(stored.root)
run = vamos.load_run(stored.root)

print(stored.manifest.run_id)
print(loaded.F.shape)
print(run.status, run.environment["python"])
```

## Directory contents

```text
runs/zdt1-seed-7/
├── manifest.json
├── result.npz
└── environment.json
```

The manifest contains requested intent, resolved execution state, the actual
seed, provenance, outcome, artifact hashes, and a semantic self-hash. The NPZ
contains numerical result/population/archive arrays. The environment document
contains a bounded privacy-conscious runtime snapshot.

The directory may be moved as a unit. All references are relative and confined
to it. The destination passed to `save_result` must not exist; VAMOS never
overwrites or merges a run.

**Development after 1.1.0:** while it saves, VAMOS writes the run into a
sibling `.<name>.vamos-staging-<token>` directory and holds a
`.<name>.vamos-save.lock` file. The lock records the process ID, host and UTC
start time of the save. Both disappear when the save ends, but a process that
dies while saving leaves them behind, and VAMOS never removes them itself. A
later save to that destination then fails with `OutputCollisionError`. The
error names the lock, its holder and any staging directories of that
destination, and says to delete them if that process is no longer running.
In VAMOS 1.1.0 the error said only that another save owned the destination.

## Requested and generated seeds

An explicit seed, including zero, is used unchanged:

```python
result = vamos.optimize("zdt1", seed=0)
print(result.meta["seed"])  # 0
```

Pass `None` to ask VAMOS to generate a seed. Generation happens before any
stochastic execution object is constructed:

```python
result = vamos.optimize("zdt1", seed=None)
actual_seed = result.meta["seed"]
stored = vamos.save_result(result, "runs/generated-seed")

print(stored.manifest.requested_spec["defaults"]["seed"])  # None
print(stored.manifest.resolved_spec["seed"])                # actual integer
```

## Manual results

`OptimizationResult` instances created by `vamos.optimize` already carry their
complete run context. A manually constructed result must provide both the
requested and resolved specs:

```python
stored = vamos.save_result(
    manual_result,
    "runs/manual",
    requested_spec=requested_spec,
    resolved_spec=resolved_spec,
)
```

Without both mappings, saving raises `vamos.IncompleteRunMetadataError`. VAMOS
does not invent a seed or execution configuration from arrays.

## Loading and verification

Loading is data-only. It does not rerun optimization, resolve plugins, import
recorded custom components, use pickle, invoke a shell, or access the network.
Inspection and verification have the same inert trust boundary. Replay is an
explicit separate operation.

Verification modes are:

- `verify="manifest"`: validate manifest syntax, semantics, task ID, and
  self-hash;
- `verify="required"` (default): also verify artifacts required for loading;
- `verify="all"`: verify every known referenced artifact.

`load_run` returns an immutable manifest and lazy `.result`/`.environment`
access. `load_result` is the convenient numerical path. A failed run remains
inspectable through `load_run`, while `.result` raises `IncompleteRunError`.

## Inspect and verify

Inspect metadata without loading full arrays:

```bash
vamos results inspect runs/zdt1-seed-7
vamos results inspect runs/zdt1-seed-7 --json
```

The summary includes identity, problem/algorithm/backend, requested and
resolved seed, population/budget/outcome, array shapes/dtypes, replayability,
and lineage. It explicitly reports that full artifact verification was not
performed.

Full verification checks integrity, schema/path/NPZ safety, material
environment compatibility, built-in reconstructability, and effective
replayability as separate dimensions:

```bash
vamos results verify runs/zdt1-seed-7
vamos results verify runs/zdt1-seed-7 --require-level exact
```

The Python equivalent is:

```python
from vamos import verify_run

verification = verify_run("runs/zdt1-seed-7", require_level="exact")
print(verification.environment.level)
```

Exact compatibility requires matching VAMOS implementation content, Python
major/minor, material NumPy/SciPy/backend evidence, OS/architecture, BLAS, and
allowlisted thread controls. Missing material evidence does not qualify as
exact. Verification never installs or contacts anything.

## Exact built-in replay

```bash
vamos reproduce runs/zdt1-seed-7
vamos reproduce runs/zdt1-seed-7 --output runs/replays/zdt1-seed-7
```

Or from Python:

```python
from vamos import reproduce

replay = reproduce(
    "runs/zdt1-seed-7",
    output="runs/replays/zdt1-seed-7",
)
print(replay.exact, replay.output_root)
```

Replay never modifies the source. It creates a new canonical run, uses the
persisted resolved configuration and concrete seed, and never substitutes
current defaults. `F`, `X`, and deterministic auxiliary arrays are compared by
dtype, shape, logical order, and exact bytes. The new manifest stores bounded
source/root lineage and per-array comparison evidence.

This slice executes only same-environment, same-backend registered built-ins.
Custom Python, plugins, cross-backend execution, and best-effort environments
are outside the exact-replay contract, and it never installs dependencies. A pre-execution
refusal creates nothing; a failure after execution begins creates an inspectable
failed canonical attempt.

## Numerical safety and limits

NPZ loading always uses `allow_pickle=False`. V1 accepts fixed-width boolean,
integer, unsigned integer, and floating arrays. It rejects executable/object,
string, structured, complex, and datetime dtypes.

Readers validate bounded JSON, artifact count/size, ZIP member layout, NPY
headers, shapes, dtype, total elements/uncompressed bytes, and compression
ratio before materializing arrays. A trusted caller can explicitly provide a
different `vamos.LoadLimits`; limits are never raised automatically.

## Unsupported directories

VAMOS 1.x supports the public `vamos.run-manifest` `1.0.0` baseline. A directory
without that manifest is rejected with guidance to regenerate the run using the
current version. Schema `1.0.0` is the sole supported reader/writer path.

SHA-256 detects accidental modification; it is not a digital signature.
