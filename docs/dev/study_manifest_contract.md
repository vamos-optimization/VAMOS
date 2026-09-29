# VAMOS durable study and StudyManifest v1 contract

Status: approved pre-release contract; create/load, sequential execution, failure policy, cancellation, reconciliation, resume, retry, inspection, summary projection/output, and the single-owner lifecycle CLI implemented

Primary document identity: `vamos.study-manifest`

Schema version: `1.0.0`

Normative acceptance inventory: [SA-001 through SA-074](study_manifest_acceptance_tests.md)

## 1. Scope and authority

This document freezes the only durable-study contract that VAMOS will implement.
It covers study intent, immutable planning, task and attempt identity, durable
state, failure policy, retry, resume, crash recovery, local concurrency,
canonical run references, inspection, and derived summaries. The contract is
implementation-independent. The immutable models, deterministic planner,
read-only `plan_study` report and `vamos study plan` command, atomic creator,
data-only loader, journal derivation, and single-process
sequential `Study.run()` slice with persisted failure policy and graceful
cancellation, explicit reconciliation writes, resume, bounded retry, the
shared immutable `Study.inspect()`/`Study.summarize()` projection, and the
single-owner lifecycle CLI with explicit derived summary output, and canonical
package/research caller integration are implemented. Locks, leases, and
parallelism remain deferred.

VAMOS 1.0.0 establishes version `1.0.0` as the first public study schema. There is no
second reader, detector, fallback layout, alias, or deprecation period. Git
history is the historical record.

The canonical per-attempt artifact remains
`vamos.run-manifest` version `1.0.0`, governed by the
[run-artifact contract](run_artifact_contract.md). This contract references
that format and never redefines it.

### Goals

- Preserve what the user requested and the complete resolved task set before
  execution.
- Make task identity independent of directory names and matrix order.
- Record every claim and attempt without overwriting history.
- Resume without rerunning verified successes or re-resolving current defaults.
- Make partial failure and interruption inspectable and actionable.
- Prevent duplicate execution within the declared local concurrency boundary.
- Keep summaries removable and reproducible from canonical records.
- Keep loading, inspection, and summary generation data-only.

### Non-goals

The current bounded implementation does not provide parallel workers,
distributed coordination, new algorithms, or statistical reporting. It does
not support plan mutation, successful-task force retry, custom-code loading,
or a second persisted format.

## 2. Evidence from the current implementation

The current flow is:

```text
StudySpec
    -> plan_study / create_study through one resolver
    -> Study.run / resume / retry through one execution service
    -> verified canonical RunManifest per attempt
    -> Study.inspect / Study.summarize through one projection
    -> optional explicit derived caller table
```

A deterministic three-seed reproduction establishes:

- execution follows canonical task-ID order while summaries use `plan_index`;
- each successful task writes only `manifest.json`, `result.npz`, and
  `environment.json` beneath its run directory;
- derived JSON/CSV appears only after an explicit output request and never
  drives recovery;
- fail-fast and continue retain every durable attempt and account for later
  pending or completed tasks;
- retry creates a new attempt while successful work remains immutable;
- foreground interruption publishes a durable cancelled state before return.

Ablation and benchmark partition heterogeneous campaign inputs into homogeneous
`StudySpec` values. Studio and study analysis traverse StudyManifest references
and derive caller views from `StudySummary`; ordinary RunManifest discovery
remains a separate run-analysis workflow. `create_study` publishes the frozen
plan, `Study.run()` reconstructs it one task at a time, and `load_study` remains
data-only.

## 3. Target lifecycle

```text
StudySpec (user intent)
    -> resolve once
ResolvedStudyPlan (immutable task set)
    -> create StudyManifest and pending TaskRecords atomically
    -> claim task and create unique AttemptRecord
    -> canonical RunManifest publication
    -> verify and publish attempt outcome
    -> replay event journal into task/study checkpoints
    -> inspect, resume, retry, and regenerate summaries
```

No operation after creation consults current defaults to change the plan.

### 3.1 Implemented sequential slice

The current `Study.run()` accepts only a pristine `created` study and returns a
freshly loaded immutable `Study`. It executes in ascending `task_id` order;
`plan_index` remains presentation metadata. Each attempt is durable and
`running` before objective evaluation. Each successful RunManifest is reopened
with full verification before `attempt_succeeded` enters the journal.

This slice deliberately has no coordination path, lock, lease, heartbeat,
worker identity, cross-process ownership guarantee, or parallelism. The CLI is
a thin consumer of this service and explicit summary files remain derived.
Cancellation is durable but limited to an idle transition or a
cooperative request in the active runner's process. Obvious same-process
reentry is rejected. An empty study follows the direct `created` to `completed`
transition and creates no attempt or run.

Reconstruction or objective failure after attempt start publishes a verified
failed run when possible and commits the failed attempt/task. The persisted
policy then either pauses fail-fast execution with later work still `pending`,
or continues and eventually records `completed_with_failures`. Publication or
verification interruption is infrastructure failure: it publishes no outcome
event, cannot link an unverified run, stops either policy, and raises a typed
error. Graceful cancellation finalizes active work without a fabricated run,
cancels unclaimed tasks, and records `study_cancelled`.

## 4. Users and decision impact

| Decision | User A: small study | User B: expensive campaign |
|---|---|---|
| Separate spec and resolved plan | Compact matrix input stays approachable; inspect shows exact expanded tasks. | Defaults cannot drift during a long campaign or resume. |
| Random study ID plus content-derived plan/task IDs | Output instances are easy to distinguish. | Repeated plans and tasks can be compared without conflating executions. |
| Per-task and per-attempt records | Progress and failure locations are explicit. | Thousands of tasks avoid one giant rewritten ledger. |
| Event journal plus checkpoints | Inspect is fast and messages explain the next action. | Crash recovery has an authoritative, auditable transition history. |
| Default fail-fast, explicit continue | A first run stops at the first unexpected task failure. | Continue can be selected before execution and partial completion remains explicit. |
| No implicit retry; bounded explicit retry | Repeated work is never surprising. | Failed and interrupted attempts remain attributable and expensive successes are protected. |
| Strict resume reconciliation | Completed tasks are not rerun. | Published runs are recovered after crashes before any new expensive attempt. |
| Local leases with fencing | Normal single-process use stays invisible. | Local workers cannot publish duplicate or stale success. |
| Derived summaries only | CSV and tables remain convenient. | Deleting reports never loses resumable scientific state. |
| Data-only load/inspect/summarize | Opening a study is safe and predictable. | Campaign inspection does not execute plugins, shell, network, or optimization. |
| Separate create and run API/CLI | The resolved plan can be reviewed before spending compute. | Output collision and plan mismatch are detected before task execution. |

Framework extenders add a task dimension through `StudySpec` resolution and
the canonical task projection, an execution backend through the existing run
resolution boundary, and a derived report through the summary service. They do
not change the state machine. Coding agents use this contract and its
[acceptance specification](study_manifest_acceptance_tests.md) rather than
inferring persistence from callers.

## 5. Frozen conceptual model and ownership

| Concept | Sole responsibility | Must not own |
|---|---|---|
| `StudySpec` | User matrix intent, labels, metadata, and initial execution/retry policy. | Resolved task truth or current state. |
| `ResolvedStudyPlan` | Immutable, fully resolved requested/resolved run inputs and the canonical task set. | Attempts, run outputs, or mutable status. |
| `StudyManifest` | Study identity, plan/spec descriptors, current checkpoint, aggregate counts, policy, and event head. | Complete run manifests, arrays, or copied resolved specs. |
| `TaskRecord` | One stable task, current validated checkpoint, attempt descriptors, selected success, claim epoch, and reason/retry classification. | Complete run specifications or numerical output. |
| `AttemptRecord` | One claim/execution attempt, timestamps, execution/lease identity, terminal status, failure classification, and optional run reference. | Arrays, environment content, or another attempt's state. |
| `StudyEvent` | One immutable, ordered state transition sufficient to replay checkpoints. | Numerical output or arbitrary logs. |
| `StudyReport` | Implemented immutable inspection view of verified canonical state; recovery operations continue to return `Study`. | Persistence authority or human rendering. |
| `StudySummary` | Regenerable JSON/CSV/DataFrame/publication view. | Resume decisions or canonical status. |
| `RunManifest` reference | Relative path, run ID, task ID, semantic manifest hash, file hash, and byte length. | A copied run spec, environment, provenance, replay evidence, or arrays. |

### 5.1 StudySpec

`StudySpec` records problems, algorithms, seeds, run backends, budgets,
population/operator settings, user labels, bounded JSON metadata, and policy.
Its defaults are `on_error="fail_fast"`, no automatic retry, and
`max_attempts_per_task=3`. Labels and presentation metadata are not scientific
identity. The spec is immutable after study publication.

The implemented V1 JSON encoding is closed. `matrix` contains exactly
`problems`, `algorithms`, and explicit integer `seeds`. `run_defaults` contains
exactly `max_evaluations`, `pop_size`, `engine`, `eval_strategy`, `n_var`,
`n_obj`, `problem_kwargs`, and per-algorithm `algorithm_configs`. `policy`
contains exactly `on_error` and `max_attempts_per_task`; `labels` and
`metadata` are bounded JSON objects. Optional run defaults are stored as
`null`, not omitted, so one writer form exists.

### 5.2 ResolvedStudyPlan

Resolution occurs once, before output publication. Every plan task contains the
requested run input and the complete resolved run specification required by the
canonical run writer. Plan tasks are serialized in ascending `task_id` order.
This is the sole pre-execution owner of resolved run input inside the study;
StudyManifest, TaskRecord, and AttemptRecord refer to its hash rather than copy
it. A produced RunManifest independently owns the actual run's requested and
resolved specs under the run-artifact contract.

### 5.3 StudyManifest and checkpoints

The root manifest is a small mutable checkpoint. Its status and counts must
equal replay of the immutable event sequence through its declared event head.
It is canonical and integrity-checked, but it is reconstructible; the event
journal wins if a crash leaves a newer valid event than the checkpoint.

Task and nonterminal attempt files are analogous per-entity checkpoints.
Terminal attempt records are immutable.

## 6. Document identities and required fields

Every document rejects duplicate keys, unknown fields, non-finite numbers, and
unsupported identity/version.

| File role | `document_type` | Required core fields |
|---|---|---|
| User intent | `vamos.study-spec` | `schema_version`, `study_id`, matrix, policy, labels, metadata, integrity. |
| Resolved plan | `vamos.resolved-study-plan` | `schema_version`, `plan_id`, sorted tasks, task count, integrity. |
| Root checkpoint | `vamos.study-manifest` | `schema_version`, IDs, status, policy, timestamps, spec/plan descriptors, counts, checkpoint, integrity. |
| Task checkpoint | `vamos.study-task` | `schema_version`, IDs, plan index, state, attempt descriptors, selected success, retry/reason, claim epoch, integrity. |
| Attempt | `vamos.study-attempt` | `schema_version`, IDs, attempt number, execution ID, status, timestamps, lease evidence, failure, run reference, integrity. |
| Event | `vamos.study-event` | `schema_version`, sequence, event ID/type, entity, transition, execution ID, timestamp, reason/payload, previous hash, integrity. |

All use `schema_version="1.0.0"`. RFC 3339 UTC timestamps use a `Z` suffix.
Durations are nonnegative integer milliseconds. Metadata is bounded JSON and
contains no executable references or secrets.

## 7. Identity and canonicalization

Canonical JSON uses sorted object keys, UTF-8, no insignificant whitespace,
duplicate-key rejection, and no non-finite values. Semantic self-hashes omit
only their own integrity hash field.

### 7.1 Study and execution identity

- `study_id` is a lowercase UUIDv4. It identifies one persisted study instance
  and is unrelated to its name or directory.
- `execution_id` is a new UUIDv4 for each `run`, `resume`, or `retry` command.
  A resumed execution records `parent_execution_id` when initiated from a known
  prior paused/interrupted execution.
- Moving or copying the study does not change either identity. Copying and then
  independently operating both roots is outside v1 and must be refused when an
  active origin/coordination token proves divergence.

### 7.2 Plan identity

`plan_id` is `sha256:` plus the SHA-256 of a canonical projection containing
schema identity and the set of task projections sorted by `task_id`. Display
order, labels, output path, study ID, timestamps, execution policy, and worker
count are excluded. Reordering equivalent matrix inputs therefore preserves
the plan ID.

The published plan is immutable. V1 has no append, delete, patch, extension, or
plan-version operation. Any scientifically relevant change creates a new study
and plan identity.

The V1 task projection is exactly `{ "task_id": <canonical task ID> }`.
Because that ID already hashes the complete resolved run specification, the
projection neither duplicates scientific fields nor admits a second identity
definition.

### 7.3 Task identity

`task_id` is exactly the canonical RunManifest task ID: `sha256:` plus the
SHA-256 of the complete canonical resolved run specification. It is independent
of matrix order, labels, study/output path, worker identity, and attempt count.
It distinguishes seed, problem dimensions/configuration, algorithm/config,
operators, evaluation/kernel backend, budget/termination, and every other
scientifically material resolved field.

Filesystem task directories use only the 64 lowercase hexadecimal digest,
without the `sha256:` prefix, so they are portable on Windows.

### 7.4 Attempt identity

`attempt_id` is a lowercase UUIDv4. `attempt_number` starts at 1 and increases
monotonically within a task. Retry retains the task ID and creates both a new
attempt ID and, once execution begins, a distinct RunManifest `run_id`.
Attempt and run IDs are deliberately separate because a claim exists before a
run can be published.

## 8. State machines

Statuses for study, task, and attempt are separate. No transition is inferred
from a directory name or summary file.

### 8.1 Attempt states

`created`, `running`, `succeeded`, `failed`, `interrupted`, and `cancelled`.
Succeeded, failed, interrupted, and cancelled are terminal for that attempt.

| From | To | Trigger and durable requirement |
|---|---|---|
| none | created | Task claim event reserves attempt ID/number and expected run path. |
| created | running | Worker owns the current lease; `attempt_started` is committed before optimization. |
| created | cancelled | Cancellation is observed before execution begins. |
| running | succeeded | A succeeded canonical run is fully published and verified; success event commits last. |
| running | failed | A failed canonical run is published and verified with sanitized failure evidence. |
| running | interrupted | Lease becomes stale or recovery finds no terminal canonical run. |
| running | cancelled | Graceful cancellation publishes a cancelled run or records a bounded cancellation boundary. |

### 8.2 Task states

`pending`, `running`, `succeeded`, `failed`, `interrupted`, `cancelled`, and
`skipped`. Retryability is a separate `{retryable, category, attempts_remaining}`
record, never a task status.

| From | To | Trigger and durable requirement |
|---|---|---|
| none | pending | Atomic study creation publishes one TaskRecord per plan task. |
| pending | running | Valid claim creates attempt and lease. |
| pending | skipped | Explicit applicability decision records stable reason; fail-fast never does this. |
| pending | cancelled | Study/user cancellation before claim. |
| running | succeeded | Current attempt succeeds and its verified run is selected. |
| running | failed | Current attempt fails terminally. |
| running | interrupted | Current attempt is reconciled as interrupted. |
| running | cancelled | Current attempt or study is cancelled. |
| failed | running | Explicit eligible retry claims a new attempt below the limit. |
| interrupted | running | Resume claims a new attempt after reconciliation and below the limit. |

`succeeded`, `skipped`, and `cancelled` tasks are terminal in v1. Successful
tasks have no force-retry operation.

### 8.3 Study states

`created`, `running`, `paused`, `completed`, `completed_with_failures`,
`failed`, and `cancelled`.

| From | To | Trigger and durable requirement |
|---|---|---|
| none | created | Complete spec, plan, pending tasks, initial event, and terminal creation manifest publish atomically. |
| created | running | First nonempty execution commits `execution_started`; future coordinated execution also acquires the study lock. |
| created | completed | An empty plan appends only `study_completed` and creates no attempt or run. |
| created | cancelled | User cancels before execution. |
| running | paused | Fail-fast observes task failure, graceful interruption occurs, or runnable work remains after reconciliation. |
| running | completed | Every task succeeded. |
| running | completed_with_failures | No runnable/pending task remains and at least one task failed, interrupted, or skipped under continue policy. |
| running | failed | A study-infrastructure failure is durably recorded while state remains trustworthy. |
| running | cancelled | Cancellation completes and all unclaimed tasks are cancelled. |
| paused | running | Resume starts a new execution after verification/reconciliation. |
| paused | cancelled | User abandons the paused study. |
| completed_with_failures | running | Explicit eligible retry starts a new execution. |

`completed`, `failed`, and `cancelled` are terminal in v1.

### 8.4 Invalid transitions and derivation

Every nonlisted transition raises `InvalidStudyTransitionError` before a write.
The error identifies operation, entity ID, current/required state, whether any
execution occurred, and a safe next command. It is never silently idempotent.
Idempotent read/resume with no runnable work returns an unchanged report; it
does not manufacture a transition.

Attempt state is stored in its record. Task state is stored as a checkpoint and
must equal derivation from its attempts and task events. Study state/counts are
stored as a checkpoint and must equal derivation from task states and study
events. A valid newer journal event derives the effective loaded view without a
write; any checkpoint that leads or contradicts the journal is corruption.

## 9. Failure, cancellation, and partial completion

### 9.1 Fail-fast

`fail_fast` is the default persisted policy. After a failed attempt is
durable, the scheduler stops issuing new claims. Already-running local tasks may
finish and publish under their leases. Unclaimed tasks remain `pending`, not
`skipped`. The study becomes `paused`, names the failed task/attempt, and gives
`vamos study resume STUDY_DIR` for pending work or
`vamos study retry STUDY_DIR --failed` for an explicit failed-task retry.

### 9.2 Continue

`continue` keeps claiming independent pending tasks after a task failure. When
none remain, all successes and failures are retained and the study becomes
`completed_with_failures`. That is a valid completed study but not successful
execution.

Policy is fixed in StudySpec before the first attempt and resume cannot change
it. A different policy requires a new study.

### 9.3 Cancellation and process termination

- A user cancellation request stops new claims, asks active workers to cancel,
  records terminal attempt outcomes, marks unclaimed tasks `cancelled`, then
  marks the study `cancelled`.
- A graceful process signal follows the same protocol when time permits.
- A forced process death writes nothing after death. Recovery treats an active
  lease as running and a stale lease as interrupted only after reconciliation.
- Worker cancellation affects its current attempt; study policy decides whether
  the study pauses or continues.

An infrastructure failure is not a task failure. If canonical state is still
writable it transitions the study to `failed`; otherwise the next data-only load
reports corruption without pretending that a transition committed.

## 10. Retry policy

V1 performs no implicit or automatic retry. `max_attempts_per_task` defaults to
3 and is persisted before execution. Explicit retry:

- applies to selected `failed` or `interrupted` tasks only;
- preserves task ID and every prior terminal attempt;
- creates the next attempt number and a new attempt/execution/run identity;
- refuses when the limit is reached;
- requires a retryable failure classification.

Transient execution/backend unavailability, worker loss, and interruption are
retryable after their precondition is corrected. Invalid specification,
unsupported component, plan/run integrity failure, invalid transition,
deterministic configuration error, and exact numerical mismatch are
nonretryable. Changing scientific configuration creates a new task in a new
study; it is never a retry. Successful tasks cannot be retried in v1.

## 11. Resume and reconciliation

Resume executes this ordered protocol:

1. Load all referenced JSON with duplicate-key rejection and finite limits.
2. Verify document identities, hashes, confined paths, spec, and immutable plan.
3. Verify the event hash chain and replay events beyond checkpoints.
4. Inspect the one prior running attempt and reject detectable same-process
   ownership.
5. For each expected run path, use the canonical data-only run verifier.
6. Publish a recovered success/failure if a complete matching run exists;
   otherwise mark a stale running attempt `interrupted`.
7. Recompute task/study checkpoints atomically.
8. Identify pending tasks and interrupted tasks with attempts remaining.
9. Optionally include failed tasks only when `retry_failed=True` was explicit.
10. Create fresh attempt/run identities and execute only that eligible set
    sequentially.

Succeeded tasks are never rerun. A succeeded task whose referenced run is
missing, corrupt, has the wrong run/task ID, or no longer matches its recorded
manifest hash is an actionable integrity error, not pending work. A changed plan
or current-default resolution is rejected. A relocated complete root works
because every load path is relative.

Resume requires the persisted built-in component IDs and complete resolved
inputs to remain supported. A materially different implementation environment
is refused by default with `ResumeEnvironmentIncompatibilityError`. A future
execution Goal may implement one explicit `accept_environment_change` option;
it may proceed only after full component/spec validation and must record both
environment fingerprints in an event. It never changes task or plan identity.

If no task is runnable, Python returns a fresh unchanged `Study` and writes no
bytes. The future CLI/report Goal will add explicit changed/next-action and exit
status presentation. Repeated load/resume remains data-stable.

## 12. Canonical relocatable directory

```text
<study>/
├── study-manifest.json
├── study-spec.json
├── plan.json
├── events/
│   └── 00000000000000000001.json
├── tasks/
│   └── <task-digest>/
│       ├── task.json
│       └── attempts/
│           └── <attempt-uuid>.json
├── runs/
│   └── <run-uuid>/
│       ├── manifest.json
│       ├── result.npz
│       └── environment.json
├── coordination/
│   ├── study.lock
│   └── leases/
│       └── <task-digest>.json
└── derived/
    ├── summary.json
    ├── tasks.csv
    └── metrics.csv
```

`study-spec.json`, `plan.json`, events, task records, attempt records, and run
directories are canonical. Root/task/nonterminal-attempt documents are mutable
canonical checkpoints and atomically replaceable. Events and terminal attempt
records are immutable. Coordination files are operational and excluded from
scientific hashes. Everything below `derived/` is optional and safe to delete
and regenerate.

The immutable plan may contain thousands of task projections because it is
written once. Each transition rewrites only one event, one affected entity
checkpoint, and the small root manifest; it never rewrites every task.

## 13. Event journal and checkpoint protocol

V1 uses an event stream, but not shared `events.jsonl`. Each event is one
canonical JSON file named by a zero-padded 20-digit sequence. A writer allocates
the next sequence, sets `previous_event_sha256`, writes/fsyncs a sibling
temporary file, and atomically renames it. The current sequential slice provides
only same-process reentry exclusion; the future coordination Goal places this
commit under the short study lock. One event per file avoids partial-line append
behavior across Windows and POSIX.

Normative event types are:

`study_created`, `execution_started`, `task_claimed`, `attempt_started`,
`attempt_succeeded`, `attempt_failed`, `attempt_interrupted`,
`attempt_cancelled`, `task_skipped`, `lease_reclaimed`, `study_paused`,
`study_completed`, `study_completed_with_failures`, `study_failed`, and
`study_cancelled`.

Events carry only bounded transition data and artifact descriptors sufficient
to replay checkpoints. Lease heartbeats and summary generation are not events.
The root checkpoint stores the latest applied event sequence and hash. Gaps,
duplicates, a broken previous hash, or an invalid transition are corruption.

## 14. Atomic writes and crash recovery

All JSON writes use a uniquely owned sibling temporary file, file fsync,
atomic replace, and parent-directory fsync where supported. The implementation
must document when a platform cannot fsync a directory; semantic recovery still
depends on atomic replace and the journal. No valid manifest ever claims
success before its referenced run is terminal and verified.

Study creation builds a sibling staging directory, writes spec/plan/tasks and
the initial event, writes the terminal root manifest last, fsyncs, then renames
to an absent destination. Every existing destination, including an empty or
partial directory, raises `StudyOutputCollisionError` without execution.

| Crash boundary | Recovery rule |
|---|---|
| Before claim event | No attempt exists; task remains pending. |
| After claim event, before attempt checkpoint | Replay creates the `created` attempt checkpoint. |
| After attempt start, before run directory | Active lease means running; stale lease becomes interrupted. |
| During canonical run staging | Run writer owns/removes only its staging path; no success is inferred. |
| After run publication, before success event | Reconcile expected run ID/path/hash, verify it, then publish outcome once. |
| After event, before task/root checkpoint | Data-only load derives the event-authoritative view; a later recovery Goal may refresh checkpoints. |
| During checkpoint replace | Reader sees an old or new complete checkpoint and derives from the event head without writing. |
| During derived summary write | Delete the incomplete derived file and regenerate; canonical state is unchanged. |

## 15. Local concurrency, locks, and leases

This section is the future coordinated-execution contract. The implemented
sequential slice writes no operational coordination state and makes no
cross-process exclusion guarantee.

The v1 schema supports concurrent processes on one host using one local
filesystem. Network filesystems, cross-host workers, object stores, and
distributed consensus are explicitly unsupported until a coordination backend
proves equivalent compare-and-set and fencing semantics.

- A short-lived `coordination/study.lock` serializes event sequence allocation,
  checkpoint commit, and task claim. It uses atomic exclusive creation, a
  random worker UUID, a random token, acquisition/expiry timestamps, and a
  30-second expiry. Long metadata work renews it; optimization never holds it.
- A task claim atomically increments `claim_epoch`, creates the AttemptRecord,
  and creates `coordination/leases/<task-digest>.json` under the study lock.
- Attempt leases expire after 300 seconds and heartbeat every 60 seconds. They
  contain worker UUID, attempt ID, claim epoch, token, acquired/heartbeat/expiry
  timestamps, and no hostname, username, secret, or absolute path.
- Publication reacquires the study lock and compare-and-sets task ID, attempt
  ID, claim epoch, and lease token. A worker that lost its lease cannot publish.
- An active lease cannot be stolen. After expiry, recovery first reconciles the
  expected run; only then may it emit `lease_reclaimed`, interrupt the attempt,
  increment the epoch, and create a new claim.

Atomic create/replace and advisory waiting behavior must be tested separately
on Windows and POSIX local filesystems. Unsupported filesystem semantics fail
before execution. Event sequence reflects durable commit order; concurrent
completion order need not be deterministic, but the resulting audit is total
and unambiguous.

## 16. Integrity, paths, relocation, and security

Every canonical JSON document has a lowercase SHA-256 semantic self-hash.
References record normalized root-relative POSIX path, exact byte length, file
SHA-256, semantic document hash, role, and required operation. The plan ID,
task-spec hash/task ID, event hash chain, and referenced RunManifest semantic
hash are independently checked.

Absolute paths, drive prefixes, backslashes, traversal, empty components, URI
forms, NUL bytes, and symlink escapes are rejected. Canonical paths are resolved
beneath the study root without following symlinks. Absolute source paths may be
bounded provenance text but are never load targets. Relocating the complete
study directory preserves all references and permits inspect/resume.

Load, inspect, verification, and summary generation are bounded, data-only
operations. They do not optimize, resolve registries/plugins, import custom
code, deserialize pickle, execute shell commands, contact a network, or access
outside the root. Execution resolves only providers permitted by the canonical
run path and never imports a manifest-provided module name.

The implemented reader defaults are: manifest/spec 8 MiB each, plan 64 MiB,
task 1 MiB, attempt/event 2 MiB each, 100,000 tasks, 300,000 referenced
documents, 512 MiB total JSON read, JSON depth 64, and 64 KiB per string.
Trusted callers may pass an explicit `StudyLoadLimits`; loading never raises a
limit automatically after rejection.

Failures store a stable category/code, bounded sanitized message, retryability,
and safe action. They exclude uncontrolled tracebacks, environment dumps,
credentials, tokens, arbitrary environment variables, usernames, hostnames,
and personal absolute paths.

Unknown fields and future schemas are rejected actionably. SHA-256 detects
corruption; it is not authentication.

## 17. Canonical RunManifest integration

An attempt reserves `runs/<run-id>/` and ultimately references
`runs/<run-id>/manifest.json`. The attempt reference contains run ID, the same
task ID, relative path, exact bytes, file SHA-256, and the RunManifest semantic
`integrity.manifest_sha256`.

Study records do not copy requested/resolved run specifications, arrays,
environment documents, provenance, outcome, or replay evidence. Fast indexes
may cache bounded scalar status/metric values, but readers verify them against
the run and never treat them as authority.

Success publication order is: canonical run directory atomically published;
run verified; IDs/hashes matched; success event committed; terminal attempt and
task/root checkpoints updated. Failed publication follows the same order with a
canonical failed RunManifest. An execution that cannot publish a terminal run
remains/reconciles as `interrupted`, never fabricated `failed` or `succeeded`.

A missing or corrupt referenced run makes a selected successful task corrupt;
resume does not rerun it silently. Exact replay remains a separate explicit run
operation and never mutates study state. A replay run becomes a study attempt
only through a future explicit new task/plan, not by directory discovery.

## 18. Derived reports and summaries

`StudyReport` and `StudySummary` are immutable Python values built by one
shared projection service from verified canonical records. `Study.inspect()`
reloads current state and reports structured run-reference issues without
repairing them. `Study.summarize()` returns exactly one `plan_index`-ordered row
per task. Both operations verify RunManifest metadata and referenced bytes
without materializing `F` or `X`, execute no component, and write no file.
Future writers may serialize the same `StudySummary` under `derived/` or an
explicit external destination; they may not reinterpret canonical state.

Every summary identifies study ID, plan ID, generation timestamp, source root
manifest semantic hash, and applied event head. Missing, failed, interrupted,
cancelled, and skipped tasks remain explicit. Aggregation calls canonical study
and run readers; it does not infer from directory names. Deleting every derived
file changes neither inspection nor resume. The in-memory summary uses the
persisted effective-state update timestamp as its deterministic generation
timestamp.

## 19. Public Python API

Three alternatives were considered:

- free functions are simple but obscure the persisted study identity;
- `Study.open(...)` makes construction/loading less discoverable;
- a thin persisted `Study` handle from explicit top-level factories separates
  creation from execution while remaining notebook-friendly.

The currently supported public surface is:

```python
from vamos import StudySpec, create_study, load_study, plan_study

spec = StudySpec(
    problems=["zdt1", "zdt2"],
    algorithms=["nsgaii"],
    seeds=[0, 1],
    max_evaluations=10_000,
)
report = plan_study(spec, output="studies/comparison-01")  # read-only
created = create_study(spec, output="studies/comparison-01")
completed = created.run()
loaded = load_study("studies/comparison-01")  # data-only
inspection = loaded.inspect()  # immutable StudyReport, zero writes
summary = loaded.summarize()  # immutable StudySummary, zero writes
```

`plan_study` and `create_study` use the same canonical resolver and therefore
produce identical plan IDs, task IDs, resolved inputs, and ordering.
`plan_study` publishes nothing and any output availability result is advisory.
`create_study` resolves and atomically publishes but executes nothing.
`load_study` is data-only. `run()` is valid only on a pristine `created` handle,
accepts no arguments, follows the persisted policy, and returns a new loaded
`Study`. `Study.cancel()` durably cancels an idle created/paused study and is a
cooperative request when this process owns the active sequential runner.
`Study.resume(retry_failed=False)` explicitly reconciles first, runs pending and
eligible interrupted tasks, and includes retryable failures only after explicit
consent. `Study.retry(failed_only=True)` retries eligible failed tasks; passing
`failed_only=False` also selects interrupted tasks. Both return a fresh loaded
`Study`; a no-runnable result writes nothing. `inspect()` and `summarize()`
reload through one metadata-only projection service and never reconcile or
write. The stateful single-owner study CLI delegates to these services. Internal journal types are not public, and no
lock/lease type exists in the implemented slice.

The top-level multi-seed result collection returned by `optimize` remains
distinct from persisted `StudyReport` and `StudySummary` projections.

## 20. Public CLI and output

The current single-owner lifecycle is exposed through one command group:

```text
vamos study plan CONFIG [--output STUDY_DIR] [--json]
vamos study create CONFIG --output STUDY_DIR [--json]
vamos study run STUDY_DIR [--json]
vamos study inspect STUDY_DIR [--json]
vamos study resume STUDY_DIR [--retry-failed] [--json]
vamos study retry STUDY_DIR --failed [--json]
vamos study summarize STUDY_DIR [--format {json,csv}] [--output PATH] [--json]
```

Planning accepts the documented JSON `StudySpec` field set, resolves the same
plan as `create_study`, never reserves output, and never evaluates an objective.
Creation and execution remain deliberately separate. There is no cross-process
cancel command; foreground interruption uses the existing graceful in-process
cancellation path.

Human mode reports study/plan IDs, sanitized root, policy, task-state counts,
whether execution or canonical state changed, and safe next commands; summary
adds a compact every-task table. Output collision is checked before resolution
publication and no `--force`, overwrite, or layout-detection option exists.

Every command supports `--json`. JSON mode emits one UTF-8 document on stdout
with `document_type="vamos.study-command-result"`, `schema_version="1.0.0"`,
operation, exit code, IDs, status, changed flag, payload, warnings, errors, and
next actions. Warnings use stderr in both rendering modes; noninteractive JSON
has no prompts, ANSI text, decorative stdout output, or raw traceback.

## 21. Typed errors and CLI exits

All typed errors expose operation, entity ID, current state, required state,
`execution_occurred`, stable reason/category, and safe next command.

| Exit | Meaning | Representative errors/states |
|---:|---|---|
| 0 | Valid success or healthy data-only/idempotent report. | Created, completed, inspect/summary without a partial or interrupted state, completed resume with no work. |
| 2 | Usage or invalid StudySpec/config. | `InvalidStudySpecError`. |
| 3 | Malformed, unsafe, missing, or corrupt data/path. | `MalformedStudyError`, duplicate key, missing task/run, hash mismatch. |
| 4 | Unsupported schema, plan mismatch, invalid transition, or invalid no-runnable operation. | `UnsupportedStudySchemaError`, `PlanMismatchError`, `InvalidStudyTransitionError`. |
| 5 | Collision or active ownership conflict. | `StudyOutputCollisionError`, `StudyLockedError`, `TaskAlreadyClaimedError`, active lease. |
| 6 | Valid partial execution with task failures. | Fail-fast paused, `completed_with_failures`, retry-required no-runnable report. |
| 7 | Study infrastructure or resume environment failure. | `StudyInfrastructureError`, `ResumeEnvironmentIncompatibilityError`, stale-lease protocol failure. |
| 8 | Cancellation or process interruption. | Graceful cancelled report or interrupted invocation. |

Additional typed failures include `ReferencedRunMissingError`,
`ReferencedRunCorruptError`, `RetryNotAllowedError`, `RetryLimitError`,
`NoRunnableTasksError`, and `LeaseLostError`. A continue-policy study with task
failures is valid `completed_with_failures` and exits 6; infrastructure failure
exits 7 and never masquerades as task failure.

## 22. Machine-readable examples

Sanitized fixtures live in
[`study_manifest_examples/`](study_manifest_examples/README.md). A fixture
envelope packages a virtual canonical file set for compact documentation; the
envelope is not a supported persistence schema. Valid fixture documents use the
identities, relative paths, semantic hashes, byte lengths, transitions, and run
placeholders frozen here. Invalid fixtures declare one expected error and fail
only for that reason.

## 23. Canonical caller surface

| Responsibility | Current code/caller |
|---|---|
| Per-run authority | Canonical run-artifact writer/readers, RunManifest identity/task ID, and `OptimizationResult`. |
| Durable orchestration | `StudySpec`, one resolver, one journal, one execution service, and one projection service. |
| Ablation/benchmark | Matrix inputs become homogeneous studies; indicator computation and tables are derived from summaries. |
| Studio/analysis | Data-only StudyManifest traversal; ordinary run discovery remains separate. |
| CLI | One command-result envelope delegates all transitions and projections to the same Python services. |

Experiment, ablation, benchmark, Studio, analysis, research, tests, docs, and
agent guidance all use this surface; no CSV-to-study import exists.

## 24. Bounded implementation roadmap

Every Goal keeps touched production modules mypy-clean, forbids new/increased
baseline diagnostics, and removes resolved baseline entries in the same
change.

| Goal | Scope | Explicit non-goals | SA unlocked | Compatibility risk | Rollback strategy |
|---|---|---|---|---|---|
| 1. Models and atomic planned-study round trip | V1 models, canonical JSON, validation, data-only loader, atomic creation, spec/plan/pending task records. | No task execution, resume, retry, CLI, or concurrency. | SA-001..020 and the create/load portion of SA-069. | New public names and frozen schema. | Internal development history only; no pre-1.0 persisted compatibility retained. |
| 2. Sequential durable runner (implemented) | `Study.run()` reserves one task at a time, reconstructs persisted science, publishes verified canonical runs, and commits attempts/events/checkpoints. | No configurable policy, cancellation, retry, resume, coordination, or workers. | SA-021..026 and SA-061..065 under the bounded-slice clarifications. | Adds the durable path without delegating to the old caller path; caller replacement remains deferred. | Revert before public release and regenerate test studies. |
| 3. Failure policy and cancellation (implemented) | `fail_fast`, `continue`, task-vs-infrastructure errors, graceful cancellation. | No retry/resume or parallelism. | SA-027..030, task-selection portions of SA-031..032, SA-064..065, and SA-068; stale-process reconciliation remains Goal 4. | Changes durable runner failure behavior intentionally; caller migration remains deferred. | Revert policy Goal; no dual policy. |
| 4. Reconciliation, resume, and explicit retry (implemented) | Reconciliation writes, interrupted-attempt recovery, pending resume, bounded failed/interrupted retry; reuse the implemented data-only journal replay. | No local parallel scheduling or environment override. | SA-033..047 and SA-049..055 beyond the already implemented data-only derivation; SA-048 remains its explicitly future override. | Prevents old rerun behavior and enforces plan identity. | Revert as one slice; preserved canonical studies remain inspectable only by the current schema revision in development. |
| 5. Inspect/summary API, stateful CLI, and caller cleanup | Immutable report/summary projection, the single-owner CLI, one command envelope, exact exits, and explicit derived summary output are implemented; ablation/benchmark/Studio/analysis migration and superseded-path deletion remain. | No parallel or distributed workers. | SA-067 and applicable SA-069..073 behavior are implemented; SA-074 completes with coordinated caller migration. | Removes pre-release APIs and directory heuristics. | Revert the coordinated caller migration; do not add aliases. |
| 6. Local locking and parallel workers | Study lock, task leases, heartbeats, fencing, path confinement, Windows/POSIX local filesystem tests. | No network filesystem or cross-host execution. | SA-056..060 and SA-066. | Platform filesystem semantics. | Disable/revert the parallel executor; retain sequential service on the same schema. |
| 7. Distributed coordination | External compare-and-set/lease backend conforming to the same tokens/events. | No schema fork or fallback filesystem claims. | New distributed acceptance IDs in that Goal. | Backend partitions/clock behavior. | Remove the provider; local contract remains canonical. |

Dependencies are strictly ordered: Goal 1 precedes all others; Goal 2 precedes
policies; Goal 3 precedes resume; Goals 2–4 precede public caller cleanup; Goal
5 precedes parallel UX; distributed work follows proven local fencing.

The first four vertical slices and the Goal 5 in-memory projection foundation are implemented: atomic planned-study creation
and data-only loading, the bounded sequential durable runner, persisted failure
policy with single-process graceful cancellation, and explicit single-owner
reconciliation/resume/retry, plus immutable inspection and summary views. Later roadmap rows do not become available by
implication.

## 25. Frozen first-slice decisions

No design choice remains open for Goal 1: identities, schema, canonical JSON,
hash projections, immutable plan, directory names, relative-path rules,
duplicate/unknown-field rejection, atomic creation, output collision, data-only
loading, public create/load names, and planned-study states are fixed here.

The later distributed coordination provider is deliberately undecided. Its
boundary is not: it must supply compare-and-set, expiring leases, fencing, and
the same event semantics without changing persisted scientific documents.

## 26. Frozen sequential-slice decisions

Goal 2 fixes `Study.run()` with no arguments, ascending `task_id` execution,
explicit persisted built-in reconstruction, one attempt and distinct canonical
run per selected task, full verification before success, a freshly loaded
return handle, event-authoritative data-only loading, and direct empty-study
completion. The slice writes no coordination or derived files.

Goal 3 makes the frozen `on_error` field authoritative: fail-fast pauses after a
durable task failure, continue retains failures and reaches
`completed_with_failures`, and infrastructure failure stops both policies.
Graceful cancellation is durable and single-process; no cross-process claim is
made. Goal 4 follows with explicit reconciliation, resume, and retry
operations.
Goal 4 records an expected run UUID in each claim event, recovers only the exact
verified matching run, interrupts an unresolved prior attempt, creates fresh
attempt/run/execution identities, and enforces the persisted attempt limit. It
does not implement SA-048's future environment-change acceptance option or any
cross-process coordination claim.
For the Goal 4 bounded meanings of SA-031 and SA-036, an explicit operation
reconciles the ended prior single-process owner immediately; no lease expiry is
modeled. SA-038 is limited to detectable same-process reentry refusal. Durable
locks and live cross-process ownership remain Goal 6 work.
Read-only planning uses the same resolver as creation and is governed by the separate
[PL-001 through PL-021 specification](study_plan_acceptance_tests.md).
