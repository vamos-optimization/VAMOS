# Study planning preflight acceptance specification

Status: normative for the read-only `plan_study` / `vamos study plan` slice

Contract: [Durable study and StudyManifest v1](study_manifest_contract.md)

These cases are separate from the persisted-state `SA-*` inventory. Planning
may resolve supported built-ins and inspect a proposed destination, but it
creates no directory, evaluates no objective, and publishes no canonical
state. The IDs are contiguous and must not be renumbered.

| ID | Operation | Input | Expected result | Invariant |
|---|---|---|---|---|
| PL-001 | Python plan | Minimal supported `StudySpec` | Immutable ready report with plan/task identities | Planning uses the canonical resolver |
| PL-002 | Python plan | Explicit empty matrix | Ready zero-task, zero-budget report | Empty plans remain representable |
| PL-003 | Python plan | Same spec twice | Equal plan ID and task IDs | Resolution is deterministic |
| PL-004 | Python plan | Equivalent reordered matrix | Equal plan ID and task set | Scientific identity is order-independent |
| PL-005 | Budget summary | Several resolved tasks | Exact sum of per-task evaluation budgets | No runtime estimate is fabricated |
| PL-006 | Seed summary | Explicit seed matrix | Every concrete resolved seed reported | Seeds are never regenerated |
| PL-007 | Backend summary | Explicit or resolved backend | Exact kernel/evaluation component IDs | No backend is substituted |
| PL-008 | Algorithm validation | NSGA-III reference directions and population | Compatible cardinality resolves; mismatch is actionable | Invalid execution shape fails before publication |
| PL-009 | Duplicate validation | Duplicate canonical matrix task | `DUPLICATE_CANONICAL_TASK` | Duplicate work is rejected |
| PL-010 | Component validation | Unsupported problem or algorithm | Typed field/matrix resolution error | No arbitrary provider import |
| PL-011 | Budget validation | Budget below initial population | `INVALID_EVALUATION_BUDGET` | Impossible initial execution is rejected |
| PL-012 | Backend validation | Unavailable backend | Typed resolution error | No silent fallback |
| PL-013 | Output inspection | Absent path | `available`, with race advisory | The path is not created or reserved |
| PL-014 | Output inspection | File, empty directory, canonical study, unrelated or invalid directory | Distinct occupied classifications | Every existing path matches create collision semantics |
| PL-015 | Side-effect audit | Missing parent/output | Tree remains byte-identical | Planning performs zero filesystem writes |
| PL-016 | Execution audit | Objective raises if evaluated | Valid report and zero evaluations | Planning never optimizes |
| PL-017 | Python/CLI equivalence | Same JSON fields and `StudySpec` | Equal plan/task identities and summaries | CLI delegates to the public planner |
| PL-018 | JSON automation | Success, collision, and invalid input with `--json` | One `vamos.study-command-result` v1 document on stdout | No mixed human stdout or traceback |
| PL-019 | Installed wheel | Python and console command in a clean environment | Both preflight paths succeed without repository imports | Packaging includes the complete feature |
| PL-020 | Plan/create equivalence | Same `StudySpec` | Exact plan ID, task IDs, ordering, specs, seeds, operators, backend, population and budget | No second resolver exists |
| PL-021 | Architecture quality | Public API, typing, architecture and remnant checks | All development ratchets pass | No legacy path, dependency, or typing debt is added |
| PL-022 | Executability validation | `eval_strategy` `multiprocessing` or `dask`, or a negative seed | `UNSUPPORTED_EVALUATION_BACKEND` or `INVALID_STUDY_SPEC` from plan and create; nothing is published; every plannable `eval_strategy` runs to completion | A plan never promises a task that `Study.run()` cannot execute |

## Stable JSON envelope

Both successful and invalid command outcomes use:

```text
document_type = "vamos.study-command-result"
schema_version = "1.0.0"
operation = "plan"
```

The common envelope always includes operation, status, exit code, study/plan
identities, changed, payload, warnings, structured errors, and next actions.
The plan payload retains validity, execution/write flags, task identities,
component summaries, and advisory output status. Invalid input leaves
identity fields empty rather than inventing partial state.
