# Run artifact contract examples

These sanitized fixtures illustrate the canonical v1 document shapes. They are
documentation artifacts, not outputs from a real person or machine. IDs,
timestamps, versions, labels, and environment values are fixed examples.

Contents:

- `nsgaii-success/`: successful deterministic built-in NSGA-II run;
- `moead-success/`: successful MOEA/D run with explicit reference directions;
- `failed-run/`: failure before numerical arrays exist;
- `custom-manual/`: stored result from a notebook-local custom problem that is
  safe to load and records caller-supplied complete execution context.
- `replay-success/`: exact replay with source/root lineage and per-array hashes;
- `replay-mismatch/`: completed replay whose F bytes differ and is not exact;
- `failed-replay/`: inspectable replay attempt that failed after execution began;
- `verification-exact.json`: sanitized exact-compatible verification report;
- `verification-incompatible.json`: sanitized Python-version incompatibility
  report.

The five result-bearing directories contain small `result.npz` files using
only numerical arrays and no object dtype or pickle. Manifest artifact byte
lengths and SHA-256 digests match the committed files. Manifest self-hashes use
the canonicalization rule from the contract with
`integrity.manifest_sha256` omitted during hashing.

`python tools/generate_run_replay_examples.py` rebuilds the replay and
verification examples from `nsgaii-success/` and normalizes the environment of
the base examples to the fixed example version.

All artifact references use relative POSIX paths. No example contains a secret,
personal path, hostname, account, or executable source payload.
