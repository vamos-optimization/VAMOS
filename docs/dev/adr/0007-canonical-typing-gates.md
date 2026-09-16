# ADR 0007: Canonical Typing Gates

## Status

Accepted

## Context

VAMOS had three incompatible typing signals: a selected CI scope, an unenforced total-count file, and full-source mypy in local health. They used no single reproducible interpretation of passing typing.

## Decision

- `tools/typecheck.py` is the sole typecheck entry point.
- Python 3.12, compiled mypy 2.3.1, typing-extensions 4.16.0, `pyproject.toml`, and `constraints/ci.txt` define the supported environment.
- Strict typing covers at least the former CI paths and requires zero diagnostics.
- Full development typing enforces one structured multiset baseline. Fingerprints exclude source locations but include path, error code, and normalized message; multiplicity cannot increase.
- Changed production files must contain no baseline debt.
- Resolved fingerprints are removed from the baseline in the same change.
- Stable typing checks every supported public facade and requires zero diagnostics.
- Release typing combines strict/stable zero, the exact full-source ratchet, and health.
- Full-zero checks all production source without accepting inherited diagnostics and remains the explicit debt-removal objective.
- Health and CI run the same strict and full commands. Release workflows run the composite release command.

## Consequences

Development health can pass while accurately reporting remaining structured debt, but it cannot hide a new diagnostic or stale allowance. VAMOS 1.0 can release only when strict and stable are zero, the full-source ratchet is exact, and health passes. Full-source zero remains visible and unfinished. Environment or configuration changes require explicit baseline review.
