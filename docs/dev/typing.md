# Canonical typing policy

VAMOS uses one reproducible typecheck entry point: `tools/typecheck.py`. Direct mypy commands are implementation details of that tool, not additional validation contracts.

## Toolchain

The canonical typing environment is:

- Python 3.12;
- compiled mypy 1.15.0;
- typing-extensions 4.16.0;
- `pyproject.toml` as the only mypy configuration;
- `constraints/ci.txt` as the dependency constraint set;
- no separately installed stub distributions;
- no optional compute, research, analysis, Studio, tuning, or model-provider distribution in the typing environment.

Create or select a Python 3.12 environment and install exactly:

```bash
python -m pip install -c constraints/ci.txt -e ".[dev]"
```

The entry point rejects Python, mypy, build-kind, typing-extensions, stub-set, optional-distribution, configuration, or constraints drift that could silently change diagnostic identity. The mypy configuration owns narrow third-party import boundaries, so optional runtime extras do not belong in this environment. Every invocation disables mypy's incremental cache and anchors imports to this checkout's `src` directory.

## Scopes

Strict:

```bash
python tools/typecheck.py --scope strict
```

Strict covers at least the former CI inventory: algorithm configuration and registry, experiment configuration, evaluation, CLI common plumbing, optimization results, and unified optimization. It requires zero diagnostics.

Stable public API:

```bash
python tools/typecheck.py --scope stable
```

Stable checks the supported public facade modules and requires zero diagnostics.
The exact stable symbol inventory is frozen separately under
`tests/compatibility/v1_0_0/`.

Full development:

```bash
python tools/typecheck.py --scope full
```

Full runs over all of `src/vamos`. It compares normalized diagnostics with `typing/mypy-baseline.json` as a multiset. A fingerprint contains repository-relative path, error code, and normalized semantic message; line and column are retained only as location metadata. New fingerprints, increased multiplicity, new error-code families, stale resolved entries, environment drift, and baseline debt in a changed production file all fail.

Development full-source typing passes the structured no-regression ratchet. It
does not claim that the complete source tree is free of diagnostics.

Release:

```bash
python tools/typecheck.py --scope release
```

Release combines strict zero, stable-facade zero, an exact full-source ratchet,
and the canonical health suite. Both release workflows invoke this command
before building or publishing.

Full-source zero:

```bash
python tools/typecheck.py --scope full-zero
```

Full-zero runs over all production source and accepts no diagnostics. It remains
the explicit public debt-removal objective and is an informational failure for
VAMOS 1.0.0.

## Current structured debt

After establishing strict=0, the structured baseline contains 1,422 diagnostics in 150 files and 176 stable fingerprints:

| Layer | Diagnostics |
|---|---:|
| engine | 810 |
| foundation | 522 |
| ux | 67 |
| experiment | 22 |
| package root | 1 |

The dominant family is 1,399 `type-arg` diagnostics, primarily unparameterized NumPy arrays in packages that now run with strict mypy settings. The remaining families are small protocol/optional/narrowing/decorator issues. VAMOS 1.0.0 enforces zero typing errors on the strict/stable surface and an exact no-regression ratchet over the complete source tree. The complete source tree is not yet globally free of mypy diagnostics.

## Reduction and baseline updates

Every production file changed by a Goal must finish with zero diagnostics. Reduce common upstream causes before editing large numbers of annotations. After a safe reduction:

1. run strict and full; full must fail only because resolved entries remain in the baseline;
2. inspect the normalized removed diagnostics and confirm there are no new or increased entries;
3. update the baseline with the reviewed base commit:

   ```bash
   python tools/typecheck.py --scope full --update-baseline --generation-commit <reviewed-base-commit>
   ```

4. review and commit the source correction and baseline reduction together;
5. rerun strict, full, health, and the focused behavioral tests.

If the supported environment or effective configuration intentionally changes, add `--review-environment-change` to the update command after reviewing the diagnostic diff. The tool records the drift and still refuses new or increased diagnostics or debt in changed production files. Never replace this process with total-count edits, broad ignores, exclusions, disabled error codes, or `Any` inserted only to silence mypy.

The reduction order is: remaining non-NumPy structural errors; shared NumPy type aliases and protocols by layer; engine algorithms/operators; foundation numerical modules; UX; then verification with the explicit full-zero scope.

```agent-docs
path: tools/typecheck.py
path: typing/mypy-baseline.json
path: pyproject.toml
path: constraints/ci.txt
path: .github/workflows/ci.yml
path: .github/workflows/release.yml
path: .github/workflows/upload_pypi.yml
command: python tools/typecheck.py --scope strict
command: python tools/typecheck.py --scope stable
command: python tools/typecheck.py --scope full
command: python tools/typecheck.py --scope release
command: python tools/typecheck.py --scope full-zero
```
