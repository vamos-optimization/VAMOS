# Release verification

## Canonical repository and publisher identity

Release work originates from [vamos-optimization/VAMOS](https://github.com/vamos-optimization/VAMOS).
The [repository governance policy](project/repository-governance.md) defines the
personal mirror and the one-way synchronization of canonical history.
Only the organization repository may recover publication input, request OIDC
publishing credentials, or create official releases. The `upload_pypi.yml`
repository guard rejects mirrors and forks before artifact recovery; every
subsequent publication job depends on that guard.

After the fresh official freeze, a maintainer must configure these Trusted
Publisher identities through the TestPyPI and PyPI account interfaces:

| Registry | Project | Owner | Repository | Workflow | Environment |
| --- | --- | --- | --- | --- | --- |
| TestPyPI | vamos-optimization | vamos-optimization | VAMOS | upload_pypi.yml | testpypi |
| PyPI | vamos-optimization | vamos-optimization | VAMOS | upload_pypi.yml | pypi |

Do not authorize the personal mirror or introduce API tokens. Create the
`testpypi` and `pypi` GitHub environments in the organization repository, restrict
deployment to `v1.0.0`, and require production approval where supported.
Confirm both publisher registrations before creating the official release tag.

Before the public release, enable **Private vulnerability reporting** under
the canonical repository's **Settings > Advanced Security**, so the security
policy's private reporting link is available. See GitHub's
[repository reporting setup](https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/configure-vulnerability-reporting/configure-for-a-repository).

## Fresh official 1.0.0 freeze

The artifact freeze made under the personal-repository identity is superseded.
Its wheel, sdist, checksums, SBOM, provenance, artifact manifest, release-check
report, and frozen Actions artifact are historical validation evidence only.
Do not reuse those artifacts or rerun the superseded workflow run. Validation
artifacts from the repository cutover are also not the authoritative freeze.

Use the final merged organization `main` commit as the immutable release source.
The release date in `CITATION.cff`, the changelog, release notes, and version
consistency test is synchronized to `2026-09-06`. Verify that this remains the
actual publication date before tagging or uploading. If publication moves to
another date, update those surfaces through a reviewed commit and create a new
freeze from the resulting canonical `main` commit. Keep version `1.0.0`.
Build new distributions and regenerate all evidence from the final source
commit with `repository = vamos-optimization/VAMOS` in provenance. Run full
local and hosted release validation before configuring publishers, creating
the annotated tag, and publishing the identical validated bytes to TestPyPI,
then PyPI, then the organization GitHub Release.
After publication and public verification succeed, fast-forward the personal
mirror to canonical `main` and copy the existing official annotated tag.

## Organization Pages

In the organization repository, open **Settings > Pages > Build and deployment**
and set **Source** to **GitHub Actions**. The guarded `docs.yml` workflow builds
both documentation sites strictly and deploys documentation only to
`https://vamos-optimization.github.io/VAMOS/`. It runs on the official tag or a
manual workflow dispatch. The generated portal mirrors the canonical `.org`
layout: the current documentation is built at clean root paths, immutable
release documentation lives under `docs/<version>/`, and the multilingual
website lives under `website/`. Compatibility trees under `docs/stable/`,
`latest/`, and root-level `<version>/` contain redirects rather than a second
canonical documentation surface. The personal mirror must not deploy canonical
Pages. Confirm the deployment and canonical `.org` metadata during the final
release Goal.

`tools/build_release_docs.py --version 1.0.0 --output <new-directory>` builds
the complete deployment layout. It builds the current documentation and the
immutable release separately so each has the correct canonical URL, preserves
previous immutable `docs/<version>/` trees when an archive input is supplied,
generates the compatibility redirects, and refuses an existing output
directory. The documentation smoke tests verify that every emitted canonical
URL resolves to a file in that deployment layout.

## Canonical validation

`tools/release_check.py` is the canonical, fail-closed VAMOS release gate. It
validates the source tree, stable contracts, typing policy, full test suite,
both documentation sites, distributions, installed wheel, dependencies, and
release evidence in one command:

```bash
python tools/release_check.py --version 1.0.0
```

Run it from a clean release branch with the development, documentation,
compute, and release tooling installed. Keep a second dependency-minimal
environment for the canonical typing and health gates; optional compute,
analysis, and Studio distributions intentionally do not enter that environment.
Build and release tools are pinned and audited separately from runtime
dependencies:

```bash
python -m pip install -c constraints/ci.txt -e ".[dev,docs,compute,analysis,examples,studio]"
python -m pip install -r release/requirements-build.txt
python -m pip install -r release/requirements-tools.txt
python -m venv <typing-venv>
<typing-venv>/bin/python -m pip install -r release/requirements-build.txt
<typing-venv>/bin/python -m pip install --no-build-isolation -c constraints/ci.txt -e ".[dev]"
```

For an auditable candidate, bind the check to the intended identity and write
evidence outside the repository:

```bash
python tools/release_check.py \
  --version 1.0.0 \
  --expected-branch release/final-1.0.0 \
  --expected-commit <full-commit-sha> \
  --tag-state pre-tag \
  --typing-python <typing-venv>/bin/python \
  --output-dir <new-empty-evidence-directory>
```

Use `--json` when a machine consumer requires exactly one JSON document on
standard output. `--list-checks` exposes the ordered check inventory. Evidence
directories are append-never: the checker refuses to overwrite a non-empty
directory.

## Frozen distributions

The compatibility gate runs all tests under `tests/compatibility/`, including
the permanent structural snapshots and the 1.0.0 run/study artifact corpus.
Those checks load, verify, inspect, and summarize the committed canonical bytes
and compare public study JSON command results without regenerating fixtures.
Fixture maintenance is documented in `tests/compatibility/v1_0_0/README.md`.

The candidate workflow builds one wheel and one sdist exactly once. Every
downstream platform downloads those same bytes. When validating an existing
candidate, pass its directory instead of rebuilding it:

```bash
python tools/release_check.py \
  --version 1.0.0 \
  --artifacts <directory-with-one-wheel-and-one-sdist> \
  --output-dir <new-empty-evidence-directory>
```

The evidence contains SHA-256 checksums, a CycloneDX SBOM, the installed
runtime lock, dependency-audit reports, artifact metadata/content inspection,
and provenance tied to the source commit. Publication consumes the frozen
workflow artifact and verifies every byte again; it never rebuilds from the
tag.

## Installed-wheel smoke

`tools/release_smoke.py` is designed to run from a clean, non-editable wheel
environment with no repository import path. Its full mode exercises public
imports, optimization, byte-exact replay, durable study execution, controlled
failure/resume/retry, relocation, and human and JSON CLI contracts while
blocking network access.

The release checker clears the checkout's `PYTHONPATH` for wheel installation,
dependency inventory, and smoke execution. This prevents source-tree metadata
from making pip treat an uninstalled wheel as already installed. Run the full
smoke with:

```bash
python tools/release_smoke.py --version 1.0.0 --mode full
```

`--mode core` omits the optional-backend failure scenario and is used by the
minimal and non-primary platform jobs. The hosted matrix covers Linux, Windows,
and macOS; minimum Python, primary Python, and the declared maximum tested
Python; a minimal wheel and the supported optional dependency set.

## Gate policy

Strict and stable typing must be clean. Full-source typing must match its exact
no-regression baseline; `full-zero` remains informational for VAMOS 1.0.0 and
does not claim that inherited typing debt is resolved. Runtime dependency
findings block the release. Build and release-tool audits are recorded
separately and surface findings as warnings for owner review; the pinned 1.0.0
tool sets are expected to audit clean.

Ruff lint is clean across production, tests, and release tools. Ruff formatting
uses the repository's explicit ratcheted formatting budget, so this release does
not disguise inherited formatting debt as global formatter compliance.

Use `--tag-state pre-normalization` only while the archived internal tags still
exist. After history normalization and before the first official tag, use
`--tag-state pre-tag`; it requires that no public remote version tag exists and
refuses a local candidate tag while allowing explicitly archived local history.
After the official tag is created on the release commit, use `--tag-state normalized`.
Do not publish if repository identity, artifact hashes, TestPyPI installation,
or any critical gate differs from the frozen candidate.

TestPyPI is published first using trusted publishing. Only after its exact
wheel installs and passes the full smoke may the same immutable wheel and sdist
be sent to PyPI and attached to the GitHub release with their checksums, SBOM,
manifest, provenance, stability contract, and known limitations.
