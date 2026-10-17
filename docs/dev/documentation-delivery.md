# Documentation delivery

VAMOS keeps documentation generation, preview validation, privileged preview publication, production deployment, and fallback mirroring as separate stages. This page describes the CI contract around the clean current site plus immutable archives in [Documentation versions and archive](../project/documentation-versioning.md) and the hosting contract in [Hosting and domains](../project/hosting.md).

## Pull request preview artifact

Changes that affect the documentation portal trigger `.github/workflows/docs-preview.yml`. The workflow installs the constrained documentation environment, builds the complete portable portal with `tools/build_release_docs.py`, validates the generated route/canonical contract with `tools/check_docs_portal.py`, and uploads the resulting static tree as a short-lived GitHub Actions artifact.

The preview build uses `https://vamos-optimization.org/` as its canonical base. The build workflow has read-only repository permissions and **does not receive Cloudflare credentials**. It also uploads a small metadata artifact containing only the pull-request number, head commit, and source repository so a later trusted workflow can bind the static artifact to the workflow run that produced it.

The read-only build workflow does not deploy.

Pull requests that change `release/requirements-tools.txt` also run the portal preview and Zensical compatibility workflows, so release-tool updates receive both documentation checks before merging.

Changes to either Cloudflare publication workflow also trigger the read-only Zensical compatibility build. This validates documentation compatibility before merging hosting-action updates without requiring a production deployment.

## Privileged Cloudflare preview publisher

`.github/workflows/docs-cloudflare-preview.yml` is triggered through `workflow_run` only after the read-only preview workflow succeeds. The workflow itself lives on trusted `main`, checks out `main` rather than pull-request code, downloads the prior run's artifacts into the runner temporary directory, validates their metadata and portal structure, and never executes files from the downloaded portal.

Only same-repository pull requests are eligible for publication. If `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` are configured, the trusted workflow uploads the validated static tree to the dedicated `vamos-docs-preview` Worker with a `pr-<number>` preview alias and posts or updates the resulting `workers.dev` URL on the pull request. Canonical metadata in that preview still points to the production `.org` domain. If the credentials are absent, the downloadable artifact remains the only preview surface.

This privilege split is intentional: Cloudflare credentials are never exposed to the workflow that builds pull-request code.

## Cloudflare production publication

`.github/workflows/docs-cloudflare.yml` is the canonical production path. It is manual, runs in the `cloudflare-production` GitHub environment, rebuilds the mutable current portal with `https://vamos-optimization.org/` as the canonical base, requires an explicit prior trusted portal artifact, validates the generated tree, deploys the `vamos-docs` Worker, and then verifies the live custom-domain contract.

Publishing a package to PyPI or merging release metadata does not update this
host. Documentation publication is a separate release step. Manual publication
and verification require an explicit version, without a preselected historical
release. Both publication workflows reject a requested version that differs
from the package installed from the checked-out source.

Before publishing a subsequent release:

1. Confirm the released PyPI version, source tag, and version named by the current
   installation guide. Run the initial examples against the installed PyPI wheel
   in a clean environment, without an editable checkout or `PYTHONPATH=src`.
2. Select a successful prior production run and its unexpired portal artifact as
   `archive_run_id` and `archive_artifact_name`. Verify the downloaded artifact's
   digest. A preview without the old archives is not an archive handoff source.
3. Build with `--archive-from` and run both portal checkers. Compare the retained
   `docs/<old-version>/` file hashes with the trusted artifact, allowing only the
   explicit withdrawn-route policy. Preserve the artifact for the next release.
4. Publish through this workflow with the selected version and archive inputs.
   Require the live check to pass, then confirm that the homepage, installation
   guide, and `docs/versions.json` identify the same stable release. If a retrieved
   page differs, compare it with the deployed artifact and workflow source SHA
   before attributing the discrepancy to a cache.

Source examples linked from the current [Examples](../examples.md) page use the
release tag matching the documented PyPI package. Historical version mentions
and figure provenance remain unchanged unless the corresponding result is
regenerated and verified.

The archive handoff is required on every production republish, including a same-version republish such as the clean-URL cutover of `1.0.0`. If the trusted prior artifact already contains `docs/<version>/`, the builder carries that immutable directory forward byte-for-byte and rebuilds only the mutable current tree and compatibility aliases. This prevents a documentation-only deployment from silently rewriting an already published scientific reference.

The production Wrangler configuration attaches `vamos-optimization.org`, `www.vamos-optimization.org`, `vamos-optimization.dev`, and `www.vamos-optimization.dev` as Custom Domains. Edge redirect logic sends the three aliases to the apex `.org` host while preserving path and query. On the apex host, `docs/stable/...`, `latest/...`, and `docs/` redirect to the corresponding clean current path; immutable `docs/<version>/...` routes remain unchanged.

## GitHub Pages fallback mirror

`.github/workflows/docs.yml` remains available as a secondary publication path and archive handoff source. It builds the same portal, but now uses `https://vamos-optimization.org/` as the canonical base before packaging the GitHub Pages artifact. This intentionally makes Pages a mirror rather than a competing canonical origin.

Before its Pages artifact can be deployed, the workflow verifies the generated clean-current, immutable, legacy-redirect, and canonical routes; uploads a reusable `vamos-docs-portal-<version>` artifact; and deploys only from the canonical repository job. A manual republish requires the prior trusted portal artifact for the same reason as Cloudflare production: already published immutable versions must be carried forward rather than regenerated. The original `v1.0.0` tag path remains able to create the first immutable archive when no prior publication exists.

The current tag trigger remains intentionally limited to `v1.0.0`; arbitrary later semantic-version tag deployment is not enabled without explicit archive preservation.

## Archive handoff for releases and republishes

Archive preservation is explicit through `--archive-from`. Production and manually dispatched fallback delivery require `archive_run_id` and `archive_artifact_name`. The supplied artifact is downloaded from the same repository and fed into the builder before the current site is produced.

When the prior artifact contains the requested semantic version, that same-version `docs/<version>/` directory is reused unchanged apart from the declared withdrawn routes described in [Documentation versions and archive](../project/documentation-versioning.md#withdrawn-routes). When the requested version is genuinely new, the builder creates its immutable tree after copying all older version directories. Therefore both later releases and same-version publication-layout changes preserve the already published archive contract.

This GitHub Actions artifact handoff is a migration-stage mechanism, not the final long-term archive store. A future retention policy may copy immutable releases to a durable object store, but no deployment is allowed to silently drop or regenerate an existing `docs/<version>/` tree.

## Shared validation

Preview, the GitHub Pages mirror, and Cloudflare production all use `tools/check_docs_portal.py`. The checker verifies the generated `docs/versions.json`, the clean current tree, immutable version trees, compatibility redirects, canonical targets, retained immutable version directories, and the absence of withdrawn routes from every published tree, sitemap, and search index.

Source-level Markdown links remain protected by strict MkDocs and Zensical builds. The portal checker is intentionally a generated-artifact gate: it catches publication-layout mistakes that source-only validation cannot see.

## Reader experience and publication identity

The preview also runs `tools/check_docs_site.py` against the generated current
tree. It checks local links and fragments, downloadable files, page language,
one main heading, image alternatives, description/social metadata, searchable
pages and search-result anchors, and recovery from the 404 page. Use
`--baseline <previous-built-site>` when changing structure to prove that prior
public routes and bookmarks remain available. Immutable archives keep their
original reader experience and are not subjected to new editorial requirements.

`tools/check_docs_browser.py` exercises the built portal in Chromium at 1440,
390 and 320 CSS pixels. Its isolated environment is pinned in
`tools/docs-browser-requirements.txt`; it adds no VAMOS runtime dependency.
The scenarios follow T01–T03 and T06–T08, verify a downloaded script's bytes,
search for algorithms and study terms, check a query with no results, operate
the mobile menu and wide tables by keyboard, inspect dark mode and reduced
motion, and recover from a nested missing address. Critical learning links and
download controls are also checked with JavaScript disabled. Reports and
screenshots go to the `docs-qa-<PR>` artifact, including failure evidence.
These are targeted accessibility checks, not a complete WCAG certification or
a claim of testing every mobile device.

For a local browser regression run after building the portal:

```bash
python -m venv /tmp/vamos-docs-browser
/tmp/vamos-docs-browser/bin/python -m pip install -r tools/docs-browser-requirements.txt
/tmp/vamos-docs-browser/bin/python -m playwright install --with-deps chromium
/tmp/vamos-docs-browser/bin/python tools/check_docs_browser.py --root public --output /tmp/vamos-docs-browser-evidence
```

Keep evidence outside the portal and select a new output directory. On Windows,
use the environment's `Scripts/python.exe` executable instead of `bin/python`.

Production publication additionally requires `expected_source_sha`, the full
reviewed commit. The workflow refuses a different checkout. Before deployment,
`tools/check_docs_artifact.py` records SHA-256 hashes for every portal file and
the source commit in a separate `vamos-docs-identity-<version>` artifact. After
the existing live route check, it compares **every current file** served by the
primary host with those recorded bytes, including HTML, search, scripts, styles,
figures and downloads. The verifier retries an individual mismatch or transport
failure three times, and fails the workflow if the host still differs. It does
not replace archive preservation, validate immutable archive bytes over HTTP,
automatically roll back a deployment, or accept a changed source commit.

Learning-path metadata and download controls live in each lesson's front matter;
the shared template renders their links relative to the current documentation
version. The 404 template uses root-relative links under that version's base,
so recovery from a deeply nested address also works on an isolated preview.
Search tokenization includes hyphen, underscore and dot separators for API
names, while preserving the slash in MOEA/D as part of one algorithm name.
Introductory lessons and routing pages receive a modest boost.
