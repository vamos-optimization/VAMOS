# Documentation delivery

VAMOS keeps documentation generation, preview validation, privileged preview publication, production deployment, and the GitHub Pages redirect as separate stages. This page describes the CI contract around the clean current site plus immutable archives in [Documentation versions and archive](../project/documentation-versioning.md) and the hosting contract in [Hosting and domains](../project/hosting.md).

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

`.github/workflows/docs-cloudflare.yml` is the canonical production path. It is manual, runs in the `cloudflare-production` GitHub environment, rebuilds the mutable current portal with `https://vamos-optimization.org/` as the canonical base, requires one trusted archive source, validates the generated tree, deploys the `vamos-docs` Worker, verifies the live custom-domain contract, and then attaches the published portal to the GitHub Release of that version.

Publishing a package to PyPI or merging release metadata does not update this
host. Documentation publication is a separate release step. Manual publication
and verification require an explicit version, without a preselected historical
release. Production rejects a requested version that differs from the package
installed from the checked-out source, and it stops before building when the
GitHub Release `v<version>` does not exist.

Before publishing a subsequent release:

1. Confirm the released PyPI version, source tag, and version named by the current
   installation guide. Run the initial examples against the installed PyPI wheel
   in a clean environment, without an editable checkout or `PYTHONPATH=src`.
2. Select the archive source: `archive_release` set to the previous release tag,
   whose `vamos-docs-portal-<version>.tar.gz` asset holds every published
   `docs/<version>/` tree. If that release has no portal asset, use the run inputs
   described in [Recovery](#recovery) instead. A preview without the old archives
   is not an archive source.
3. Download that asset and its `.sha256` file, run `sha256sum --check`, build with
   `--archive-from <tarball>`, and run both portal checkers. Compare the retained
   `docs/<old-version>/` file hashes with the archive, allowing only the explicit
   withdrawn-route policy.
4. Publish through this workflow with the selected version and archive input.
   Require the live check to pass, then confirm that the homepage, installation
   guide, and `docs/versions.json` identify the same stable release. If a retrieved
   page differs, compare it with the deployed artifact and workflow source SHA
   before attributing the discrepancy to a cache.
5. Confirm that release `v<version>` now lists `vamos-docs-portal-<version>.tar.gz`
   and its `.sha256` file. That asset is the archive source of the next release.

Source examples linked from the current [Examples](../examples.md) page use the
release tag matching the documented PyPI package. Historical version mentions
and figure provenance remain unchanged unless the corresponding result is
regenerated and verified.

The archive handoff is required on every production republish, including a same-version republish such as the clean-URL cutover of `1.0.0`. If the archive source already contains `docs/<version>/`, the builder carries that immutable directory forward byte-for-byte and rebuilds only the mutable current tree and compatibility aliases. This prevents a documentation-only deployment from silently rewriting an already published scientific reference.

The production Wrangler configuration attaches `vamos-optimization.org`, `www.vamos-optimization.org`, `vamos-optimization.dev`, and `www.vamos-optimization.dev` as Custom Domains. Edge redirect logic sends the three aliases to the apex `.org` host while preserving path and query. On the apex host, `docs/stable/...`, `latest/...`, and `docs/` redirect to the corresponding clean current path; immutable `docs/<version>/...` routes remain unchanged.

## GitHub Pages redirect

`https://vamos-optimization.github.io/VAMOS/` serves no documentation. `.github/workflows/docs.yml` is dispatched manually, without inputs, and deploys two pages built by `tools/build_pages_redirect.py`:

- `index.html` declares `https://vamos-optimization.org/` as canonical, redirects there with a meta refresh, and shows a link to it;
- `404.html` answers every other Pages address. A small script maps `/VAMOS/<path>?<query>#<hash>` to `https://vamos-optimization.org/<path>?<query>#<hash>`. The production host already redirects `latest/...`, `docs/stable/...`, and root-level `<version>/...`, so an old Pages bookmark reaches the same page. Without JavaScript, the page links to `https://vamos-optimization.org/`.

Both pages declare `.org` as canonical, so Pages is not a competing canonical origin. The workflow builds no portal, downloads no archive, and is not an archive source. `tools/check_repository_identity.py` keeps it guarded to the canonical repository, manual, and limited to this redirect. Redeploy it only when the redirect changes.

## Archive handoff for releases and republishes

Archive preservation is explicit through `--archive-from`, which accepts a portal directory or a `.tar.gz` portal archive with `docs/` at its root. Production requires exactly one archive source: `archive_release`, or both `archive_run_id` and `archive_artifact_name`. A release asset is downloaded with its `.sha256` file and checked with `sha256sum --check`; an Actions artifact is downloaded from the same repository. The builder extracts a tarball with the `data` filter of `tarfile` and then uses it exactly as a directory, before the current site is produced.

When the archive source contains the requested semantic version, that same-version `docs/<version>/` directory is reused unchanged apart from the declared withdrawn routes described in [Documentation versions and archive](../project/documentation-versioning.md#withdrawn-routes). When the requested version is genuinely new, the builder creates its immutable tree after copying all older version directories. Therefore both later releases and same-version publication-layout changes preserve the already published archive contract.

The durable store is the GitHub Release of each version. Only after the live checks pass, production packages the published `public/` tree as `vamos-docs-portal-<version>.tar.gz` (sorted names, numeric owner 0, and the source commit time as every modification time, so the bytes depend only on the tree), checks that the archive extracts to the approved file inventory, writes `vamos-docs-portal-<version>.tar.gz.sha256`, and uploads both to release `v<version>` with `gh release upload --clobber`. Release assets do not expire; the `vamos-docs-cloudflare-<version>` and `vamos-docs-identity-<version>` Actions artifacts that the workflow still uploads expire after 90 days. A same-version republish replaces the asset, which is safe because the immutable trees inside it are carried forward byte-for-byte. No deployment is allowed to silently drop or regenerate an existing `docs/<version>/` tree.

### Recovery

- To rebuild or republish from the durable store, dispatch the production workflow with `archive_release` set to the tag whose asset holds the trees: the previous release for a new version, or the same release for a same-version republish.
- To inspect or rebuild an archive locally, run `gh release download v<version> --pattern 'vamos-docs-portal-*'`, check it with `sha256sum --check vamos-docs-portal-<version>.tar.gz.sha256`, and pass the tarball to `python tools/build_release_docs.py --archive-from`.
- Only the latest published release needs the asset, because each portal contains every earlier `docs/<version>/` tree. That release has none if it was deployed before production attached its portal, or if an upload failed: `--clobber` deletes the old asset before uploading the new one. Republish that version once with `archive_run_id` and `archive_artifact_name` naming a successful production run and its `vamos-docs-cloudflare-<version>` artifact, before that artifact expires; the deployment attaches the missing asset.

## Shared validation

Preview and Cloudflare production both use `tools/check_docs_portal.py`. The checker verifies the generated `docs/versions.json`, the clean current tree, immutable version trees, compatibility redirects, canonical targets, retained immutable version directories, and the absence of withdrawn routes from every published tree, sitemap, and search index.

Source-level Markdown links remain protected by strict MkDocs and Zensical builds. The portal checker is intentionally a generated-artifact gate: it catches publication-layout mistakes that source-only validation cannot see.

The GitHub Pages redirect holds no documentation tree, so no portal checker runs on it. `tools/check_repository_identity.py` checks its workflow, and the documentation tests check the generated pages and run the address mapping of `404.html`.

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
