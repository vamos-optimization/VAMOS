# Documentation delivery

VAMOS keeps documentation generation, preview validation, privileged preview publication, production deployment, and fallback mirroring as separate stages. This page describes the CI contract around the clean current site plus immutable archives in [Documentation versions and archive](../project/documentation-versioning.md) and the hosting contract in [Hosting and domains](../project/hosting.md).

## Pull request preview artifact

Changes that affect the documentation portal trigger `.github/workflows/docs-preview.yml`. The workflow installs the constrained documentation environment, builds the complete portable portal with `tools/build_release_docs.py`, validates the generated route/canonical contract with `tools/check_docs_portal.py`, and uploads the resulting static tree as a short-lived GitHub Actions artifact.

The preview build uses `https://vamos-optimization.org/` as its canonical base. The build workflow has read-only repository permissions and **does not receive Cloudflare credentials**. It also uploads a small metadata artifact containing only the pull-request number, head commit, and source repository so a later trusted workflow can bind the static artifact to the workflow run that produced it.

The read-only build workflow does not deploy.

## Privileged Cloudflare preview publisher

`.github/workflows/docs-cloudflare-preview.yml` is triggered through `workflow_run` only after the read-only preview workflow succeeds. The workflow itself lives on trusted `main`, checks out `main` rather than pull-request code, downloads the prior run's artifacts into the runner temporary directory, validates their metadata and portal structure, and never executes files from the downloaded portal.

Only same-repository pull requests are eligible for publication. If `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` are configured, the trusted workflow uploads the validated static tree to the dedicated `vamos-docs-preview` Worker with a `pr-<number>` preview alias and posts or updates the resulting `workers.dev` URL on the pull request. Canonical metadata in that preview still points to the production `.org` domain. If the credentials are absent, the downloadable artifact remains the only preview surface.

This privilege split is intentional: Cloudflare credentials are never exposed to the workflow that builds pull-request code.

## Cloudflare production publication

`.github/workflows/docs-cloudflare.yml` is the canonical production path. It is manual, runs in the `cloudflare-production` GitHub environment, rebuilds the portal with `https://vamos-optimization.org/` as the canonical base, reuses the explicit prior-artifact handoff for releases after `1.0.0`, validates the generated tree, deploys the `vamos-docs` Worker, and then verifies the live custom-domain contract.

The production Wrangler configuration attaches `vamos-optimization.org`, `www.vamos-optimization.org`, `vamos-optimization.dev`, and `www.vamos-optimization.dev` as Custom Domains. Edge redirect logic sends the three aliases to the apex `.org` host while preserving path and query. On the apex host, `docs/stable/...`, `latest/...`, and `docs/` redirect to the corresponding clean current path; immutable `docs/<version>/...` routes remain unchanged.

## GitHub Pages fallback mirror

`.github/workflows/docs.yml` remains available as a secondary publication path and archive handoff source. It builds the same portal, but now uses `https://vamos-optimization.org/` as the canonical base before packaging the GitHub Pages artifact. This intentionally makes Pages a mirror rather than a competing canonical origin.

Before its Pages artifact can be deployed, the workflow verifies the generated clean-current, immutable, legacy-redirect, and canonical routes; uploads a reusable `vamos-docs-portal-<version>` artifact; and deploys only from the canonical repository job.

The current tag trigger remains intentionally limited to `v1.0.0`; arbitrary later semantic-version tag deployment is not enabled without explicit archive preservation.

## Archive handoff for later versions

Archive preservation is explicit through `--archive-from`. Both release delivery paths require `archive_run_id` and `archive_artifact_name` for a manually dispatched version after `1.0.0`. The supplied artifact is downloaded from the same repository and fed into the builder before the new current release is produced.

This GitHub Actions artifact handoff is a migration-stage mechanism, not the final long-term archive store. A future retention policy may copy immutable releases to a durable object store, but no deployment is allowed to silently drop an existing `docs/<version>/` tree.

## Shared validation

Preview, the GitHub Pages mirror, and Cloudflare production all use `tools/check_docs_portal.py`. The checker verifies the generated `docs/versions.json`, the clean current tree, immutable version trees, compatibility redirects, canonical targets, retained immutable version directories, and the separate legacy `website/` tree.

Source-level Markdown links remain protected by strict MkDocs and Zensical builds. The portal checker is intentionally a generated-artifact gate: it catches publication-layout mistakes that source-only validation cannot see.
