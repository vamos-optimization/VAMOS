# Documentation delivery

VAMOS keeps documentation generation, preview validation, and public deployment as separate stages. This page describes the CI contract around the versioned portal prepared in [Documentation versions and archive](../project/documentation-versioning.md).

## Pull request preview artifact

Changes that affect the documentation portal trigger `.github/workflows/docs-preview.yml`. The workflow installs the constrained documentation environment, builds the complete portable portal with `tools/build_release_docs.py`, validates the generated route/canonical contract with `tools/check_docs_portal.py`, and uploads the resulting static tree as a short-lived GitHub Actions artifact.

The preview workflow has read-only repository permissions and **does not deploy**. It has no Pages write permission, no OIDC write permission, and no `deploy-pages` step. This keeps pull requests from mutating the public documentation site.

At this stage the preview is a downloadable build artifact rather than a public website URL. A real branch/PR preview URL belongs to the hosting work in Goal 7, where it can be isolated from the stable/version archive.

## Release portal gate

`.github/workflows/docs.yml` remains the GitHub Pages publication workflow. Before its Pages artifact can be deployed, the workflow now:

1. builds the versioned portal with the production base URL;
2. verifies the generated stable, immutable, legacy, and canonical routes;
3. uploads a reusable `vamos-docs-portal-<version>` artifact for audit and archive handoff;
4. uploads the separately packaged GitHub Pages artifact; and
5. deploys only from the canonical repository job.

The current tag trigger remains intentionally limited to `v1.0.0`. Goal 6 does not silently enable arbitrary future semantic-version tag deployments.

## Archive handoff for later versions

Goal 5 made archive preservation explicit through `--archive-from`. Goal 6 connects that input to manual release delivery without allowing a later version to overwrite the archive accidentally.

For a manually dispatched version after `1.0.0`, the release workflow requires both `archive_run_id` and `archive_artifact_name`. It downloads that prior trusted `vamos-docs-portal-<version>` artifact from the same repository and supplies it to the builder. Missing archive metadata fails before the new portal is built.

This GitHub Actions artifact handoff is a migration-stage mechanism, not the final long-term archive store. Goal 7 should replace or supplement it with the hosting pipeline's durable immutable-version storage before generic automatic release deployment is enabled.

## Shared validation

Both preview and release use `tools/check_docs_portal.py`. The checker verifies the generated `docs/versions.json`, stable/current release relationship, root and legacy redirects, canonical targets, retained immutable version directories, and the separate legacy `website/` tree.

Source-level Markdown links remain protected by strict MkDocs and Zensical builds. The portal checker is intentionally a generated-artifact gate: it catches publication-layout mistakes that source-only validation cannot see.

## Non-goals

Goal 6 does not switch the production generator from MkDocs to Zensical, change DNS, configure Cloudflare, or publish a custom-domain preview. Those remain hosting/deployment concerns for Goal 7.
