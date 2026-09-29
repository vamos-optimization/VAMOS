# Documentation versions and archive

VAMOS publishes documentation as a versioned scientific reference rather than treating the current website as the only copy. The public canonical host is `https://vamos-optimization.org/`, while immutable released documentation remains available independently of the current-site URLs.

## Route contract

A built portal has two kinds of documentation routes:

| Route | Meaning |
| --- | --- |
| `/...` | Current documentation for the release designated stable, exposed at clean user-facing URLs. |
| `docs/<version>/...` | Immutable documentation for one released semantic version. |

For example, the current NSGA-II guide is published at:

`https://vamos-optimization.org/algorithms/nsgaii/`

A page that is part of the frozen VAMOS 1.0.0 archive remains available at its release-qualified path, for example:

`https://vamos-optimization.org/docs/1.0.0/guide/getting-started/`

The current site may acquire documentation pages after an immutable release archive was first published; those newer pages are not backfilled into the historical `docs/<version>/` tree.

The root current site is intentionally the normal browsing and search-engine surface. Users therefore do not need to carry an implementation-oriented `docs/stable` prefix through every URL.

The former `docs/stable/...` and `latest/...` routes remain compatibility aliases and permanently redirect to the corresponding clean current route. The `docs/` root also redirects to `/`. Existing links therefore continue to work while new links use the clean route scheme.

There is deliberately no public `docs/dev/` channel. Development previews belong to CI/preview infrastructure, where they are temporary and branch-specific instead of becoming a second quasi-stable documentation surface.

## Legacy routes

For every retained release, the builder produces redirect mirrors from the old root-level `<version>/...` route into `docs/<version>/...`.

The old moving aliases are normalized as follows:

- `docs/stable/...` -> `/...`
- `latest/...` -> `/...`
- `docs/` -> `/`

## Withdrawn routes

A small, explicit list of routes is withdrawn from every published tree, including carried-forward immutable archives. It is `WITHDRAWN_ROUTES` in `tools/check_docs_portal.py`:

- `audit/` and `topics/engineering_audit/`: raw internal audit evidence, which belongs outside the product tree rather than in published documentation;
- `website/`: the retired legacy multilingual site, superseded by the canonical documentation.

When the builder carries an archive forward, it deletes these routes and removes their entries from that archive's `sitemap.xml`, `sitemap.xml.gz`, and search index. Every other archived byte is left unchanged. The portal checker rejects any tree, sitemap, or search index that still references a withdrawn route. Withdrawn URLs therefore return the site's 404 page, and the live verifier `tools/check_docs_live.py` confirms that after every production deployment. Adding a route to this list is a deliberate publication decision, not a general way to edit released documentation.

## Archive preservation

`tools/build_release_docs.py` accepts an optional `--archive-from` artifact. When supplied, every immutable semantic-version directory already present under `docs/<version>/` is copied forward before publication. If the requested stable version already exists in that trusted artifact, the builder reuses that directory unchanged rather than rebuilding it from the current checkout. If the requested version is new, only that new immutable tree is built after older archives have been copied.

This distinction is essential because the clean current tree is mutable while a released `docs/<version>/` tree is not. A same-version documentation-layout change may update `/...`, `docs/stable/...`, `latest/...`, redirects, navigation, or hosting behavior, but it must not rewrite the historical bytes already published under `docs/<version>/`, except to remove the [withdrawn routes](#withdrawn-routes).

Archive preservation is therefore an explicit deployment input rather than relying on the hosting provider to retain files after a deployment. Production and manual republish workflows require a prior trusted portal artifact. The initial release-tag path can create the first immutable archive when no previous publication exists.

The current clean tree and an immutable `docs/<version>/` tree are produced separately so each emits the correct canonical URLs. The current tree points to clean root paths, while each immutable archive points to its version-qualified path.

## Hosting independence

The builder accepts `--base-url`, but its default is now the canonical public base:

`https://vamos-optimization.org/`

The current documentation entry point is therefore simply:

`https://vamos-optimization.org/`

Cloudflare Workers is the primary production host. GitHub Pages remains available as a secondary mirror, but its generated canonical metadata also points to the `.org` domain so search engines and citations converge on one public identity.
