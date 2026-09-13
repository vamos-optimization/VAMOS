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

while the immutable VAMOS 1.0.0 copy is published at:

`https://vamos-optimization.org/docs/1.0.0/algorithms/nsgaii/`

The root current site is intentionally the normal browsing and search-engine surface. Users therefore do not need to carry an implementation-oriented `docs/stable` prefix through every URL.

The former `docs/stable/...` and `latest/...` routes remain compatibility aliases and permanently redirect to the corresponding clean current route. The `docs/` root also redirects to `/`. Existing links therefore continue to work while new links use the clean route scheme.

There is deliberately no public `docs/dev/` channel. Development previews belong to CI/preview infrastructure, where they are temporary and branch-specific instead of becoming a second quasi-stable documentation surface.

## Legacy routes

For every retained release, the builder produces redirect mirrors from the old root-level `<version>/...` route into `docs/<version>/...`.

The old moving aliases are normalized as follows:

- `docs/stable/...` -> `/...`
- `latest/...` -> `/...`
- `docs/` -> `/`

The legacy `website/` tree remains separately built for compatibility. It is not promoted to the canonical documentation source.

## Archive preservation

`tools/build_release_docs.py` accepts an optional `--archive-from` artifact. When supplied, every immutable semantic-version directory already present under `docs/<version>/` is copied forward before publication. If the requested stable version already exists in that trusted artifact, the builder reuses that directory unchanged rather than rebuilding it from the current checkout. If the requested version is new, only that new immutable tree is built after older archives have been copied.

This distinction is essential because the clean current tree is mutable while a released `docs/<version>/` tree is not. A same-version documentation-layout change may update `/...`, `docs/stable/...`, `latest/...`, redirects, navigation, or hosting behavior, but it must not rewrite the historical bytes already published under `docs/<version>/`.

Archive preservation is therefore an explicit deployment input rather than relying on the hosting provider to retain files after a deployment. Production and manual republish workflows require a prior trusted portal artifact. The initial release-tag path can create the first immutable archive when no previous publication exists.

The current clean tree and an immutable `docs/<version>/` tree are produced separately so each emits the correct canonical URLs. The current tree points to clean root paths, while each immutable archive points to its version-qualified path.

## Hosting independence

The builder accepts `--base-url`, but its default is now the canonical public base:

`https://vamos-optimization.org/`

The current documentation entry point is therefore simply:

`https://vamos-optimization.org/`

Cloudflare Workers is the primary production host. GitHub Pages remains available as a secondary mirror, but its generated canonical metadata also points to the `.org` domain so search engines and citations converge on one public identity.
