# Documentation versions and archive

VAMOS publishes documentation as a versioned scientific reference rather than treating the current website as the only copy. The public canonical host is now `https://vamos-optimization.org/`, while the version/archive model remains independent of the hosting provider.

## Route contract

A built portal has two canonical documentation routes:

| Route | Meaning |
| --- | --- |
| `docs/<version>/` | Immutable documentation for one released semantic version. |
| `docs/stable/` | Mutable alias for the release currently designated stable. |

The portal root and `docs/` root redirect to `docs/stable/`. The stable tree is byte-for-byte copied from the current immutable release, while its canonical metadata continues to point at the immutable version. This keeps citations and indexed references anchored to a specific release.

There is deliberately no public `docs/dev/` channel. Development previews belong to CI/preview infrastructure, where they are temporary and branch-specific instead of becoming a second quasi-stable documentation surface.

## Legacy routes

For every retained release, the builder produces redirect mirrors from the old `<version>/...` route into `docs/<version>/...`. The old `latest/...` route redirects into `docs/stable/...`.

The legacy `website/` tree remains separately built for compatibility. It is not promoted to the canonical documentation source.

## Archive preservation

`tools/build_release_docs.py` accepts an optional `--archive-from` artifact. When supplied, immutable `docs/<version>/` directories from that previous portal are copied forward before the current release is built. The generated `docs/versions.json` records the retained versions and the current stable release.

This makes archive preservation an explicit input rather than relying on the hosting provider to retain files after a deployment. Release delivery must supply the previous trusted artifact when publishing a newer release.

## Hosting independence

The builder accepts `--base-url`, but its default is now the canonical public base:

`https://vamos-optimization.org/`

The stable entry point is therefore:

`https://vamos-optimization.org/docs/stable/`

Cloudflare Workers is the primary production host. GitHub Pages remains available as a secondary mirror, but its generated canonical metadata also points to the `.org` domain so search engines and citations converge on one public identity.
