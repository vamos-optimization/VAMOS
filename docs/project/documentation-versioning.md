# Documentation versions and archive

VAMOS publishes documentation as a versioned scientific reference rather than treating the current website as the only copy. The release builder now prepares a portable portal layout that can remain on GitHub Pages today and move behind the project domains later without changing the version model.

## Route contract

A built portal has two canonical documentation routes:

| Route | Meaning |
| --- | --- |
| `docs/<version>/` | Immutable documentation for one released semantic version. |
| `docs/stable/` | Mutable alias for the release currently designated stable. |

The portal root and `docs/` root redirect to `docs/stable/`. The stable tree is byte-for-byte copied from the current immutable release, while its canonical metadata continues to point at the immutable version. This keeps citations and indexed references anchored to a specific release.

There is deliberately no public `docs/dev/` channel in this Goal. Development previews belong to CI/preview infrastructure, where they can be temporary and branch-specific instead of becoming a second quasi-stable documentation surface.

## Legacy routes

Existing GitHub Pages links remain part of the migration contract. For every retained release, the builder produces redirect mirrors from the old `<version>/...` route into `docs/<version>/...`. The old `latest/...` route redirects into `docs/stable/...`.

The legacy `website/` tree remains separately built for now. It is not promoted to a canonical source, and Goal 5 does not remove its URLs.

## Archive preservation

`tools/build_release_docs.py` accepts an optional `--archive-from` artifact. When supplied, immutable `docs/<version>/` directories from that previous portal are copied forward before the current release is built. The generated `docs/versions.json` records the retained versions and the current stable release.

This makes archive preservation an explicit input rather than relying on the hosting provider to retain files after a deployment. The CI/deployment Goal is responsible for supplying the previous trusted artifact when publishing a newer release.

## Hosting independence

The builder also accepts `--base-url`. Its default remains the current GitHub Pages base URL, so the existing deployment workflow is still valid. A later deployment can build the same route tree for `https://vamos-optimization.org/` without changing documentation source paths.

For the current GitHub Pages base, a newly deployed artifact prepared by this builder would expose the stable portal at:

`https://vamos-optimization.github.io/VAMOS/docs/stable/`

This page describes the artifact contract; merging this Goal does not itself trigger a deployment, change DNS, or deploy to Cloudflare.
