# Hosting and domains

VAMOS uses Cloudflare Workers Static Assets as the primary public host for the documentation portal. The version/archive model remains independent of the hosting provider: current documentation is served directly from the site root, while immutable released documentation remains under `docs/<version>/`.

## Public host contract

The canonical public hostname is:

`https://vamos-optimization.org/`

The production Worker is `vamos-docs`. Its Wrangler configuration declares four Custom Domains:

| Host | Behavior |
| --- | --- |
| `vamos-optimization.org` | Primary documentation origin. |
| `www.vamos-optimization.org` | Permanent redirect to the apex `.org` host. |
| `vamos-optimization.dev` | Permanent redirect to the apex `.org` host. |
| `www.vamos-optimization.dev` | Permanent redirect to the apex `.org` host. |

Redirects preserve the path and query string. The current documentation is served directly at clean paths such as `/algorithms/nsgaii/`. Legacy `docs/stable/...`, `docs/`, and `latest/...` routes are normalized at the Worker edge to the equivalent clean current route. Legacy root-level semantic-version routes such as `/1.0.0/...` are normalized to the immutable `/docs/1.0.0/...` archive described in [Documentation versions and archive](documentation-versioning.md).

VAMOS 1.0.0 has been deployed through this Worker and the live verification gate covers the apex host, clean current documentation, immutable documentation routes, the version manifest, legacy redirects, and all three redirect hosts. Canonical project metadata therefore points to the `.org` domain.

## Pull request previews

Pull-request builds remain privilege-separated. `.github/workflows/docs-preview.yml` builds and validates the portal with read-only repository permissions and uploads it as data. It never receives Cloudflare credentials.

`.github/workflows/docs-cloudflare-preview.yml` consumes a successful preview artifact through `workflow_run`. The privileged workflow checks out only trusted `main` hosting code, treats the downloaded portal as untrusted static data, validates it again, and publishes it through the dedicated `vamos-docs-preview` Worker.

Each same-repository pull request receives a stable alias of the form:

`https://pr-<number>-vamos-docs-preview.<account-subdomain>.workers.dev`

The alias is updated when the pull request receives new commits. Fork pull requests are not published to Cloudflare. Canonical metadata inside previews points to the production `.org` domain rather than the preview hostname.

## Production publication

`.github/workflows/docs-cloudflare.yml` is deliberately manual. It builds the canonical portal with `https://vamos-optimization.org/` as its base URL, verifies the generated artifact, preserves earlier immutable versions through the same archive handoff used by the GitHub Pages mirror, deploys Wrangler, and then verifies the real public host before reporting success.

The production job uses the `cloudflare-production` GitHub environment as an approval boundary. It does not run on pull requests or arbitrary pushes.

The repository expects two GitHub secrets:

- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN`

Use a scoped Cloudflare token suitable for Workers deployment and restrict resources to the VAMOS account and the `vamos-optimization.org` / `vamos-optimization.dev` zones where possible.

A separate read-only `Verify Cloudflare documentation host` workflow can repeat the live verification without redeploying or requiring Cloudflare credentials.

## Canonical metadata contract

The production host has passed live verification, and the canonical metadata contract is now:

- package Documentation URL: `https://vamos-optimization.org/`;
- canonical MkDocs current-site URL: `https://vamos-optimization.org/`;
- immutable release URLs: `https://vamos-optimization.org/docs/<version>/...`;
- legacy multilingual site URL: `https://vamos-optimization.org/website/`;
- release, preview, and fallback artifacts emit current canonical links at clean root paths and immutable canonical links under `docs/<version>/`.

GitHub Pages remains available as a fallback mirror and archive-delivery mechanism, but it is no longer the canonical documentation origin. Keeping the mirror does not change the public identity of the project because its generated canonical metadata points back to `.org`.
