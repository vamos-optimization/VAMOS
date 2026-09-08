# Hosting and domains

VAMOS uses Cloudflare Workers Static Assets as the target public host for the documentation portal. The version/archive model remains independent of the hosting provider, so immutable documentation stays under `docs/<version>/` and the moving stable alias stays under `docs/stable/`.

## Public host contract

The primary canonical hostname is:

`https://vamos-optimization.org/`

The production Worker is `vamos-docs`. Its Wrangler configuration declares four Custom Domains:

| Host | Behavior |
| --- | --- |
| `vamos-optimization.org` | Primary documentation origin. |
| `www.vamos-optimization.org` | Permanent redirect to the apex `.org` host. |
| `vamos-optimization.dev` | Permanent redirect to the apex `.org` host. |
| `www.vamos-optimization.dev` | Permanent redirect to the apex `.org` host. |

Redirects preserve the path and query string. Root, `docs/`, legacy `latest/`, and legacy root-level semantic-version routes are normalized at the Worker edge to the versioned portal routes prepared in [Documentation versions and archive](documentation-versioning.md).

Cloudflare Custom Domains are used because the Worker is the origin. When production deployment is authorized, Cloudflare creates the corresponding DNS records and certificates for zones already active in the same account.

## Pull request previews

Pull-request builds remain privilege-separated. `.github/workflows/docs-preview.yml` builds and validates the portal with read-only repository permissions and uploads it as data. It never receives Cloudflare credentials.

After this hosting infrastructure is present on `main`, `.github/workflows/docs-cloudflare-preview.yml` may consume a successful preview artifact through `workflow_run`. The privileged workflow checks out only trusted `main` hosting code, treats the downloaded portal as untrusted static data, validates it again, and publishes it through the dedicated `vamos-docs-preview` Worker.

Each same-repository pull request receives a stable alias of the form:

`https://pr-<number>-vamos-docs-preview.<account-subdomain>.workers.dev`

The alias is updated when the pull request receives new commits. Fork pull requests are not published to Cloudflare. If Cloudflare credentials are not configured, the downloadable GitHub Actions artifact remains available and the privileged publisher exits without deploying.

## Production publication

`.github/workflows/docs-cloudflare.yml` is deliberately manual. It builds the canonical portal with `https://vamos-optimization.org/` as its base URL, verifies the generated artifact, preserves earlier immutable versions through the same archive handoff used by the GitHub Pages bridge, and only then runs Wrangler.

The production job uses the `cloudflare-production` GitHub environment as an approval boundary. It does not run on pull requests or arbitrary pushes.

The repository expects two GitHub secrets:

- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN`

Use a scoped Cloudflare token suitable for Workers deployment. The Cloudflare `Edit Cloudflare Workers` token template includes Workers Scripts write access at account scope and Workers Routes write access at zone scope; restrict resources to the VAMOS account and the `vamos-optimization.org` / `vamos-optimization.dev` zones where possible.

## Migration boundary

GitHub Pages remains the publication bridge until the Cloudflare production workflow has been run successfully and the custom domains have been verified. The existing `site_url` and package Documentation metadata therefore remain on the GitHub Pages URL during this Goal; they should move to `https://vamos-optimization.org/` only after the canonical host is demonstrably live.

This avoids publishing metadata that points to an unavailable host and makes the final cutover a small, independently verifiable change rather than coupling it to infrastructure creation.
