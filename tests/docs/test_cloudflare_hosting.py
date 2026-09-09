from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_json(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_cloudflare_production_worker_owns_primary_and_redirect_hosts() -> None:
    config = _load_json("wrangler.jsonc")

    assert config["name"] == "vamos-docs"
    assert config["main"] == "./cloudflare/worker.js"
    assert config["workers_dev"] is False
    assert config["preview_urls"] is False
    assert config["compatibility_date"] == "2026-09-08"

    routes = config["routes"]
    assert isinstance(routes, list)
    assert {item["pattern"] for item in routes} == {
        "vamos-optimization.org",
        "www.vamos-optimization.org",
        "vamos-optimization.dev",
        "www.vamos-optimization.dev",
    }
    assert all(item["custom_domain"] is True for item in routes)

    assets = config["assets"]
    assert isinstance(assets, dict)
    assert assets == {
        "directory": "./public",
        "binding": "ASSETS",
        "run_worker_first": True,
        "not_found_handling": "404-page",
        "html_handling": "auto-trailing-slash",
    }


def test_cloudflare_preview_worker_has_no_custom_domains() -> None:
    config = _load_json("wrangler.preview.jsonc")

    assert config["name"] == "vamos-docs-preview"
    assert config["workers_dev"] is True
    assert config["preview_urls"] is True
    assert "routes" not in config
    assets = config["assets"]
    assert isinstance(assets, dict)
    assert assets["directory"] == "./preview"
    assert assets["binding"] == "ASSETS"
    assert assets["run_worker_first"] is True


def test_edge_worker_preserves_version_routes_and_domain_redirect_policy() -> None:
    source = (ROOT / "cloudflare" / "worker.js").read_text(encoding="utf-8")

    assert 'const CANONICAL_HOST = "vamos-optimization.org";' in source
    assert '"www.vamos-optimization.org"' in source
    assert '"vamos-optimization.dev"' in source
    assert '"www.vamos-optimization.dev"' in source
    assert 'return "/docs/stable/";' in source
    assert 'pathname.startsWith("/latest/")' in source
    assert "LEGACY_VERSION" in source
    assert "Response.redirect(target.toString(), 308)" in source
    assert "env.ASSETS.fetch(request)" in source


def test_cloudflare_preview_uses_privilege_separation() -> None:
    build = (ROOT / ".github" / "workflows" / "docs-preview.yml").read_text(encoding="utf-8")
    publish = (ROOT / ".github" / "workflows" / "docs-cloudflare-preview.yml").read_text(encoding="utf-8")

    assert yaml.load(build, Loader=yaml.BaseLoader) is not None
    assert yaml.load(publish, Loader=yaml.BaseLoader) is not None
    assert "CLOUDFLARE_API_TOKEN" not in build
    assert "docs-preview-meta-${{ github.event.pull_request.number }}" in build
    assert "workflow_run:" in publish
    assert "pull_request_target" not in publish
    assert "head_repository.full_name == github.repository" in publish
    assert "ref: main" in publish
    assert "persist-credentials: false" in publish
    assert "Revalidate generated portal as untrusted data" in publish
    assert "--preview-alias \"$alias\"" in publish
    assert "wrangler@${WRANGLER_VERSION}" in publish
    assert 'WRANGLER_VERSION: "4.129.1"' in publish


def test_cloudflare_production_deploy_is_manual_and_guarded() -> None:
    workflow = (ROOT / ".github" / "workflows" / "docs-cloudflare.yml").read_text(encoding="utf-8")
    parsed = yaml.load(workflow, Loader=yaml.BaseLoader)

    assert parsed is not None
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "push:" not in workflow
    assert "cloudflare-production" in workflow
    assert "CLOUDFLARE_ACCOUNT_ID" in workflow
    assert "CLOUDFLARE_API_TOKEN" in workflow
    assert "https://vamos-optimization.org/" in workflow
    assert "tools/check_docs_portal.py" in workflow
    assert "wrangler.jsonc" in workflow
    assert 'WRANGLER_VERSION: "4.129.1"' in workflow


def test_hosting_contract_uses_proven_canonical_domain_and_keeps_pages_as_mirror() -> None:
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    hosting = (ROOT / "docs" / "project" / "hosting.md").read_text(encoding="utf-8")

    assert "site_url: https://vamos-optimization.org/docs/stable/" in mkdocs
    assert 'Documentation = "https://vamos-optimization.org/"' in pyproject
    assert "VAMOS 1.0.0 has been deployed" in hosting
    assert "GitHub Pages remains available as a fallback mirror" in hosting
