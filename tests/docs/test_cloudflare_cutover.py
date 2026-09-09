from __future__ import annotations

import json
from pathlib import Path

import yaml

from tools.check_docs_live import PRIMARY_HOST, REDIRECT_HOSTS, ResponseSnapshot, check_live

ROOT = Path(__file__).resolve().parents[2]


def _snapshot(status: int, *, location: str | None = None, body: bytes = b"") -> ResponseSnapshot:
    headers = {} if location is None else {"location": location}
    return ResponseSnapshot(status=status, headers=headers, body=body)


def test_live_cutover_checker_accepts_expected_public_contract() -> None:
    version = "1.0.0"
    base_url = f"https://{PRIMARY_HOST}/"
    stable_url = f"{base_url}docs/stable/"
    immutable_url = f"{base_url}docs/{version}/"
    homepage = f'<link rel="canonical" href="{immutable_url}">'.encode()

    responses = {
        (PRIMARY_HOST, "/"): _snapshot(308, location=stable_url),
        (PRIMARY_HOST, "/latest/?cutover=1"): _snapshot(
            308,
            location=f"{stable_url}?cutover=1",
        ),
        (PRIMARY_HOST, f"/{version}/?cutover=1"): _snapshot(
            308,
            location=f"{immutable_url}?cutover=1",
        ),
        (PRIMARY_HOST, "/docs/versions.json"): _snapshot(
            200,
            body=json.dumps({"stable": version, "versions": [version]}).encode(),
        ),
        (PRIMARY_HOST, "/docs/stable/"): _snapshot(200, body=homepage),
        (PRIMARY_HOST, f"/docs/{version}/"): _snapshot(200, body=homepage),
    }
    for host in REDIRECT_HOSTS:
        responses[(host, "/docs/stable/?cutover=1")] = _snapshot(
            308,
            location=f"{stable_url}?cutover=1",
        )

    def fake_request(host: str, target: str, timeout: float) -> ResponseSnapshot:
        assert timeout == 5.0
        return responses[(host, target)]

    result = check_live(version, requester=fake_request, timeout=5.0)

    assert result["primary"] == base_url
    assert result["stable"] == version
    assert len(result["checked"]) == 9


def test_cutover_workflows_separate_deployment_from_read_only_reverification() -> None:
    production = (ROOT / ".github" / "workflows" / "docs-cloudflare.yml").read_text(encoding="utf-8")
    verify = (ROOT / ".github" / "workflows" / "docs-cloudflare-verify.yml").read_text(encoding="utf-8")

    assert yaml.load(production, Loader=yaml.BaseLoader) is not None
    assert yaml.load(verify, Loader=yaml.BaseLoader) is not None
    assert "Verify live custom-domain contract" in production
    assert "python tools/check_docs_live.py" in production
    assert "--attempts 24" in production

    assert "workflow_dispatch:" in verify
    assert "python tools/check_docs_live.py" in verify
    assert "CLOUDFLARE_ACCOUNT_ID" not in verify
    assert "CLOUDFLARE_API_TOKEN" not in verify
    assert "wrangler@" not in verify
    assert "pull_request:" not in verify
    assert "push:" not in verify


def test_post_cutover_contract_keeps_pages_only_as_fallback() -> None:
    hosting = (ROOT / "docs" / "project" / "hosting.md").read_text(encoding="utf-8")
    delivery = (ROOT / "docs" / "dev" / "documentation-delivery.md").read_text(encoding="utf-8")

    assert "## Canonical metadata cutover" in hosting
    assert "GitHub Pages remains available as a fallback mirror" in hosting
    assert "Cloudflare production publication" in delivery
    assert "GitHub Pages fallback mirror" in delivery
    assert "competing canonical origin" in delivery
