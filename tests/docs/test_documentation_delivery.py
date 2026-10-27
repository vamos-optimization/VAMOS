from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
UPLOAD_ARTIFACT = "actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9"
DOWNLOAD_ARTIFACT = "actions/download-artifact@9000827ccba6bdab643e8b6fd33ac0654aef8333"
BASE_URL = "https://vamos-optimization.org/"
VERSION = "1.0.0"


def test_publication_rejects_version_that_differs_from_source() -> None:
    workflow = yaml.load((ROOT / ".github/workflows/docs-cloudflare.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    steps = [step for job in workflow["jobs"].values() for step in job["steps"]]
    guard = next(step for step in steps if step.get("name") == "Check requested documentation version against source")
    commands = guard["run"]
    source = commands.split("<<'PYVERSION'\n", 1)[1].split("\nPYVERSION", 1)[0]
    installed = version("vamos-optimization")
    for requested, succeeds in ((installed, True), (f"{installed}-mismatch", False)):
        result = subprocess.run(
            [sys.executable, "-c", source],
            env={**os.environ, "DOC_VERSION": requested},
            capture_output=True,
            text=True,
            check=False,
        )
        assert (result.returncode == 0) is succeeds, result.stderr
        if not succeeds:
            assert "differs from source version" in result.stderr


def _artifact_action_steps(action: str) -> list[dict[str, object]]:
    steps: list[dict[str, object]] = []
    for workflow_path in (ROOT / ".github" / "workflows").glob("*.yml"):
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        for job in workflow.get("jobs", {}).values():
            for step in job.get("steps", []):
                if str(step.get("uses", "")).startswith(f"actions/{action}-artifact@"):
                    steps.append(step)
    return steps


def test_artifact_transport_actions_are_pinned_and_archive_compatible() -> None:
    upload_steps = _artifact_action_steps("upload")
    download_steps = _artifact_action_steps("download")

    assert upload_steps
    assert download_steps
    assert {step["uses"] for step in upload_steps} == {UPLOAD_ARTIFACT}
    assert {step["uses"] for step in download_steps} == {DOWNLOAD_ARTIFACT}

    for step in upload_steps:
        inputs = step.get("with", {})
        assert isinstance(inputs, dict)
        assert str(inputs.get("archive", "true")).lower() != "false"

    for step in download_steps:
        inputs = step.get("with", {})
        assert isinstance(inputs, dict)
        assert str(inputs.get("digest-mismatch", "error")).lower() == "error"


def test_preview_workflow_is_read_only_and_artifact_only() -> None:
    workflow = (ROOT / ".github" / "workflows" / "docs-preview.yml").read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "runs-on: ubuntu-24.04" in workflow
    assert "python tools/build_release_docs.py" in workflow
    assert "python tools/check_docs_portal.py" in workflow
    assert UPLOAD_ARTIFACT in workflow
    assert "retention-days: 14" in workflow
    assert "pages: write" not in workflow
    assert "id-token: write" not in workflow
    assert "actions/deploy-pages@" not in workflow
    assert f"DOC_BASE_URL: {BASE_URL}" in workflow


def test_release_workflow_checks_portal_and_requires_an_archive_for_every_publication() -> None:
    workflow = (ROOT / ".github" / "workflows" / "docs-cloudflare.yml").read_text(encoding="utf-8")

    assert "archive_release:" in workflow
    assert "archive_run_id:" in workflow
    assert "archive_artifact_name:" in workflow
    assert "if: steps.archive.outputs.source == 'release'" in workflow
    assert "if: steps.archive.outputs.source == 'artifact'" in workflow
    assert DOWNLOAD_ARTIFACT in workflow
    assert '--archive-from "$ARCHIVE_PATH"' in workflow
    assert "python tools/check_docs_portal.py" in workflow
    assert UPLOAD_ARTIFACT in workflow
    assert "vamos-docs-cloudflare-${{ env.DOC_VERSION }}" in workflow
    assert "pages: write" not in workflow
    assert "actions/deploy-pages@" not in workflow
    assert f"DOC_BASE_URL: {BASE_URL}" in workflow


def _workflow(filename: str) -> tuple[str, dict]:
    text = (ROOT / ".github" / "workflows" / filename).read_text(encoding="utf-8")
    # BaseLoader preserves GitHub's `on` key and keeps every scalar as text.
    return text, yaml.load(text, Loader=yaml.BaseLoader)


def _step_names(job: dict) -> list[str]:
    return [str(step.get("name", "")) for step in job["steps"]]


def _named_step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_production_publication_attaches_the_portal_to_the_release() -> None:
    text, workflow = _workflow("docs-cloudflare.yml")

    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert inputs["archive_release"]["required"] == "false"
    assert inputs["archive_run_id"]["required"] == "false"
    assert inputs["archive_artifact_name"]["required"] == "false"
    assert workflow["permissions"] == {"actions": "read", "contents": "read"}
    job = workflow["jobs"]["deploy"]
    assert job["permissions"] == {"actions": "read", "contents": "write"}
    names = _step_names(job)

    release = _named_step(job, "Require the GitHub Release of the documented version")
    assert 'gh release view "v$DOC_VERSION"' in release["run"]
    assert release["env"]["GH_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
    assert names.index(release["name"]) < names.index("Build canonical Cloudflare portal")
    assert names.index("Build canonical Cloudflare portal") < names.index("Deploy Worker and custom domains")

    artifact = _named_step(job, "Download previous trusted portal artifact")
    assert artifact["uses"] == DOWNLOAD_ARTIFACT
    asset = _named_step(job, "Download previous trusted portal release asset")
    assert asset["env"]["GH_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
    assert 'gh release download "$ARCHIVE_RELEASE"' in asset["run"]
    assert 'sha256sum --check --strict "$asset.sha256"' in asset["run"]
    build = _named_step(job, "Build canonical Cloudflare portal")
    assert '--archive-from "$ARCHIVE_PATH"' in build["run"]

    live = names.index("Compare deployed current files with the approved artifact")
    package = _named_step(job, "Package the published portal as a release asset")
    upload = _named_step(job, "Attach the portal archive to the GitHub Release")
    assert live < names.index(package["name"]) < names.index(upload["name"])
    for option in ("--sort=name", "--owner=0", "--group=0", "--numeric-owner", "--mtime="):
        assert option in package["run"]
    assert "gzip -n" in package["run"]
    assert 'asset="vamos-docs-portal-${DOC_VERSION}.tar.gz"' in package["run"]
    assert 'sha256sum "$asset" > "$asset.sha256"' in package["run"]
    assert upload["env"]["GH_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
    assert 'gh release upload "v$DOC_VERSION"' in upload["run"]
    assert "--clobber" in upload["run"]

    assert "name: vamos-docs-identity-${{ env.DOC_VERSION }}" in text
    assert "name: vamos-docs-cloudflare-${{ env.DOC_VERSION }}" in text
    summary = _named_step(job, "Summarize production deployment")["run"]
    assert "vamos-docs-portal-${DOC_VERSION}.tar.gz" in summary


def _workflow_bash() -> str:
    bash = shutil.which("bash")
    if sys.platform == "win32":
        # Prefer Git's bash over the WSL launcher that may shadow it on PATH.
        git = shutil.which("git")
        parents = Path(git).resolve().parents if git is not None else ()
        candidates = (parent / relative for parent in parents for relative in ("bin/bash.exe", "usr/bin/bash.exe"))
        bash = next((str(candidate) for candidate in candidates if candidate.is_file()), None)
    if bash is None:
        pytest.skip("bash is needed to run the workflow step")
    return bash


@pytest.mark.parametrize(
    ("release", "run_id", "artifact", "expected"),
    [
        ("v1.1.0", "", "", {"source": "release", "path": "previous-release/vamos-docs-portal-1.1.0.tar.gz"}),
        ("", "39917026051", "vamos-docs-cloudflare-1.1.0", {"source": "artifact", "path": "previous-public"}),
        ("", "", "", None),
        ("", "39917026051", "", None),
        ("", "", "vamos-docs-cloudflare-1.1.0", None),
        ("v1.1.0", "39917026051", "", None),
        ("v1.1.0", "", "vamos-docs-cloudflare-1.1.0", None),
        ("1.1.0", "", "", None),
        ("v1.1.0/../x", "", "", None),
    ],
)
def test_production_publication_requires_exactly_one_archive_source(
    tmp_path: Path, release: str, run_id: str, artifact: str, expected: dict[str, str] | None
) -> None:
    _, workflow = _workflow("docs-cloudflare.yml")
    step = _named_step(workflow["jobs"]["deploy"], "Select one trusted archive source")
    assert step["env"] == {
        "ARCHIVE_RELEASE": "${{ inputs.archive_release }}",
        "ARCHIVE_RUN_ID": "${{ inputs.archive_run_id }}",
        "ARCHIVE_ARTIFACT_NAME": "${{ inputs.archive_artifact_name }}",
    }
    outputs = tmp_path / "github-output"
    outputs.write_text("", encoding="utf-8")
    environment = {
        **os.environ,
        "ARCHIVE_RELEASE": release,
        "ARCHIVE_RUN_ID": run_id,
        "ARCHIVE_ARTIFACT_NAME": artifact,
        "GITHUB_OUTPUT": outputs.as_posix(),
    }
    completed = subprocess.run(
        [_workflow_bash(), "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if expected is None:
        assert completed.returncode != 0
        assert completed.stderr.strip()
        assert outputs.read_text(encoding="utf-8") == ""
    else:
        assert completed.returncode == 0, completed.stderr
        lines = outputs.read_text(encoding="utf-8").splitlines()
        assert dict(line.split("=", 1) for line in lines) == expected


def test_pages_mirror_points_to_org(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    text, workflow = _workflow("docs.yml")

    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert not workflow["on"]["workflow_dispatch"]
    assert "build_release_docs.py" not in text
    assert DOWNLOAD_ARTIFACT not in text
    assert UPLOAD_ARTIFACT not in text
    build = workflow["jobs"]["build"]
    commands = [str(step.get("run", "")).strip() for step in build["steps"]]
    assert f"python tools/build_pages_redirect.py --base-url {BASE_URL} --output public" in commands
    assert "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9" in text
    assert "actions/deploy-pages@368f82528645a54fb793d4d04e342629a3f51346" in text
    assert workflow["jobs"]["deploy"]["environment"]["name"] == "github-pages"

    output = tmp_path / "pages"
    completed = subprocess.run(
        [sys.executable, "tools/build_pages_redirect.py", "--base-url", BASE_URL, "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert sorted(path.name for path in output.iterdir()) == ["404.html", "index.html"]

    index = (output / "index.html").read_text(encoding="utf-8")
    assert f'<link rel="canonical" href="{BASE_URL}">' in index
    assert f'<meta http-equiv="refresh" content="0; url={BASE_URL}">' in index
    assert f'<a href="{BASE_URL}">' in index

    missing = (output / "404.html").read_text(encoding="utf-8")
    assert f'<link rel="canonical" href="{BASE_URL}">' in missing
    assert 'http-equiv="refresh"' not in missing
    assert f'<noscript><p>Continue to <a href="{BASE_URL}">' in missing
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    redirect = importlib.import_module("build_pages_redirect")
    assert f".replace(/{redirect.REPOSITORY_SEGMENT}/, " in missing
    assert "\\.github\\.io$" in missing
    assert "window.location.replace(" in missing


_NODE_PAGES_HARNESS = """
const vm = require("node:vm");
const input = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
const targets = input.locations.map((href) => {
  const url = new URL(href);
  const replaced = [];
  const location = {
    hostname: url.hostname, pathname: url.pathname, search: url.search, hash: url.hash,
    replace: (value) => replaced.push(value),
  };
  vm.runInNewContext(input.script, { window: { location } });
  return replaced;
});
process.stdout.write(JSON.stringify(targets));
"""


def test_pages_not_found_script_maps_repository_paths_to_org(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    redirect = importlib.import_module("build_pages_redirect")
    pages = importlib.import_module("check_repository_identity").PAGES_URL
    cases = {
        pages: BASE_URL,
        pages.rstrip("/"): BASE_URL,
        f"{pages}algorithms/nsgaii/?q=front#usage": f"{BASE_URL}algorithms/nsgaii/?q=front#usage",
        f"{pages}latest/guide/getting-started/": f"{BASE_URL}latest/guide/getting-started/",
        f"{pages}docs/stable/reference/?v=1": f"{BASE_URL}docs/stable/reference/?v=1",
        f"{pages}1.0.0/guide/#install": f"{BASE_URL}1.0.0/guide/#install",
        f"{pages}docs/1.1.0/algorithms/moead/": f"{BASE_URL}docs/1.1.0/algorithms/moead/",
        "https://vamos-optimization.github.io/": BASE_URL,
        "https://mirror.example/VAMOS/latest/": BASE_URL,
    }
    for location, expected in cases.items():
        assert redirect.redirect_target(location, BASE_URL) == expected, location

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to run the generated redirect script")
    output = tmp_path / "pages"
    redirect.build_pages_redirect(BASE_URL, output)
    page = (output / "404.html").read_text(encoding="utf-8")
    script = page.split("<script>", 1)[1].split("</script>", 1)[0]
    completed = subprocess.run(
        [node, "-e", _NODE_PAGES_HARNESS],
        input=json.dumps({"script": script, "locations": list(cases)}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [[expected] for expected in cases.values()]


def test_documentation_delivery_page_is_discoverable() -> None:
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    nav_text = yaml.safe_dump(config["nav"], sort_keys=False)
    delivery = (ROOT / "docs" / "dev" / "documentation-delivery.md").read_text(encoding="utf-8")

    assert "dev/documentation-delivery.md" in nav_text
    assert "does not deploy" in delivery
    assert "archive_release" in delivery
    assert "archive_run_id" in delivery
    assert "vamos-docs-portal-<version>.tar.gz" in delivery
    assert "### Recovery" in delivery
    assert "## GitHub Pages redirect" in delivery
    assert "migration-stage" not in delivery
    assert "canonical production path" in delivery
    assert "clean current" in delivery
    assert "same-version" in delivery


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _canonical(
    target: str,
    *,
    body_refresh: str | None = None,
    include_canonical: bool = True,
) -> str:
    canonical = "" if not include_canonical else f'<link rel="canonical" href="{target}">\n'
    body = "" if body_refresh is None else f'<meta http-equiv="refresh" content="0; url={body_refresh}">\n'
    return f"<!doctype html><html><head>\n{canonical}</head><body>\n{body}</body></html>\n"


def _redirect(target: str) -> str:
    return (
        "<!doctype html><html><head>\n"
        f'<link rel="canonical" href="{target}">\n'
        f'<meta http-equiv="refresh" content="0; url={target}">\n'
        "</head><body></body></html>\n"
    )


def _build_minimal_clean_portal(root: Path, *, include_deep_current: bool = False) -> None:
    immutable_url = f"{BASE_URL}docs/{VERSION}/"
    _write(root, "index.html", _canonical(BASE_URL))
    _write(root, f"docs/{VERSION}/index.html", _canonical(immutable_url))
    _write(root, "docs/versions.json", json.dumps({"stable": VERSION, "versions": [VERSION]}))
    _write(root, "docs/index.html", _redirect(BASE_URL))
    _write(root, "docs/stable/index.html", _redirect(BASE_URL))
    _write(root, "latest/index.html", _redirect(BASE_URL))
    _write(root, f"{VERSION}/index.html", _redirect(immutable_url))
    if include_deep_current:
        target = f"{BASE_URL}algorithms/nsgaii/"
        _write(root, "algorithms/nsgaii/index.html", _canonical(target))
        _write(root, "docs/stable/algorithms/nsgaii/index.html", _redirect(target))
        _write(root, "latest/algorithms/nsgaii/index.html", _redirect(target))


def _run_portal_check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "tools/check_docs_portal.py",
            "--root",
            str(root),
            "--version",
            VERSION,
            "--base-url",
            BASE_URL,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_portal_checker_accepts_minimal_clean_current_contract(tmp_path: Path) -> None:
    root = tmp_path / "portal"
    _build_minimal_clean_portal(root, include_deep_current=True)

    completed = _run_portal_check(root)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["stable"] == VERSION
    assert payload["versions"] == [VERSION]
    assert payload["current_url"] == BASE_URL
    assert payload["canonical_links_checked"] >= 7


def test_portal_checker_rejects_missing_current_homepage(tmp_path: Path) -> None:
    root = tmp_path / "missing-home"
    _build_minimal_clean_portal(root, include_deep_current=True)
    (root / "index.html").unlink()
    (root / "docs" / "stable" / "index.html").unlink()
    (root / "latest" / "index.html").unlink()

    completed = _run_portal_check(root)
    assert completed.returncode != 0
    assert "Missing required portal file" in completed.stderr


def test_portal_checker_rejects_missing_deep_compatibility_route(tmp_path: Path) -> None:
    root = tmp_path / "missing-alias"
    _build_minimal_clean_portal(root, include_deep_current=True)
    (root / "latest" / "algorithms" / "nsgaii" / "index.html").unlink()

    completed = _run_portal_check(root)
    assert completed.returncode != 0
    assert "route inventory does not match its source tree" in completed.stderr


def test_portal_checker_requires_canonical_on_every_current_html_page(tmp_path: Path) -> None:
    root = tmp_path / "missing-canonical"
    _build_minimal_clean_portal(root, include_deep_current=True)
    target = f"{BASE_URL}algorithms/nsgaii/"
    _write(root, "algorithms/nsgaii/index.html", _canonical(target, include_canonical=False))

    completed = _run_portal_check(root)
    assert completed.returncode != 0
    assert "Canonical mismatch" in completed.stderr


@pytest.mark.parametrize(
    "relative",
    [
        "website/index.html",
        "audit/commands_used/index.html",
        f"docs/{VERSION}/audit/findings.csv",
        f"docs/{VERSION}/topics/engineering_audit/index.html",
        "dev/adr/index.html",
        f"docs/{VERSION}/dev/adr/0008-durable-study-manifest-contract/index.html",
    ],
)
def test_portal_checker_rejects_withdrawn_routes(tmp_path: Path, relative: str) -> None:
    root = tmp_path / "withdrawn"
    _build_minimal_clean_portal(root)
    _write(root, relative, "withdrawn")

    completed = _run_portal_check(root)
    assert completed.returncode != 0
    assert "Withdrawn route is still published" in completed.stderr


@pytest.mark.parametrize(
    ("relative", "href"),
    [
        ("algorithms/nsgaii/index.html", "../../dev/adr/"),
        (f"docs/{VERSION}/index.html", "dev/adr/0006-run-artifact-and-replay-contract/#decision"),
        (f"docs/{VERSION}/index.html", f"{BASE_URL}docs/{VERSION}/audit/commands_used/"),
    ],
)
def test_portal_checker_rejects_links_to_withdrawn_routes(tmp_path: Path, relative: str, href: str) -> None:
    root = tmp_path / "withdrawn-link"
    _build_minimal_clean_portal(root, include_deep_current=True)
    canonical = f"{BASE_URL}{relative.removesuffix('index.html')}"
    _write(root, relative, _canonical(canonical).replace("<body>", f'<body><a href="{href}">ADR</a>'))

    completed = _run_portal_check(root)
    assert completed.returncode != 0
    assert "Page links to withdrawn route" in completed.stderr


def test_portal_checker_accepts_links_that_only_resemble_withdrawn_routes(tmp_path: Path) -> None:
    root = tmp_path / "similar-link"
    _build_minimal_clean_portal(root, include_deep_current=True)
    target = f"{BASE_URL}algorithms/nsgaii/"
    body = '<body><a href="../../dev/adr-notes/">notes</a><a href="#dev/adr">anchor</a>'
    _write(root, "algorithms/nsgaii/index.html", _canonical(target).replace("<body>", body))

    completed = _run_portal_check(root)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_portal_checker_rejects_withdrawn_sitemap_and_search_entries(tmp_path: Path) -> None:
    root = tmp_path / "withdrawn-index"
    _build_minimal_clean_portal(root)
    _write(
        root,
        f"docs/{VERSION}/sitemap.xml",
        f"<urlset><url><loc>{BASE_URL}docs/{VERSION}/audit/commands_used/</loc></url></urlset>\n",
    )
    completed = _run_portal_check(root)
    assert completed.returncode != 0
    assert "Sitemap lists withdrawn route" in completed.stderr

    (root / "docs" / VERSION / "sitemap.xml").unlink()
    _write(
        root,
        f"docs/{VERSION}/search/search_index.json",
        json.dumps({"docs": [{"location": "topics/engineering_audit/#summary", "text": ""}]}),
    )
    completed = _run_portal_check(root)
    assert completed.returncode != 0
    assert "Search index lists withdrawn route" in completed.stderr


def test_portal_checker_rejects_body_meta_refresh_on_current_page(tmp_path: Path) -> None:
    root = tmp_path / "body-refresh"
    _build_minimal_clean_portal(root, include_deep_current=True)
    target = f"{BASE_URL}algorithms/nsgaii/"
    _write(
        root,
        "algorithms/nsgaii/index.html",
        _canonical(target, body_refresh="https://evil.invalid/"),
    )

    completed = _run_portal_check(root)
    assert completed.returncode != 0
    assert "must not meta-refresh" in completed.stderr


def test_portal_checker_rejects_canonical_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "path-escape"
    _build_minimal_clean_portal(root)
    outside = tmp_path / "outside.txt"
    outside.write_text("not part of the portal", encoding="utf-8")
    malicious = f"{BASE_URL}docs/{VERSION}/%2e%2e/%2e%2e/%2e%2e/outside.txt"
    _write(root, f"docs/{VERSION}/index.html", _canonical(malicious))

    completed = _run_portal_check(root)
    assert completed.returncode != 0
    assert "Canonical URL escapes portal root" in completed.stderr


def test_portal_checker_rejects_commented_redirect_spoof(tmp_path: Path) -> None:
    root = tmp_path / "spoof"
    _build_minimal_clean_portal(root)
    spoofed = (
        "<!doctype html><html><head>\n"
        f'<!-- <link rel="canonical" href="{BASE_URL}">'
        f'<meta http-equiv="refresh" content="0; url={BASE_URL}"> -->\n'
        '<link rel="canonical" href="https://evil.invalid/">\n'
        '<meta http-equiv="refresh" content="0; url=https://evil.invalid/">\n'
        "</head><body></body></html>\n"
    )
    _write(root, "docs/stable/index.html", spoofed)

    completed = _run_portal_check(root)
    assert completed.returncode != 0
    assert "Redirect canonical mismatch" in completed.stderr


def test_portal_checker_rejects_duplicate_redirect_attributes(tmp_path: Path) -> None:
    root = tmp_path / "duplicate-attrs"
    _build_minimal_clean_portal(root)
    duplicated = (
        "<!doctype html><html><head>\n"
        f'<link rel="canonical" href="https://evil.invalid/" href="{BASE_URL}">\n'
        f'<meta http-equiv="refresh" content="0; url=https://evil.invalid/" content="0; url={BASE_URL}">\n'
        "</head><body></body></html>\n"
    )
    _write(root, "docs/stable/index.html", duplicated)

    completed = _run_portal_check(root)
    assert completed.returncode != 0
    assert "duplicate 'href' attribute" in completed.stderr
    assert "duplicate 'content' attribute" in completed.stderr


def test_portal_checker_ignores_redirect_directives_inside_template(tmp_path: Path) -> None:
    root = tmp_path / "template-spoof"
    _build_minimal_clean_portal(root)
    inert = (
        "<!doctype html><html><head><template>\n"
        f'<link rel="canonical" href="{BASE_URL}">\n'
        f'<meta http-equiv="refresh" content="0; url={BASE_URL}">\n'
        "</template></head><body></body></html>\n"
    )
    _write(root, "docs/stable/index.html", inert)

    completed = _run_portal_check(root)
    assert completed.returncode != 0
    assert "Redirect canonical mismatch" in completed.stderr
