"""Build the clean current documentation site plus immutable version archives and redirects."""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from check_repository_identity import DOCUMENTATION_URL
from mkdocs.commands.build import build
from mkdocs.config import load_config

_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")


def _normalize_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("Documentation base URL must be an absolute HTTPS URL without query or fragment")
    return value.rstrip("/") + "/"


def _version_key(value: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in value.split("."))  # type: ignore[return-value]


def _page_url(base_url: str, prefix: str, relative: Path) -> str:
    prefix = prefix.strip("/")
    route_base = f"{base_url}{prefix}/" if prefix else base_url
    if relative.name == "index.html":
        parent = relative.parent.as_posix()
        suffix = "" if parent == "." else f"{parent}/"
    else:
        suffix = relative.as_posix()
    return f"{route_base}{suffix}"


def _write_redirect(path: Path, target: str) -> None:
    escaped = html.escape(target, quote=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">\n'
        f'<link rel="canonical" href="{escaped}">\n'
        f'<meta http-equiv="refresh" content="0; url={escaped}">\n'
        "<title>VAMOS documentation redirect</title></head>\n"
        f'<body><p>Moved to <a href="{escaped}">{escaped}</a>.</p></body></html>\n',
        encoding="utf-8",
    )


def _build_site(root: Path, configuration: str, site_dir: Path, site_url: str) -> None:
    config = load_config(
        config_file=str(root / configuration),
        strict=True,
        site_dir=str(site_dir),
        site_url=site_url,
    )
    config.plugins.on_startup(command="build", dirty=False)
    try:
        build(config)
    finally:
        config.plugins.on_shutdown()


def _copy_tree_contents(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for source_path in source.iterdir():
        target_path = target / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)


def _copy_archived_versions(archive_from: Path, docs_root: Path, current_version: str) -> None:
    archive_docs = archive_from / "docs"
    if not archive_docs.is_dir():
        raise FileNotFoundError(f"Archived documentation root not found: {archive_docs}")
    for source in sorted(archive_docs.iterdir(), key=lambda path: path.name):
        if not source.is_dir() or _VERSION_RE.fullmatch(source.name) is None:
            continue
        if source.name == current_version:
            continue
        shutil.copytree(source, docs_root / source.name)


def _published_versions(docs_root: Path) -> list[str]:
    versions = [
        path.name
        for path in docs_root.iterdir()
        if path.is_dir() and _VERSION_RE.fullmatch(path.name) is not None
    ]
    return sorted(versions, key=_version_key)


def _write_legacy_alias(source: Path, target: Path, *, base_url: str, canonical_prefix: str) -> None:
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        target_path = target / relative
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.suffix.lower() == ".html":
            _write_redirect(target_path, _page_url(base_url, canonical_prefix, relative))
        else:
            shutil.copy2(source_path, target_path)


def _write_versions_manifest(docs_root: Path, *, stable: str, versions: list[str]) -> None:
    payload = {"stable": stable, "versions": versions}
    (docs_root / "versions.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_release_docs(
    version: str,
    output: Path,
    *,
    base_url: str = DOCUMENTATION_URL,
    archive_from: Path | None = None,
) -> None:
    if _VERSION_RE.fullmatch(version) is None:
        raise ValueError("Documentation version must be a numeric major.minor.patch value")
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite documentation output: {output}")
    base_url = _normalize_base_url(base_url)
    docs_root = output / "docs"
    docs_root.mkdir(parents=True)

    if archive_from is not None:
        _copy_archived_versions(archive_from.resolve(), docs_root, version)

    with TemporaryDirectory(prefix="vamos-docs-current-") as temporary:
        current_dir = Path(temporary) / "current"
        _build_site(root, "mkdocs.yml", current_dir, base_url)
        _copy_tree_contents(current_dir, output)

        immutable_dir = docs_root / version
        _build_site(root, "mkdocs.yml", immutable_dir, f"{base_url}docs/{version}/")
        _build_site(root, "website/mkdocs.yml", output / "website", f"{base_url}website/")

        versions = _published_versions(docs_root)
        _write_versions_manifest(docs_root, stable=version, versions=versions)

        for published_version in versions:
            _write_legacy_alias(
                docs_root / published_version,
                output / published_version,
                base_url=base_url,
                canonical_prefix=f"docs/{published_version}",
            )

        # Keep old moving aliases as redirect mirrors for bookmarks and static mirrors.
        _write_legacy_alias(
            current_dir,
            docs_root / "stable",
            base_url=base_url,
            canonical_prefix="",
        )
        _write_legacy_alias(
            current_dir,
            output / "latest",
            base_url=base_url,
            canonical_prefix="",
        )

    _write_redirect(docs_root / "index.html", base_url)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=DOCUMENTATION_URL)
    parser.add_argument(
        "--archive-from",
        type=Path,
        help="Previous portal artifact whose immutable docs/<version>/ directories should be preserved",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_release_docs(
        args.version,
        args.output,
        base_url=args.base_url,
        archive_from=args.archive_from,
    )


if __name__ == "__main__":
    main()
