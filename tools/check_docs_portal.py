"""Validate a built VAMOS documentation portal before preview or deployment."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
_CANONICAL_RE = re.compile(r'<link\s+rel="canonical"\s+href="([^"]+)"')


class PortalCheckError(RuntimeError):
    """Raised when a built documentation portal violates its route contract."""


def _normalize_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise PortalCheckError("Base URL must be an absolute HTTPS URL without query or fragment")
    return value.rstrip("/") + "/"


def _read(path: Path) -> str:
    if not path.is_file():
        raise PortalCheckError(f"Missing required portal file: {path}")
    return path.read_text(encoding="utf-8")


def _page_url(base_url: str, prefix: str, relative: Path) -> str:
    prefix = prefix.strip("/")
    route_base = f"{base_url}{prefix}/" if prefix else base_url
    if relative.name == "index.html":
        parent = relative.parent.as_posix()
        suffix = "" if parent == "." else f"{parent}/"
    else:
        suffix = relative.as_posix()
    return f"{route_base}{suffix}"


def _target_file(root: Path, *, base_url: str, target_url: str) -> Path:
    base = urlsplit(base_url)
    target = urlsplit(html.unescape(target_url))
    if target.scheme != base.scheme or target.netloc != base.netloc:
        raise PortalCheckError(f"Canonical URL leaves configured host: {target_url}")
    if not target.path.startswith(base.path):
        raise PortalCheckError(f"Canonical URL leaves configured base path: {target_url}")
    relative = unquote(target.path[len(base.path) :].lstrip("/"))
    candidate = root / relative
    if target.path.endswith("/"):
        candidate /= "index.html"
    return candidate


def _check_redirect(path: Path, *, expected_target: str) -> None:
    content = _read(path)
    if f'rel="canonical" href="{expected_target}"' not in content:
        raise PortalCheckError(f"Redirect canonical mismatch in {path}: expected {expected_target}")
    if f"url={expected_target}" not in content:
        raise PortalCheckError(f"Redirect refresh mismatch in {path}: expected {expected_target}")


def _check_redirect_tree(tree: Path, *, base_url: str, canonical_prefix: str) -> int:
    if not tree.is_dir():
        raise PortalCheckError(f"Missing required portal directory: {tree}")
    count = 0
    for page in tree.rglob("*.html"):
        relative = page.relative_to(tree)
        _check_redirect(
            page,
            expected_target=_page_url(base_url, canonical_prefix, relative),
        )
        count += 1
    if count == 0:
        raise PortalCheckError(f"No redirect pages found under {tree}")
    return count


def _check_canonical_tree(
    root: Path,
    tree: Path,
    *,
    base_url: str,
    expected_prefix: str,
) -> int:
    if not tree.is_dir():
        raise PortalCheckError(f"Missing required portal directory: {tree}")
    count = 0
    for page in tree.rglob("*.html"):
        for raw_url in _CANONICAL_RE.findall(page.read_text(encoding="utf-8")):
            canonical = html.unescape(raw_url)
            if not canonical.startswith(expected_prefix):
                raise PortalCheckError(
                    f"Unexpected canonical in {page}: {canonical!r}; expected prefix {expected_prefix!r}"
                )
            target = _target_file(root, base_url=base_url, target_url=canonical)
            if not target.is_file():
                raise PortalCheckError(f"Canonical target is absent for {page}: {canonical}")
            count += 1
    if count == 0:
        raise PortalCheckError(f"No canonical URLs found under {tree}")
    return count


def _check_current_tree(root: Path, *, base_url: str) -> int:
    count = 0
    for page in root.rglob("*.html"):
        relative = page.relative_to(root)
        if relative.parts:
            first = relative.parts[0]
            if first in {"docs", "website", "latest"} or _VERSION_RE.fullmatch(first):
                continue
        expected = _page_url(base_url, "", relative)
        for raw_url in _CANONICAL_RE.findall(page.read_text(encoding="utf-8")):
            canonical = html.unescape(raw_url)
            if canonical != expected:
                raise PortalCheckError(
                    f"Unexpected current-site canonical in {page}: {canonical!r}; expected {expected!r}"
                )
            target = _target_file(root, base_url=base_url, target_url=canonical)
            if not target.is_file():
                raise PortalCheckError(f"Canonical target is absent for {page}: {canonical}")
            count += 1
    if count == 0:
        raise PortalCheckError("No canonical URLs found in the clean current documentation tree")
    return count


def check_portal(root: Path, *, version: str, base_url: str) -> dict[str, object]:
    if _VERSION_RE.fullmatch(version) is None:
        raise PortalCheckError("Version must be a numeric major.minor.patch value")
    root = root.resolve()
    if not root.is_dir():
        raise PortalCheckError(f"Portal root does not exist: {root}")
    base_url = _normalize_base_url(base_url)

    manifest = json.loads(_read(root / "docs" / "versions.json"))
    versions = manifest.get("versions")
    if manifest.get("stable") != version or not isinstance(versions, list) or version not in versions:
        raise PortalCheckError("versions.json does not identify the requested version as stable")
    if any(not isinstance(item, str) or _VERSION_RE.fullmatch(item) is None for item in versions):
        raise PortalCheckError("versions.json contains an invalid semantic version")

    immutable = root / "docs" / version
    stable_alias = root / "docs" / "stable"
    if not immutable.is_dir() or not stable_alias.is_dir():
        raise PortalCheckError("Immutable documentation and the legacy stable redirect tree must both exist")

    canonical_count = 0
    canonical_count += _check_current_tree(root, base_url=base_url)
    canonical_count += _check_canonical_tree(
        root,
        immutable,
        base_url=base_url,
        expected_prefix=f"{base_url}docs/{version}/",
    )
    canonical_count += _check_redirect_tree(
        stable_alias,
        base_url=base_url,
        canonical_prefix="",
    )
    canonical_count += _check_redirect_tree(
        root / "latest",
        base_url=base_url,
        canonical_prefix="",
    )
    canonical_count += _check_canonical_tree(
        root,
        root / "website",
        base_url=base_url,
        expected_prefix=f"{base_url}website/",
    )

    _check_redirect(root / "docs" / "index.html", expected_target=base_url)

    for published in versions:
        archived = root / "docs" / published
        legacy = root / published
        if not archived.is_dir():
            raise PortalCheckError(f"Manifest version is missing from archive: {published}")
        canonical_count += _check_redirect_tree(
            legacy,
            base_url=base_url,
            canonical_prefix=f"docs/{published}",
        )

    return {
        "stable": version,
        "versions": versions,
        "canonical_links_checked": canonical_count,
        "current_url": base_url,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    result = check_portal(args.root, version=args.version, base_url=args.base_url)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
