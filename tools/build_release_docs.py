"""Build the clean current documentation site plus immutable version archives and redirects."""

from __future__ import annotations

import argparse
import gzip
import html
import json
import logging
import re
import shutil
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from check_docs_portal import WITHDRAWN_ROUTES, is_withdrawn, links_to_withdrawn
from check_repository_identity import DOCUMENTATION_URL
from mkdocs.commands.build import build
from mkdocs.config import load_config

_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
_SITEMAP_ENTRY_RE = re.compile(r"[ \t]*<url>\s*<loc>\s*([^<\s]+)\s*</loc>.*?</url>[ \t]*(?:\r?\n)?", re.DOTALL)


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
    javascript_target = json.dumps(target).replace("</", "<\\/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">\n'
        f'<link rel="canonical" href="{escaped}">\n'
        "<script>\n"
        f"const redirectBase = {javascript_target};\n"
        "window.location.replace(redirectBase + window.location.search + window.location.hash);\n"
        "</script>\n"
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


def _ensure_current_error_page_canonical(site_dir: Path, base_url: str) -> None:
    """Give the generated current-site 404 document an explicit clean canonical URL."""
    page = site_dir / "404.html"
    if not page.is_file():
        raise FileNotFoundError(f"Generated current documentation is missing its 404 page: {page}")
    content = page.read_text(encoding="utf-8")
    if 'rel="canonical"' in content:
        return
    marker = "</head>"
    if marker not in content:
        raise ValueError("Generated current documentation 404 page has no </head> marker")
    canonical = html.escape(f"{base_url}404.html", quote=True)
    content = content.replace(marker, f'<link rel="canonical" href="{canonical}">\n{marker}', 1)
    page.write_text(content, encoding="utf-8")


def _copy_tree_contents(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for source_path in source.iterdir():
        target_path = target / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)


class _WithdrawnLinkLocator(HTMLParser):
    """Locate markup in one generated page that links to withdrawn routes.

    Navigation items whose links all target withdrawn routes are removed whole,
    ``<link rel="prev|next">`` hints to them are removed, and any other link to
    them is unwrapped so its text remains.
    """

    def __init__(self, text: str, is_withdrawn_href: Callable[[str], bool]) -> None:
        super().__init__(convert_charrefs=True)
        self._text = text
        self._line_starts = [0, *(match.end() for match in re.finditer(r"\n", text))]
        self._is_withdrawn_href = is_withdrawn_href
        # [start or -1 for non-navigation items, links, withdrawn links]
        self._items: list[list[int]] = []
        self._anchors: list[tuple[int, int, bool]] = []
        self.removals: list[tuple[int, int]] = []

    def _position(self) -> int:
        line, column = self.getpos()
        return self._line_starts[line - 1] + column

    def _end_tag_end(self, start: int) -> int:
        return self._text.index(">", start) + 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = dict(attrs)
        start = self._position()
        end = start + len(self.get_starttag_text() or "")
        href = data.get("href")
        if tag == "li":
            navigation = "md-nav__item" in (data.get("class") or "").split()
            self._items.append([start if navigation else -1, 0, 0])
        elif tag == "a":
            withdrawn = href is not None and self._is_withdrawn_href(href)
            if href is not None:
                for item in self._items:
                    item[1] += 1
                    item[2] += int(withdrawn)
            self._anchors.append((start, end, withdrawn))
        elif tag == "link" and href is not None:
            if {"prev", "next"} & set((data.get("rel") or "").split()) and self._is_withdrawn_href(href):
                self.removals.append((start, end))

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._anchors:
            start, tag_end, withdrawn = self._anchors.pop()
            if withdrawn:
                close = self._position()
                self.removals.extend([(start, tag_end), (close, self._end_tag_end(close))])
        elif tag == "li" and self._items:
            start, links, withdrawn = self._items.pop()
            if start >= 0 and links and links == withdrawn:
                self.removals.append((start, self._end_tag_end(self._position())))


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _unlink_withdrawn_routes(tree: Path, *, tree_url: str) -> None:
    for page in sorted(tree.rglob("*.html")):
        page_url = _page_url(tree_url, "", page.relative_to(tree))
        original = page.read_bytes().decode("utf-8")
        locator = _WithdrawnLinkLocator(
            original,
            lambda href, page_url=page_url: links_to_withdrawn(href, page_url=page_url, tree_url=tree_url),
        )
        locator.feed(original)
        locator.close()
        if locator.removals:
            page.write_bytes(_remove_spans(original, locator.removals).encode("utf-8"))


def _withdraw_archived_routes(tree: Path, *, tree_url: str) -> None:
    """Remove withdrawn routes, links to them, and their sitemap/search entries from an archive.

    Files are rewritten only when they reference a withdrawn route, so an archive
    without withdrawn content is preserved byte-for-byte.
    """
    for route in WITHDRAWN_ROUTES:
        target = tree / route
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    _unlink_withdrawn_routes(tree, tree_url=tree_url)

    sitemap = tree / "sitemap.xml"
    if sitemap.is_file():
        original = sitemap.read_bytes().decode("utf-8")

        def keep(match: re.Match[str]) -> str:
            location = match.group(1)
            withdrawn = location.startswith(tree_url) and is_withdrawn(location[len(tree_url) :])
            return "" if withdrawn else match.group(0)

        pruned = _SITEMAP_ENTRY_RE.sub(keep, original)
        if pruned != original:
            sitemap.write_bytes(pruned.encode("utf-8"))
            compressed = tree / "sitemap.xml.gz"
            if compressed.is_file():
                compressed.write_bytes(gzip.compress(pruned.encode("utf-8"), mtime=0))

    index = tree / "search" / "search_index.json"
    if index.is_file():
        payload = json.loads(index.read_bytes().decode("utf-8"))
        entries = payload.get("docs", [])
        kept = [entry for entry in entries if not is_withdrawn(str(entry.get("location", "")))]
        if len(kept) != len(entries):
            payload["docs"] = kept
            index.write_bytes(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def _copy_archived_versions(archive_from: Path, docs_root: Path, *, base_url: str) -> None:
    archive_docs = archive_from / "docs"
    if not archive_docs.is_dir():
        raise FileNotFoundError(f"Archived documentation root not found: {archive_docs}")
    for source in sorted(archive_docs.iterdir(), key=lambda path: path.name):
        if not source.is_dir() or _VERSION_RE.fullmatch(source.name) is None:
            continue
        target = docs_root / source.name
        shutil.copytree(source, target)
        _withdraw_archived_routes(target, tree_url=f"{base_url}docs/{source.name}/")


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
        _copy_archived_versions(archive_from.resolve(), docs_root, base_url=base_url)

    with TemporaryDirectory(prefix="vamos-docs-current-") as temporary:
        current_dir = Path(temporary) / "current"
        _build_site(root, "mkdocs.yml", current_dir, base_url)
        _ensure_current_error_page_canonical(current_dir, base_url)
        _copy_tree_contents(current_dir, output)

        immutable_dir = docs_root / version
        if not immutable_dir.exists():
            _build_site(root, "mkdocs.yml", immutable_dir, f"{base_url}docs/{version}/")

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
        help=(
            "Previous trusted portal artifact whose immutable docs/<version>/ directories "
            "must be preserved; if it already contains the requested version, that archive "
            "is reused byte-for-byte instead of rebuilt, except that withdrawn routes are removed"
        ),
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
