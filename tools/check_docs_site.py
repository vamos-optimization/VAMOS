"""Validate reader-facing links, search, metadata, downloads and preserved bookmarks."""

from __future__ import annotations

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from check_docs_portal import _current_relative_files


class SiteCheckError(ValueError):
    """The generated current site cannot support a documented reader journey."""


class Page(HTMLParser):
    def __init__(self, text: str) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.downloads: list[str] = []
        self.images_without_alt = 0
        self.h1 = 0
        self.lang = ""
        self.meta: dict[str, str] = {}
        self.feed(text)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = dict(attrs)
        if data.get("id"):
            self.ids.add(str(data["id"]))
        if tag == "a" and data.get("name"):
            self.ids.add(str(data["name"]))
        if tag == "h1":
            self.h1 += 1
        if tag == "html":
            self.lang = data.get("lang") or ""
        if tag == "img" and "alt" not in data:
            self.images_without_alt += 1
        if tag == "meta":
            key = data.get("name") or data.get("property")
            if key:
                self.meta[key] = data.get("content") or ""
        for key in ("href", "src", "action"):
            if data.get(key):
                self.links.append(str(data[key]))
        if tag == "a" and "download" in data and data.get("href"):
            self.downloads.append(str(data["href"]))


def route(relative: Path) -> str:
    value = relative.as_posix()
    return value.removesuffix("index.html") if relative.name == "index.html" else value


def local_target(base_url: str, page_route: str, link: str) -> tuple[Path, str] | None:
    base = urlsplit(base_url)
    target = urlsplit(urljoin(urljoin(base_url, page_route), link))
    if (target.scheme, target.netloc) != (base.scheme, base.netloc):
        return None
    if not target.path.startswith(base.path):
        return None
    raw = unquote(target.path[len(base.path) :])
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise SiteCheckError(f"Link escapes the site: {link}")
    if not raw or raw.endswith("/"):
        path /= "index.html"
    return path, unquote(target.fragment)


def check_site(root: Path, *, base_url: str, baseline: Path | None = None) -> dict[str, int]:
    root = root.resolve()
    base_url = base_url.rstrip("/") + "/"
    files = _current_relative_files(root)
    pages = {p: Page((root / p).read_text(encoding="utf-8")) for p in files if p.suffix == ".html"}
    if Path("index.html") not in pages or Path("404.html") not in pages:
        raise SiteCheckError("Current home and recovery page are required")
    links = downloads = 0
    for path, page in pages.items():
        if page.h1 != 1 or not page.lang or page.images_without_alt:
            raise SiteCheckError(f"Invalid heading, language or image alternative in {path}")
        for key in ("description", "og:title", "og:description"):
            if not page.meta.get(key, "").strip():
                raise SiteCheckError(f"Missing {key} in {path}")
        for link in page.links:
            target = local_target(base_url, route(path), link)
            if target is None:
                continue
            destination, fragment = target
            if not (root / destination).is_file():
                raise SiteCheckError(f"Broken local link: {path} -> {link}")
            # Built-in UI anchors can refer to an empty hash; real fragments must exist.
            if fragment and destination in pages and fragment not in pages[destination].ids:
                raise SiteCheckError(f"Missing bookmark: {path} -> {link}")
            links += 1
        for link in page.downloads:
            target = local_target(base_url, route(path), link)
            if target is None or not (root / target[0]).is_file() or (root / target[0]).stat().st_size == 0:
                raise SiteCheckError(f"Unavailable local download: {path} -> {link}")
            downloads += 1

    error = pages[Path("404.html")]
    if "noindex" not in error.meta.get("robots", "") or "recovery-query" not in error.ids:
        raise SiteCheckError("404 must provide search recovery and noindex metadata")
    index = json.loads((root / "search/search_index.json").read_text(encoding="utf-8"))
    indexed: set[Path] = set()
    for document in index["docs"]:
        target = local_target(base_url, "", document["location"])
        if target is None or target[0] not in pages:
            raise SiteCheckError(f"Search points outside the current pages: {document['location']}")
        path, fragment = target
        if fragment and fragment not in pages[path].ids:
            raise SiteCheckError(f"Search points to a missing bookmark: {document['location']}")
        indexed.add(path)
    missing = set(pages) - indexed - {Path("404.html")}
    if missing:
        raise SiteCheckError(f"Pages missing from search: {sorted(str(p) for p in missing)}")
    preserved = 0
    if baseline is not None:
        for path in _current_relative_files(baseline):
            if path.suffix != ".html" or path.name == "404.html":
                continue
            old = Page((baseline / path).read_text(encoding="utf-8"))
            if path not in pages or not old.ids.issubset(pages[path].ids):
                raise SiteCheckError(f"Removed public route or bookmark: {path}")
            preserved += 1
    return {"pages": len(pages), "links": links, "downloads": downloads, "search_pages": len(indexed), "preserved_pages": preserved}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--base-url", default="https://vamos-optimization.org/")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    print(json.dumps(check_site(args.root, base_url=args.base_url, baseline=args.baseline), sort_keys=True))


if __name__ == "__main__":
    main()
