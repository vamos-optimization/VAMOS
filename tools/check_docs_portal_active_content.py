"""Fail-closed scan for active meta-refresh directives in built docs portals.

The canonical portal validator verifies route inventories, canonicals, and expected
redirects. This companion scan deliberately treats SVG and MathML descendants as
potentially active HTML integration contexts, so a refresh hidden under constructs
such as ``svg > foreignObject`` cannot bypass validation. Only ``template`` content
is treated as inert.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path

from check_docs_portal import (
    PortalCheckError,
    _current_relative_files,
    _manifest,
    _page_url,
)

_REFRESH_INERT_TAGS = {"template"}


class _ActiveRefreshScanner(HTMLParser):
    """Collect active or potentially active meta refreshes outside templates."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.refreshes: list[str] = []
        self.errors: list[str] = []
        self._inert_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.casefold()
        if lowered in _REFRESH_INERT_TAGS:
            self._inert_stack.append(lowered)
            return
        if self._inert_stack or lowered != "meta":
            return

        normalized = [(name.casefold(), value) for name, value in attrs]
        for relevant in ("http-equiv", "content"):
            if sum(1 for name, _ in normalized if name == relevant) > 1:
                self.errors.append(f"duplicate {relevant!r} attribute on <meta>")
        data: dict[str, str | None] = {}
        for name, value in normalized:
            data.setdefault(name, value)
        if (data.get("http-equiv") or "").casefold() == "refresh":
            self.refreshes.append(data.get("content") or "")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered not in _REFRESH_INERT_TAGS:
            return
        if not self._inert_stack or self._inert_stack[-1] != lowered:
            self.errors.append(f"mismatched </{lowered}> inert-subtree close")
            return
        self._inert_stack.pop()

    def finish(self) -> None:
        if self._inert_stack:
            self.errors.append("unclosed inert subtree")


def _active_refreshes(page: Path) -> list[str]:
    scanner = _ActiveRefreshScanner()
    scanner.feed(page.read_text(encoding="utf-8"))
    scanner.close()
    scanner.finish()
    if scanner.errors:
        raise PortalCheckError(f"Ambiguous meta-refresh markup in {page}: {scanner.errors!r}")
    return scanner.refreshes


def _require_no_refresh(page: Path) -> None:
    refreshes = _active_refreshes(page)
    if refreshes:
        raise PortalCheckError(f"Canonical content page must not meta-refresh: {page}; found {refreshes!r}")


def _require_redirect(page: Path, expected_target: str) -> None:
    expected = f"0; url={expected_target}"
    refreshes = _active_refreshes(page)
    if refreshes != [expected]:
        raise PortalCheckError(
            f"Compatibility redirect refresh mismatch in {page}: expected exactly {expected!r}, got {refreshes!r}"
        )


def check_active_content(root: Path, *, version: str, base_url: str) -> None:
    """Validate refresh behavior for every canonical and compatibility surface."""
    root = root.resolve()
    if not root.is_dir():
        raise PortalCheckError(f"Portal root does not exist: {root}")
    base_url = base_url.rstrip("/") + "/"
    versions = _manifest(root, version=version)

    current_files = _current_relative_files(root)
    current_pages = sorted(path for path in current_files if path.suffix.lower() == ".html")
    for relative in current_pages:
        _require_no_refresh(root / relative)

    for page in (root / "website").rglob("*.html"):
        _require_no_refresh(page)

    for published in versions:
        archive_root = root / "docs" / published
        legacy_root = root / published
        for page in archive_root.rglob("*.html"):
            _require_no_refresh(page)
            relative = page.relative_to(archive_root)
            _require_redirect(
                legacy_root / relative,
                _page_url(base_url, f"docs/{published}", relative),
            )

    for alias_root in (root / "docs" / "stable", root / "latest"):
        for relative in current_pages:
            _require_redirect(alias_root / relative, _page_url(base_url, "", relative))

    _require_redirect(root / "docs" / "index.html", base_url)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    check_active_content(args.root, version=args.version, base_url=args.base_url)


if __name__ == "__main__":
    main()
