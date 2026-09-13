"""Supplement the transition validator with strict clean-current content checks.

The temporary transition validator accepts both the legacy portal and the proposed clean-current
portal. Legacy root pages are themselves redirects, so these rules apply only after the artifact
has been classified as ``clean-current-transition``.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path

from check_docs_portal_transition import (
    PortalCheckError,
    _INERT_TAGS,
    _current_relative_files,
    _directives,
    _page_url,
    check_transition_portal,
)


class _ActiveRefreshScanner(HTMLParser):
    """Find active meta-refresh directives anywhere outside inert subtrees."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.refreshes: list[str] = []
        self.errors: list[str] = []
        self._inert_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.casefold()
        if lowered in _INERT_TAGS:
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
        if lowered not in _INERT_TAGS:
            return
        if not self._inert_stack or self._inert_stack[-1] != lowered:
            self.errors.append(f"mismatched </{lowered}> inert-subtree close")
            return
        self._inert_stack.pop()

    def finish(self) -> None:
        if self._inert_stack:
            self.errors.append("unclosed inert or foreign subtree")


def _active_refreshes(page: Path) -> list[str]:
    scanner = _ActiveRefreshScanner()
    scanner.feed(page.read_text(encoding="utf-8"))
    scanner.close()
    scanner.finish()
    if scanner.errors:
        raise PortalCheckError(f"Ambiguous meta-refresh markup in {page}: {scanner.errors!r}")
    return scanner.refreshes


def check_clean_content_pages(root: Path, *, version: str, base_url: str) -> None:
    """Require one canonical and no active refresh on every clean-current HTML page."""
    result = check_transition_portal(root, version=version, base_url=base_url)
    if result.get("contract") != "clean-current-transition":
        return
    root = root.resolve()
    base_url = base_url.rstrip("/") + "/"
    for relative in sorted(_current_relative_files(root)):
        if relative.suffix.lower() != ".html":
            continue
        page = root / relative
        expected = _page_url(base_url, "", relative)
        directives = _directives(page)
        if directives.canonicals != [expected]:
            raise PortalCheckError(
                f"Clean-current canonical mismatch in {page}: expected exactly {expected!r}, "
                f"got {directives.canonicals!r}"
            )
        refreshes = _active_refreshes(page)
        if refreshes:
            raise PortalCheckError(
                f"Clean-current content page must not meta-refresh: {page}; found {refreshes!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    check_clean_content_pages(args.root, version=args.version, base_url=args.base_url)


if __name__ == "__main__":
    main()
