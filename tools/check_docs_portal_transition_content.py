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
    _current_relative_files,
    _directives,
    _page_url,
    check_transition_portal,
)

# For refresh detection we intentionally fail closed. ``template`` remains inert,
# but SVG/MathML descendants are scanned as well so HTML integration points such
# as ``svg > foreignObject`` cannot hide an active meta refresh.
_REFRESH_INERT_TAGS = {"template"}


class _ActiveRefreshScanner(HTMLParser):
    """Find active or potentially active meta-refresh directives outside templates."""

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


def _check_single_expected_refresh(page: Path, expected: str) -> None:
    refreshes = _active_refreshes(page)
    expected_refresh = f"0; url={expected}"
    if refreshes != [expected_refresh]:
        raise PortalCheckError(
            f"Compatibility redirect refresh mismatch in {page}: expected exactly "
            f"{expected_refresh!r}, got {refreshes!r}"
        )


def check_clean_content_pages(root: Path, *, version: str, base_url: str) -> None:
    """Require expected canonicals and reject hidden refreshes in clean portal pages."""
    result = check_transition_portal(root, version=version, base_url=base_url)
    if result.get("contract") != "clean-current-transition":
        return
    root = root.resolve()
    base_url = base_url.rstrip("/") + "/"

    current_files = _current_relative_files(root)
    for relative in sorted(current_files):
        if relative.suffix.lower() != ".html":
            continue
        page = root / relative
        refreshes = _active_refreshes(page)
        if refreshes:
            raise PortalCheckError(
                f"Clean-current content page must not meta-refresh: {page}; found {refreshes!r}"
            )
        # MkDocs' generated 404 page intentionally has no canonical element.
        if relative.as_posix() == "404.html":
            continue
        expected = _page_url(base_url, "", relative)
        directives = _directives(page)
        if directives.canonicals != [expected]:
            raise PortalCheckError(
                f"Clean-current canonical mismatch in {page}: expected exactly {expected!r}, "
                f"got {directives.canonicals!r}"
            )

    for alias_root in (root / "docs" / "stable", root / "latest"):
        for relative in sorted(current_files):
            if relative.suffix.lower() != ".html":
                continue
            expected = _page_url(base_url, "", relative)
            _check_single_expected_refresh(alias_root / relative, expected)

    _check_single_expected_refresh(root / "docs" / "index.html", base_url)

    for published in result.get("versions", []):
        if not isinstance(published, str):
            raise PortalCheckError("Transition validator returned a non-string version")
        archive_root = root / "docs" / published
        legacy_root = root / published
        for page in archive_root.rglob("*.html"):
            relative = page.relative_to(archive_root)
            expected = _page_url(base_url, f"docs/{published}", relative)
            _check_single_expected_refresh(legacy_root / relative, expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    check_clean_content_pages(args.root, version=args.version, base_url=args.base_url)


if __name__ == "__main__":
    main()
