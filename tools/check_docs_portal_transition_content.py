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
    _relative_files,
    check_transition_portal,
)

# ``noscript`` can contain active meta refresh content for users with scripting
# disabled, so it must not be treated as inert by the refresh-specific scanner.
_REFRESH_INERT_TAGS = _INERT_TAGS - {"noscript"}


class _ActiveRefreshScanner(HTMLParser):
    """Find active meta-refresh directives anywhere outside inert subtrees."""

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
            self.errors.append("unclosed inert or foreign subtree")


def _active_refreshes(page: Path) -> list[str]:
    scanner = _ActiveRefreshScanner()
    scanner.feed(page.read_text(encoding="utf-8"))
    scanner.close()
    scanner.finish()
    if scanner.errors:
        raise PortalCheckError(f"Ambiguous meta-refresh markup in {page}: {scanner.errors!r}")
    return scanner.refreshes


def _require_no_active_refresh(page: Path, *, label: str) -> None:
    refreshes = _active_refreshes(page)
    if refreshes:
        raise PortalCheckError(f"{label} must not meta-refresh: {page}; found {refreshes!r}")


def _require_single_redirect_refresh(page: Path, *, expected_target: str) -> None:
    expected = f"0; url={expected_target}"
    refreshes = _active_refreshes(page)
    if refreshes != [expected]:
        raise PortalCheckError(
            f"Compatibility redirect refresh mismatch in {page}: expected exactly {expected!r}, "
            f"got {refreshes!r}"
        )


def _check_redirect_tree_refreshes(tree: Path, *, base_url: str, canonical_prefix: str) -> None:
    for relative in sorted(_relative_files(tree)):
        if relative.suffix.lower() != ".html":
            continue
        _require_single_redirect_refresh(
            tree / relative,
            expected_target=_page_url(base_url, canonical_prefix, relative),
        )


def check_clean_content_pages(root: Path, *, version: str, base_url: str) -> None:
    """Validate active refresh behavior across every clean-current publication surface."""
    result = check_transition_portal(root, version=version, base_url=base_url)
    if result.get("contract") != "clean-current-transition":
        return

    root = root.resolve()
    base_url = base_url.rstrip("/") + "/"

    # Current content must never redirect. MkDocs' generated 404.html is the one
    # current HTML file that intentionally has no canonical metadata.
    for relative in sorted(_current_relative_files(root)):
        if relative.suffix.lower() != ".html":
            continue
        page = root / relative
        expected = _page_url(base_url, "", relative)
        directives = _directives(page)
        if relative == Path("404.html"):
            if directives.canonicals not in ([], [expected]):
                raise PortalCheckError(
                    f"Clean-current 404 canonical mismatch in {page}: expected none or {expected!r}, "
                    f"got {directives.canonicals!r}"
                )
        elif directives.canonicals != [expected]:
            raise PortalCheckError(
                f"Clean-current canonical mismatch in {page}: expected exactly {expected!r}, "
                f"got {directives.canonicals!r}"
            )
        _require_no_active_refresh(page, label="Clean-current content page")

    # Canonical content outside the current root must likewise never gain an
    # active refresh from an untrusted preview artifact.
    versions = result.get("versions")
    if not isinstance(versions, list) or any(not isinstance(item, str) for item in versions):
        raise PortalCheckError("Transition validator returned an invalid versions list")
    for published in versions:
        for page in (root / "docs" / published).rglob("*.html"):
            _require_no_active_refresh(page, label=f"Immutable docs/{published} page")
    for page in (root / "website").rglob("*.html"):
        _require_no_active_refresh(page, label="Legacy website content page")

    # Every compatibility redirect may contain exactly the one expected refresh.
    _require_single_redirect_refresh(root / "docs" / "index.html", expected_target=base_url)
    _check_redirect_tree_refreshes(root / "docs" / "stable", base_url=base_url, canonical_prefix="")
    _check_redirect_tree_refreshes(root / "latest", base_url=base_url, canonical_prefix="")
    for published in versions:
        _check_redirect_tree_refreshes(
            root / published,
            base_url=base_url,
            canonical_prefix=f"docs/{published}",
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
