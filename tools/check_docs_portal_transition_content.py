"""Supplement the transition validator with a no-refresh rule for clean content pages.

The temporary transition validator accepts both the legacy portal and the proposed clean-current
portal. Legacy root pages are themselves redirects, so this rule applies only after the artifact
has been classified as ``clean-current-transition``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from check_docs_portal_transition import (
    PortalCheckError,
    _current_relative_files,
    _directives,
    check_transition_portal,
)


def check_clean_content_pages(
    root: Path,
    *,
    version: str,
    base_url: str,
) -> None:
    """Reject browser-active meta refreshes on every clean-current HTML content page."""
    result = check_transition_portal(root, version=version, base_url=base_url)
    if result.get("contract") != "clean-current-transition":
        return

    root = root.resolve()
    for relative in sorted(_current_relative_files(root)):
        if relative.suffix.lower() != ".html":
            continue
        page = root / relative
        directives = _directives(page)
        if directives.refreshes:
            raise PortalCheckError(
                f"Clean-current content page must not meta-refresh: {page}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    check_clean_content_pages(
        args.root,
        version=args.version,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()
