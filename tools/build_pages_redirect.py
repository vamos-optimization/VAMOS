"""Build the GitHub Pages site that sends readers to the canonical documentation host.

The site holds two pages. ``index.html`` redirects the project root to the base
URL. ``404.html`` serves every other address: on a ``*.github.io`` project site
it maps ``/<repository>/<path>?<query>#<hash>`` to ``<base-url><path>?<query>#<hash>``.
The documentation host normalizes its own aliases (``latest/...``,
``docs/stable/...`` and root-level ``<version>/...``), so the mapping only drops
the repository segment.

The tool uses only the standard library because the Pages workflow installs no
project dependencies.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

# The leading path segment of a project site. This text is both a Python and a
# JavaScript regular expression, so the generated script and `redirect_target`
# share one definition.
REPOSITORY_SEGMENT = r"^\/[^\/]*\/?"
PAGES_HOST_SUFFIX = ".github.io"


def normalize_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("Documentation base URL must be an absolute HTTPS URL without query or fragment")
    return value.rstrip("/") + "/"


def redirect_target(location: str, base_url: str) -> str:
    """Return the documentation address that the generated 404 page sends ``location`` to."""
    parts = urlsplit(location)
    if not (parts.hostname or "").endswith(PAGES_HOST_SUFFIX):
        return base_url
    path = re.sub(REPOSITORY_SEGMENT, "", parts.path, count=1)
    query = f"?{parts.query}" if parts.query else ""
    fragment = f"#{parts.fragment}" if parts.fragment else ""
    return f"{base_url}{path}{query}{fragment}"


def _page(base_url: str, *, head: list[str], body: list[str]) -> str:
    escaped = html.escape(base_url, quote=True)
    lines = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<link rel="canonical" href="{escaped}">',
        "<title>VAMOS documentation has moved</title>",
        *head,
        "</head><body>",
        *body,
        "</body></html>",
    ]
    return "\n".join(lines) + "\n"


def index_page(base_url: str) -> str:
    escaped = html.escape(base_url, quote=True)
    return _page(
        base_url,
        head=[f'<meta http-equiv="refresh" content="0; url={escaped}">'],
        body=[f'<p>The VAMOS documentation is at <a href="{escaped}">{escaped}</a>.</p>'],
    )


def not_found_page(base_url: str) -> str:
    escaped = html.escape(base_url, quote=True)
    javascript_base = json.dumps(base_url).replace("</", "<\\/")
    script = [
        "<script>",
        "(function () {",
        f"  var base = {javascript_base};",
        "  var here = window.location;",
        "  var target = base;",
        f"  if (/{re.escape(PAGES_HOST_SUFFIX)}$/i.test(here.hostname)) {{",
        f'    target = base + here.pathname.replace(/{REPOSITORY_SEGMENT}/, "") + here.search + here.hash;',
        "  }",
        "  window.location.replace(target);",
        "})();",
        "</script>",
    ]
    body = [
        "<p>Redirecting to the VAMOS documentation&hellip;</p>",
        f'<noscript><p>Continue to <a href="{escaped}">{escaped}</a>.</p></noscript>',
    ]
    return _page(base_url, head=script, body=body)


def build_pages_redirect(base_url: str, output: Path) -> None:
    base_url = normalize_base_url(base_url)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Pages output: {output}")
    output.mkdir(parents=True)
    (output / "index.html").write_text(index_page(base_url), encoding="utf-8")
    (output / "404.html").write_text(not_found_page(base_url), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="canonical documentation host that the Pages site redirects to")
    parser.add_argument("--output", type=Path, required=True, help="new directory for the Pages site")
    args = parser.parse_args()
    build_pages_redirect(args.base_url, args.output)


if __name__ == "__main__":
    main()
