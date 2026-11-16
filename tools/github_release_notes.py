"""Write a release's notes for its GitHub Release page, with every link absolute.

``docs/project/release-notes-<version>.md`` links to other documentation pages relatively, for the
website. A GitHub Release page resolves those links against itself, so they break there. This writes
a copy whose links point to the version's immutable documentation archive,
``https://vamos-optimization.org/docs/<version>/...``: a link to a page of ``docs/`` becomes that
page's URL, an in-page anchor becomes the notes page's URL with the anchor, and a link to a file
outside ``docs/`` becomes its GitHub URL at the version's tag. Absolute links do not change, nor
does anything inside code.

Usage::

    python tools/github_release_notes.py 1.3.0 --output release-notes-github.md
    python tools/github_release_notes.py 1.2.0 --input old-notes.md --output out.md
"""

from __future__ import annotations

import argparse
import posixpath
import re
import sys
from pathlib import Path

DOCS_SITE = "https://vamos-optimization.org/docs"
REPOSITORY = "https://github.com/vamos-optimization/VAMOS"
NOTES_DIR = "docs/project"

_INLINE_LINK = re.compile(r"(!?\[[^\]\n]*\]\()([^)\s]+)((?:\s+\"[^\"\n]*\")?\))")
_REFERENCE_LINK = re.compile(r"^( {0,3}\[[^\]\n]+\]:[ \t]+)(\S+)", re.MULTILINE)
_ABSOLUTE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_CODE_SPAN = re.compile(r"`[^`\n]+`")


def absolute_link(target: str, version: str) -> str:
    """The absolute form of ``target``, a link found in ``docs/project/release-notes-<version>.md``."""
    if _ABSOLUTE.match(target) or target.startswith("//"):
        return target
    path, _, anchor = target.partition("#")
    fragment = f"#{anchor}" if anchor else ""
    site = f"{DOCS_SITE}/{version}"
    if not path:
        return f"{site}/project/release-notes-{version}/{fragment}"
    resolved = posixpath.normpath(posixpath.join(NOTES_DIR, path))
    if resolved == ".." or resolved.startswith("../"):
        raise ValueError(f"link {target!r} leaves the repository")
    if not resolved.startswith("docs/"):
        return f"{REPOSITORY}/blob/v{version}/{resolved}{fragment}"
    page = resolved[len("docs/") :]
    if page.endswith(".md"):
        page = page[: -len(".md")]
        page = page[: -len("index")] if page == "index" or page.endswith("/index") else f"{page}/"
    return f"{site}/{page}{fragment}"


def _rewrite_prose(text: str, version: str) -> str:
    """Rewrite the links of ``text``, except those that start inside an inline code span."""
    spans = [code.span() for code in _CODE_SPAN.finditer(text)]

    def outside_code(match: re.Match[str], rewritten: str) -> str:
        return match.group(0) if any(start <= match.start() < end for start, end in spans) else rewritten

    text = _INLINE_LINK.sub(lambda m: outside_code(m, m.group(1) + absolute_link(m.group(2), version) + m.group(3)), text)
    spans = [code.span() for code in _CODE_SPAN.finditer(text)]
    return _REFERENCE_LINK.sub(lambda m: outside_code(m, m.group(1) + absolute_link(m.group(2), version)), text)


def github_release_notes(markdown: str, version: str) -> str:
    """``markdown`` with its relative links made absolute; fenced code blocks are left as they are."""
    out: list[str] = []
    prose: list[str] = []
    fence: str | None = None
    for line in markdown.splitlines(keepends=True):
        stripped = line.lstrip()
        marker = stripped[:3] if stripped.startswith(("```", "~~~")) else None
        if fence is not None:
            out.append(line)
            if marker == fence:
                fence = None
        elif marker is not None:
            out.append(_rewrite_prose("".join(prose), version))
            prose = []
            fence = marker
            out.append(line)
        else:
            prose.append(line)
    out.append(_rewrite_prose("".join(prose), version))
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a release's notes for its GitHub Release page, with every link absolute.")
    parser.add_argument("version", help="the release version, for example 1.3.0")
    parser.add_argument("--input", type=Path, help="the notes to convert (default: docs/project/release-notes-<version>.md)")
    parser.add_argument("--output", type=Path, required=True, help="where to write the GitHub Release body")
    args = parser.parse_args(argv)
    source = args.input or Path(NOTES_DIR) / f"release-notes-{args.version}.md"
    notes = github_release_notes(source.read_text(encoding="utf-8"), args.version)
    args.output.write_text(notes, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
