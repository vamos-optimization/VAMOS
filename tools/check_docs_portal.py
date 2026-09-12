"""Validate a built VAMOS documentation portal before preview or deployment."""

from __future__ import annotations

import argparse
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
_INERT_TAGS = {"template", "noscript", "svg", "math"}
_HEAD_ALLOWED_TAGS = {"base", "link", "meta", "title", "style", "script", "noscript", "template"}
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class PortalCheckError(RuntimeError):
    """Raised when a built documentation portal violates its route contract."""


class _HeadDirectiveParser(HTMLParser):
    """Collect browser-active head directives and reject ambiguous HTML structure."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.canonicals: list[str] = []
        self.refreshes: list[str] = []
        self.errors: list[str] = []
        self._head_open = False
        self._head_seen = False
        self._inert_stack: list[str] = []
        self._head_element_stack: list[str] = []

    @staticmethod
    def _attribute_map(
        tag: str,
        attrs: list[tuple[str, str | None]],
        relevant: set[str],
    ) -> tuple[dict[str, str | None], list[str]]:
        normalized = [(name.casefold(), value) for name, value in attrs]
        duplicates = sorted(
            name
            for name in relevant
            if sum(1 for current, _ in normalized if current == name) > 1
        )
        data: dict[str, str | None] = {}
        for name, value in normalized:
            data.setdefault(name, value)
        return data, [f"duplicate {name!r} attribute on <{tag}>" for name in duplicates]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.casefold()
        if lowered in _INERT_TAGS:
            self._inert_stack.append(lowered)
            if self._head_open and lowered not in _HEAD_ALLOWED_TAGS:
                self.errors.append(f"foreign <{lowered}> subtree inside <head>")
            return
        if lowered == "head":
            if self._inert_stack:
                self.errors.append("<head> inside an inert or foreign subtree")
                return
            if self._head_seen or self._head_open:
                self.errors.append("multiple or nested <head> elements")
                return
            self._head_seen = True
            self._head_open = True
            return
        if not self._head_open or self._inert_stack:
            return
        if lowered not in _HEAD_ALLOWED_TAGS:
            self.errors.append(f"unexpected <{lowered}> element inside <head>")
            if lowered not in _VOID_TAGS:
                self._head_element_stack.append(lowered)
            return
        if self._head_element_stack:
            return
        if lowered == "link":
            data, errors = self._attribute_map(lowered, attrs, {"rel", "href"})
            self.errors.extend(errors)
            rel = data.get("rel") or ""
            if "canonical" in {token.casefold() for token in rel.split()}:
                href = data.get("href")
                if href is not None:
                    self.canonicals.append(href)
        elif lowered == "meta":
            data, errors = self._attribute_map(lowered, attrs, {"http-equiv", "content"})
            self.errors.extend(errors)
            if (data.get("http-equiv") or "").casefold() == "refresh":
                content = data.get("content")
                if content is not None:
                    self.refreshes.append(content)
        elif lowered not in _VOID_TAGS:
            self._head_element_stack.append(lowered)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in _INERT_TAGS:
            if not self._inert_stack or self._inert_stack[-1] != lowered:
                self.errors.append(f"mismatched </{lowered}> inert-subtree close")
            else:
                self._inert_stack.pop()
            return
        if lowered == "head":
            if not self._head_open:
                self.errors.append("closing </head> without an active <head>")
                return
            if self._inert_stack or self._head_element_stack:
                self.errors.append("closing </head> with an unclosed nested element")
            self._head_open = False
            self._head_element_stack.clear()
            return
        if self._head_open and self._head_element_stack:
            if self._head_element_stack[-1] != lowered:
                self.errors.append(f"mismatched </{lowered}> inside <head>")
            else:
                self._head_element_stack.pop()

    def finish(self) -> None:
        if self._head_open:
            self.errors.append("unclosed <head> element")
        if self._inert_stack:
            self.errors.append("unclosed inert or foreign subtree")


def _normalize_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise PortalCheckError("Base URL must be an absolute HTTPS URL without query or fragment")
    return value.rstrip("/") + "/"


def _read(path: Path) -> str:
    if not path.is_file():
        raise PortalCheckError(f"Missing required portal file: {path}")
    return path.read_text(encoding="utf-8")


def _directives(path: Path) -> _HeadDirectiveParser:
    parser = _HeadDirectiveParser()
    parser.feed(_read(path))
    parser.close()
    parser.finish()
    if parser.errors:
        raise PortalCheckError(f"Ambiguous or inert redirect markup in {path}: {parser.errors!r}")
    return parser


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
    directives = _directives(path)
    if directives.canonicals != [expected_target]:
        raise PortalCheckError(
            f"Redirect canonical mismatch in {path}: expected exactly {expected_target!r}, "
            f"got {directives.canonicals!r}"
        )
    expected_refresh = f"0; url={expected_target}"
    if directives.refreshes != [expected_refresh]:
        raise PortalCheckError(
            f"Redirect refresh mismatch in {path}: expected exactly {expected_refresh!r}, "
            f"got {directives.refreshes!r}"
        )


def _relative_files(tree: Path) -> set[Path]:
    if not tree.is_dir():
        raise PortalCheckError(f"Missing required portal directory: {tree}")
    return {path.relative_to(tree) for path in tree.rglob("*") if path.is_file()}


def _current_relative_files(root: Path) -> set[Path]:
    files: set[Path] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        first = relative.parts[0]
        if first in {"docs", "website", "latest"} or _VERSION_RE.fullmatch(first):
            continue
        files.add(relative)
    if not files:
        raise PortalCheckError("Clean current documentation tree is empty")
    return files


def _require_same_routes(source_files: set[Path], alias_tree: Path, *, label: str) -> None:
    alias_files = _relative_files(alias_tree)
    if alias_files != source_files:
        missing = sorted(str(path) for path in source_files - alias_files)[:5]
        extra = sorted(str(path) for path in alias_files - source_files)[:5]
        raise PortalCheckError(
            f"{label} route inventory does not match its source tree; "
            f"missing={missing!r}, extra={extra!r}"
        )


def _check_redirect_tree(tree: Path, *, base_url: str, canonical_prefix: str) -> int:
    files = _relative_files(tree)
    pages = sorted(path for path in files if path.suffix.lower() == ".html")
    if not pages:
        raise PortalCheckError(f"No redirect pages found under {tree}")
    for relative in pages:
        _check_redirect(
            tree / relative,
            expected_target=_page_url(base_url, canonical_prefix, relative),
        )
    return len(pages)


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
        directives = _directives(page)
        if len(directives.canonicals) > 1:
            raise PortalCheckError(f"Multiple active canonical URLs in {page}: {directives.canonicals!r}")
        if not directives.canonicals:
            continue
        canonical = directives.canonicals[0]
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
    for relative in sorted(_current_relative_files(root)):
        if relative.suffix.lower() != ".html":
            continue
        page = root / relative
        directives = _directives(page)
        if len(directives.canonicals) > 1:
            raise PortalCheckError(f"Multiple active canonical URLs in {page}: {directives.canonicals!r}")
        if not directives.canonicals:
            continue
        expected = _page_url(base_url, "", relative)
        canonical = directives.canonicals[0]
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


def _manifest(root: Path, *, version: str) -> list[str]:
    try:
        payload = json.loads(_read(root / "docs" / "versions.json"))
    except json.JSONDecodeError as exc:
        raise PortalCheckError("docs/versions.json is not valid JSON") from exc
    versions = payload.get("versions")
    if payload.get("stable") != version or not isinstance(versions, list) or version not in versions:
        raise PortalCheckError("versions.json does not identify the requested version as stable")
    if any(not isinstance(item, str) or _VERSION_RE.fullmatch(item) is None for item in versions):
        raise PortalCheckError("versions.json contains an invalid semantic version")
    return versions


def _check_version_aliases(
    root: Path,
    *,
    versions: list[str],
    base_url: str,
) -> int:
    count = 0
    for published in versions:
        archived = root / "docs" / published
        legacy = root / published
        archived_files = _relative_files(archived)
        _require_same_routes(archived_files, legacy, label=f"Legacy {published} alias")
        count += _check_redirect_tree(
            legacy,
            base_url=base_url,
            canonical_prefix=f"docs/{published}",
        )
    return count


def check_portal(root: Path, *, version: str, base_url: str) -> dict[str, object]:
    if _VERSION_RE.fullmatch(version) is None:
        raise PortalCheckError("Version must be a numeric major.minor.patch value")
    root = root.resolve()
    if not root.is_dir():
        raise PortalCheckError(f"Portal root does not exist: {root}")
    base_url = _normalize_base_url(base_url)
    versions = _manifest(root, version=version)

    immutable = root / "docs" / version
    stable_alias = root / "docs" / "stable"
    current_files = _current_relative_files(root)
    _require_same_routes(current_files, stable_alias, label="docs/stable compatibility alias")
    _require_same_routes(current_files, root / "latest", label="latest compatibility alias")

    canonical_count = _check_current_tree(root, base_url=base_url)
    canonical_count += _check_canonical_tree(
        root,
        immutable,
        base_url=base_url,
        expected_prefix=f"{base_url}docs/{version}/",
    )
    canonical_count += _check_redirect_tree(stable_alias, base_url=base_url, canonical_prefix="")
    canonical_count += _check_redirect_tree(root / "latest", base_url=base_url, canonical_prefix="")
    canonical_count += _check_canonical_tree(
        root,
        root / "website",
        base_url=base_url,
        expected_prefix=f"{base_url}website/",
    )

    _check_redirect(root / "docs" / "index.html", expected_target=base_url)
    canonical_count += _check_version_aliases(root, versions=versions, base_url=base_url)

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
