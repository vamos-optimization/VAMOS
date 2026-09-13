from __future__ import annotations

from pathlib import Path

from tools.check_docs_portal import _directives


def test_noscript_meta_refresh_is_treated_as_active(tmp_path: Path) -> None:
    page = tmp_path / "page.html"
    page.write_text(
        "<!doctype html><html><head>"
        '<link rel="canonical" href="https://vamos-optimization.org/example/">'
        "</head><body><noscript>"
        '<meta http-equiv="refresh" content="0; url=https://evil.invalid/">'
        "</noscript></body></html>",
        encoding="utf-8",
    )

    directives = _directives(page)
    assert directives.canonicals == ["https://vamos-optimization.org/example/"]
    assert directives.refreshes == ["0; url=https://evil.invalid/"]
