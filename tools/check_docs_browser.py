"""Exercise the built documentation in desktop/mobile Chromium and retain QA evidence.

Run in the isolated environment from docs-browser-requirements.txt. This checks
representative tasks and accessibility mechanics, not complete WCAG conformance.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROUTES = (
    "",
    "tutorials/",
    "tutorials/t01-first-result/",
    "tutorials/t03-pareto-decision/",
    "tutorials/t06-fair-study/",
    "tutorials/t07-quality-and-budget/",
    "tutorials/t08-study-report/",
    "tutorials/engineering-design/",
    "reference/algorithms/",
    "reference/api/optimization/",
    "guide/installation/",
)


class PortalHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        if code != 404:
            super().send_error(code, message, explain)
            return
        body = (Path(self.directory) / "404.html").read_bytes()
        self.send_response(404)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


def assert_layout(page) -> dict:
    result = page.evaluate("""() => ({
      width: innerWidth,
      scrollWidth: document.documentElement.scrollWidth,
      h1: document.querySelectorAll('main h1').length,
      missingAlt: document.querySelectorAll('main img:not([alt])').length,
      brokenImages: [...document.querySelectorAll('main img')].filter(i => !i.complete || !i.naturalWidth).length
    })""")
    assert result["scrollWidth"] <= result["width"] + 1, f"Horizontal page overflow at {page.url}: {result}"
    assert result["h1"] == 1 and not result["missingAlt"] and not result["brokenImages"], (page.url, result)
    return result


def search(page, query: str, expected: str | None) -> None:
    field = page.locator(".md-search__input")
    if not page.locator("#__search").is_checked():
        opener = page.locator('.md-header label[for="__search"]')
        if opener.is_visible():
            opener.click()
        else:
            field.click()
    field.fill(query)
    if expected is None:
        page.get_by_text("No matching documents", exact=True).wait_for()
    else:
        page.locator(f'.md-search-result__list a[href*="{expected}"]').first.wait_for()


def run_journey(browser, base: str, root: Path, output: Path, width: int) -> dict:
    context = browser.new_context(viewport={"width": width, "height": 960 if width > 600 else 844}, reduced_motion="reduce")
    page = context.new_page()
    page.set_default_timeout(15000)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    results = []
    try:
        for index, route in enumerate(ROUTES):
            page.goto(base + route, wait_until="networkidle")
            results.append({"route": route, **assert_layout(page)})
            if index in (0, 1, 2, 7, 8):
                page.screenshot(path=str(output / f"{width}-{index}.png"), full_page=True)

        page.goto(base + "tutorials/t01-first-result/", wait_until="networkidle")
        page.keyboard.press("Tab")
        assert page.evaluate("document.activeElement.textContent").strip() == "Skip to content"
        page.keyboard.press("Enter")
        with page.expect_download() as download:
            page.locator(".vamos-download").first.click()
        saved = output / f"{width}-t01.py"
        download.value.save_as(str(saved))
        assert (
            hashlib.sha256(saved.read_bytes()).digest()
            == hashlib.sha256((root / "assets/tutorials/t01-first-result.py").read_bytes()).digest()
        )
        for slug in ("t02-constrained-problem", "t03-pareto-decision", "t06-fair-study", "t07-quality-and-budget", "t08-study-report"):
            page.locator('.vamos-lesson-nav a[rel="next"]').click()
            page.wait_for_url(f"**/tutorials/{slug}/")

        for query, target in (
            ("MOEA/D", "algorithms/moead/"),
            ("NSGA-II", "algorithms/nsgaii/"),
            ("StudySpec", "study"),
            ("hypervolume", "t07-quality-and-budget/"),
            ("constraints", "constraints"),
            ("zzqnonexistenttopicqq", None),
        ):
            search(page, query, target)
        page.locator(".md-search__input").press("Escape")

        page.goto(base + "reference/algorithms/", wait_until="networkidle")
        if width < 600:
            page.locator('.md-header label[for="__drawer"]').click()
            assert page.locator(".md-sidebar--primary").is_visible()
            page.locator('.md-overlay[for="__drawer"]').click(position={"x": width - 5, "y": 100})
            region = page.locator('.md-typeset__scrollwrap[tabindex="0"]').first
            region.focus()
            before = region.evaluate("el => el.scrollLeft")
            region.press("ArrowRight")
            page.wait_for_function("document.activeElement.scrollLeft > 0")
            assert region.evaluate("el => el.scrollLeft") > before

        page.locator('label[title="Switch to dark mode"]').click()
        assert page.locator("body").get_attribute("data-md-color-scheme") == "slate"
        page.screenshot(path=str(output / f"{width}-dark.png"), full_page=True)
        assert_layout(page)
        assert page.evaluate("getComputedStyle(document.documentElement).scrollBehavior") == "auto"

        response = page.goto(base + "missing/nested/page/", wait_until="networkidle")
        assert response.status == 404
        page.get_by_role("heading", name="Let's find your next step").wait_for()
        assert_layout(page)
        page.screenshot(path=str(output / f"{width}-404.png"), full_page=True)
        page.get_by_label("What are you looking for?").fill("MOEA/D")
        page.get_by_role("button", name="Search", exact=True).click()
        page.wait_for_url("**/?q=MOEA%2FD")
        page.locator('.md-search-result__list a[href*="algorithms/moead/"]').first.wait_for()
        assert not errors, errors
        return {
            "viewport": width,
            "pages": results,
            "tasks": "navigation, download bytes, search, keyboard, overflow, dark mode, reduced motion, 404 recovery",
            "javascript_errors": errors,
        }
    except Exception:
        page.screenshot(path=str(output / f"{width}-failure.png"), full_page=True)
        (output / f"{width}-failure.html").write_text(page.content(), encoding="utf-8")
        raise
    finally:
        context.close()


def main() -> None:
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.is_relative_to(root):
        parser.error("QA evidence must stay outside the portal")
    output.mkdir(parents=True, exist_ok=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(PortalHandler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                result = [
                    run_journey(browser, f"http://127.0.0.1:{server.server_port}/", root, output, width) for width in (1440, 390, 320)
                ]
                # Critical links and local downloads remain usable without JavaScript.
                context = browser.new_context(java_script_enabled=False)
                page = context.new_page()
                page.goto(f"http://127.0.0.1:{server.server_port}/tutorials/t01-first-result/")
                assert page.locator(".vamos-download").is_visible()
                page.locator('.vamos-lesson-nav a[rel="next"]').click()
                assert urlsplit(page.url).path.endswith("/tutorials/t02-constrained-problem/")
                context.close()
                (output / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({"viewports": [1440, 390, 320], "journeys": len(result), "javascript_disabled": "passed"}))
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
