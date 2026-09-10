# Maintaining the API documentation

The [API Reference](../reference/api_reference.md) is a short task index. The canonical Python reference lives in `docs/reference/api/`, alongside the existing [documentation architecture](documentation-architecture.md). The legacy `website/docs/` tree is not a second source of API signatures.

## Page contract

Each public page gives a short purpose, the public import, links to a relevant guide, and generated signatures. Render public objects through the supported facades. Select individual symbols rather than dumping unrelated implementation modules into the user index. Keep `show_root_full_path: false` and `show_source: false`: implementation paths and full source listings should not dominate the reading surface.

Algorithm configuration gets one page per built-in algorithm. The fluent builder is obtained through `Config.builder()`, not through an import of its private class. Its method signatures are still generated from the implementation; do not replace them with a manually copied table.

Stable public references, experimental APIs, and internal implementation references have distinct navigation and labels. Documenting an internal object must not promote it to a stable interface.

## Bookmark compatibility

The former `reference/api_reference/` URL remains the API index. `reference/api/legacy-bookmarks.json` freezes 152 former section and symbol anchors, recorded from the documentation at commit `ed100ea43a3f5dec0204c8297e7defdd5e020bcc`. The index contains a matching directory of ordinary HTML links with the old IDs. This directory stays collapsed for ordinary reading and is usable without JavaScript.

`javascripts/api-reference.js` forwards a recognized fragment to its exact new destination. It keeps the current host, documentation version, and query string. It does not redirect unrecognized fragments, other pages, or unsafe targets. Fragments are handled in the browser because HTTP servers do not receive them.

When reorganizing an object, update its destination in both the inventory and the index. Do not delete a frozen old ID. The tests compare both representations and check every target fragment against the generated HTML in both `docs/stable/` and `docs/1.0.0/`.

## Validation

```bash
python -m pytest tests/docs/test_modular_api.py tests/docs/test_documentation_architecture.py
node --test tests/docs/api_reference_redirects.test.cjs
python -m mkdocs build --strict
```

The existing isolated Zensical compatibility workflow must also succeed. Review the built index, a function page, and an algorithm configuration page at desktop and mobile widths, with keyboard navigation and JavaScript disabled. A passing source test is not a visual review.

This restructuring changes documentation organization only. It does not alter Python contracts, package metadata, hosting, or the existing archive-preservation rules. No production deployment is triggered by this document.

## Next documentation increments

The next independently reviewable increments are a canonical NSGA-II tutorial and algorithm catalogue; a Quickstart that produces and explains a figure; a complete reproducible-comparison tutorial; and problem, operator, indicator, and backend catalogues. Those increments must check implementation capabilities and use executable source for examples, rather than copying claims or defaults from another framework.
