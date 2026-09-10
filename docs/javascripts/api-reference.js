/* Keep old API #bookmarks useful without crossing origins or documentation versions. */
(function () {
  "use strict";

  function forwardLegacyBookmark() {
    const directory = document.getElementById("api-legacy-bookmarks");
    if (!directory || !window.location.hash) return;
    let fragment;
    try {
      fragment = decodeURIComponent(window.location.hash.slice(1));
    } catch (_) {
      return; // A malformed or unknown fragment must not break the index.
    }
    const link = document.getElementById(fragment);
    if (!link || !directory.contains(link) || !link.hasAttribute("data-api-legacy")) return;
    const current = new URL(window.location.href);
    const base = new URL(current.href);
    base.pathname = base.pathname.replace(/\/index\.html$/, "/");
    if (!base.pathname.endsWith("/")) base.pathname += "/";
    const target = new URL(link.getAttribute("href"), base);
    const apiRoot = new URL("../api/", base);
    if (target.origin !== current.origin || !target.pathname.startsWith(apiRoot.pathname)) return;
    target.search = current.search;
    window.location.replace(target.href);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", forwardLegacyBookmark);
  } else {
    forwardLegacyBookmark();
  }
  window.addEventListener("hashchange", forwardLegacyBookmark);
  // Also handle a future activation of Material/Zensical instant navigation.
  if (typeof document$ !== "undefined") document$.subscribe(forwardLegacyBookmark);
})();
