/* Progressive enhancements: readable overflow regions and local example downloads. */
(function () {
  "use strict";
  let observer;

  function enhance() {
    if (observer) observer.disconnect();
    const regions = document.querySelectorAll(".md-typeset__scrollwrap");
    function markOverflow(region) {
      const overflowing = region.scrollWidth > region.clientWidth + 1;
      if (overflowing) {
        region.setAttribute("tabindex", "0");
        region.setAttribute("role", "region");
        region.setAttribute("aria-label", "Scrollable table; use arrow keys to see more columns");
      } else {
        region.removeAttribute("tabindex");
        region.removeAttribute("role");
        region.removeAttribute("aria-label");
      }
    }
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(entries => entries.forEach(entry => markOverflow(entry.target)));
    }
    regions.forEach(region => {
      markOverflow(region);
      if (observer) observer.observe(region);
    });
    document.querySelectorAll('.md-content a[href]').forEach(link => {
      const target = new URL(link.getAttribute("href"), document.baseURI);
      if (target.origin === window.location.origin && /\/assets\/(tutorials|studies|coverage)\/[^/]+\.py$/.test(target.pathname)) {
        link.setAttribute("download", "");
      }
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", enhance);
  else enhance();
  if (typeof document$ !== "undefined") document$.subscribe(enhance);
})();
