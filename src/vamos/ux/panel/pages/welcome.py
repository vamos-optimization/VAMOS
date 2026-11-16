"""Welcome page for VAMOS Studio."""

from __future__ import annotations

from collections.abc import Sequence

import panel as pn

from vamos.ux.panel.shell import CODE_EXECUTION_PAGES, PAGES


def welcome_markdown(offered: Sequence[str] | None = None) -> str:
    """The welcome text; it lists the ``offered`` pages of the shell's page registry (all by default)."""
    pages = [PAGES[slug] for slug in (PAGES if offered is None else offered) if slug != "welcome"]
    steps = "\n".join(f"{index}. **{label}** — {summary}" for index, (label, summary) in enumerate(pages, start=1))
    return f"""
# Welcome to VAMOS Studio

**Vectorized Architecture for Multi-Objective Optimization Studies**

VAMOS Studio is an experimental workbench for browsing the registered
multi-objective problems, running VAMOS algorithms on them and exploring
saved studies.

---

### Pages

{steps}

Use the sidebar on the left to navigate between pages.
"""


def render_welcome(offered: Sequence[str] | None = None) -> pn.Column:
    """Render the welcome / onboarding page for the ``offered`` pages (all by default)."""
    hero = pn.pane.Markdown(welcome_markdown(offered), sizing_mode="stretch_width")
    if offered is None or set(CODE_EXECUTION_PAGES) <= set(offered):
        tip = (
            "**Tip:** for long or many runs, use **Batch Comparison**: it runs a durable study in a background process "
            "that keeps going if you close the browser. Run Experiment runs one run in the background while you "
            "watch it, and **Stop** ends it; leaving the page stops it too."
        )
    else:
        tip = (
            "**Batch Comparison** is off: this Studio serves a non-loopback address, and that page runs study "
            "workers with the account that serves it. Its owner can offer it by starting Studio with "
            "`--allow-remote-code-execution`."
        )
    return pn.Column(hero, pn.pane.Alert(tip, alert_type="info"), sizing_mode="stretch_both")
