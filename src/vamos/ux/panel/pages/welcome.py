"""Welcome page for VAMOS Studio."""

from __future__ import annotations

import panel as pn

from vamos.ux.panel.shell import PAGES


def welcome_markdown() -> str:
    """The welcome text; its page list comes from the shell's page registry."""
    pages = [(label, summary) for slug, (label, summary) in PAGES.items() if slug != "welcome"]
    steps = "\n".join(f"{index}. **{label}** — {summary}" for index, (label, summary) in enumerate(pages, start=1))
    return f"""
# Welcome to VAMOS Studio

**Vectorized Architecture for Multi-Objective Optimization Studies**

VAMOS Studio is an experimental workbench for defining multi-objective
problems, running VAMOS algorithms on them and exploring saved studies.

---

### Pages

{steps}

Use the sidebar on the left to navigate between pages.
"""


def render_welcome() -> pn.Column:
    """Render the welcome / onboarding page."""
    hero = pn.pane.Markdown(welcome_markdown(), sizing_mode="stretch_width")
    tips = pn.pane.Alert(
        "**Tip:** for long or many runs, use **Batch Comparison**: it runs a durable study in a background process "
        "that keeps going if you close the browser. Problem Builder and Run Experiment run in the Studio server "
        "process, and their page updates when the run finishes.",
        alert_type="info",
    )
    return pn.Column(hero, tips, sizing_mode="stretch_both")
