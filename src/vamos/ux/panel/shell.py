"""Theme, CSS and navigation helpers for the VAMOS Studio Panel app."""

from __future__ import annotations

import panel as pn

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
ACCENT = "#0d9488"
ACCENT_WARM = "#c86d1f"
FONT_STACK = '"Segoe UI", "Aptos", system-ui, -apple-system, sans-serif'
DISPLAY_FONT = '"Aptos Display", "Bahnschrift", "Segoe UI", system-ui, sans-serif'

_GLOBAL_CSS = f"""
/* ---------- Typography ---------- */
body, #content {{
    font-family: {FONT_STACK};
}}
h1, h2, h3, h4, h5 {{
    font-family: {DISPLAY_FONT};
    letter-spacing: -0.02em;
}}

/* ---------- Buttons ---------- */
.bk-btn-primary {{
    background: {ACCENT} !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-weight: 600;
    transition: all 0.2s ease;
}}
.bk-btn-primary:hover {{
    filter: brightness(1.1);
    transform: translateY(-1px);
}}
.bk-btn-success {{
    border-radius: 8px !important;
    font-weight: 600;
}}
.bk-btn-warning {{
    border-radius: 8px !important;
    font-weight: 600;
}}
.bk-btn-light {{
    border-radius: 8px !important;
    transition: all 0.2s ease;
}}

/* ---------- Sidebar nav buttons ---------- */
.sidenav-btn {{
    width: 100%;
    text-align: left;
    margin-bottom: 6px;
    padding: 10px 16px !important;
    font-size: 0.92rem;
}}
.sidenav-btn.bk-btn-primary {{
    border-left: 3px solid {ACCENT} !important;
}}

/* ---------- Inputs ---------- */
.bk-input {{
    border-radius: 6px !important;
}}
.bk-input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 2px rgba(15,108,189,0.15) !important;
}}

/* ---------- Labels ---------- */
.bk label {{
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}}

/* ---------- Alerts ---------- */
.alert-info {{
    border-radius: 8px !important;
}}

/* ---------- Code editor ---------- */
.ace_editor {{
    border-radius: 8px !important;
}}

/* ---------- Tabulator (tables) ---------- */
.pnx-tabulator .tabulator {{
    border-radius: 8px !important;
}}

/* ---------- Plotly ---------- */
.plotly .main-svg {{
    background: transparent !important;
}}
"""

# ---------------------------------------------------------------------------
# Page registry — slug -> (label, summary); the navigation, the welcome page
# and the app's renderers all follow it
# ---------------------------------------------------------------------------
PAGES: dict[str, tuple[str, str]] = {
    "welcome": ("Welcome", "What Studio does and where to start."),
    "catalog": (
        "Problems & Algorithms",
        "Browse the registered problems by encoding, dimensions and reference front, and the built-in algorithms "
        "with their configurable parameters, and open any of them in Run Experiment.",
    ),
    "problem_builder": (
        "Problem Builder",
        "Define a problem in Python or start from a template, preview its Pareto front, run it with a configurable "
        "algorithm and export a ready-to-run script.",
    ),
    "experiment": (
        "Run Experiment",
        "Run any registered problem with an algorithm whose operators and parameters come from its VAMOS "
        "configuration space, and browse the population generation by generation.",
    ),
    "batch": (
        "Batch Comparison",
        "Run several algorithms on several problems for many seeds as a durable study in the background; follow, "
        "stop and resume it, and compare normalized hypervolume with CSV and LaTeX exports.",
    ),
    "compare": (
        "Statistical Comparison",
        "Test whether the algorithms of a study differ on each problem and across problems: Mann-Whitney with Holm "
        "or Benjamini-Hochberg correction, A12 effect sizes, Friedman ranks with a critical-difference diagram, "
        "attainment surfaces, and LaTeX and SAES exports.",
    ),
    "results": (
        "Explore Results",
        "Load a saved study with the data-only readers, compare the fronts of each problem, filter them by "
        "feasibility and objective ranges, rank solutions with MCDM methods, export the ranking, and run a run again.",
    ),
}


def build_template(title: str = "VAMOS Studio") -> pn.template.FastListTemplate:
    """Return a configured FastListTemplate with the VAMOS theme.

    Light theme by default, with a toggle to switch to dark.
    """
    return pn.template.FastListTemplate(
        title=title,
        site="",
        accent_base_color=ACCENT,
        header_background=ACCENT,
        sidebar_width=310,
        main_layout=None,
        theme="default",
        theme_toggle=True,
        raw_css=[_GLOBAL_CSS],
    )


def build_sidebar_nav(current: str = "welcome") -> pn.Column:
    """Build the sidebar navigation column. Returns a Column of buttons."""
    buttons: list[pn.widgets.Button] = []
    for slug, (label, _summary) in PAGES.items():
        btn = pn.widgets.Button(
            name=label,
            button_type="primary" if slug == current else "light",
            css_classes=["sidenav-btn"],
            sizing_mode="stretch_width",
        )
        btn.tags = [slug]  # used to identify page on click
        buttons.append(btn)
    return pn.Column(*buttons, sizing_mode="stretch_width")
