"""Main entry point for the VAMOS Studio Panel application."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import panel as pn
import param

from vamos.ux.panel.shell import PAGES, build_sidebar_nav, build_template

if TYPE_CHECKING:
    from vamos.ux.studio.experiment_setup import ExperimentSetup


def _ensure_panel_extension() -> None:
    pn.extension("codeeditor", "plotly", "tabulator", sizing_mode="stretch_width")


class StudioApp(param.Parameterized):
    """Top-level application state coordinating page navigation."""

    page = param.Selector(
        default="welcome",
        objects=list(PAGES.keys()),
        doc="Currently active page slug.",
    )
    study_dir = param.String(default="results", doc="Root directory for study results.")
    focus_study = param.String(default="", doc="Study that Explore Results loads when it opens next.")
    pending_setup = param.Parameter(default=None, precedence=-1, doc="Settings that Run Experiment loads when it opens next.")

    # ---- internal ----
    _nav_col = param.ClassSelector(class_=pn.Column, precedence=-1)
    _main_area = param.ClassSelector(class_=pn.Column, precedence=-1)

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._nav_col = build_sidebar_nav(self.page)
        self._main_area = pn.Column(sizing_mode="stretch_both")
        self._exit_hooks: list[Callable[[], None]] = []
        self._wire_nav_buttons()
        self._refresh_main()

    # ---- navigation wiring ----

    def _wire_nav_buttons(self) -> None:
        for btn in self._nav_col:
            if isinstance(btn, pn.widgets.Button):
                btn.on_click(self._on_nav_click)

    def _on_nav_click(self, event: Any) -> None:
        slug = event.obj.tags[0] if event.obj.tags else "welcome"
        self.page = slug

    @param.depends("page", watch=True)
    def _refresh_main(self) -> None:
        # Update sidebar button styles
        for btn in self._nav_col:
            if isinstance(btn, pn.widgets.Button):
                slug = btn.tags[0] if btn.tags else ""
                btn.button_type = "primary" if slug == self.page else "light"

        # Let the page being left stop its background work, then lazy-load the selected page
        hooks, self._exit_hooks = self._exit_hooks, []
        for hook in hooks:
            hook()
        self._main_area.clear()
        self._main_area.append(self._load_page(self.page))

    def _load_page(self, slug: str) -> pn.viewable.Viewable:
        renderer = PAGE_RENDERERS.get(slug)
        if renderer is None:
            return pn.pane.Markdown(f"## Unknown page: {slug}")
        try:
            return renderer(self)
        except Exception as exc:
            return pn.pane.Alert(f"**{PAGES[slug][0]}** could not be opened: {exc}", alert_type="danger")

    # ---- page services ----

    def on_page_exit(self, hook: Callable[[], None]) -> None:
        """Run ``hook`` when the user leaves the current page."""
        self._exit_hooks.append(hook)

    def open_results(self, study_root: str) -> None:
        """Show ``study_root`` in Explore Results."""
        self.focus_study = study_root
        self.page = "results"

    def open_compare(self, study_root: str) -> None:
        """Analyze ``study_root`` in Statistical Comparison."""
        self.focus_study = study_root
        self.page = "compare"

    def open_experiment(self, setup: ExperimentSetup) -> None:
        """Open Run Experiment with a catalog problem or the settings of a stored run."""
        self.pending_setup = setup
        self.page = "experiment"

    # ---- public: composable template ----

    def view(self) -> pn.template.FastListTemplate:
        from vamos.ux.studio.environment import environment_markdown, environment_report

        tmpl = build_template()
        tmpl.sidebar.append(
            pn.pane.Markdown(
                "### VAMOS Studio",
                align="center",
                styles={"text-align": "center", "padding": "0.5rem 0"},
            )
        )
        tmpl.sidebar.append(pn.layout.Divider())
        tmpl.sidebar.append(self._nav_col)
        tmpl.sidebar.append(
            pn.Card(
                pn.pane.Markdown(environment_markdown(environment_report()), sizing_mode="stretch_width"),
                title="Environment",
                collapsed=True,
                sizing_mode="stretch_width",
            )
        )
        tmpl.main.append(self._main_area)
        return tmpl


# ---------------------------------------------------------------------------
# Page renderers — one per entry of the shell's page registry
# ---------------------------------------------------------------------------


def _render_welcome(_app: StudioApp) -> pn.viewable.Viewable:
    from vamos.ux.panel.pages.welcome import render_welcome

    return render_welcome()


def _render_catalog(app: StudioApp) -> pn.viewable.Viewable:
    from vamos.ux.panel.pages.catalog import render_catalog

    return render_catalog(open_experiment=app.open_experiment)


def _render_problem_builder(app: StudioApp) -> pn.viewable.Viewable:
    from vamos.ux.panel.pages.problem_builder import render_problem_builder

    return render_problem_builder(on_exit=app.on_page_exit)


def _render_experiment(app: StudioApp) -> pn.viewable.Viewable:
    from vamos.ux.panel.pages.experiment import render_experiment

    setup, app.pending_setup = app.pending_setup, None
    return render_experiment(on_exit=app.on_page_exit, setup=setup)


def _render_batch(app: StudioApp) -> pn.viewable.Viewable:
    from vamos.ux.panel.pages.batch import render_batch

    return render_batch(
        study_dir=Path(app.study_dir), open_results=app.open_results, open_compare=app.open_compare, on_exit=app.on_page_exit
    )


def _render_compare(app: StudioApp) -> pn.viewable.Viewable:
    from vamos.ux.panel.pages.compare import render_compare

    focus, app.focus_study = app.focus_study, ""
    return render_compare(study_dir=Path(app.study_dir), focus_study=focus or None)


def _render_results(app: StudioApp) -> pn.viewable.Viewable:
    from vamos.ux.panel.pages.results import render_results

    focus, app.focus_study = app.focus_study, ""
    return render_results(study_dir=Path(app.study_dir), focus_study=focus or None, open_experiment=app.open_experiment)


PAGE_RENDERERS: dict[str, Callable[[StudioApp], pn.viewable.Viewable]] = {
    "welcome": _render_welcome,
    "catalog": _render_catalog,
    "problem_builder": _render_problem_builder,
    "experiment": _render_experiment,
    "batch": _render_batch,
    "compare": _render_compare,
    "results": _render_results,
}


# ---------------------------------------------------------------------------
# Server entry point (``vamos studio`` runs ``python -m vamos.ux.panel.app``)
# ---------------------------------------------------------------------------


def create_app(study_dir: str = "results") -> pn.template.FastListTemplate:
    """Build one Studio session; Panel calls this for every browser session."""
    _ensure_panel_extension()
    app = StudioApp(study_dir=study_dir)
    return app.view()


def parse_serve_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m vamos.ux.panel.app", description="Serve VAMOS Studio with Panel.")
    parser.add_argument("--study-dir", default="results")
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--show", action="store_true", help="Open a browser tab.")
    parser.add_argument(
        "--allow-websocket-origin",
        action="append",
        default=[],
        help="Extra host[:port] allowed to open the session websocket (repeatable).",
    )
    return parser.parse_args(argv)


def websocket_origins(address: str, port: int, extra: Sequence[str] = ()) -> list[str]:
    """The ``host:port`` origins whose pages may open a Studio session.

    A browser sends the host it used to reach Studio. A specific address is
    reached through itself; a wildcard address (``0.0.0.0`` or ``::``) through
    any name or address of this computer, so those are allowed. Names this
    computer cannot see, such as a proxy or DNS alias, go in ``extra``.
    """
    origins = [f"localhost:{port}", *extra]
    if address.strip("[]") in {"0.0.0.0", "::", ""}:
        origins += [f"{host}:{port}" for host in _local_host_names() if f"{host}:{port}" not in origins]
    elif ":" not in address:  # Bokeh's allowlist cannot express IPv6 literals; browse those via localhost.
        origins.append(f"{address}:{port}")
    return origins


def _local_host_names() -> list[str]:
    """This computer's host names and IPv4 addresses, as browsers on the network address it."""
    import socket

    names = {"127.0.0.1", socket.gethostname(), socket.getfqdn()}
    try:
        names.update(str(sockaddr[0]) for *_, sockaddr in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:  # An unresolvable host name still allows the names above.
        logging.getLogger(__name__).debug("Cannot list this computer's addresses.", exc_info=True)
    return sorted(name for name in names if name and ":" not in name)


def serve(argv: Sequence[str] | None = None) -> int:
    """Serve Studio at ``/app`` until the process is stopped."""
    args = parse_serve_args(argv)
    origins = websocket_origins(args.address, args.port, args.allow_websocket_origin)
    pn.serve(
        {"app": lambda: create_app(args.study_dir)},
        address=args.address,
        port=args.port,
        show=args.show,
        websocket_origin=origins,
        title="VAMOS Studio",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(serve(sys.argv[1:]))
