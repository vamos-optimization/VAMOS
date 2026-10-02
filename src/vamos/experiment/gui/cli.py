"""``vamos gui``: argument parsing and launch of the experimental NiceGUI interface.

Parsing, validation, and ``--help`` never import NiceGUI; the optional
dependencies are checked only when the server is about to start. The bind
address defaults to loopback, and a non-loopback address requires an explicit
``--allow-remote-binding`` acknowledgement because the GUI has no
authentication and can launch runs on the host.
"""

from __future__ import annotations

import argparse
import importlib.util
import ipaddress
import sys
from collections.abc import Sequence
from pathlib import Path

LOOPBACK_ADDRESS = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_RESULTS_ROOT = "results"
JOBS_DIRNAME = "gui-runs"
REQUIRED_MODULES = ("nicegui", "plotly")
INSTALL_HINT = 'Install the optional GUI dependencies with: pip install "vamos-optimization[gui]"'


def is_loopback_address(value: str) -> bool:
    """Return whether ``value`` names a loopback interface."""
    address = value.strip().strip("[]")
    if address.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


def missing_dependencies() -> list[str]:
    """Return the optional GUI modules that are not importable."""
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vamos gui",
        description=(
            "Launch the experimental VAMOS GUI: explore canonical runs and launch built-in "
            "problem/algorithm runs with live progress. Requires the optional 'gui' extra."
        ),
    )
    parser.add_argument(
        "results_root",
        nargs="?",
        default=DEFAULT_RESULTS_ROOT,
        help=f"Directory scanned for canonical runs (default: ./{DEFAULT_RESULTS_ROOT}).",
    )
    parser.add_argument(
        "--jobs-dir",
        default=None,
        help=f"Directory for runs launched from the GUI (default: <results_root>/{JOBS_DIRNAME}).",
    )
    parser.add_argument("--address", default=LOOPBACK_ADDRESS, help=f"Bind address (default: loopback {LOOPBACK_ADDRESS}).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to listen on (default: {DEFAULT_PORT}).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab on start.")
    parser.add_argument("--native", action="store_true", help="Open a native window instead of a browser tab (requires pywebview).")
    parser.add_argument(
        "--allow-remote-binding",
        action="store_true",
        help="Acknowledge the risk and allow a non-loopback bind address (the GUI has no authentication).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not 0 < args.port < 65536:
        parser.error(f"--port must be between 1 and 65535, got {args.port}")
    remote = not is_loopback_address(args.address)
    if remote and not args.allow_remote_binding:
        parser.error("non-loopback binding is disabled; pass --allow-remote-binding to acknowledge the exposure risk")
    missing = missing_dependencies()
    if missing:
        sys.stderr.write(f"vamos gui: missing optional dependencies: {', '.join(missing)}. {INSTALL_HINT}\n")
        return 2
    results_root = Path(args.results_root).expanduser().resolve()
    jobs_root = Path(args.jobs_dir).expanduser().resolve() if args.jobs_dir else results_root / JOBS_DIRNAME
    if remote:
        sys.stderr.write(
            "WARNING: the VAMOS GUI is binding beyond loopback. Anyone who can reach this address can launch runs "
            "on this machine; there is no authentication.\n"
        )
    from .app import GuiSettings, run_gui

    run_gui(
        GuiSettings(results_root=results_root, jobs_root=jobs_root),
        host=args.address,
        port=args.port,
        show=not args.no_browser,
        native=args.native,
    )
    return 0


__all__ = ["DEFAULT_PORT", "JOBS_DIRNAME", "LOOPBACK_ADDRESS", "build_parser", "is_loopback_address", "main", "missing_dependencies"]
