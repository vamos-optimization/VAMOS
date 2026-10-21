"""Launcher for VAMOS Studio via the Panel runtime."""

from __future__ import annotations

import argparse
import ipaddress
import subprocess
import sys
from collections.abc import Sequence

STUDIO_APP_MODULE = "vamos.ux.panel.app"

LOOPBACK_ADDRESS = "127.0.0.1"


def _is_loopback_address(value: str) -> bool:
    address = value.strip().strip("[]")
    if address.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _build_panel_command(
    study_dir: str,
    panel_args: Sequence[str],
    *,
    address: str = LOOPBACK_ADDRESS,
    port: int = 5006,
    show: bool = True,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        STUDIO_APP_MODULE,
        f"--address={address}",
        f"--port={port}",
        f"--study-dir={study_dir}",
        *panel_args,
    ]
    if show:
        command.append("--show")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vamos studio", description="Launch VAMOS Studio (Panel).")
    parser.add_argument("--study-dir", default="results", help="A StudyManifest directory, or a folder to search for studies.")
    parser.add_argument("--address", default=LOOPBACK_ADDRESS, help="Bind address (default: loopback 127.0.0.1).")
    parser.add_argument("--port", type=_port, default=5006, help="Port for the Panel server.")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab.")
    parser.add_argument(
        "--allow-remote-binding",
        action="store_true",
        help="Acknowledge the risk and allow a non-loopback bind address.",
    )
    parser.add_argument(
        "--allow-websocket-origin",
        action="append",
        default=[],
        metavar="HOST:PORT",
        help=(
            "Allow pages served under HOST:PORT to open Studio sessions (repeatable). With a wildcard --address, "
            "this computer's names and addresses are allowed already; add proxies or DNS aliases here."
        ),
    )
    args, panel_args = parser.parse_known_args(argv)
    panel_args = [*(f"--allow-websocket-origin={origin}" for origin in args.allow_websocket_origin), *panel_args]

    if not _is_loopback_address(args.address) and not args.allow_remote_binding:
        parser.error("non-loopback binding is disabled; pass --allow-remote-binding to acknowledge the exposure risk")
    if not _is_loopback_address(args.address):
        sys.stderr.write(
            "WARNING: VAMOS Studio is binding beyond loopback. Anyone with network access may reach this experimental local app.\n"
        )

    try:
        import panel  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("VAMOS Studio requires the 'studio' extra: pip install -e \".[studio]\"") from exc

    cmd = _build_panel_command(
        args.study_dir,
        panel_args,
        address=args.address,
        port=args.port,
        show=not args.no_browser,
    )

    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


__all__ = ["LOOPBACK_ADDRESS", "main"]
