"""Who can open VAMOS Studio, and which pages a server bound beyond loopback offers."""

from __future__ import annotations

import asyncio
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("panel")

import panel as pn  # noqa: E402

from vamos.ux.panel import app as studio_app  # noqa: E402
from vamos.ux.panel import launcher  # noqa: E402

pytestmark = pytest.mark.studio

CODE_EXECUTION_LABELS = {"Problem Builder", "Batch Comparison"}


def _record_serve(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_serve(panels: dict[str, Any], **kwargs: Any) -> None:
        calls.append({**kwargs, "template": panels["app"]()})

    monkeypatch.setattr(studio_app.pn, "serve", fake_serve)
    return calls


def _nav_labels(template: Any) -> set[str]:
    buttons = [button for item in template.sidebar for button in item.select(pn.widgets.Button)]
    return {button.name for button in buttons if "sidenav-btn" in button.css_classes}


def test_serve_rejects_non_loopback_without_acknowledgement(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_serve(monkeypatch)

    with pytest.raises(SystemExit) as exited:
        studio_app.serve(["--address", "0.0.0.0"])

    assert exited.value.code == 2
    assert calls == []


def test_remote_binding_hides_code_execution_pages(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls = _record_serve(monkeypatch)

    studio_app.serve(["--address=0.0.0.0", "--port=5010", "--allow-remote-binding"])
    warning = capsys.readouterr().err
    studio_app.serve(["--address=0.0.0.0", "--port=5010", "--allow-remote-binding", "--allow-remote-code-execution"])
    studio_app.serve(["--port=5010"])
    remote, remote_with_code, local = (_nav_labels(call["template"]) for call in calls)

    assert CODE_EXECUTION_LABELS.isdisjoint(remote)
    assert "Run Experiment" in remote
    assert CODE_EXECUTION_LABELS <= remote_with_code
    assert CODE_EXECUTION_LABELS <= local
    assert "--allow-remote-code-execution" in warning
    app = studio_app.StudioApp(code_execution=False)
    for slug in ("problem_builder", "batch"):
        page = app._load_page(slug)
        assert type(page).__name__ == "Alert"
        assert "--allow-remote-code-execution" in page.object


def test_every_launch_draws_its_own_token_unless_told_not_to(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls = _record_serve(monkeypatch)

    studio_app.serve(["--port=5010"])
    studio_app.serve(["--port=5010"])
    tokens = re.findall(r"/app\?token=([\w-]+)", capsys.readouterr().out)
    studio_app.serve(["--port=5010", "--no-auth"])
    unprotected = capsys.readouterr()

    assert [call["auth_provider"].token for call in calls[:2]] == tokens
    assert len(set(tokens)) == 2
    assert "auth_provider" not in calls[2]
    assert "http://127.0.0.1:5010/app\n" in unprotected.out
    assert "WARNING" in unprotected.err and "--no-auth" in unprotected.err


def test_the_browser_receives_the_token_in_a_private_file(monkeypatch: pytest.MonkeyPatch) -> None:
    from vamos.ux.panel import _access

    opened: list[tuple[str, str]] = []

    def fake_open(uri: str, new: int = 0) -> bool:
        path = Path(urllib.request.url2pathname(urllib.parse.urlsplit(uri).path))
        opened.append((uri, path.read_text(encoding="utf-8")))
        return True

    def fake_serve(_panels: dict[str, Any], *, loop: Any, **_kwargs: Any) -> None:
        loop.run_sync(lambda: asyncio.sleep(0.05))  # Panel runs the loop's callbacks once the server listens.

    monkeypatch.setattr(_access.webbrowser, "open", fake_open)
    monkeypatch.setattr(studio_app.pn, "serve", fake_serve)
    loop = asyncio.new_event_loop()  # The event loop of a fresh `python -m vamos.ux.panel.app` process.
    asyncio.set_event_loop(loop)
    try:
        studio_app.serve(["--port=5010", "--show"])
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    [(uri, page)] = opened
    assert uri.startswith("file:") and "token" not in uri
    assert re.search(r"http://127\.0\.0\.1:5010/app\?token=[\w-]+", page)
    assert not Path(urllib.request.url2pathname(urllib.parse.urlsplit(uri).path)).exists()


def test_launcher_rejects_unknown_arguments(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def started(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("the launcher started the server")

    monkeypatch.setattr(launcher.subprocess, "run", started)

    with pytest.raises(SystemExit) as exited:
        launcher.main(["--no-browser", "--dev"])

    assert exited.value.code == 2
    assert "vamos studio: error: unrecognized arguments: --dev" in capsys.readouterr().err


def test_launcher_forwards_the_access_options(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        launcher.subprocess, "run", lambda command, check: commands.append(command) or subprocess.CompletedProcess(command, 0)
    )

    argv = ["--address=0.0.0.0", "--allow-remote-binding", "--allow-remote-code-execution", "--no-auth", "--no-browser"]
    assert launcher.main(argv) == 0

    for option in ("--address=0.0.0.0", "--allow-remote-binding", "--allow-remote-code-execution", "--no-auth"):
        assert option in commands[0]
    assert studio_app.parse_serve_args(commands[0][3:]).no_auth is True


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_port(port: int, server: subprocess.Popen[str], timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            pytest.fail(f"Studio exited with code {server.returncode}")
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    pytest.fail("Studio did not start listening")


async def _first_message(port: int, cookie: str | None) -> str:
    """Open a Studio session the way a script would: no page, no Origin, a made-up session id."""
    from bokeh.util.token import generate_jwt_token, generate_session_id
    from tornado.httpclient import HTTPRequest
    from tornado.websocket import websocket_connect

    request = HTTPRequest(f"ws://127.0.0.1:{port}/app/ws", headers={"Cookie": cookie} if cookie else {})
    connection = await websocket_connect(request, subprotocols=["bokeh", generate_jwt_token(generate_session_id())])
    try:
        message = await asyncio.wait_for(connection.read_message(), timeout=60)
    finally:
        connection.close()
    return str(message)


def test_studio_rejects_session_without_secret(tmp_path: Path) -> None:
    from tornado.httpclient import HTTPClientError

    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-u", "-m", "vamos.ux.panel.app", f"--port={port}", f"--study-dir={tmp_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line) for line in server.stdout or ()], daemon=True).start()
    try:
        _wait_for_port(port, server)

        with pytest.raises(HTTPClientError) as refused:
            message = asyncio.run(_first_message(port, cookie=None))
            pytest.fail(f"a session opened without the launch token: {message}")
        assert refused.value.code == 403

        token = _printed_token(lines)
        with pytest.raises(urllib.error.HTTPError) as wrong:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/login?token=not-{token}", timeout=60)
        assert wrong.value.code == 403
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/app?token={token}", timeout=60) as page:
            cookie = "; ".join(header.split(";", 1)[0] for header in page.headers.get_all("Set-Cookie", []))

        assert '"msgtype": "ACK"' in asyncio.run(_first_message(port, cookie=cookie))
    finally:
        server.terminate()
        server.wait(timeout=30)


def _printed_token(lines: queue.Queue[str], timeout: float = 30.0) -> str:
    seen: list[str] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            seen.append(lines.get(timeout=0.5))
        except queue.Empty:
            continue
        if found := re.search(r"/app\?token=([\w-]+)", seen[-1]):
            return found.group(1)
    pytest.fail("Studio printed no link with its access token:\n" + "".join(seen))
