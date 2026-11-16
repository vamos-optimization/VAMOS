"""Who may open VAMOS Studio: a token per launch, which a browser trades for a cookie.

Anyone who can reach the server's port, including other users of the same
computer, could otherwise open a session and run code with the account that
serves Studio. Each launch draws a token, and a browser that presents it once,
in the printed link or on the login page, keeps a cookie that admits its later
requests and session websockets.
"""

from __future__ import annotations

import hmac
import html
import os
import secrets
import tempfile
import webbrowser
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from bokeh.server.auth_provider import AuthProvider
from tornado.web import HTTPError, RequestHandler

LOGIN_PATH = "/login"
_USER = "studio"

_LOGIN_PAGE = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>VAMOS Studio</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 36rem; margin: 4rem auto; padding: 0 1rem;">
<h1>VAMOS Studio</h1>
<p>{message}</p>
<form method="get" action="login">
<input type="hidden" name="next" value="{next_page}">
<label>Access token <input name="token" type="password" autocomplete="off" autofocus></label>
<button type="submit">Open Studio</button>
</form>
</body>
</html>
"""

_REDIRECT_PAGE = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={url}"><title>VAMOS Studio</title></head>
<body><p>Opening <a href="{url}">VAMOS Studio</a>.</p></body>
</html>
"""


def _same(given: str, expected: str) -> bool:
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


class TokenAuth(AuthProvider):
    """Admit the browsers that present this launch's token, and nobody else."""

    def __init__(self, token: str, port: int) -> None:
        self.token = token
        self.cookie_name = f"vamos-studio-{port}"  # Browsers share cookies across the ports of a host.
        self._cookie_value = secrets.token_urlsafe(32)
        super().__init__()

    @property
    def get_user(self) -> Callable[[RequestHandler], str | None]:
        return self.user

    @property
    def login_url(self) -> str:
        return LOGIN_PATH

    @property
    def login_handler(self) -> type[RequestHandler]:
        return LoginHandler

    def user(self, handler: RequestHandler) -> str | None:
        """The user of a request with this launch's cookie or token, else ``None``."""
        cookie = handler.get_cookie(self.cookie_name)
        if cookie is not None and _same(cookie, self._cookie_value):
            return _USER
        token = handler.get_query_argument("token", None)
        return _USER if token is not None and self.admit(handler, token) else None

    def admit(self, handler: RequestHandler, token: str) -> bool:
        """Give ``handler``'s browser the cookie if ``token`` is this launch's."""
        if not _same(token, self.token):
            return False
        handler.set_cookie(self.cookie_name, self._cookie_value, httponly=True, samesite="Lax")
        return True


class LoginHandler(RequestHandler):
    """Ask for the token, and send a browser that gives it to the page it wanted."""

    def get(self) -> None:
        auth = getattr(self.application, "auth_provider", None)
        if not isinstance(auth, TokenAuth):
            raise HTTPError(404)
        next_page = _local_path(self.get_query_argument("next", "/app"))
        token = self.get_query_argument("token", None)
        if token is not None and auth.admit(self, token):
            self.redirect(next_page)
            return
        if token is None:
            message = "Open Studio with the link that <code>vamos studio</code> printed when it started, or enter its token."
        else:
            self.set_status(403)
            message = "That is not the access token of this Studio server. Use the link it printed when it started."
        self.set_header("Cache-Control", "no-store")
        self.finish(_LOGIN_PAGE.format(message=message, next_page=html.escape(next_page, quote=True)))


def _local_path(value: str) -> str:
    """``value`` as a path on this server; anything else, such as another site, becomes ``/app``."""
    parts = urlsplit(value)
    path = parts.path
    if parts.scheme or parts.netloc or not path.startswith("/") or path.startswith("//") or "\\" in path:
        return "/app"
    return path


def new_token() -> str:
    """A fresh access token for one launch."""
    return secrets.token_urlsafe(32)


def open_in_browser(url: str) -> Path:
    """Open ``url`` through a private redirect file, which keeps a token off the browser's command line.

    Other users can read a process's command line on many systems, but not this
    file, which only its owner can read. The caller deletes it when Studio stops.
    """
    descriptor, name = tempfile.mkstemp(prefix="vamos-studio-", suffix=".html")
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(_REDIRECT_PAGE.format(url=html.escape(url, quote=True)))
    path = Path(name)
    webbrowser.open(path.as_uri(), new=2)
    return path


__all__ = ["LOGIN_PATH", "LoginHandler", "TokenAuth", "new_token", "open_in_browser"]
