"""The studio's defences: Host and Origin checks and the per-run token.

These are the rules for every chatomon tool that opens a server (CLAUDE.md, "Local tools").
"""

from __future__ import annotations

import ipaddress
import secrets
import socket

from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
"""Hosts that only this machine can reach."""


_UNSAFE = frozenset({"POST", "PATCH", "PUT", "DELETE"})

SECURITY_HEADERS = {
    "content-security-policy": (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
}
"""Sent with every response: only the studio's own files run, and no other page can frame it."""

LOCKED_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>chatomon studio</title>
<style>
body {
  margin: 0; min-height: 100vh; display: grid; place-items: center;
  font: 16px/1.5 system-ui, sans-serif; background: #f8f8f9; color: #1a1c26;
}
@media (prefers-color-scheme: dark) { body { background: #121319; color: #efeff3; } }
main { max-width: 32rem; padding: 24px; }
</style>
</head>
<body>
<main>
<h1>chatomon studio</h1>
<p>Open the link printed in the terminal where you started the studio.
It carries the key for this run.</p>
</main>
</body>
</html>
"""


def is_bind_all(host: str) -> bool:
    """Tell whether `host` listens on every address, however it is spelled.

    `0`, `0.0`, `::0` or `::ffff:0.0.0.0` bind everything just like `0.0.0.0`,
    so the host is resolved and its addresses are checked.

    Args:
        host: The address or name the server is asked to listen on.

    Returns:
        Whether any address `host` resolves to is the unspecified one.
    """
    if not host:
        return True
    try:
        found = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return False
    for *_, sockaddr in found:
        address = ipaddress.ip_address(str(sockaddr[0]).split("%")[0])
        mapped = getattr(address, "ipv4_mapped", None)
        if address.is_unspecified or (mapped is not None and mapped.is_unspecified):
            return True
    return False


def new_token() -> str:
    """Return a fresh token for one run of the studio."""
    return secrets.token_urlsafe(32)


def cookie_name(port: int) -> str:
    """Return the cookie name; browsers do not separate cookies by port."""
    return f"chatomon_studio_{port}"


def allowed_hosts(host: str, port: int) -> frozenset[str]:
    """Return the `Host` header values the studio answers to."""
    names = {"127.0.0.1", "localhost", "[::1]"}
    if host not in LOOPBACK:
        names.add(f"[{host}]" if ":" in host else host.lower())
    hosts = {f"{name}:{port}" for name in names}
    if port == 80:
        hosts |= names  # browsers leave the default port out of Host
    return frozenset(hosts)


def _error(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"type": kind, "message": message}}, status_code=status)


def _same(given: str, token: str) -> bool:
    return secrets.compare_digest(given.encode(), token.encode())


class Guard:
    """ASGI middleware: checks Host on every request, Origin on changes, and the token."""

    def __init__(self, app: ASGIApp, *, token: str, host: str, port: int) -> None:
        self.app = app
        self.token = token
        self.hosts = allowed_hosts(host, port)
        self.cookie = cookie_name(port)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            # The studio has no websocket: refuse rather than pass an unchecked scope on.
            await receive()
            await send({"type": "websocket.close", "code": 1008})
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
                if "cache-control" not in headers:
                    headers["cache-control"] = "no-cache"
            await send(message)

        response = self._check(Request(scope))
        if response is None:
            await self.app(scope, receive, send_with_headers)
        else:
            await response(scope, receive, send_with_headers)

    def _check(self, request: Request) -> Response | None:
        host = request.headers.get("host", "").lower()
        if host not in self.hosts:
            return _error(400, "BadHost", "the studio only answers on the address it printed")
        origin = (request.headers.get("origin") or "").lower()
        if request.method in _UNSAFE and origin != f"http://{host}":
            return _error(403, "BadOrigin", "changes must come from the studio's own page")
        given = request.query_params.get("token")
        if request.url.path == "/" and request.method == "GET" and given is not None:
            if not _same(given, self.token):
                return HTMLResponse(LOCKED_PAGE, status_code=401)
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(self.cookie, self.token, httponly=True, samesite="strict", path="/")
            return response
        if _same(request.cookies.get(self.cookie, ""), self.token):
            return None
        if request.url.path == "/":
            return HTMLResponse(LOCKED_PAGE, status_code=401)
        return _error(401, "Unauthorized", "open the link printed in the terminal")
