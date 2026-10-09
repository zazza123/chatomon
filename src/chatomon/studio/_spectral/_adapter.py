"""The adapter: a small server, for this machine only, that answers requests with turns of the atom.

It is the server `spectral-bridge` hands every request to. It is not made for a
browser: it has no cookie, and takes a key in the first segment of its path.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import Awaitable, Callable

from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from chatomon.studio._spectral import _shapes as shapes

MAX_RUNNING = 4
"""How many requests run their turn at the same time; the others wait."""

MAX_BODY = 16 * 1024 * 1024
"""The largest body that is read: the largest frame the relay sends."""

_NO_STORE = {"cache-control": "no-store"}

Answer = Callable[[shapes.Asked, str | None], Awaitable[str]]
"""Runs a turn on what a request asks and returns the reply, or raises `Unanswered`.

It is also given the id the answer will have when a later request can name it to
continue the conversation, and `None` on a route that has no way to name one.
"""


class Unanswered(Exception):
    """A request that gets no reply: its turn left none, or it could not be run.

    Args:
        status: The status of the answer.
        kind: A name for the error.
        message: A sentence of the studio's own: never the text of an exception.
    """

    def __init__(self, status: int, kind: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.kind = kind
        self.message = message


def _refusal(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse(shapes.error(kind, message), status_code=status, headers=_NO_STORE)


async def _body(request: Request) -> bytes | None:
    """Read the body of a request.

    Returns:
        The body, or `None` when it is larger than `MAX_BODY`.
    """
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


async def _gone(receive: Receive) -> None:
    """Return when the caller closes the connection."""
    while (await receive())["type"] != "http.disconnect":
        pass


def create_adapter(
    *, secret: str, port: int, answer: Answer, rejected: Callable[[], None]
) -> ASGIApp:
    """Build the adapter's ASGI app.

    Args:
        secret: The key of this connection: the first segment of every path.
        port: The port the adapter listens on, on `127.0.0.1`.
        answer: Runs a turn and returns its reply.
        rejected: Called for every request that is refused for its shape.

    Returns:
        An app that answers `POST /<secret>/v1/responses` and
        `POST /<secret>/v1/chat/completions`, and refuses everything else.
    """
    slots = asyncio.Semaphore(MAX_RUNNING)
    host = f"127.0.0.1:{port}"

    async def turn(asked: shapes.Asked, kept_as: str | None) -> str:
        async with slots:
            return await answer(asked, kept_as)

    async def respond(request: Request, receive: Receive) -> Response | None:
        given, slash, rest = request.url.path.removeprefix("/").partition("/")
        route = slash + rest
        known = secrets.compare_digest(given.encode(), secret.encode())
        if not known or request.method != "POST" or route not in shapes.ROUTES:
            return _refusal(404, "NotFound", "Nothing answers here.")
        if request.headers.get("host", "").lower() != host:
            return _refusal(400, "BadHost", "The adapter answers only on its own address.")
        if "origin" in request.headers:
            # A page always says where it comes from; the program this is made for never does.
            return _refusal(403, "BadOrigin", "The adapter does not answer pages.")
        kind = request.headers.get("content-type", "").partition(";")[0].strip().lower()
        if kind != "application/json":
            return _refusal(415, "UnsupportedMediaType", "Send the request as application/json.")
        raw = await _body(request)
        if raw is None:
            return _refusal(413, "TooLarge", "The request is larger than 16 MiB.")
        try:
            body = json.loads(raw)
        except ValueError:
            return _refusal(400, "BadJSON", "The body is not JSON.")
        try:
            asked = shapes.read(route, body)
        except shapes.ShapeError as exc:
            rejected()
            return _refusal(exc.status, "BadRequest", exc.message)
        answer_id = shapes.new_id(route)
        # Only a response can be named by a later request: a chat brings all of itself.
        kept_as = answer_id if route == shapes.RESPONSES else None
        running = asyncio.ensure_future(turn(asked, kept_as))
        gone = asyncio.ensure_future(_gone(receive))
        await asyncio.wait({running, gone}, return_when=asyncio.FIRST_COMPLETED)
        gone.cancel()
        await asyncio.gather(gone, return_exceptions=True)
        if not running.done():
            # The caller went away: nobody is left to read a reply.
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
            return None
        try:
            text = running.result()
        except Unanswered as exc:
            return _refusal(exc.status, exc.kind, exc.message)
        except Exception:
            return _refusal(500, "AdapterFailed", "The adapter failed.")
        return JSONResponse(shapes.reply(route, text, answer_id), headers=_NO_STORE)

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await receive()
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            return
        try:
            response = await respond(Request(scope, receive), receive)
        except ClientDisconnect:
            # The caller went away while its request was being read: nobody is left to answer.
            return
        if response is not None:
            await response(scope, receive, send)

    return app
