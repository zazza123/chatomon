"""The studio's page, its API and the event stream of a turn."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from importlib.resources import files
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from chatomon.errors import ChatomonConfigError
from chatomon.studio._events import Event
from chatomon.studio._security import Guard
from chatomon.studio._session import Session, StudioBusyError, UnknownOrbitalError

_STATIC = files("chatomon.studio") / "static"
_NO_STORE = {"Cache-Control": "no-store"}


def _error(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"type": kind, "message": message}}, status_code=status)


async def _body(request: Request) -> dict[str, Any] | None:
    try:
        body = await request.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


async def _stream(queue: asyncio.Queue[Event | None]) -> AsyncIterator[str]:
    while (event := await queue.get()) is not None:
        yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app(session: Session, *, token: str, host: str, port: int) -> Starlette:
    """Build the studio's ASGI app for one session.

    Args:
        session: The session the page drives.
        token: The token of this run.
        host: The address the server listens on.
        port: The port the server listens on.

    Returns:
        The Starlette app, guarded by Host, Origin and token checks.
    """

    async def index(request: Request) -> Response:
        page = (_STATIC / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(page, headers=_NO_STORE)

    async def get_state(request: Request) -> Response:
        return JSONResponse(session.state(), headers=_NO_STORE)

    async def post_turn(request: Request) -> Response:
        body = await _body(request)
        message = body.get("message") if body is not None else None
        if not isinstance(message, str) or not message.strip():
            return _error(422, "EmptyMessage", 'write a message first: {"message": "..."}')
        try:
            queue = session.start_turn(message)
        except StudioBusyError as exc:
            return _error(409, "Busy", str(exc))
        return StreamingResponse(_stream(queue), media_type="text/event-stream", headers=_NO_STORE)

    async def patch_orbital(request: Request) -> Response:
        body = await _body(request)
        enabled = body.get("enabled") if body is not None else None
        if not isinstance(enabled, bool):
            return _error(422, "InvalidBody", 'send {"enabled": true} or {"enabled": false}')
        try:
            session.set_enabled(request.path_params["side"], request.path_params["name"], enabled)
        except UnknownOrbitalError as exc:
            return _error(404, "UnknownOrbital", str(exc))
        except StudioBusyError as exc:
            return _error(409, "Busy", str(exc))
        except ChatomonConfigError as exc:
            return _error(409, "Refused", str(exc))
        return JSONResponse(session.state())

    async def post_reset(request: Request) -> Response:
        try:
            session.reset()
        except StudioBusyError as exc:
            return _error(409, "Busy", str(exc))
        return JSONResponse(session.state())

    routes = [
        Route("/", index, methods=["GET"]),
        Route("/api/state", get_state, methods=["GET"]),
        Route("/api/turns", post_turn, methods=["POST"]),
        Route("/api/orbitals/{side}/{name}", patch_orbital, methods=["PATCH"]),
        Route("/api/conversation/reset", post_reset, methods=["POST"]),
        Mount("/static", StaticFiles(packages=[("chatomon.studio", "static")]), name="static"),
    ]
    middleware = [Middleware(Guard, token=token, host=host, port=port)]
    return Starlette(routes=routes, middleware=middleware)
