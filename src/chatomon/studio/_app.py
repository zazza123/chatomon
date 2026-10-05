"""The studio's page, its API and the event stream of a turn."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from importlib.resources import files
from typing import Any

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from chatomon.errors import ChatomonError
from chatomon.studio._chains import InvalidRequestError, UnknownOrbitalError
from chatomon.studio._events import Event
from chatomon.studio._kinds import InvalidSettingError
from chatomon.studio._security import LOOPBACK, Guard
from chatomon.studio._session import Session, StudioBusyError

_STATIC = files("chatomon.studio") / "static"
_NO_STORE = {"Cache-Control": "no-store"}


_PATCH_HELP = 'send {"side": "input", "name": "..."} and one of "enabled", "settings" or "position"'


def _error(status: int, kind: str, message: str, **extra: Any) -> JSONResponse:
    body = {"error": {"type": kind, "message": message, **extra}}
    return JSONResponse(body, status_code=status)


def _refusal(exc: ChatomonError) -> JSONResponse:
    """Turn an error of the session into its response.

    Returns:
        The JSON error, with the status that says who refused and why.
    """
    if isinstance(exc, UnknownOrbitalError):
        return _error(404, "UnknownOrbital", str(exc))
    if isinstance(exc, StudioBusyError):
        return _error(409, "Busy", str(exc))
    if isinstance(exc, InvalidRequestError):
        return _error(422, "InvalidBody", str(exc))
    if isinstance(exc, InvalidSettingError):
        extra = {} if exc.setting is None else {"setting": exc.setting}
        return _error(422, "InvalidSetting", str(exc), **extra)
    # A refusal by the orbital or the session, or by the chain when the atom is built.
    return _error(409, "Refused", str(exc))


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

    def state() -> dict[str, Any]:
        current = session.state()
        current["studio"]["local"] = host in LOOPBACK
        return current

    async def index(request: Request) -> Response:
        page = (_STATIC / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(page, headers=_NO_STORE)

    async def get_state(request: Request) -> Response:
        return JSONResponse(state(), headers=_NO_STORE)

    async def post_turn(request: Request) -> Response:
        body = await _body(request)
        message = body.get("message") if body is not None else None
        if not isinstance(message, str) or not message.strip():
            return _error(422, "EmptyMessage", 'write a message first: {"message": "..."}')
        try:
            message.encode("utf-8")
        except UnicodeEncodeError:
            return _error(422, "InvalidMessage", "the message has characters that are not text")
        try:
            queue = session.start_turn(message)
        except StudioBusyError as exc:
            return _error(409, "Busy", str(exc))
        return StreamingResponse(_stream(queue), media_type="text/event-stream", headers=_NO_STORE)

    async def patch_orbital(request: Request) -> Response:
        body = await _body(request) or {}
        side, name = body.get("side"), body.get("name")
        given = [key for key in ("enabled", "settings", "position") if key in body]
        if not isinstance(side, str) or not isinstance(name, str) or len(given) != 1:
            return _error(422, "InvalidBody", _PATCH_HELP)
        value = body[given[0]]
        try:
            if given[0] == "enabled" and isinstance(value, bool):
                session.switch(side, name, value)
            elif given[0] == "settings" and isinstance(value, dict) and value:
                session.edit(side, name, value)
            elif given[0] == "position" and isinstance(value, int) and not isinstance(value, bool):
                session.move(side, name, value)
            else:
                return _error(422, "InvalidBody", _PATCH_HELP)
        except ChatomonError as exc:
            return _refusal(exc)
        return JSONResponse(state())

    async def post_orbital(request: Request) -> Response:
        body = await _body(request) or {}
        side, kind, settings = body.get("side"), body.get("kind"), body.get("settings", {})
        if not isinstance(side, str) or not isinstance(kind, str) or not isinstance(settings, dict):
            return _error(
                422, "InvalidBody", 'send {"side": "input", "kind": "...", "settings": {...}}'
            )
        try:
            session.add(side, kind, settings)
        except ChatomonError as exc:
            return _refusal(exc)
        return JSONResponse(state())

    async def delete_orbital(request: Request) -> Response:
        body = await _body(request) or {}
        side, name = body.get("side"), body.get("name")
        if not isinstance(side, str) or not isinstance(name, str):
            return _error(422, "InvalidBody", 'send {"side": "input", "name": "..."}')
        try:
            session.delete(side, name)
        except ChatomonError as exc:
            return _refusal(exc)
        return JSONResponse(state())

    async def reset_orbitals(request: Request) -> Response:
        body = await _body(request) or {}
        side, name = body.get("side"), body.get("name")
        one = isinstance(side, str) and isinstance(name, str)
        if not one and (side is not None or name is not None):
            return _error(422, "InvalidBody", 'send {} or {"side": "input", "name": "..."}')
        try:
            session.reset_orbitals(side, name) if one else session.reset_orbitals()
        except ChatomonError as exc:
            return _refusal(exc)
        return JSONResponse(state())

    async def undo(request: Request) -> Response:
        try:
            session.undo()
        except ChatomonError as exc:
            return _refusal(exc)
        return JSONResponse(state())

    async def get_code(request: Request) -> Response:
        return JSONResponse(session.code(), headers=_NO_STORE)

    async def post_reset(request: Request) -> Response:
        try:
            session.reset_conversation()
        except StudioBusyError as exc:
            return _error(409, "Busy", str(exc))
        return JSONResponse(state())

    routes = [
        Route("/", index, methods=["GET"]),
        Route("/api/state", get_state, methods=["GET"]),
        Route("/api/turns", post_turn, methods=["POST"]),
        Route("/api/code", get_code, methods=["GET"]),
        Route("/api/orbitals", patch_orbital, methods=["PATCH"]),
        Route("/api/orbitals", post_orbital, methods=["POST"]),
        Route("/api/orbitals", delete_orbital, methods=["DELETE"]),
        Route("/api/orbitals/reset", reset_orbitals, methods=["POST"]),
        Route("/api/orbitals/undo", undo, methods=["POST"]),
        Route("/api/conversation/reset", post_reset, methods=["POST"]),
        Mount("/static", StaticFiles(packages=[("chatomon.studio", "static")]), name="static"),
    ]

    async def http_error(request: Request, exc: Exception) -> Response:
        status = exc.status_code if isinstance(exc, HTTPException) else 500
        kinds = {404: "NotFound", 405: "MethodNotAllowed"}
        detail = exc.detail if isinstance(exc, HTTPException) else "error"
        return _error(status, kinds.get(status, "HTTPError"), str(detail))

    async def server_error(request: Request, exc: Exception) -> Response:
        # Starlette raises the exception again after this response, so the server logs it.
        return _error(500, "InternalError", "the studio failed; the terminal has the details")

    middleware = [Middleware(Guard, token=token, host=host, port=port)]
    handlers = {HTTPException: http_error, Exception: server_error}
    return Starlette(routes=routes, middleware=middleware, exception_handlers=handlers)
