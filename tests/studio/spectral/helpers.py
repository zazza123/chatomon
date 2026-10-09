"""What the tests of the add-on share."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from starlette.applications import Starlette
from starlette.types import Message

from chatomon import Atom
from chatomon.connectors.pydantic_ai import PydanticAIConnector

TOKEN = "the-token-of-this-run"
BASE = "http://127.0.0.1:8016"
ORIGIN = {"origin": BASE}
NAMED = {"api_key_env": "SPECTRAL_BRIDGE_API_KEY"}


async def eventually(check: Callable[[], Any], *, timeout: float = 5) -> None:
    """Wait until `check` gives something true; it can be a coroutine function."""
    deadline = time.monotonic() + timeout
    while True:
        result = check()
        if inspect.isawaitable(result):
            result = await result
        if result:
            return
        assert time.monotonic() < deadline, "it did not happen in time"
        await asyncio.sleep(0.02)


def gated() -> tuple[Atom, asyncio.Event]:
    """An atom whose nucleus waits for the returned event before it answers `Hi`."""
    gate = asyncio.Event()

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await gate.wait()
        return ModelResponse(parts=[TextPart(content="Hi")])

    connector = PydanticAIConnector(FunctionModel(respond))
    return Atom(name="shop", instructions="Be brief.", connector=connector), gate


@contextlib.asynccontextmanager
async def started(app: Starlette) -> AsyncIterator[None]:
    """Run the app's lifespan, as the server does around the requests it serves."""
    inbox: asyncio.Queue[Message] = asyncio.Queue()
    outbox: asyncio.Queue[Message] = asyncio.Queue()
    running = asyncio.ensure_future(app({"type": "lifespan"}, inbox.get, outbox.put))
    await inbox.put({"type": "lifespan.startup"})
    assert (await outbox.get())["type"] == "lifespan.startup.complete"
    try:
        yield
    finally:
        await inbox.put({"type": "lifespan.shutdown"})
        assert (await outbox.get())["type"] == "lifespan.shutdown.complete"
        await running


@contextlib.asynccontextmanager
async def page(app: Starlette, base: str = BASE) -> AsyncIterator[httpx.AsyncClient]:
    """A browser that opened the link of the run: it has the cookie."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url=base, follow_redirects=True
    ) as client:
        await client.get(f"/?token={TOKEN}")
        yield client
