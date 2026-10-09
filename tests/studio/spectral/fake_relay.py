"""A relay on loopback that speaks the protocol of spectral-bridge, for tests.

It listens on `127.0.0.1` on a free port, so nothing leaves the machine. It is
`ws://`, not `wss://`: a test builds its client with the allowance for that.
"""

from __future__ import annotations

import asyncio
import http
import json
import threading
from typing import Any
from uuid import uuid4

from websockets.asyncio.server import Server, ServerConnection, serve


class FakeRelay:
    """A relay with one key, that can refuse, drop the connection and send requests."""

    def __init__(self, key: str = "good-key") -> None:
        self.key = key
        self.refuse: str | None = None
        """`"http-401"`, `"close-4001"` or `"close-4002"`: how to refuse the next clients."""
        self.connections = 0
        self.url = ""
        self._port = 0
        self._server: Server | None = None
        self._client: ServerConnection | None = None
        self._waiting: dict[str, asyncio.Future[dict[str, Any]]] = {}

    async def start(self) -> None:
        """Listen, on a free port the first time and on the same one after a stop."""
        self._server = await serve(self._serve, "127.0.0.1", self._port, process_request=self._gate)
        self._port = next(iter(self._server.sockets)).getsockname()[1]
        self.url = f"ws://127.0.0.1:{self._port}/connect"

    async def stop(self) -> None:
        """Stop listening and close the connection, as a relay that goes down does."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    def _gate(self, connection: ServerConnection, request: Any) -> Any:
        if self.refuse == "http-401":
            return connection.respond(http.HTTPStatus.UNAUTHORIZED, "no\n")
        return None

    async def _serve(self, client: ServerConnection) -> None:
        self.connections += 1
        headers = client.request.headers if client.request is not None else {}
        if self.refuse == "close-4002" or headers.get("Spectral-Bridge-Protocol") != "2":
            await client.close(4002, "unsupported protocol version")
            return
        if self.refuse == "close-4001" or headers.get("Authorization") != f"Bearer {self.key}":
            await client.close(4001, "authentication failed")
            return
        await client.send(json.dumps({"type": "connected", "protocol": 2}))
        self._client = client
        try:
            async for raw in client:
                frame = json.loads(raw)
                if frame.get("type") != "response":
                    continue
                await client.send(json.dumps({"type": "ack", "request_id": frame["request_id"]}))
                waiting = self._waiting.pop(frame["request_id"], None)
                if waiting is not None and not waiting.done():
                    waiting.set_result(frame["payload"])
        finally:
            self._client = None

    async def request(self, path: str, body: Any, *, timeout: float = 10) -> dict[str, Any]:
        """Send a request to the client and return the payload of its response."""
        assert self._client is not None, "no client is connected"
        request_id = uuid4().hex
        answer: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._waiting[request_id] = answer
        payload = {"method": "POST", "path": path, "headers": {}, "body": body}
        frame = {"type": "request", "request_id": request_id, "payload": payload}
        await self._client.send(json.dumps(frame))
        return await asyncio.wait_for(answer, timeout)

    @property
    def has_client(self) -> bool:
        """Whether a client is connected now."""
        return self._client is not None


class RelayThread:
    """A fake relay in a thread of its own, for a test that drives another process."""

    def __init__(self, key: str = "good-key") -> None:
        self.relay = FakeRelay(key)
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        await self.relay.start()
        self._ready.set()
        await self._stop.wait()
        await self.relay.stop()

    def ask(self, path: str, body: Any, *, timeout: float = 10) -> dict[str, Any]:
        """Send a request from the thread of the test and return the payload of its response."""
        assert self._loop is not None, "the fake relay is not running"
        asked = asyncio.run_coroutine_threadsafe(
            self.relay.request(path, body, timeout=timeout), self._loop
        )
        return asked.result(timeout + 1)

    def __enter__(self) -> FakeRelay:
        self._thread.start()
        assert self._ready.wait(timeout=10), "the fake relay did not start"
        return self.relay

    def __exit__(self, *exc: object) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=10)
