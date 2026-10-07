"""A stand-in for the Principled Intelligence services, for tests that need a real HTTP call.

It listens on loopback, on a free port, so nothing leaves the machine. It answers
the two routes the `api` backend of the `orbitals` library calls.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

SCOPE_GUARD = "/orbitals/scope-guard-v2/validate"
CLAIM_EXTRACTOR = "/orbitals/claim-extractor/extract"


@dataclass
class FakeService:
    """What the fake service answers, and what it was asked."""

    url: str
    key: str | None = None
    """The key a request must carry in `X-API-Key`; any request passes when `None`."""
    status: int | None = None
    """Answer every request with this status instead."""
    delay: float = 0.0
    """Seconds to wait before answering."""
    scope_class: str = "Directly Supported"
    suggested_response: str | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)
    """Every request received: its path, its headers and its JSON body."""

    def answer(self, path: str) -> dict[str, Any] | None:
        """Return the body of a successful answer, or `None` for an unknown path."""
        if path == SCOPE_GUARD:
            return {
                "scope_class": self.scope_class,
                "evidences": ["Parcel tracking assistant"],
                "reasoning": "because",
                "suggested_response": self.suggested_response,
                "model": "fake-scope-guard",
            }
        if path == CLAIM_EXTRACTOR:
            claims = [{"subtype": "Factoid", "content": "Open 24/7", "evidences": []}]
            return {
                "extractions": {"intents": [], "claims": claims},
                "model": "fake-claim-extractor",
                "usage": None,
            }
        return None


def _handler(service: FakeService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return None

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            service.requests.append(
                {"path": self.path, "headers": dict(self.headers.items()), "body": body}
            )
            if service.delay:
                time.sleep(service.delay)
            answer = service.answer(self.path)
            if service.status is not None:
                status = service.status
            elif service.key is not None and self.headers.get("X-API-Key") != service.key:
                status = 401
            else:
                status = 200 if answer is not None else 404
            payload = json.dumps(answer if status == 200 else {"detail": "refused"}).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except OSError:
                # The caller stopped waiting, as after a timeout.
                return None

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        return None


@contextmanager
def running() -> Iterator[FakeService]:
    """Run the fake service for the length of a `with` block."""
    service = FakeService(url="")
    server = _Server(("127.0.0.1", 0), _handler(service))
    service.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield service
    finally:
        server.shutdown()
        server.server_close()


def closed_url() -> str:
    """Return a loopback address on which nothing listens."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}"
