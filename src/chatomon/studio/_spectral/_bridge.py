"""The bridge: the studio's atom, reachable from Spectral for as long as it is connected.

It puts the pieces together: the adapter that answers requests, the copy of the
atom the requests run on, the counts of what its orbitals decided, and the
connection to the relay.

Spectral sends the first message of a conversation and then, turn after turn, the new
message with the id of the answer before it. So the bridge keeps the conversation each
answer left, as the runner gives it: in memory, for as long as the connection is open,
and nowhere else.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import re
import secrets
import socket
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import uvicorn
from opentelemetry.trace import NoOpTracerProvider

from chatomon.schemas import TurnReport
from chatomon.studio._chains import SIDES, RefusedError
from chatomon.studio._runners import InterruptedTurnError, Runner, runner_for
from chatomon.studio._session import SAMPLE, Session, StudioBusyError
from chatomon.studio._spectral import _relay
from chatomon.studio._spectral import _shapes as shapes
from chatomon.studio._spectral._adapter import Unanswered, create_adapter
from chatomon.studio._spectral._counts import Chain, Counts
from chatomon.studio._spectral._relay import KEY_ENV, RELAY_ENV, Relay
from chatomon.studio._values import InvalidSettingError

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ACTIVE_FOR = 2.0
"""For how many seconds after its last request a connection is still said to be active."""

_SENTENCES = {
    ("off", None): "Spectral is not connected",
    ("connecting", None): "Connecting to Spectral",
    ("connecting", "unreachable"): "The relay is not answering. Trying again.",
    ("connected", None): "Connected to Spectral",
    ("lost", None): "The connection to Spectral was lost. Trying again.",
    ("closing", None): "Disconnecting from Spectral",
    ("refused", "key"): "Spectral refused the key. Check the variable and connect again.",
    ("refused", "protocol"): (
        "Spectral speaks another version of the protocol. Upgrade spectral-bridge."
    ),
    ("refused", "stopped"): "The connection to Spectral stopped.",
}
_ACTIVE = "Spectral is sending requests"
_NOT_KEPT = (
    "The answer this request continues is not kept here: send the whole conversation in input."
)
_REPLACED = "The atom was replaced by one of another framework: start the conversation again."

_KEPT = 1000
"""How many answers have their conversation kept to be continued; the oldest go first."""

_KEPT_CHARS = 32 * 1024 * 1024
"""How many characters the kept conversations hold in all; the oldest go first.

A count of answers alone bounds nothing: a request can be as large as the adapter
reads, and a conversation that is kept grows with every turn.
"""

# The reports of the turn that runs in this context: how a report finds its request.
_reports: contextvars.ContextVar[list[TurnReport] | None] = contextvars.ContextVar(
    "chatomon_spectral_reports", default=None
)


@dataclass(frozen=True)
class _Copy:
    """The agent the requests run on, and what it was built from."""

    revision: int
    agent: Any
    runner: Runner
    chain: Chain


@dataclass(frozen=True)
class _Kept:
    """The conversation an answer left, as the runner that ran it gives it."""

    runner: Runner
    conversation: Any
    weight: int
    """About how many characters it holds: the messages and the replies of its turns."""


@dataclass(frozen=True)
class _Turn:
    """How a turn went: its reply, or why there is none to send."""

    reply: str
    report: TurnReport | None
    duration_ms: float
    failure: Unanswered | None


def _address(url: str) -> tuple[str, str]:
    """Read the scheme and the host of an address.

    Returns:
        The two, in lower case; both empty for an address that cannot be read.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return "", ""
    return parts.scheme.lower(), parts.hostname or ""


def _collect(report: TurnReport) -> None:
    turn = _reports.get()
    if turn is not None:
        turn.append(report)


class Bridge:
    """Connects the session's atom to Spectral and counts what the requests meet.

    Args:
        session: The studio's session. It is read through `revision`,
            `build_copy()` and `settled()`: it knows nothing of the bridge.
    """

    insecure = False
    """Whether `ws://` is taken for a relay. Only tests set it, in code."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._env = KEY_ENV
        self._relay: Relay | None = None
        self._server: uvicorn.Server | None = None
        self._serving: asyncio.Task[None] | None = None
        self._port: int | None = None
        self._since: datetime | None = None
        self._counts: Counts | None = None
        self._copy: _Copy | None = None
        self._kept: OrderedDict[str, _Kept] = OrderedDict()
        self._weight = 0
        self._test: dict[str, Any] | None = None
        self._opening = False
        self._running = 0
        self._last = float("-inf")

    # The copy of the atom

    def _current(self) -> _Copy:
        """Return the copy for the session as it is now, built again after a change.

        Nothing is awaited between reading the revision and taking the runner:
        the three belong to the same atom.
        """
        session = self._session
        if self._copy is None or self._copy.revision != session.revision:
            chain: Chain = {
                side: [(orb.name, type(orb).__name__) for orb in session.chains.enabled(side)]
                for side in SIDES
            }
            agent = session.build_copy(
                on_turn=_collect, capture_content="none", tracer_provider=NoOpTracerProvider()
            )
            runner = runner_for(session.atom.connector)
            self._copy = _Copy(session.revision, agent, runner, chain)
        return self._copy

    async def _run(
        self,
        asked: shapes.Asked,
        *,
        counts: Counts | None,
        kept: _Kept | None = None,
        keep_as: str | None = None,
    ) -> _Turn:
        """Run one turn on the copy; `counts` is told when given.

        A request that brings its own history runs on those texts. Any other runs as a
        turn of a conversation the runner keeps: the one in `kept`, or a new one. What
        that turn leaves is kept under `keep_as`, when it is given and there is a reply.

        Returns:
            How the turn went: its reply, or why there is none to send.

        Raises:
            asyncio.CancelledError: When the turn is cancelled; it is counted as failed.
        """
        await self._session.settled()
        reports: list[TurnReport] = []
        token = _reports.set(reports)
        if counts is not None:
            counts.arrived()
            self._running += 1
        started = time.perf_counter()
        copy: _Copy | None = None
        left: _Kept | None = None
        reply = ""
        failure: Unanswered | None = None
        cancelled = False
        try:
            # A copy that cannot be built is a request that failed, like a turn that raised.
            copy = self._current()
            if asked.history:
                reply = await copy.runner.run_from_history(copy.agent, asked.history, asked.message)
            elif kept is not None and kept.runner is not copy.runner:
                # A conversation of one framework means nothing to another.
                failure = Unanswered(409, "AtomReplaced", _REPLACED)
            else:
                before = kept.conversation if kept is not None else None
                outcome = await copy.runner.run_turn(copy.agent, asked.message, before)
                # What a tool returned is not a reply: no output orbital checked it.
                reply = outcome.reply if outcome.by_nucleus else ""
                weight = (kept.weight if kept is not None else 0) + len(asked.message) + len(reply)
                left = _Kept(copy.runner, outcome.conversation, weight)
        except InterruptedTurnError:
            failure = Unanswered(500, "AtomInterrupted", "The atom stopped to wait for an answer.")
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            # The class of the error, never its text: it can quote the conversation.
            failure = Unanswered(502, "AtomFailed", f"The atom failed with {type(exc).__name__}.")
        else:
            if failure is None and not reply:
                failure = Unanswered(500, "AtomFailed", "The atom gave no reply.")
        finally:
            _reports.reset(token)
            duration_ms = (time.perf_counter() - started) * 1000
            report = reports[-1] if reports else None
            if counts is not None:
                self._running -= 1
                self._last = time.monotonic()
                counts.done(
                    revision=copy.revision if copy is not None else self._session.revision,
                    chain=copy.chain if copy is not None else {},
                    report=report,
                    failed=cancelled or failure is not None,
                    duration_ms=duration_ms,
                )
        if failure is None and left is not None and keep_as is not None:
            self._keep(keep_as, left)
        return _Turn(reply, report, duration_ms, failure)

    def _keep(self, answer_id: str, left: _Kept) -> None:
        """Keep the conversation an answer left, and forget the oldest beyond the limits.

        A conversation that is larger than the limit by itself is forgotten at once.
        """
        self._kept[answer_id] = left
        self._weight += left.weight
        while self._kept and (len(self._kept) > _KEPT or self._weight > _KEPT_CHARS):
            _, gone = self._kept.popitem(last=False)
            self._weight -= gone.weight

    async def _answer(self, asked: shapes.Asked, kept_as: str | None) -> str:
        """Answer a request of the adapter with a turn of the atom.

        Returns:
            The reply of the atom.

        Raises:
            Unanswered: When the connection is closing, the answer the request
                continues is not kept, or the turn left no reply.
        """
        relay = self._relay
        if relay is None or relay.state()[0] in {"closing", "off", "refused"}:
            raise Unanswered(503, "Closing", "The studio is disconnecting.")
        relay.seen()
        kept: _Kept | None = None
        if asked.previous is not None:
            kept = self._kept.get(asked.previous)
            if kept is None:
                # Refused before any turn, like a request that cannot be read.
                self._rejected()
                raise Unanswered(400, "UnknownConversation", _NOT_KEPT)
        turn = await self._run(asked, counts=self._counts, kept=kept, keep_as=kept_as)
        if turn.failure is not None:
            raise turn.failure
        return turn.reply

    def _rejected(self) -> None:
        if self._counts is not None:
            self._counts.rejected()

    # What the page asks

    def _notes(self) -> list[str]:
        """Return what the runner says of the atom as it is now, to read before connecting."""
        connector = self._session.atom.connector
        return list(runner_for(connector).notes(connector))

    def state(self) -> dict[str, Any]:
        """Return what the page shows of the connection, as JSON-ready data.

        The key and the key of the adapter are never part of it, nor is any text of
        the conversations that are kept: only how many they are.
        """
        status, reason = self._relay.state() if self._relay is not None else ("off", None)
        if status != "connected":
            # The time is of the connection that is open now: a lost one has none.
            self._since = None
        elif self._since is None:
            self._since = datetime.now(UTC)
        recent = time.monotonic() - self._last < _ACTIVE_FOR
        active = status == "connected" and (self._running > 0 or recent)
        since = self._since.strftime("%Y-%m-%dT%H:%M:%SZ") if self._since is not None else None
        adapter = None
        if self._port is not None:
            adapter = {"port": self._port, "routes": list(shapes.ROUTES)}
        return {
            "status": status,
            "active": active,
            "reason": reason,
            "message": _ACTIVE if active else _SENTENCES[status, reason],
            "local": self._session.local,
            "notes": self._notes(),
            "api_key_env": self._env,
            "key_set": bool(os.environ.get(self._env, "").strip()),
            "relay": _address(_relay.relay_url())[1],
            "adapter": adapter,
            "since": since,
            "test": dict(self._test) if self._test is not None else None,
            "counts": self._counts.to_json() if self._counts is not None else None,
            "conversations": len(self._kept),
        }

    async def connect(self, api_key_env: str) -> None:
        """Open the adapter and connect to the relay with the key a variable holds.

        Args:
            api_key_env: The name of the environment variable that holds the key.

        Raises:
            RefusedError: If the studio listens on a network address, a
                connection is already open, or the relay's address is not `wss://`.
            InvalidSettingError: If the name is not the name of a variable, or
                the variable is not set.
            MissingExtraError: If `spectral-bridge` is not installed.
        """  # noqa: DOC502 - MissingExtraError is raised by the relay
        if not self._session.local:
            raise RefusedError(
                "Spectral can be connected only when the studio listens on this machine"
            )
        held = self._relay is not None and self._relay.state()[0] not in {"off", "refused"}
        if self._opening or held:
            raise RefusedError("Spectral is already connected: disconnect first")
        name = api_key_env.strip()
        if not _NAME.match(name):
            # What was sent may be the key itself: it is not kept and not repeated.
            raise InvalidSettingError(
                "api_key_env takes the name of an environment variable, not its value",
                setting="api_key_env",
            )
        key = os.environ.get(name, "").strip()
        if not key:
            # A key can read as a name: what names no variable is not kept or repeated either.
            raise InvalidSettingError(
                "the environment variable that api_key_env names is not set",
                setting="api_key_env",
            )
        self._env = name
        url = _relay.relay_url()
        scheme, host = _address(url)
        if scheme != "wss" and not (self.insecure and scheme == "ws"):
            raise RefusedError(f"the relay must be a wss:// address: check {RELAY_ENV}")
        # Nothing is awaited before this: a second request to connect finds the first.
        self._opening = True
        try:
            await self._open(url, key)
        finally:
            self._opening = False
        print(f"chatomon studio: connecting this atom to Spectral through {host}", file=sys.stderr)

    async def _open(self, url: str, key: str) -> None:
        """Start the adapter on a port of its own, then the client that calls it.

        Raises:
            RefusedError: If the adapter does not start, or the client refuses the address.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
        except OSError:
            sock.close()
            raise RefusedError("the adapter could not start") from None
        port = sock.getsockname()[1]
        secret = secrets.token_urlsafe(32)
        app = create_adapter(secret=secret, port=port, answer=self._answer, rejected=self._rejected)
        config = uvicorn.Config(
            app, log_config=None, access_log=False, lifespan="off", timeout_graceful_shutdown=3
        )
        server = uvicorn.Server(config)
        serving = asyncio.get_running_loop().create_task(server.serve(sockets=[sock]))
        while not server.started:
            if serving.done():
                sock.close()
                if not serving.cancelled():
                    serving.exception()  # read, so that asyncio does not report it
                raise RefusedError("the adapter could not start")
            await asyncio.sleep(0.01)
        self._server, self._serving, self._port = server, serving, port
        try:
            self._relay = Relay(
                url,
                key,
                f"http://127.0.0.1:{port}/{secret}",
                secret=secret,
                on_end=self._ended,
                insecure=self.insecure,
            )
        except ValueError:
            self._ended(said=False)
            raise RefusedError(
                f"the relay's address is not one the client takes: check {RELAY_ENV}"
            ) from None
        # The counts start again with a connection that exists; no request can come before this.
        self._counts = Counts()
        self._since = None
        self._last = float("-inf")

    def _ended(self, *, said: bool = True) -> None:
        """Close the adapter when the connection is over, for whatever reason."""
        if self._server is not None:
            self._server.should_exit = True
        self._server = None
        self._port = None
        self._since = None
        # Nothing can continue them any more: their texts are not kept past the connection.
        self._kept.clear()
        self._weight = 0
        if said:
            print("chatomon studio: Spectral can no longer reach this atom.", file=sys.stderr)

    def disconnect(self) -> None:
        """Ask the connection to close: requests that run finish first, for a few seconds."""
        if self._relay is not None:
            self._relay.stop()

    async def test(self) -> None:
        """Run one turn on a sample message, built like a request, and keep how it went.

        The turn is real: it calls the atom's models and tools. It is not counted.

        Raises:
            StudioBusyError: While another test runs.
        """
        if self._test is not None and self._test["outcome"] == "running":
            raise StudioBusyError("a test is running; wait for it to finish")
        self._test = {"outcome": "running"}
        try:
            turn = await self._run(shapes.read(shapes.RESPONSES, {"input": SAMPLE}), counts=None)
        except BaseException:
            self._test = None
            raise
        failed = turn.failure is not None
        self._test = {
            "outcome": "failed" if failed else "answered",
            "final_action": turn.report.final_action if turn.report is not None else None,
            "message": turn.failure.message if turn.failure is not None else "The atom answered.",
            "duration_ms": round(turn.duration_ms),
        }

    def reset_counts(self) -> None:
        """Start the counts again; requests that run finish in the counts they started in."""
        if self._counts is not None:
            self._counts = Counts()

    async def close(self) -> None:
        """Close the connection and the adapter now: the studio is stopping."""
        if self._relay is not None:
            # Asked twice: the second time the client does not wait for the requests that run.
            self._relay.stop()
            self._relay.stop()
            await self._relay.wait(3)
        if self._server is not None:
            self._server.should_exit = True
        if self._serving is not None:
            await asyncio.wait({self._serving}, timeout=3)
