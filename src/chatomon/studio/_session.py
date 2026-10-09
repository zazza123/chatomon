"""The studio session: the developer's atom, the studio's copy of it and one conversation."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from pydantic_core import to_json

from chatomon._version import __version__
from chatomon.atom import Atom
from chatomon.engine import TurnCallback
from chatomon.errors import ChatomonConfigError, ChatomonError
from chatomon.orbitals.base import Orbital
from chatomon.schemas import CaptureContent, OrbitalContext, Side, TurnReport, Verdict
from chatomon.studio import _codegen as codegen
from chatomon.studio import _files as files
from chatomon.studio import _kinds as kinds
from chatomon.studio import _loading as loading
from chatomon.studio import _presets as presets
from chatomon.studio import _testgen as testgen
from chatomon.studio import _values as values
from chatomon.studio._chains import SIDES, Chains, Entry, InvalidRequestError, RefusedError
from chatomon.studio._events import Event, EventProcessor
from chatomon.studio._runners import Runner, runner_for

_T = TypeVar("_T")


def _always_recording_provider() -> TracerProvider:
    """Return a provider whose spans are always recorded.

    The studio's live events are spans. They must not depend on the developer's
    OpenTelemetry environment: a sampler that drops spans, an unsampled span of
    their own tracing in the context, or `OTEL_SDK_DISABLED`, which the SDK
    reads when a provider is created.
    """
    disabled = os.environ.pop("OTEL_SDK_DISABLED", None)
    try:
        return TracerProvider(sampler=ALWAYS_ON)
    finally:
        if disabled is not None:
            os.environ["OTEL_SDK_DISABLED"] = disabled


def _copy(
    a: Atom[Any],
    chains: Chains,
    *,
    on_turn: TurnCallback,
    capture_content: CaptureContent,
    tracer_provider: trace.TracerProvider,
) -> Any:
    """Build an agent from a copy of an atom that has the orbitals `chains` has on.

    The atom that is given is read, never changed.

    Returns:
        The native agent of the atom's framework.
    """
    copy: Atom[Any] = Atom(
        name=a.name,
        instructions=a.instructions,
        connector=a.connector,
        knowledge=a.knowledge,
        input_orbitals=chains.enabled("input"),
        output_orbitals=chains.enabled("output"),
        refusal=a.refusal,
        blocked_placeholder=a.blocked_placeholder,
        blocked_guidance=a.blocked_guidance,
        max_retries=a.max_retries,
        on_turn=on_turn,
        capture_content=capture_content,
        tracer_provider=tracer_provider,
    )
    return copy.build()


def _in_thread(function: Callable[[], _T]) -> asyncio.Future[_T]:
    """Run a function in a thread of its own; the future has what it returns or raises.

    The thread is as close as it can be to the one the command starts in. It
    has an event loop of its own as its current one, not running, for code that
    asks for the loop while it is imported. And it is a daemon: code that hangs
    while it is loaded must not keep the studio from stopping.

    Returns:
        A future of the running loop.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[_T] = loop.create_future()

    def deliver(settle: Callable[[], None]) -> None:
        def settled() -> None:
            if not future.done():
                settle()

        # When the studio stopped while the function ran, the loop is closed: nobody waits.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(settled)

    def run() -> None:
        own = asyncio.new_event_loop()
        asyncio.set_event_loop(own)
        try:
            result = function()
        except BaseException as exc:
            deliver(lambda error=exc: future.set_exception(error))
        else:
            deliver(lambda: future.set_result(result))
        finally:
            asyncio.set_event_loop(None)
            own.close()

    threading.Thread(target=run, name="chatomon-studio-reload", daemon=True).start()
    return future


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _notes(runner: Runner, atom: Atom[Any]) -> list[str]:
    """Return what the developer should know at start: about the atom, then about the library."""
    return [*runner.notes(atom.connector), *kinds.import_notes()]


SAMPLE = "Hello"
"""The message a test of an orbital sends. The server writes it: the page sends no text."""

_HINTS = {
    401: " It did not accept the credentials.",
    403: " It did not accept the credentials.",
    404: " Check the address.",
}


@dataclass(frozen=True)
class Probe:
    """How one check of an orbital on a sample message went."""

    outcome: str
    """`answered`, `timeout`, `http`, `unreachable` or `failed`."""
    message: str
    duration_ms: float
    action: str | None = None
    reason: str | None = None
    status: int | None = None

    def to_json(self) -> dict[str, Any]:
        """Return the outcome as the page receives it."""
        data: dict[str, Any] = {
            "outcome": self.outcome,
            "message": self.message,
            "duration_ms": self.duration_ms,
        }
        if self.outcome == "answered":
            data.update(action=self.action, reason=self.reason)
        if self.status is not None:
            data["status"] = self.status
        return data


def _links(exc: BaseException) -> list[BaseException]:
    """Return an exception and what caused it, a few links down.

    A framework wraps the error of its HTTP client, which wraps the socket's: the
    refused connection is three or four links below what the check raises.
    """
    found: list[BaseException] = []
    queue: list[Any] = [exc]
    while queue and len(found) < 12:
        link = queue.pop(0)
        if not isinstance(link, BaseException) or any(link is seen for seen in found):
            continue
        found.append(link)
        queue += [link.__cause__, link.__context__, getattr(link, "os_error", None)]
    return found


def _failure(exc: BaseException, timeout: float, waited: float) -> tuple[str, str, int | None]:
    """Say how a check failed, from the kind of exception and never from its text.

    A client's error can quote the request it made, and so a key: the sentence is
    the studio's own. Exceptions are told apart by Python's base classes and by
    the attributes HTTP clients share, so no client is imported here.

    Args:
        exc: What the check raised.
        timeout: The orbital's timeout, in seconds.
        waited: How long the check ran, in seconds.

    Returns:
        The outcome, its sentence and the HTTP status when there is one.
    """
    if isinstance(exc, TimeoutError):
        if waited < timeout * 0.9:
            # The client has a timeout of its own, shorter than the orbital's.
            early = f"The client stopped waiting before the orbital's timeout of {timeout:g} s."
            return "timeout", early, None
        wait = f"No answer within {timeout:g} s. A service that is starting can take longer."
        return "timeout", wait, None
    if isinstance(exc, ChatomonError):
        return "failed", f"The check failed with {type(exc).__name__}: {exc}.", None
    for held in (getattr(exc, "status", None), getattr(exc, "status_code", None)):
        # A status below 400 on an error is an answer the client could not use, not a refusal.
        if isinstance(held, int) and not isinstance(held, bool) and 400 <= held <= 599:
            return "http", f"The service answered {held}.{_HINTS.get(held, '')}", held
    errors = [link for link in _links(exc) if isinstance(link, OSError)]
    if any(isinstance(error, ConnectionRefusedError) for error in errors):
        return "unreachable", "The connection was refused.", None
    if any(isinstance(error, socket.gaierror) for error in errors):
        return "unreachable", "The address was not found.", None
    if errors:
        return "unreachable", "Could not connect.", None
    return "failed", f"The check failed with {type(exc).__name__}.", None


NETWORK = (
    "the studio listens on a network address: where an orbital connects is set in your code, "
    "or in a preset given when the studio starts, not from the page"
)


class StudioBusyError(ChatomonError):
    """A turn is running, so the session cannot start another one or change its orbitals."""


class UnknownTurnError(ChatomonError):
    """The conversation has no turn with that number."""


@dataclass
class TurnRecord:
    """A finished turn, as the page shows it."""

    n: int
    message: str
    chain: dict[str, list[str]]
    changes: list[str] = field(default_factory=list)
    """How the session differed from the code when the turn started."""
    models: frozenset[str] = frozenset()
    """The enabled orbitals that call a model, by name, when the turn started."""
    signature: tuple[tuple[str, str, str, bool], ...] = ()
    """Every orbital of the session when the turn started: side, name, fingerprint, on."""
    reloads: int = 0
    """How many times the code had been loaded again when the turn started."""
    reply: str | None = None
    report: TurnReport | None = None
    error: dict[str, str] | None = None

    def to_json(self) -> dict[str, Any]:
        """Return the turn as JSON-ready data."""
        return {
            "n": self.n,
            "message": self.message,
            "reply": self.reply,
            "report": self.report.model_dump(mode="json") if self.report is not None else None,
            "chain": self.chain,
            "changes": self.changes,
            "cannot_save": testgen.refusal(self),
            "error": self.error,
        }


@dataclass(frozen=True)
class Reloaded:
    """What a reload from code did that the developer should be told."""

    dropped: list[str]
    """One sentence for each change of the studio that no longer applies."""
    restarted: str | None
    """Why a new conversation was started, or `None` when it goes on."""
    notes: list[str]
    """The notes about the reloaded atom that the studio had not given before."""


@dataclass(frozen=True)
class _Loaded:
    """Everything a reload replaces, prepared before anything is replaced."""

    atom: Atom[Any]
    namespace: Mapping[str, object]
    runner: Runner
    notes: list[str]
    chains: Chains
    agent: Any
    dropped: list[str]


class Session:
    """Runs turns on a copy of the developer's atom; one conversation, one turn at a time.

    Args:
        atom: The developer's atom; it is read, never changed.
        source: How the studio was started: `module:attribute` or `serve()`.
        namespace: The variables of the atom's module, or `None` without a module.
        preset: A preset to start with, as JSON text; its warnings join the notes.

    Raises:
        ChatomonConfigError: If the studio has no runner for the atom's connector, or
            the atom does not build.
        InvalidPresetError: With every problem of the preset; no session is created.
    """

    def __init__(
        self,
        atom: Atom[Any],
        *,
        source: str,
        namespace: Mapping[str, object] | None = None,
        preset: str | None = None,
    ) -> None:
        runner = runner_for(atom.connector)
        self.atom = atom
        self.source = source
        self.namespace = namespace
        self.root = Path.cwd().resolve()
        self._runner = runner
        self.processor = EventProcessor()
        self.provider = _always_recording_provider()
        self.provider.add_span_processor(self.processor)
        self.conversation: Any = None
        self.turns: list[TurnRecord] = []
        self.notes: list[str] = _notes(runner, atom)
        self.preset_connections: list[str] = []
        """Where the preset given at the start makes orbitals connect: the page shows it."""
        self.local = True
        """Whether the studio listens only on this machine: the app says so. On a network
        address the page cannot say where an orbital connects."""
        self._task: asyncio.Task[None] | None = None
        self._reload: asyncio.Task[Reloaded] | None = None
        self._probing = False
        self.reloads = 0
        self.revision = 0
        """Grows by one each time the chains or the atom are replaced: an agent built
        from the session before a change is told from one built after it by this number."""
        self._running: TurnRecord | None = None
        self._report: TurnReport | None = None
        self.chains = Chains.from_atom(atom)
        self._undo: tuple[Chains, str] | None = None
        self._agent: Any = self._build(atom, self.chains)
        if preset is not None:
            noted = self.import_preset(preset)
            self.notes = [*self.notes, *noted.connections, *noted.warnings]
            self.preset_connections = list(noted.connections)
            # The session starts here: there is no earlier state of it to go back to.
            self._undo = None

    def _build(self, a: Atom[Any], chains: Chains) -> Any:
        return _copy(
            a,
            chains,
            on_turn=self._record,
            capture_content="full",
            tracer_provider=self.provider,
        )

    def build_copy(
        self,
        *,
        on_turn: TurnCallback,
        capture_content: CaptureContent,
        tracer_provider: trace.TracerProvider,
    ) -> Any:
        """Build one more agent from the atom and the chains as they are now.

        It is how something other than the page runs the atom. The copy has the
        atom's connector and the orbitals of the chains, the same objects the
        studio's own agent has, and shares nothing else with it: it reports its
        turns to `on_turn`, never to the session, and its spans go to
        `tracer_provider`. The atom's redactor is not applied, as in the studio's
        own agent: with `capture_content="full"` the reports carry the text as it
        is. The copy stays as it was built: after a change build another one, and
        tell the two apart by `revision`.

        Args:
            on_turn: Receives the report of every turn the copy runs.
            capture_content: How much text those reports carry.
            tracer_provider: Where the spans of the copy go.

        Returns:
            The native agent of the atom's framework.
        """
        return _copy(
            self.atom,
            self.chains,
            on_turn=on_turn,
            capture_content=capture_content,
            tracer_provider=tracer_provider,
        )

    def _adopt(self, candidate: Chains, *, undo: str | None = None) -> None:
        """Take the candidate chains if the atom builds with them; otherwise change nothing.

        With `undo`, the previous chains are kept under that label; any other
        change drops what was kept.
        """
        agent = self._build(self.atom, candidate)
        previous = self.chains
        self.chains, self._agent = candidate, agent
        self._undo = (previous, undo) if undo is not None else None
        self.revision += 1

    def switch(self, side: str, name: str, enabled: bool) -> None:
        """Turn an orbital on or off for the next turns.

        Raises:
            StudioBusyError: While a turn runs.
            UnknownOrbitalError: If the atom has no such orbital.
            ChatomonConfigError: If the atom would be invalid; nothing changes.
        """  # noqa: DOC502 - raised by the helpers
        self._require_idle()
        self._adopt(self.chains.switched(side, name, enabled))

    def edit(self, side: str, name: str, settings: Mapping[str, Any]) -> None:
        """Change settings of an orbital, or the name of one added in the studio.

        Raises:
            StudioBusyError: While a turn runs.
            UnknownOrbitalError: If the atom has no such orbital.
            RefusedError: If the orbital is read-only or its name comes from code.
            InvalidSettingError: If a value is refused.
            ChatomonConfigError: If the atom would be invalid; nothing changes.
        """  # noqa: DOC502 - raised by the helpers
        self._require_idle()
        where, index = self.chains.find(side, name)
        self._keep_connections(type(self.chains.work[where][index].orbital), settings, new=False)
        self._adopt(self.chains.edited(side, name, settings))

    def move(self, side: str, name: str, position: int) -> None:
        """Move an orbital to a 0-based position in its chain.

        Raises:
            StudioBusyError: While a turn runs.
            UnknownOrbitalError: If the atom has no such orbital.
            InvalidRequestError: If the position is outside the chain.
            ChatomonConfigError: If the atom would be invalid; nothing changes.
        """  # noqa: DOC502 - raised by the helpers
        self._require_idle()
        self._adopt(self.chains.moved(side, name, position))

    def add(self, side: str, kind: str, settings: Mapping[str, Any]) -> None:
        """Add an orbital of a built-in class at the end of a chain.

        Raises:
            StudioBusyError: While a turn runs.
            RefusedError: If the class cannot be added, or not on that side.
            InvalidSettingError: If the name or a value is refused.
            ChatomonConfigError: If the atom would be invalid; nothing changes.
        """  # noqa: DOC502 - raised by the helpers
        self._require_idle()
        self._keep_connections(kind, settings, new=True)
        self._adopt(self.chains.added(side, kind, settings))

    def delete(self, side: str, name: str) -> None:
        """Delete an orbital added in the studio; it can be undone.

        Raises:
            StudioBusyError: While a turn runs.
            UnknownOrbitalError: If the atom has no such orbital.
            RefusedError: If the orbital is defined in the code.
            ChatomonConfigError: If another orbital reads it; nothing changes.
        """  # noqa: DOC502 - raised by the helpers
        self._require_idle()
        self._adopt(self.chains.deleted(side, name), undo=f"{name} deleted")

    def reset_orbitals(self, side: str | None = None, name: str | None = None) -> None:
        """Go back to the code, for one orbital or for all; it can be undone.

        Raises:
            StudioBusyError: While a turn runs.
            UnknownOrbitalError: If the atom has no such orbital.
            RefusedError: If the orbital was added in the studio.
            ChatomonConfigError: If the atom would be invalid; nothing changes.
        """  # noqa: DOC502 - raised by the helpers
        self._require_idle()
        label = "Reset to code" if side is None or name is None else f"{name} reset to code"
        self._adopt(self.chains.reset(side, name), undo=label)

    def undo(self) -> None:
        """Bring back the chains from before the last deletion, reset or import.

        Raises:
            StudioBusyError: While a turn runs.
            RefusedError: If there is nothing to undo.
        """  # noqa: DOC502 - StudioBusyError is raised by _require_idle()
        self._require_idle()
        if self._undo is None:
            raise RefusedError("nothing to undo")
        self._adopt(self._undo[0])

    async def probe(
        self,
        side: str,
        *,
        name: str | None = None,
        kind: str | None = None,
        settings: Mapping[str, Any] | None = None,
    ) -> Probe:
        """Run one check of an orbital on a sample message, and keep nothing of it.

        The orbital is one of the session, named by `name`, or one that is not
        added yet, built from `kind` and `settings` as an addition would build
        it. The check is not a turn: it does not go through the engine, and it
        leaves the chains, the conversation and the turns as they were.

        Args:
            side: `"input"` or `"output"`.
            name: The name of an orbital of the session.
            kind: The class name of an orbital to build for the test.
            settings: Its settings, when `kind` is given.

        Returns:
            How the check went: an answer, or why there was none.

        Raises:
            StudioBusyError: While another test runs.
            UnknownOrbitalError: If the session has no orbital with that name.
            InvalidRequestError: If the side does not exist, or neither or both of
                `name` and `kind` are given.
            InvalidSettingError: If the orbital to build is refused, as when the
                environment variable it names is not set.
            RefusedError: If the class cannot be added, the orbital calls no model,
                or it reads the annotations of other orbitals.
        """  # noqa: DOC502 - raised by the helpers
        if self._probing:
            raise StudioBusyError("a test is running; wait for it to finish")
        orb = self._probed(side, name, kind, settings)
        if not orb.calls_model:
            raise RefusedError(f"{orb.name!r} calls no model: there is nothing to test")
        if orb.reads:
            read = " and ".join(repr(other) for other in orb.reads)
            raise RefusedError(
                f"{orb.name!r} reads the annotations of {read}: it can be tried only in a turn"
            )
        self._probing = True
        started = time.perf_counter()
        try:
            # As Atom.build() does: a judge reaches its model through the atom's connector.
            orb.bind(self.atom.connector)
            ctx = OrbitalContext(side=self.chains.find_side(side), message=SAMPLE, turn_id="probe")
            verdict = await asyncio.wait_for(orb.check(ctx), timeout=orb.timeout)
        except Exception as exc:
            waited = time.perf_counter() - started
            outcome, message, status = _failure(exc, orb.timeout, waited)
            return Probe(outcome, message, _elapsed(started), status=status)
        finally:
            self._probing = False
        if not isinstance(verdict, Verdict):
            # In a turn the engine raises a TypeError for a check that returns anything else.
            return Probe("failed", "The check failed with TypeError.", _elapsed(started))
        return Probe(
            "answered",
            "The orbital answered.",
            _elapsed(started),
            action=verdict.action,
            reason=verdict.reason,
        )

    def _keep_connections(
        self, cls: type[Orbital] | str, settings: Mapping[str, Any], *, new: bool
    ) -> None:
        """Refuse a connection the page gives while the studio listens on a network address.

        Whoever has the link could otherwise have a variable of this machine sent to
        an address of their choice, with a test or a turn.

        Args:
            cls: The orbital's class, or the name of one that can be added.
            settings: What the page sent.
            new: Whether the orbital is being created: it may leave a connection out.

        Raises:
            InvalidSettingError: Naming the setting that says where the orbital connects.
        """
        if self.local:
            return
        if isinstance(cls, str):
            found = next((k for k in kinds.addable() if k.cls.__name__ == cls), None)
        else:
            found = kinds.describe(cls)
        if found is None:
            return
        for key, value in settings.items():
            setting = found.setting(key)
            if setting is not None and values.connects(setting) and not (new and value is None):
                raise values.InvalidSettingError(NETWORK, setting=key)

    def _probed(
        self, side: str, name: str | None, kind: str | None, settings: Mapping[str, Any] | None
    ) -> Orbital:
        if (name is None) == (kind is None):
            raise InvalidRequestError("name the orbital to test, or the class to build for it")
        if name is not None:
            where, index = self.chains.find(side, name)
            return self.chains.work[where][index].orbital
        # Built as an addition would build it, on chains that are never adopted.
        self._keep_connections(kind or "", settings or {}, new=True)
        candidate = self.chains.added(side, kind or "", settings or {})
        return candidate.work[self.chains.find_side(side)][-1].orbital

    def code(self) -> dict[str, Any]:
        """Return Copy as code: the text, the changes, the placeholders and the notes."""
        result = codegen.generate(self.chains, self.namespace)
        return {
            "code": result.code,
            "changes": [change.text for change in self.chains.changes()],
            "placeholders": result.placeholders,
            "notes": result.notes,
        }

    def export_preset(self) -> dict[str, Any]:
        """Return the session's orbital changes as a preset."""
        return presets.exported(self.chains, atom=self.atom.name)

    def import_preset(self, text: str) -> presets.Notes:
        """Replace the session's orbital changes with a preset's; it can be undone.

        Args:
            text: The preset, as JSON text.

        Returns:
            Where the preset makes an orbital connect, and the warnings: what
            differs from the exported session.

        Raises:
            StudioBusyError: While a turn runs.
            InvalidPresetError: With every problem; nothing changes.
        """  # noqa: DOC502 - raised by the helpers
        self._require_idle()
        candidate, noted = presets.applied(text, self.chains, atom=self.atom.name)
        if noted.connections and not self.local:
            raise presets.InvalidPresetError([NETWORK])
        try:
            self._adopt(candidate, undo="Preset imported")
        except ChatomonConfigError as exc:
            raise presets.InvalidPresetError([str(exc)]) from exc
        return noted

    def signature(self, *, code: bool = False) -> tuple[tuple[str, str, str, bool], ...]:
        """Return what identifies the session's orbitals now.

        Two sessions with the same signature build the same chains: the sentences
        of `changes` cannot tell a setting changed once from one changed again.

        Args:
            code: Whether to identify the orbitals as the code has them, with none
                of the studio's changes.

        Returns:
            Side, name, fingerprint and whether it is on, for every orbital in order.
        """
        chains = self.chains.initial() if code else self.chains
        return tuple(
            (side, entry.orbital.name, entry.orbital.fingerprint(), entry.enabled)
            for side in SIDES
            for entry in chains.work[side]
        )

    def _saved(self, n: int, alone: bool) -> testgen.SavedTest:
        if not 1 <= n <= len(self.turns):
            raise UnknownTurnError(f"there is no turn {n} in this conversation")
        return testgen.generate(
            self.turns[:n],
            alone=alone,
            source=self.source,
            namespace=self.namespace,
            signature=self.signature(),
            code_signature=self.signature(code=True),
            reloads=self.reloads,
            framework=self._runner.saved_test(self.atom.connector),
        )

    def saved_test(self, n: int, *, alone: bool = False) -> dict[str, Any]:
        """Return turn `n` as a test: the code, where it would go and what to know.

        Args:
            n: The turn's number in this conversation.
            alone: Whether to leave out the turns before it.

        Returns:
            The code, the proposed path and the root (both `None` when the studio
            cannot write), and the notes and warnings.

        Raises:
            UnknownTurnError: If the conversation has no such turn.
            RefusedError: If the turn failed.
        """  # noqa: DOC502 - raised by the helpers
        test = self._saved(n, alone)
        writable = self.namespace is not None and test.importable
        return {
            "code": test.code,
            "path": files.proposed(self.root, test.name) if writable else None,
            "root": str(self.root) if writable else None,
            "writable": writable,
            "integration": test.integration,
            "earlier": test.earlier,
            "notes": test.notes,
            "warnings": test.warnings,
            # What the page sends back to write: the turn's number alone can name another
            # turn once the conversation is started again.
            "turn": self._turn_id(n),
        }

    def _turn_id(self, n: int) -> str | None:
        report = self.turns[n - 1].report
        return report.turn_id if report is not None else None

    def write_test(
        self,
        n: int,
        path: str,
        *,
        turn: str | None = None,
        alone: bool = False,
        overwrite: bool = False,
    ) -> str:
        """Write turn `n` as a test under the directory the studio was started from.

        The content is generated here: the caller chooses the turn and the path,
        never what is written.

        Args:
            n: The turn's number in this conversation.
            path: Where to write, relative to that directory.
            turn: The id of the turn the test was shown for; when given, it must
                still be turn `n`.
            alone: Whether to leave out the turns before it.
            overwrite: Whether to replace a test the studio wrote earlier.

        Returns:
            The written path, relative to that directory.

        Raises:
            UnknownTurnError: If the conversation has no such turn.
            InvalidPathError: If the path is not one the studio writes to.
            OwnFileExistsError: If the file is the studio's and `overwrite` is false.
            RefusedError: If the turn failed or is no longer the one shown, the studio
                was started from Python, or the file exists and the studio did not
                write it.
        """  # noqa: DOC502 - raised by the helpers
        test = self._saved(n, alone)
        if turn is not None and turn != self._turn_id(n):
            raise RefusedError(
                f"turn {n} is not the turn this test was made from: the conversation "
                "changed. Open the turn again"
            )
        if self.namespace is None:
            raise RefusedError(
                "the studio was started from Python and does not know where the atom "
                "lives: copy the test instead"
            )
        if not test.importable:
            raise RefusedError(
                "the import of this test has to be completed by hand: copy the test instead"
            )
        return files.write(self.root, path, test.code, header=testgen.HEADER, overwrite=overwrite)

    async def reload(self) -> Reloaded:
        """Load the atom from the developer's code again and carry the session over to it.

        The modules of the project are imported again in a thread, where no event
        loop runs, as when the studio started. The atom, its chains and its agent
        are replaced together once everything is ready, or not at all. The
        session's orbital changes are laid over the new code; the conversation
        goes on when the new atom is run by the same runner, and what could be
        undone is forgotten.

        Returns:
            What was dropped, whether the conversation started again, and the
            notes that are new.

        Raises:
            StudioBusyError: While a turn or another reload runs.
            RefusedError: If the studio was started from Python.
            ChatomonConfigError: If the code does not give an atom the studio can
                run with the session's changes; nothing changes, and the error an
                import or a factory raised is its cause.
        """  # noqa: DOC502 - raised by the helpers
        self._require_idle()
        if self.namespace is None:
            raise RefusedError(
                "the studio was started from Python and does not know where the atom "
                "lives: restart it to load your changes"
            )
        # Owned by the session, like a turn: a page that goes away does not stop it half way.
        task = asyncio.get_running_loop().create_task(self._reloaded())
        task.add_done_callback(lambda done: done.cancelled() or done.exception())
        self._reload = task
        return await asyncio.shield(task)

    async def settled(self) -> None:
        """Wait until no reload is running; return at once when none is.

        What a reload raises belongs to whoever asked for it: this only waits,
        also for a reload that starts as the one before ends. It is how a turn
        that does not come from the page avoids an atom that is being replaced.
        """
        while (reload := self._reload) is not None and not reload.done():
            await asyncio.wait({reload})

    async def _reloaded(self) -> Reloaded:
        loaded = await _in_thread(self._loaded)
        restarted = None
        if loaded.runner is not self._runner:
            # Only the runner that returned a conversation can read it.
            self.conversation, self.turns = None, []
            restarted = (
                f"The atom is built with {type(loaded.atom.connector).__name__} now, "
                "another framework: a new conversation was started."
            )
        told = [note for note in loaded.notes if note not in self.notes]
        self.atom, self.namespace = loaded.atom, loaded.namespace
        self._runner, self.notes = loaded.runner, loaded.notes
        self.chains, self._agent = loaded.chains, loaded.agent
        self._undo = None
        self.reloads += 1
        self.revision += 1
        return Reloaded(dropped=loaded.dropped, restarted=restarted, notes=told)

    def _loaded(self) -> _Loaded:
        """Import the project again and prepare what a reload replaces; the session is not touched.

        Returns:
            The new atom with its runner, its chains and its agent.

        Raises:
            ChatomonError: A refusal of chatomon's own, as it was raised.
            ChatomonConfigError: If the import, the factory or the framework's build
                fails. In every case the modules are back as they were.
        """
        module = self.source.partition(":")[0]
        with loading.reimporting(self.root) as removed:
            if module in sys.modules and module not in removed:
                # It would be found as it is, and the page would say it was loaded again.
                raise ChatomonConfigError(
                    f"{module!r} is outside the directory the studio was started from, so it "
                    "is not loaded again: restart the studio to load its changes"
                )
            try:
                atom = loading.load(self.source)
                return self._prepared(atom)
            except ChatomonError:
                raise
            except SystemExit as exc:
                # In the loop it would end the studio, with the session and its conversation.
                raise ChatomonConfigError(
                    f"the code called sys.exit({exc.code!r}) while it was loaded"
                ) from exc
            except Exception as exc:
                # What a framework raises when it builds, such as a model whose key is not set.
                raise ChatomonConfigError(
                    f"the reloaded atom could not be built: {type(exc).__name__}: {exc}"
                ) from exc

    def _prepared(self, atom: Atom[Any]) -> _Loaded:
        runner = runner_for(atom.connector)
        code = Chains.from_atom(atom)
        candidate, dropped = presets.carried(self.chains, code)
        try:
            agent = self._build(atom, candidate)
        except ChatomonConfigError as exc:
            # When the atom does not build on its own either, the error is its own.
            self._build(atom, code)
            # No cause: this is the studio's refusal, and nothing of the developer's code failed.
            raise ChatomonConfigError(
                f"the reloaded atom does not build with the studio's changes: {exc}. "
                "Change them or reset to code, then reload again"
            ) from None
        return _Loaded(
            atom=atom,
            namespace=loading.namespace(self.source),
            runner=runner,
            notes=_notes(runner, atom),
            chains=candidate,
            agent=agent,
            dropped=dropped,
        )

    def reset_conversation(self) -> None:
        """Start a new conversation.

        Raises:
            StudioBusyError: While a turn runs.
        """  # noqa: DOC502 - raised by _require_idle()
        self._require_idle()
        self.conversation = None
        self.turns = []

    def _record(self, report: TurnReport) -> None:
        self._report = report

    @property
    def busy(self) -> bool:
        """Whether a turn is running."""
        return self._task is not None and not self._task.done()

    def _require_idle(self) -> None:
        if self._reload is not None and not self._reload.done():
            raise StudioBusyError("the atom is being reloaded; wait for it to finish")
        if self.busy:
            raise StudioBusyError("a turn is running; wait for it to finish")

    def start_turn(self, message: str) -> asyncio.Queue[Event | None]:
        """Start a turn as a task owned by the session and return its event queue.

        A reader that stops early does not cancel the turn.

        Args:
            message: The user message.

        Returns:
            The queue of the turn's events, which ends with `None`.

        Raises:
            StudioBusyError: While another turn runs.
        """  # noqa: DOC502 - raised by _require_idle()
        self._require_idle()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Event | None] = asyncio.Queue()

        def emit(event: Event | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        enabled = {side: self.chains.enabled(side) for side in SIDES}
        record = TurnRecord(
            n=len(self.turns) + 1,
            message=message,
            chain={side: [orb.name for orb in enabled[side]] for side in SIDES},
            changes=[change.text for change in self.chains.changes()],
            models=frozenset(
                orb.name for side in SIDES for orb in enabled[side] if orb.calls_model
            ),
            signature=self.signature(),
            reloads=self.reloads,
        )
        self._task = loop.create_task(self._run(record, emit))
        return queue

    async def wait(self) -> None:
        """Wait for the running turn, if any."""
        if self._task is not None:
            await self._task

    async def _run(self, record: TurnRecord, emit: Callable[[Event | None], None]) -> None:
        emit({"type": "turn_started", "n": record.n})
        self._report = None
        self._running = record
        self.processor.attach(emit)
        try:
            try:
                outcome = await self._runner.run_turn(
                    self._agent, record.message, self.conversation
                )
            except Exception as exc:
                record.error = {"type": type(exc).__name__, "message": str(exc)}
            else:
                record.reply = outcome.reply
                # A framework can return from a turn the engine closed as aborted, one that
                # ended with no reply of the nucleus: it leaves nothing, like one that raised.
                if self._report is None or self._report.final_action != "aborted":
                    self.conversation = outcome.conversation
            record.report = self._report
            self.turns.append(record)
            emit({"type": "turn_finished", "turn": record.to_json()})
        finally:
            # Also when the task is cancelled: the stream must end and the session be free.
            self.processor.detach()
            self._running = None
            emit(None)

    def state(self) -> dict[str, Any]:
        """Return everything the page shows, as JSON-ready data (spec section 6)."""
        a = self.atom
        return {
            "studio": {
                "version": __version__,
                "capture_content": "full",
                # Without a module the studio does not know what to import again.
                "reload": self.namespace is not None,
                # The extras that would bring more orbitals to add.
                "extras": [{"module": module, "extra": extra} for module, extra in kinds.extras()],
                # The terminal has them too, but the page is what the developer looks at.
                "preset_connections": list(self.preset_connections),
            },
            "atom": {
                "name": a.name,
                "source": self.source,
                **self._runner.describe(a.connector, self._agent),
                "instructions": a.instructions,
                "knowledge": a.knowledge.text if a.knowledge is not None else None,
                "max_retries": a.max_retries,
                "refusal": a.refusal,
                "blocked_placeholder": a.blocked_placeholder,
                "blocked_guidance": a.blocked_guidance,
                "capture_content": a.content_policy.mode,
            },
            "orbitals": {
                side: [self._orbital(side, entry) for entry in self.chains.work[side]]
                for side in SIDES
            },
            "kinds": [
                {
                    "kind": kind.cls.__name__,
                    "sides": sorted(kind.cls.sides),
                    "summary": kinds.summary(kind.cls),
                    "calls_model": kind.cls.calls_model,
                    "name": self.chains.free_name(kind.cls),
                    "settings": [setting.schema() for setting in kind.settings],
                }
                for kind in kinds.addable()
                # On a network address a class that needs a connection is added in code.
                if self.local or not any(s.required and values.connects(s) for s in kind.settings)
            ],
            "changes": [change.to_json() for change in self.chains.changes()],
            "undo": self._undo[1] if self._undo is not None else None,
            "turns": [turn.to_json() for turn in self.turns],
            "busy": self.busy,
            "running": (
                {"n": self._running.n, "message": self._running.message}
                if self._running is not None
                else None
            ),
        }

    def _orbital(self, side: Side, entry: Entry) -> dict[str, Any]:
        orb = entry.orbital
        note = self.chains.note(entry)
        settings: list[dict[str, Any]] = []
        fixed: list[dict[str, Any]] = []
        if note is None:
            kind = kinds.describe(type(orb))
            now = kinds.values(orb)
            before = kinds.values(entry.base) if entry.base is not None else now
            settings = [
                {
                    **setting.schema(),
                    "value": now[setting.key],
                    "changed": now[setting.key] != before.get(setting.key),
                }
                for setting in kind.settings
                if setting.key in now
            ]
            known = codegen.names(self.namespace)
            # A block that holds an object of the developer's is listed with the objects.
            for key in (name for name in kind.order if name in {*kind.objects, *kinds.opaque(orb)}):
                value = kinds.read(orb, key, self.atom.connector)
                if value is not None:
                    # The class name only: an object's repr could carry a secret.
                    fixed.append(
                        {"key": key, "type": type(value).__name__, "name": known.get(id(value))}
                    )
        data = {
            "name": orb.name,
            "kind": type(orb).__name__,
            "side": side,
            "enabled": entry.enabled,
            "calls_model": orb.calls_model,
            "config": orb.config(),
            "fingerprint": orb.fingerprint(),
            "origin": entry.origin,
            "modified": self.chains.modified(entry),
            "editable": note is None,
            "note": note,
            "settings": settings,
            "fixed": fixed,
        }
        # In config, objects become their repr and non-finite floats strings: valid JSON.
        return json.loads(to_json(data, fallback=repr, inf_nan_mode="strings"))
