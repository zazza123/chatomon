"""The studio session: the developer's atom, the studio's copy of it and one conversation."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from pydantic_core import to_json

from chatomon._version import __version__
from chatomon.atom import Atom
from chatomon.errors import ChatomonError
from chatomon.schemas import Side, TurnReport
from chatomon.studio import _kinds as kinds
from chatomon.studio._chains import SIDES, Chains, Entry, RefusedError
from chatomon.studio._events import Event, EventProcessor
from chatomon.studio._runners import check_supported


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


class StudioBusyError(ChatomonError):
    """A turn is running, so the session cannot start another one or change its orbitals."""


@dataclass
class TurnRecord:
    """A finished turn, as the page shows it."""

    n: int
    message: str
    chain: dict[str, list[str]]
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
            "error": self.error,
        }


class Session:
    """Runs turns on a copy of the developer's atom; one conversation, one turn at a time."""

    def __init__(self, atom: Atom[Any], *, source: str) -> None:
        check_supported(atom.connector)
        from chatomon.studio._runners import pydantic_ai as runner

        self.atom = atom
        self.source = source
        self._runner = runner
        self.processor = EventProcessor()
        self.provider = _always_recording_provider()
        self.provider.add_span_processor(self.processor)
        self.history: list[Any] = []
        self.turns: list[TurnRecord] = []
        self.notes: list[str] = runner.notes(atom.connector)
        self._task: asyncio.Task[None] | None = None
        self._running: TurnRecord | None = None
        self._report: TurnReport | None = None
        self.chains = Chains.from_atom(atom)
        self._undo: tuple[Chains, str] | None = None
        self._agent: Any = self._build(self.chains)

    def _build(self, chains: Chains) -> Any:
        a = self.atom
        copy: Atom[Any] = Atom(
            name=a.name,
            instructions=a.instructions,
            connector=a.connector,
            knowledge=a.knowledge,
            input_orbitals=chains.enabled("input"),
            output_orbitals=chains.enabled("output"),
            refusal=a.refusal,
            blocked_placeholder=a.blocked_placeholder,
            max_retries=a.max_retries,
            on_turn=self._record,
            capture_content="full",
            tracer_provider=self.provider,
        )
        return copy.build()

    def _adopt(self, candidate: Chains, *, undo: str | None = None) -> None:
        """Take the candidate chains if the atom builds with them; otherwise change nothing.

        With `undo`, the previous chains are kept under that label; any other
        change drops what was kept.
        """
        agent = self._build(candidate)
        previous = self.chains
        self.chains, self._agent = candidate, agent
        self._undo = (previous, undo) if undo is not None else None

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

    def reset_conversation(self) -> None:
        """Start a new conversation.

        Raises:
            StudioBusyError: While a turn runs.
        """  # noqa: DOC502 - raised by _require_idle()
        self._require_idle()
        self.history = []
        self.turns = []

    def _record(self, report: TurnReport) -> None:
        self._report = report

    @property
    def busy(self) -> bool:
        """Whether a turn is running."""
        return self._task is not None and not self._task.done()

    def _require_idle(self) -> None:
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

        record = TurnRecord(
            n=len(self.turns) + 1,
            message=message,
            chain={side: [orb.name for orb in self.chains.enabled(side)] for side in SIDES},
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
                outcome = await self._runner.run_turn(self._agent, record.message, self.history)
            except Exception as exc:
                record.error = {"type": type(exc).__name__, "message": str(exc)}
            else:
                record.reply = outcome.reply
                self.history = outcome.history
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
            "studio": {"version": __version__, "capture_content": "full"},
            "atom": {
                "name": a.name,
                "source": self.source,
                **self._runner.describe(a.connector),
                "instructions": a.instructions,
                "knowledge": a.knowledge.text if a.knowledge is not None else None,
                "max_retries": a.max_retries,
                "refusal": a.refusal,
                "blocked_placeholder": a.blocked_placeholder,
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
                    "name": self.chains.free_name(kind.cls),
                    "settings": [setting.schema() for setting in kind.settings],
                }
                for kind in kinds.addable()
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
                    "changed": now[setting.key] != before[setting.key],
                }
                for setting in kind.settings
            ]
            for key in kind.objects:
                value = kinds.read(orb, key, self.atom.connector)
                if value is not None:
                    # The class name only: an object's repr could carry a secret.
                    fixed.append({"key": key, "type": type(value).__name__, "name": None})
        data = {
            "name": orb.name,
            "kind": type(orb).__name__,
            "side": side,
            "enabled": entry.enabled,
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
