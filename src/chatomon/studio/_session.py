"""The studio session: the developer's atom, the studio's copy of it and one conversation."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from pydantic_core import to_json

from chatomon._version import __version__
from chatomon.atom import Atom
from chatomon.errors import ChatomonConfigError, ChatomonError
from chatomon.orbitals.base import Orbital
from chatomon.schemas import Side, TurnReport
from chatomon.studio._events import Event, EventProcessor
from chatomon.studio._runners import check_supported

SIDES: tuple[Side, ...] = ("input", "output")


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


class UnknownOrbitalError(ChatomonError):
    """The atom has no orbital with that side and name."""


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
        self._enabled: dict[tuple[Side, str], bool] = {
            (side, orb.name): True for side in SIDES for orb in self._chain(side)
        }
        self.processor = EventProcessor()
        self.provider = _always_recording_provider()
        self.provider.add_span_processor(self.processor)
        self.history: list[Any] = []
        self.turns: list[TurnRecord] = []
        self._task: asyncio.Task[None] | None = None
        self._report: TurnReport | None = None
        self._agent: Any = self._build()

    def _chain(self, side: Side) -> tuple[Orbital, ...]:
        return self.atom.input_orbitals if side == "input" else self.atom.output_orbitals

    def _on(self, side: Side) -> list[Orbital]:
        return [orb for orb in self._chain(side) if self._enabled[(side, orb.name)]]

    def _build(self) -> Any:
        a = self.atom
        copy: Atom[Any] = Atom(
            name=a.name,
            instructions=a.instructions,
            connector=a.connector,
            knowledge=a.knowledge,
            input_orbitals=self._on("input"),
            output_orbitals=self._on("output"),
            refusal=a.refusal,
            blocked_placeholder=a.blocked_placeholder,
            max_retries=a.max_retries,
            on_turn=self._record,
            capture_content="full",
            tracer_provider=self.provider,
        )
        return copy.build()

    def _record(self, report: TurnReport) -> None:
        self._report = report

    @property
    def busy(self) -> bool:
        """Whether a turn is running."""
        return self._task is not None and not self._task.done()

    def _require_idle(self) -> None:
        if self.busy:
            raise StudioBusyError("a turn is running; wait for it to finish")

    def set_enabled(self, side: str, name: str, enabled: bool) -> None:
        """Turn an orbital on or off for the next turns.

        Raises:
            StudioBusyError: While a turn runs.
            UnknownOrbitalError: If the atom has no such orbital.
            ChatomonConfigError: If the atom would be invalid; nothing changes.
        """  # noqa: DOC502 - StudioBusyError is raised by _require_idle()
        self._require_idle()
        key = next((key for key in self._enabled if key == (side, name)), None)
        if key is None:
            raise UnknownOrbitalError(f"the atom has no {side} orbital named {name!r}")
        previous = self._enabled[key]
        self._enabled[key] = enabled
        try:
            self._agent = self._build()
        except ChatomonConfigError:
            self._enabled[key] = previous
            raise

    def reset(self) -> None:
        """Start a new conversation.

        Raises:
            StudioBusyError: While a turn runs.
        """  # noqa: DOC502 - raised by _require_idle()
        self._require_idle()
        self.history = []
        self.turns = []

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
            chain={side: [orb.name for orb in self._on(side)] for side in SIDES},
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
        self.processor.attach(emit)
        try:
            outcome = await self._runner.run_turn(self._agent, record.message, self.history)
        except Exception as exc:
            record.error = {"type": type(exc).__name__, "message": str(exc)}
        else:
            record.reply = outcome.reply
            self.history = outcome.history
        finally:
            self.processor.detach()
        record.report = self._report
        self.turns.append(record)
        emit({"type": "turn_finished", "turn": record.to_json()})
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
                side: [self._orbital(side, orb) for orb in self._chain(side)] for side in SIDES
            },
            "turns": [turn.to_json() for turn in self.turns],
            "busy": self.busy,
        }

    def _orbital(self, side: Side, orb: Orbital) -> dict[str, Any]:
        return {
            "name": orb.name,
            "kind": type(orb).__name__,
            "side": side,
            "enabled": self._enabled[(side, orb.name)],
            # Objects become their repr and non-finite floats strings, so the state is valid JSON.
            "config": json.loads(to_json(orb.config(), fallback=repr, inf_nan_mode="strings")),
            "fingerprint": orb.fingerprint(),
        }
