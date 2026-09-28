"""The orbital-chain engine: runs orbitals and turns verdicts into decisions.

The engine is pure Python and framework-agnostic. A connector calls it from the
framework's native hooks, in this order for every turn:

1. `start_turn`;
2. `run_input` on the user message;
3. unless the input decision is `reply`, `run_output` on every final reply of
   the nucleus, until the decision is not `regenerate`;
4. `finish_turn`.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from chatomon.errors import ChatomonConfigError, MissingAnnotationsError
from chatomon.observability import ContentPolicy, get_tracer
from chatomon.orbitals.base import Orbital
from chatomon.schemas import (
    DEFAULT_REFUSAL,
    FinalAction,
    InputDecision,
    Message,
    OrbitalContext,
    OrbitalResult,
    OutputDecision,
    Side,
    TurnReport,
    Verdict,
)

__all__ = ["Engine", "TurnCallback", "TurnState"]

logger = logging.getLogger(__name__)

TurnCallback = Callable[[TurnReport], Awaitable[None] | None]
"""Sync or async function that receives the report of every turn."""


@dataclass
class TurnState:
    """Mutable state of one turn; the connector keeps it between its hooks."""

    turn_id: str
    """Identifier of the turn."""

    conversation_id: str | None
    """Identifier of the conversation, when available."""

    started_at: datetime
    """When the turn started (UTC)."""

    started_perf: float
    """`time.perf_counter()` at the start of the turn."""

    annotations: dict[str, Any] = field(default_factory=dict)
    """Annotations produced so far, by orbital name."""

    input_annotations: dict[str, Any] = field(default_factory=dict)
    """Annotations left by the input chain; every output attempt starts from them."""

    input_results: list[OrbitalResult] = field(default_factory=list)
    """Audit records of the input chain."""

    output_results: list[list[OrbitalResult]] = field(default_factory=list)
    """Audit records of the output chain, one list per attempt."""

    input_decision: InputDecision | None = None
    """The decision of the input chain, once it ran."""

    attempt: int = 0
    """Number of output-chain runs so far."""

    user_message: str | None = None
    """The user message of this turn."""

    final_reply: str | None = None
    """The reply sent to the user, once known."""

    final_action: FinalAction | None = None
    """How the turn ended, once known."""

    escalated: bool = False
    """Whether any orbital asked for a human."""

    finished: bool = False
    """Whether the report was produced."""


class Engine:
    """Runs an atom's orbital chains; created by `Atom.build()`.

    Args:
        atom_name: Name of the atom, used in spans, logs and reports.
        input_orbitals: Orbitals run in order on the user message.
        output_orbitals: Orbitals run in order on the nucleus reply.
        refusal: Reply used when a message is blocked without a specific reply.
        max_retries: How many times the nucleus may regenerate a reply blocked
            with guidance before the refusal is used.
        on_turn: Receives the report of every turn.
        content: Content-capture policy; records nothing when `None`.
        tracer: OpenTelemetry tracer; chatomon's global tracer when `None`.

    Raises:
        ChatomonConfigError: If `max_retries` is negative.
    """

    def __init__(
        self,
        *,
        atom_name: str,
        input_orbitals: Sequence[Orbital] = (),
        output_orbitals: Sequence[Orbital] = (),
        refusal: str = DEFAULT_REFUSAL,
        max_retries: int = 2,
        on_turn: TurnCallback | None = None,
        content: ContentPolicy | None = None,
        tracer: trace.Tracer | None = None,
    ) -> None:
        if max_retries < 0:
            raise ChatomonConfigError(f"max_retries must be >= 0, got {max_retries!r}")
        self._atom_name = atom_name
        self._input = tuple(input_orbitals)
        self._output = tuple(output_orbitals)
        self._refusal = refusal
        self._max_retries = max_retries
        self._on_turn = on_turn
        self._content = content if content is not None else ContentPolicy()
        self._tracer = tracer if tracer is not None else get_tracer()
        self._fingerprints = {id(orb): orb.fingerprint() for orb in (*self._input, *self._output)}

    @property
    def atom_name(self) -> str:
        """Name of the atom."""
        return self._atom_name

    @property
    def input_orbitals(self) -> tuple[Orbital, ...]:
        """The input chain."""
        return self._input

    @property
    def output_orbitals(self) -> tuple[Orbital, ...]:
        """The output chain."""
        return self._output

    @property
    def refusal(self) -> str:
        """The default refusal."""
        return self._refusal

    @property
    def max_retries(self) -> int:
        """Maximum number of regenerations per turn."""
        return self._max_retries

    @property
    def content(self) -> ContentPolicy:
        """The content-capture policy."""
        return self._content

    def start_turn(self, *, conversation_id: str | None = None) -> TurnState:
        """Start a new turn.

        Args:
            conversation_id: Identifier of the conversation, when the framework has one.

        Returns:
            The state the connector passes to the other methods.
        """
        return TurnState(
            turn_id=uuid.uuid4().hex,
            conversation_id=conversation_id,
            started_at=datetime.now(timezone.utc),
            started_perf=time.perf_counter(),
        )

    async def run_input(
        self, turn: TurnState, *, message: str, history: Sequence[Message] = ()
    ) -> InputDecision:
        """Run the input chain on the user message.

        Args:
            turn: The turn started with `start_turn`.
            message: The user message.
            history: Earlier messages, with blocked user messages already replaced
                by the placeholder.

        Returns:
            What the connector must do next.

        Raises:
            RuntimeError: If the turn is finished or the input chain already ran.
        """
        if turn.finished:
            raise RuntimeError("this turn is already finished")
        if turn.input_decision is not None:
            raise RuntimeError("run_input was already called for this turn")
        turn.user_message = message
        guidance: list[str] = []
        decision: InputDecision | None = None
        with self._tracer.start_as_current_span(
            "chatomon.input", attributes=self._turn_attributes(turn)
        ) as span:
            for orb in self._input:
                verdict = await self._run_orbital(
                    orb,
                    turn,
                    side="input",
                    message=message,
                    history=history,
                    attempt=1,
                    results=turn.input_results,
                )
                if verdict.action == "allow":
                    if verdict.guidance:
                        guidance.append(verdict.guidance)
                    continue
                decision = self._stop_input(turn, verdict)
                break
            if decision is None:
                decision = InputDecision(kind="proceed", guidance=tuple(guidance))
            span.set_attribute("chatomon.decision.kind", decision.kind)
        turn.input_decision = decision
        turn.input_annotations = dict(turn.annotations)
        self._log_decision(turn, "input", decision.kind)
        return decision

    def _stop_input(self, turn: TurnState, verdict: Verdict) -> InputDecision:
        if verdict.action == "respond":
            reply = verdict.reply or self._refusal
            turn.final_action = "orbital_response"
            turn.final_reply = reply
            return InputDecision(kind="reply", reply=reply)
        if verdict.guidance:
            return InputDecision(
                kind="guided_refusal", guidance=(verdict.guidance,), redact_user_message=True
            )
        reply = verdict.reply or self._refusal
        turn.final_action = "blocked"
        turn.final_reply = reply
        return InputDecision(kind="reply", reply=reply, redact_user_message=True)

    async def run_output(
        self, turn: TurnState, *, reply: str, history: Sequence[Message] = ()
    ) -> OutputDecision:
        """Run the output chain on a final reply of the nucleus.

        Call it on every final reply until the decision is not `regenerate`.

        Args:
            turn: The turn whose input chain already ran.
            reply: The nucleus reply.
            history: Earlier messages including the current user message
                (replaced by the placeholder if the input decision asked for it).

        Returns:
            What the connector must do with the reply.

        Raises:
            RuntimeError: If the input chain did not run, already produced the
                reply, or the turn already has its final reply.
        """
        if turn.finished:
            raise RuntimeError("this turn is already finished")
        if turn.input_decision is None:
            raise RuntimeError("call run_input before run_output")
        if turn.input_decision.kind == "reply":
            raise RuntimeError(
                "the input chain already produced the reply; run_output must not be called"
            )
        if turn.final_action is not None:
            raise RuntimeError("this turn already has its final reply")
        turn.attempt += 1
        attempt = turn.attempt
        turn.annotations = dict(turn.input_annotations)
        results: list[OrbitalResult] = []
        turn.output_results.append(results)
        decision: OutputDecision | None = None
        with self._tracer.start_as_current_span(
            "chatomon.output",
            attributes={**self._turn_attributes(turn), "chatomon.attempt": attempt},
        ) as span:
            for orb in self._output:
                verdict = await self._run_orbital(
                    orb,
                    turn,
                    side="output",
                    message=reply,
                    history=history,
                    attempt=attempt,
                    results=results,
                )
                if verdict.action == "allow":
                    continue
                decision = self._stop_output(turn, verdict, attempt)
                break
            if decision is None:
                decision = OutputDecision(kind="accept")
                turn.final_reply = reply
                turn.final_action = (
                    "guided_refusal" if turn.input_decision.kind == "guided_refusal" else "answered"
                )
            span.set_attribute("chatomon.decision.kind", decision.kind)
        self._log_decision(turn, "output", decision.kind)
        return decision

    def _stop_output(self, turn: TurnState, verdict: Verdict, attempt: int) -> OutputDecision:
        if verdict.action == "respond":
            text = verdict.reply or self._refusal
            turn.final_action = "orbital_response"
            turn.final_reply = text
            return OutputDecision(kind="replace", text=text)
        if verdict.guidance:
            if attempt <= self._max_retries:
                return OutputDecision(kind="regenerate", guidance=(verdict.guidance,))
            turn.final_action = "fallback_refusal"
            turn.final_reply = self._refusal
            return OutputDecision(kind="replace", text=self._refusal)
        text = verdict.reply or self._refusal
        turn.final_action = "blocked"
        turn.final_reply = text
        return OutputDecision(kind="replace", text=text)

    async def finish_turn(self, turn: TurnState) -> TurnReport:
        """Close the turn, build its report and deliver it to `on_turn`.

        An exception raised by `on_turn` is logged and never propagates.

        Args:
            turn: The turn, which must have its final reply.

        Returns:
            The audit report of the turn.

        Raises:
            RuntimeError: If the turn has no final reply yet or was already finished.
        """
        if turn.finished:
            raise RuntimeError("finish_turn was already called for this turn")
        if turn.final_action is None:
            raise RuntimeError("the turn has no final reply yet")
        turn.finished = True
        report = TurnReport(
            turn_id=turn.turn_id,
            conversation_id=turn.conversation_id,
            atom_name=self._atom_name,
            started_at=turn.started_at,
            duration_ms=(time.perf_counter() - turn.started_perf) * 1000,
            final_action=turn.final_action,
            escalated=turn.escalated,
            input=tuple(turn.input_results),
            output=tuple(tuple(attempt) for attempt in turn.output_results),
            user_message=self._content.text(turn.user_message),
            final_reply=self._content.text(turn.final_reply),
        )
        fields = {
            **self._log_fields(turn),
            "final_action": report.final_action,
            "escalated": report.escalated,
        }
        level = (
            logging.DEBUG
            if report.final_action == "answered" and not report.escalated
            else logging.INFO
        )
        logger.log(level, "turn finished: %s", report.final_action, extra={"chatomon": fields})
        if self._on_turn is not None:
            try:
                result = self._on_turn(report)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception(
                    "on_turn callback failed for turn %s", turn.turn_id, extra={"chatomon": fields}
                )
        return report

    async def _run_orbital(
        self,
        orb: Orbital,
        turn: TurnState,
        *,
        side: Side,
        message: str,
        history: Sequence[Message],
        attempt: int,
        results: list[OrbitalResult],
    ) -> Verdict:
        ctx = OrbitalContext(
            side=side,
            message=message,
            history=tuple(history),
            annotations=dict(turn.annotations),
            attempt=attempt,
            turn_id=turn.turn_id,
            conversation_id=turn.conversation_id,
        )
        attributes = {
            **self._turn_attributes(turn),
            "chatomon.orbital.name": orb.name,
            "chatomon.orbital.side": side,
            "chatomon.orbital.fingerprint": self._fingerprints[id(orb)],
            "chatomon.attempt": attempt,
        }
        started = time.perf_counter()
        error: Exception | None = None
        with self._tracer.start_as_current_span(
            f"chatomon.orbital {orb.name}", attributes=attributes
        ) as span:
            try:
                missing = [name for name in orb.reads if name not in turn.annotations]
                if missing:
                    raise MissingAnnotationsError(orb.name, missing)
                verdict = await asyncio.wait_for(orb.check(ctx), timeout=orb.timeout)
                if not isinstance(verdict, Verdict):
                    raise TypeError(
                        f"orbital {orb.name!r} returned {type(verdict).__name__}, expected Verdict"
                    )
            except Exception as exc:
                error = exc
                verdict = (
                    Verdict.block(reason="orbital_error")
                    if orb.on_error == "block"
                    else Verdict.allow(reason="orbital_error")
                )
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("error.type", type(exc).__name__)
            duration_ms = (time.perf_counter() - started) * 1000
            span.set_attribute("chatomon.verdict.action", verdict.action)
            if verdict.reason:
                span.set_attribute("chatomon.verdict.reason", verdict.reason)
            span.set_attribute("chatomon.verdict.has_guidance", bool(verdict.guidance))
            span.set_attribute("chatomon.verdict.escalate", verdict.escalate)
        if error is None and verdict.annotations is not None:
            turn.annotations[orb.name] = verdict.annotations
        if verdict.escalate:
            turn.escalated = True
        result = self._result(orb, side, verdict, attempt, duration_ms, error)
        results.append(result)
        self._log_orbital(turn, result)
        if side == "output" and verdict.action == "allow" and verdict.guidance:
            logger.warning(
                "orbital %s returned guidance on the output side; it is ignored",
                orb.name,
                extra={"chatomon": self._log_fields(turn, result)},
            )
        return verdict

    def _result(
        self,
        orb: Orbital,
        side: Side,
        verdict: Verdict,
        attempt: int,
        duration_ms: float,
        error: Exception | None,
    ) -> OrbitalResult:
        content = self._content
        return OrbitalResult(
            name=orb.name,
            side=side,
            kind=type(orb).__qualname__,
            fingerprint=self._fingerprints[id(orb)],
            action=verdict.action,
            reason=verdict.reason,
            has_guidance=bool(verdict.guidance),
            escalate=verdict.escalate,
            attempt=attempt,
            duration_ms=duration_ms,
            error_type=type(error).__name__ if error is not None else None,
            error_message=(
                content.text(str(error)) if error is not None and content.mode == "full" else None
            ),
            error_policy=orb.on_error if error is not None else None,
            guidance=content.text(verdict.guidance),
            reply=content.text(verdict.reply),
            annotations=content.data(verdict.annotations),
        )

    def _turn_attributes(self, turn: TurnState) -> dict[str, str]:
        attributes = {"chatomon.atom.name": self._atom_name, "chatomon.turn.id": turn.turn_id}
        if turn.conversation_id is not None:
            attributes["chatomon.conversation.id"] = turn.conversation_id
        return attributes

    def _log_fields(self, turn: TurnState, result: OrbitalResult | None = None) -> dict[str, Any]:
        fields: dict[str, Any] = {"atom": self._atom_name, "turn_id": turn.turn_id}
        if result is not None:
            fields.update(
                orbital=result.name,
                side=result.side,
                action=result.action,
                reason=result.reason,
                attempt=result.attempt,
                error_type=result.error_type,
            )
        return fields

    def _log_orbital(self, turn: TurnState, result: OrbitalResult) -> None:
        extra = {"chatomon": self._log_fields(turn, result)}
        if result.error_type is not None:
            logger.warning(
                "orbital %s failed with %s; applying on_error=%s",
                result.name,
                result.error_type,
                result.error_policy,
                extra=extra,
            )
        else:
            logger.debug("orbital %s returned %s", result.name, result.action, extra=extra)

    def _log_decision(self, turn: TurnState, side: Side, kind: str) -> None:
        level = logging.DEBUG if kind in ("proceed", "accept") else logging.INFO
        logger.log(
            level,
            "%s decision: %s",
            side,
            kind,
            extra={"chatomon": {**self._log_fields(turn), "side": side, "decision": kind}},
        )
