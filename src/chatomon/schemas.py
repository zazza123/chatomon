"""Data contracts shared by the engine, the orbitals and the connectors.

This module imports nothing from chatomon, so every other module can depend on it
without creating import cycles.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "DEFAULT_BLOCKED_PLACEHOLDER",
    "DEFAULT_REFUSAL",
    "CaptureContent",
    "ErrorPolicy",
    "FinalAction",
    "InputDecision",
    "Message",
    "OrbitalContext",
    "OrbitalResult",
    "OutputAttempt",
    "OutputDecision",
    "OutputDecisionKind",
    "Side",
    "TurnReport",
    "Verdict",
    "VerdictAction",
]

Side = Literal["input", "output"]
"""Where an orbital runs: on the user's message (`"input"`) or on the nucleus reply (`"output"`)."""

VerdictAction = Literal["allow", "block", "respond"]
"""What an orbital decided about the text it checked."""

ErrorPolicy = Literal["block", "allow"]
"""What the engine does when an orbital fails or times out."""

CaptureContent = Literal["none", "hash", "full"]
"""How much message content chatomon records in turn reports; logs and traces carry
labels only."""

FinalAction = Literal[
    "answered", "orbital_response", "blocked", "guided_refusal", "fallback_refusal", "aborted"
]
"""How a turn ended, as recorded in the [TurnReport][chatomon.schemas.TurnReport]."""

OutputDecisionKind = Literal["accept", "replace", "regenerate"]
"""What the engine decided about a reply of the nucleus: `accept` sends it, `replace`
sends another text instead, `regenerate` asks the nucleus for a new one."""

DEFAULT_REFUSAL = "I'm sorry, I can't help with that."
"""Reply used when a message is blocked without a specific reply."""

DEFAULT_BLOCKED_PLACEHOLDER = "[message removed by chatomon]"
"""Text that replaces a blocked user message in the conversation history."""


class Message(BaseModel):
    """One message of the conversation history."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    """Who wrote the message."""

    content: str
    """The message text."""


class Verdict(BaseModel):
    """The result of an orbital's check.

    Build verdicts with [allow][chatomon.schemas.Verdict.allow],
    [block][chatomon.schemas.Verdict.block] and
    [respond][chatomon.schemas.Verdict.respond] rather than the constructor.

    Examples:
        ```python
        from chatomon import Verdict

        ok = Verdict.allow(guidance="The user sounds frustrated: be empathetic.")
        refusal = Verdict.block(reply="I can only help with parcel tracking.")
        rewrite = Verdict.block(guidance="Do not mention competitors.")
        faq = Verdict.respond("We are open Monday to Friday, 9 to 18.")

        assert ok.action == "allow"
        assert refusal.reply and rewrite.guidance and faq.reply
        ```
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    action: VerdictAction
    """`"allow"`, `"block"` or `"respond"`."""

    reply: str | None = None
    """Text shown to the user (block with a reply, or respond)."""

    guidance: str | None = None
    """Instruction for the nucleus (allow on input, or block)."""

    reason: str | None = None
    """Short label for the audit report. Must never contain user text."""

    escalate: bool = False
    """Whether the conversation needs a human."""

    annotations: Any = None
    """Data for later orbitals. Treated as content by `capture_content`."""

    @model_validator(mode="after")
    def _check_consistency(self) -> Verdict:
        if self.action == "allow" and self.reply is not None:
            raise ValueError("an 'allow' verdict cannot carry a reply")
        if self.action == "respond":
            if not self.reply or not self.reply.strip():
                raise ValueError("a 'respond' verdict needs a non-empty reply")
            if self.guidance is not None:
                raise ValueError("a 'respond' verdict cannot carry guidance")
        if self.action == "block" and self.reply is not None and self.guidance is not None:
            raise ValueError("a 'block' verdict takes a reply or guidance, not both")
        return self

    @classmethod
    def allow(
        cls, guidance: str | None = None, *, reason: str | None = None, annotations: Any = None
    ) -> Verdict:
        """Let the text through.

        Args:
            guidance: On the input side, an instruction the nucleus follows for this
                turn. Ignored (with a warning) on the output side.
            reason: Short label for the audit report.
            annotations: Data for later orbitals.

        Returns:
            An `allow` verdict.
        """
        return cls(action="allow", guidance=guidance, reason=reason, annotations=annotations)

    @classmethod
    def block(
        cls,
        reply: str | None = None,
        *,
        guidance: str | None = None,
        reason: str | None = None,
        escalate: bool = False,
        annotations: Any = None,
    ) -> Verdict:
        """Stop the text.

        With `reply`, the user receives it. With `guidance`, the nucleus writes the
        answer following it (on the input side without seeing the blocked message;
        on the output side by regenerating its reply). With neither, the atom's
        refusal is used.

        Args:
            reply: Text shown to the user.
            guidance: Instruction for the nucleus.
            reason: Short label for the audit report.
            escalate: Whether the conversation needs a human.
            annotations: Data for later orbitals.

        Returns:
            A `block` verdict.
        """
        return cls(
            action="block",
            reply=reply,
            guidance=guidance,
            reason=reason,
            escalate=escalate,
            annotations=annotations,
        )

    @classmethod
    def respond(
        cls,
        text: str,
        *,
        reason: str | None = None,
        escalate: bool = False,
        annotations: Any = None,
    ) -> Verdict:
        """Answer the user directly instead of the nucleus.

        Args:
            text: The reply shown to the user.
            reason: Short label for the audit report.
            escalate: Whether the conversation needs a human.
            annotations: Data for later orbitals.

        Returns:
            A `respond` verdict.
        """
        return cls(
            action="respond",
            reply=text,
            reason=reason,
            escalate=escalate,
            annotations=annotations,
        )


class OrbitalContext(BaseModel):
    """What an orbital sees when it checks a message."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    side: Side
    """Whether the text is the user's message (`"input"`) or the nucleus reply (`"output"`)."""

    message: str
    """The text under check."""

    history: tuple[Message, ...] = ()
    """Earlier messages, with blocked user messages already replaced by the placeholder.
    On the output side it includes the current user message."""

    annotations: Mapping[str, Any] = Field(default_factory=dict)
    """Annotations of the orbitals that already ran in this turn, by orbital name.

    The values are the objects the producing orbitals returned, shared with later
    orbitals and the report: treat them as read-only."""

    attempt: int = 1
    """On the output side, the attempt number (1 is the first reply)."""

    turn_id: str
    """Identifier of the turn."""

    conversation_id: str | None = None
    """Identifier of the conversation, when the framework provides one."""


class InputDecision(BaseModel):
    """What the connector must do after the input chain."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["proceed", "reply", "guided_refusal"]
    """`proceed`: call the nucleus with the message; `reply`: send `reply` without
    calling the nucleus; `guided_refusal`: call the nucleus without the message,
    with `guidance` only."""

    guidance: tuple[str, ...] = ()
    """Guidance to add to this turn's model call."""

    reply: str | None = None
    """The reply to send when `kind` is `reply`."""

    redact_user_message: bool = False
    """Whether the user message must be replaced by the placeholder in the history."""

    @model_validator(mode="after")
    def _check_payload(self) -> InputDecision:
        if self.kind == "reply" and self.reply is None:
            raise ValueError("a 'reply' decision needs a reply")
        if self.kind == "guided_refusal" and not self.guidance:
            raise ValueError("a 'guided_refusal' decision needs guidance")
        return self


class OutputDecision(BaseModel):
    """What the connector must do after the output chain."""

    model_config = ConfigDict(frozen=True)

    kind: OutputDecisionKind
    """`accept`: send the reply; `replace`: send `text` instead; `regenerate`: call
    the nucleus again with `guidance`, then run the output chain again."""

    text: str | None = None
    """The replacement reply when `kind` is `replace`."""

    guidance: tuple[str, ...] = ()
    """Guidance for the regeneration when `kind` is `regenerate`."""

    @model_validator(mode="after")
    def _check_payload(self) -> OutputDecision:
        if self.kind == "replace" and self.text is None:
            raise ValueError("a 'replace' decision needs text")
        if self.kind == "regenerate" and not self.guidance:
            raise ValueError("a 'regenerate' decision needs guidance")
        return self


class OrbitalResult(BaseModel):
    """Audit record of one orbital execution."""

    model_config = ConfigDict(frozen=True)

    name: str
    """Orbital name."""

    side: Side
    """Side it ran on."""

    kind: str
    """Orbital class name."""

    fingerprint: str
    """Hash of the orbital's configuration at build time."""

    action: VerdictAction
    """Action of the verdict that was applied (after `on_error`)."""

    reason: str | None = None
    """Verdict reason label."""

    has_guidance: bool = False
    """Whether the verdict carried guidance."""

    escalate: bool = False
    """Whether the verdict asked for a human."""

    attempt: int = 1
    """Attempt number (output side)."""

    duration_ms: float
    """Execution time in milliseconds."""

    error_type: str | None = None
    """Exception class name when the orbital failed or timed out."""

    error_message: str | None = None
    """Exception message; only with `capture_content="full"`."""

    error_policy: ErrorPolicy | None = None
    """The `on_error` policy applied when the orbital failed."""

    guidance: str | None = None
    """Guidance text, according to `capture_content`."""

    reply: str | None = None
    """Reply text, according to `capture_content`."""

    annotations: str | None = None
    """Annotations as JSON, according to `capture_content`."""


class OutputAttempt(BaseModel):
    """Audit record of one reply of the nucleus and of the output chain that checked it."""

    model_config = ConfigDict(frozen=True)

    nucleus_reply: str | None = None
    """The reply the nucleus wrote, according to `capture_content`. It is what the
    nucleus wrote whatever the decision: a text that took its place is in the result
    of the orbital that gave it and, when it ended the turn, in `final_reply`."""

    results: tuple[OrbitalResult, ...]
    """Results of the output chain on this reply, in order."""

    decision: OutputDecisionKind | None = None
    """What the engine decided about the reply; `None` when the turn was aborted
    while the output chain was running."""


class TurnReport(BaseModel):
    """Audit record of one conversation turn, delivered to `on_turn`."""

    model_config = ConfigDict(frozen=True)

    turn_id: str
    """Identifier of the turn."""

    conversation_id: str | None
    """Identifier of the conversation, when available."""

    atom_name: str
    """Name of the atom."""

    started_at: datetime
    """When the turn started (UTC)."""

    duration_ms: float
    """Time from the start of the turn to its report, in milliseconds."""

    final_action: FinalAction
    """How the turn ended."""

    escalated: bool
    """Whether any orbital asked for a human."""

    input: tuple[OrbitalResult, ...]
    """Results of the input chain."""

    output: tuple[OutputAttempt, ...]
    """The attempts of the nucleus, in order: one per reply the output chain checked."""

    user_message: str | None = None
    """The user message, according to `capture_content`."""

    final_reply: str | None = None
    """The reply sent to the user, according to `capture_content`."""

    error_type: str | None = None
    """Exception class name when the turn was aborted."""

    error_message: str | None = None
    """Exception message of an aborted turn; only with `capture_content="full"`."""
