"""Turn the spans chatomon already opens into the studio's live events.

The event shapes are the contract the studio page reads; the tests in
`tests/studio/test_events.py` pin them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

Event = dict[str, Any]
"""One live event: JSON-serializable, labels only, never message text."""

EventSink = Callable[[Event], None]
"""Receives the events of the running turn."""

_ORBITAL = "chatomon.orbital "


def start_event(name: str, attributes: Mapping[str, Any]) -> Event | None:
    """Return the event for a span that starts, or `None` for spans the studio ignores.

    Args:
        name: Span name.
        attributes: Span attributes at start.

    Returns:
        `orbital_started`, `reply_drafted` or `None`.
    """
    if name.startswith(_ORBITAL):
        return {
            "type": "orbital_started",
            "name": attributes.get("chatomon.orbital.name"),
            "side": attributes.get("chatomon.orbital.side"),
            "attempt": attributes.get("chatomon.attempt", 1),
        }
    if name == "chatomon.output":
        return {"type": "reply_drafted", "attempt": attributes.get("chatomon.attempt", 1)}
    return None


def end_event(name: str, attributes: Mapping[str, Any], duration_ms: float) -> Event | None:
    """Return the event for a span that ends, or `None` for spans the studio ignores.

    Args:
        name: Span name.
        attributes: Span attributes at end.
        duration_ms: Span duration.

    Returns:
        `orbital_finished`, `input_decided`, `output_decided` or `None`.
    """
    if name.startswith(_ORBITAL):
        return {
            "type": "orbital_finished",
            "name": attributes.get("chatomon.orbital.name"),
            "side": attributes.get("chatomon.orbital.side"),
            "attempt": attributes.get("chatomon.attempt", 1),
            "action": attributes.get("chatomon.verdict.action"),
            "reason": attributes.get("chatomon.verdict.reason"),
            "has_guidance": bool(attributes.get("chatomon.verdict.has_guidance", False)),
            "escalate": bool(attributes.get("chatomon.verdict.escalate", False)),
            "duration_ms": round(duration_ms, 1),
            "error_type": attributes.get("error.type"),
        }
    kind = attributes.get("chatomon.decision.kind")
    if kind is None:
        return None
    if name == "chatomon.input":
        return {"type": "input_decided", "kind": kind}
    if name == "chatomon.output":
        return {
            "type": "output_decided",
            "attempt": attributes.get("chatomon.attempt", 1),
            "kind": kind,
        }
    return None


class EventProcessor(SpanProcessor):
    """Forwards the events of chatomon's spans to the sink of the running turn."""

    def __init__(self) -> None:
        self._sink: EventSink | None = None

    def attach(self, sink: EventSink) -> None:
        """Send the events of the next spans to `sink`."""
        self._sink = sink

    def detach(self) -> None:
        """Stop forwarding events."""
        self._sink = None

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        sink = self._sink
        if sink is None:
            return
        event = start_event(span.name, span.attributes or {})
        if event is not None:
            sink(event)

    def on_end(self, span: ReadableSpan) -> None:
        sink = self._sink
        if sink is None:
            return
        start, end = span.start_time, span.end_time
        duration = (end - start) / 1_000_000 if start is not None and end is not None else 0.0
        event = end_event(span.name, span.attributes or {}, duration)
        if event is not None:
            sink(event)
