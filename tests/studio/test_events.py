from collections.abc import Callable, Sequence
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from chatomon import Atom, Orbital, Verdict
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.studio._events import Event, EventProcessor, end_event, start_event
from tests.fakes import ScriptedOrbital


def test_unrelated_spans_make_no_event():
    assert start_event("chatomon.input", {}) is None
    assert start_event("http.request", {}) is None
    assert end_event("http.request", {"chatomon.decision.kind": "proceed"}, 1.0) is None


def test_an_orbital_end_carries_its_verdict_labels():
    event = end_event(
        "chatomon.orbital judge",
        {
            "chatomon.orbital.name": "judge",
            "chatomon.orbital.side": "output",
            "chatomon.attempt": 2,
            "chatomon.verdict.action": "block",
            "chatomon.verdict.reason": "judge_block",
            "chatomon.verdict.has_guidance": True,
            "chatomon.verdict.escalate": False,
            "error.type": "TimeoutError",
        },
        12.345,
    )
    assert event == {
        "type": "orbital_finished",
        "name": "judge",
        "side": "output",
        "attempt": 2,
        "action": "block",
        "reason": "judge_block",
        "has_guidance": True,
        "escalate": False,
        "duration_ms": 12.3,
        "error_type": "TimeoutError",
    }


def test_a_detached_processor_forwards_nothing():
    events: list[Event] = []
    processor = EventProcessor()
    processor.attach(events.append)
    processor.detach()
    provider = TracerProvider()
    provider.add_span_processor(processor)
    with provider.get_tracer("t").start_as_current_span(
        "chatomon.output", attributes={"chatomon.attempt": 1}
    ):
        pass
    assert events == []


def scripted(*replies: str | Exception) -> FunctionModel:
    steps = list(replies)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        if isinstance(step, Exception):
            raise step
        return ModelResponse(parts=[TextPart(content=step)])

    return FunctionModel(respond)


def compact(event: Event) -> tuple[Any, ...]:
    kind = event["type"]
    if kind == "orbital_started":
        return (kind, event["side"], event["name"])
    if kind == "orbital_finished":
        return (kind, event["name"], event["action"])
    if kind == "input_decided":
        return (kind, event["kind"])
    if kind == "reply_drafted":
        return (kind, event["attempt"])
    if kind == "output_decided":
        return (kind, event["attempt"], event["kind"])
    return (kind,)


Factory = Callable[[], Sequence[Orbital]]


def allow(name: str) -> Factory:
    return lambda: [ScriptedOrbital(name=name)]


def none() -> Sequence[Orbital]:
    return []


# Scenario table: #58 runs the same table against the LangChain and MAF runners.
SCENARIOS: list[
    tuple[str, Factory, Factory, tuple[str | Exception, ...], list[tuple[Any, ...]]]
] = [
    (
        "answered",
        allow("guard"),
        allow("judge"),
        ("Hi",),
        [
            ("orbital_started", "input", "guard"),
            ("orbital_finished", "guard", "allow"),
            ("input_decided", "proceed"),
            ("reply_drafted", 1),
            ("orbital_started", "output", "judge"),
            ("orbital_finished", "judge", "allow"),
            ("output_decided", 1, "accept"),
        ],
    ),
    (
        "input block with a reply",
        lambda: [ScriptedOrbital(Verdict.block("No."), name="guard")],
        allow("judge"),
        ("unused",),
        [
            ("orbital_started", "input", "guard"),
            ("orbital_finished", "guard", "block"),
            ("input_decided", "reply"),
        ],
    ),
    (
        "guided refusal",
        lambda: [ScriptedOrbital(Verdict.block(guidance="Decline politely."), name="guard")],
        allow("judge"),
        ("I can't help with that.",),
        [
            ("orbital_started", "input", "guard"),
            ("orbital_finished", "guard", "block"),
            ("input_decided", "guided_refusal"),
            ("reply_drafted", 1),
            ("orbital_started", "output", "judge"),
            ("orbital_finished", "judge", "allow"),
            ("output_decided", 1, "accept"),
        ],
    ),
    (
        "regeneration",
        none,
        lambda: [
            ScriptedOrbital(Verdict.block(guidance="Shorter."), Verdict.allow(), name="judge")
        ],
        ("A long reply.", "Short."),
        [
            ("input_decided", "proceed"),
            ("reply_drafted", 1),
            ("orbital_started", "output", "judge"),
            ("orbital_finished", "judge", "block"),
            ("output_decided", 1, "regenerate"),
            ("reply_drafted", 2),
            ("orbital_started", "output", "judge"),
            ("orbital_finished", "judge", "allow"),
            ("output_decided", 2, "accept"),
        ],
    ),
    (
        "orbital response",
        lambda: [ScriptedOrbital(Verdict.respond("Open 9-18."), name="faq")],
        none,
        ("unused",),
        [
            ("orbital_started", "input", "faq"),
            ("orbital_finished", "faq", "respond"),
            ("input_decided", "reply"),
        ],
    ),
    (
        "no orbitals",
        none,
        none,
        ("Hi",),
        [("input_decided", "proceed"), ("reply_drafted", 1), ("output_decided", 1, "accept")],
    ),
]


async def run(inputs: Factory, outputs: Factory, replies: Sequence[str | Exception]) -> list[Event]:
    events: list[Event] = []
    processor = EventProcessor()
    processor.attach(events.append)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    agent = Atom(
        name="t",
        instructions="Be brief.",
        connector=PydanticAIConnector(scripted(*replies)),
        input_orbitals=inputs(),
        output_orbitals=outputs(),
        tracer_provider=provider,
    ).build()
    try:
        await agent.run("hello")
    finally:
        processor.detach()
    return events


@pytest.mark.parametrize(
    ("inputs", "outputs", "replies", "expected"),
    [scenario[1:] for scenario in SCENARIOS],
    ids=[scenario[0] for scenario in SCENARIOS],
)
async def test_event_sequence(inputs, outputs, replies, expected):
    assert [compact(event) for event in await run(inputs, outputs, replies)] == expected


async def test_an_aborted_turn_stops_after_the_input_chain():
    events: list[Event] = []
    processor = EventProcessor()
    processor.attach(events.append)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    agent = Atom(
        name="t",
        instructions="Be brief.",
        connector=PydanticAIConnector(scripted(RuntimeError("model down"))),
        input_orbitals=[ScriptedOrbital(name="guard")],
        tracer_provider=provider,
    ).build()
    with pytest.raises(RuntimeError, match="model down"):
        await agent.run("hello")
    assert [compact(event) for event in events] == [
        ("orbital_started", "input", "guard"),
        ("orbital_finished", "guard", "allow"),
        ("input_decided", "proceed"),
    ]
