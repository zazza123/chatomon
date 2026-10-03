import asyncio
import json
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from chatomon import Atom, ChatomonConfigError, Verdict
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.schemas import DEFAULT_BLOCKED_PLACEHOLDER
from chatomon.studio._session import Session, StudioBusyError, UnknownOrbitalError
from tests.fakes import FakeConnector, ScriptedOrbital


def make_atom(**kwargs: Any) -> Atom:
    kwargs.setdefault("connector", PydanticAIConnector(TestModel(custom_output_text="Hi")))
    return Atom(name="speedypack", instructions="Be brief.", **kwargs)


async def drain(queue: asyncio.Queue) -> list[dict]:
    events = []
    while (event := await queue.get()) is not None:
        events.append(event)
    return events


def test_the_developers_atom_is_left_untouched():
    def on_turn(report):
        pass

    atom = make_atom(on_turn=on_turn)
    Session(atom, source="app:atom")
    assert atom.on_turn is on_turn
    assert atom.content_policy.mode == "none"
    assert atom.tracer_provider is None


def test_an_unsupported_connector_is_refused():
    with pytest.raises(ChatomonConfigError, match="PydanticAI atoms for now"):
        Session(make_atom(connector=FakeConnector()), source="app:atom")


async def test_a_turn_streams_its_events_and_is_recorded():
    session = Session(make_atom(input_orbitals=[ScriptedOrbital(name="guard")]), source="app:atom")
    events = await drain(session.start_turn("hello"))
    assert events[0] == {"type": "turn_started", "n": 1}
    finished = events[-1]
    assert finished["type"] == "turn_finished"
    turn = finished["turn"]
    assert turn["reply"] == "Hi"
    assert turn["chain"] == {"input": ["guard"], "output": []}
    assert turn["report"]["final_action"] == "answered"
    assert turn["report"]["user_message"] == "hello"  # the studio captures content in full
    assert [record.n for record in session.turns] == [1]


async def test_history_carries_across_turns_and_keeps_the_placeholder():
    guard = ScriptedOrbital(Verdict.block("No."), Verdict.allow(), name="guard")
    session = Session(make_atom(input_orbitals=[guard]), source="app:atom")
    await drain(session.start_turn("secret"))
    await drain(session.start_turn("hello"))
    assert [message.content for message in guard.contexts[1].history] == [
        DEFAULT_BLOCKED_PLACEHOLDER,
        "No.",
    ]


async def test_one_turn_at_a_time_and_busy_in_the_state():
    gate = asyncio.Event()

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await gate.wait()
        return ModelResponse(parts=[TextPart(content="Hi")])

    session = Session(
        make_atom(connector=PydanticAIConnector(FunctionModel(respond))), source="app:atom"
    )
    queue = session.start_turn("hello")
    await asyncio.sleep(0)
    assert session.state()["busy"] is True
    with pytest.raises(StudioBusyError):
        session.start_turn("again")
    with pytest.raises(StudioBusyError):
        session.reset()
    gate.set()
    await drain(queue)
    assert session.state()["busy"] is False


async def test_an_unread_stream_does_not_cancel_the_turn():
    session = Session(make_atom(), source="app:atom")
    session.start_turn("hello")
    await session.wait()
    assert session.turns[0].reply == "Hi"


async def test_a_failing_model_is_recorded_as_an_error_turn():
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model down")

    session = Session(
        make_atom(connector=PydanticAIConnector(FunctionModel(respond))), source="app:atom"
    )
    events = await drain(session.start_turn("hello"))
    turn = events[-1]["turn"]
    assert turn["error"] == {"type": "RuntimeError", "message": "model down"}
    assert turn["reply"] is None
    assert turn["report"]["final_action"] == "aborted"
    assert session.history == []


async def test_turning_an_orbital_off_rebuilds_without_it():
    guard = ScriptedOrbital(name="guard")
    session = Session(make_atom(input_orbitals=[guard]), source="app:atom")
    fingerprint = session.state()["orbitals"]["input"][0]["fingerprint"]
    session.set_enabled("input", "guard", False)
    await drain(session.start_turn("hello"))
    assert guard.contexts == []
    orbital = session.state()["orbitals"]["input"][0]
    assert orbital["enabled"] is False
    assert orbital["fingerprint"] == fingerprint


def test_a_refused_change_leaves_the_state_as_it_was():
    first = ScriptedOrbital(Verdict.allow(annotations={"x": 1}), name="first")
    reader = ScriptedOrbital(name="reader", reads=["first"])
    session = Session(make_atom(input_orbitals=[first, reader]), source="app:atom")
    with pytest.raises(ChatomonConfigError, match="reads 'first'"):
        session.set_enabled("input", "first", False)
    assert [orbital["enabled"] for orbital in session.state()["orbitals"]["input"]] == [True, True]


def test_an_unknown_orbital_is_reported():
    session = Session(make_atom(), source="app:atom")
    with pytest.raises(UnknownOrbitalError, match="no output orbital named 'judge'"):
        session.set_enabled("output", "judge", False)


def test_the_same_instance_in_both_chains_is_switched_per_side():
    both = ScriptedOrbital(name="both")
    session = Session(make_atom(input_orbitals=[both], output_orbitals=[both]), source="app:atom")
    session.set_enabled("input", "both", False)
    orbitals = session.state()["orbitals"]
    assert orbitals["input"][0]["enabled"] is False
    assert orbitals["output"][0]["enabled"] is True


class HoldsAClient(ScriptedOrbital):
    def config(self) -> dict[str, Any]:
        return {**super().config(), "client": object()}


def test_a_config_with_objects_still_serializes():
    session = Session(make_atom(input_orbitals=[HoldsAClient(name="client")]), source="app:atom")
    state = json.loads(json.dumps(session.state()))
    assert state["orbitals"]["input"][0]["config"]["client"].startswith("<object object")


def test_the_state_describes_the_atom():
    state = Session(make_atom(knowledge="Open 9-18."), source="app:atom").state()
    assert state["studio"]["capture_content"] == "full"
    assert state["atom"]["source"] == "app:atom"
    assert state["atom"]["connector"] == "PydanticAIConnector"
    assert state["atom"]["knowledge"] == "Open 9-18."
    assert state["atom"]["capture_content"] == "none"
    assert state["turns"] == []


async def test_reset_clears_the_conversation():
    session = Session(make_atom(), source="app:atom")
    await drain(session.start_turn("hello"))
    session.reset()
    assert session.history == []
    assert session.turns == []
