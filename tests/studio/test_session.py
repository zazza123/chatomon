import asyncio
import json
import math
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from chatomon import Atom, ChatomonConfigError, Verdict
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals import JudgeOrbital, MessageLengthOrbital
from chatomon.schemas import DEFAULT_BLOCKED_PLACEHOLDER
from chatomon.studio._chains import RefusedError, UnknownOrbitalError
from chatomon.studio._session import Session, StudioBusyError
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
        session.reset_conversation()
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
    session.switch("input", "guard", False)
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
        session.switch("input", "first", False)
    assert [orbital["enabled"] for orbital in session.state()["orbitals"]["input"]] == [True, True]


def test_an_unknown_orbital_is_reported():
    session = Session(make_atom(), source="app:atom")
    with pytest.raises(UnknownOrbitalError, match="no output orbital named 'judge'"):
        session.switch("output", "judge", False)


def test_the_same_instance_in_both_chains_is_switched_per_side():
    both = ScriptedOrbital(name="both")
    session = Session(make_atom(input_orbitals=[both], output_orbitals=[both]), source="app:atom")
    session.switch("input", "both", False)
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
    session.reset_conversation()
    assert session.history == []
    assert session.turns == []


@pytest.mark.parametrize(
    "env",
    [
        {"OTEL_TRACES_SAMPLER": "always_off"},
        {"OTEL_TRACES_SAMPLER": "traceidratio", "OTEL_TRACES_SAMPLER_ARG": "0"},
        {"OTEL_SDK_DISABLED": "true"},
    ],
    ids=["sampler off", "ratio zero", "sdk disabled"],
)
async def test_live_events_do_not_depend_on_opentelemetry_settings(monkeypatch, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    session = Session(make_atom(input_orbitals=[ScriptedOrbital(name="guard")]), source="app:atom")
    events = await drain(session.start_turn("hello"))
    assert "orbital_finished" in [event["type"] for event in events]


async def test_live_events_survive_an_unsampled_span_of_the_developer():
    from opentelemetry import trace
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

    session = Session(make_atom(input_orbitals=[ScriptedOrbital(name="guard")]), source="app:atom")
    unsampled = SpanContext(trace_id=1, span_id=1, is_remote=False, trace_flags=TraceFlags(0))
    with trace.use_span(NonRecordingSpan(unsampled)):
        events = await drain(session.start_turn("hello"))
    assert "orbital_finished" in [event["type"] for event in events]


def test_the_opentelemetry_environment_is_left_as_it_was(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    Session(make_atom(), source="app:atom")
    import os

    assert os.environ["OTEL_SDK_DISABLED"] == "true"


def test_non_finite_numbers_in_a_config_still_serialize():
    slow = ScriptedOrbital(name="slow", timeout=math.inf)
    state = Session(make_atom(input_orbitals=[slow]), source="app:atom").state()
    json.dumps(state, allow_nan=False)
    assert state["orbitals"]["input"][0]["config"]["timeout"] == "Infinity"


class FailsToBind(ScriptedOrbital):
    broken = False

    def bind(self, connector: Any) -> None:
        if self.broken:
            raise RuntimeError("cannot bind")


def test_any_failure_while_rebuilding_leaves_the_switches_as_they_were():
    fragile = FailsToBind(name="fragile")
    session = Session(
        make_atom(input_orbitals=[fragile, ScriptedOrbital(name="other")]), source="app:atom"
    )
    fragile.broken = True
    with pytest.raises(RuntimeError, match="cannot bind"):
        session.switch("input", "other", False)
    assert [orbital["enabled"] for orbital in session.state()["orbitals"]["input"]] == [True, True]


def gated_session(**kwargs: Any) -> tuple[Session, asyncio.Event]:
    gate = asyncio.Event()

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await gate.wait()
        return ModelResponse(parts=[TextPart(content="Hi")])

    atom = make_atom(connector=PydanticAIConnector(FunctionModel(respond)), **kwargs)
    return Session(atom, source="app:atom"), gate


async def test_a_cancelled_turn_still_ends_its_stream():
    session, _ = gated_session()
    queue = session.start_turn("hello")
    await asyncio.sleep(0.05)
    assert session._task is not None
    session._task.cancel()
    events = await asyncio.wait_for(drain(queue), timeout=2)
    kinds = [event["type"] for event in events]
    assert kinds[0] == "turn_started"
    assert "turn_finished" not in kinds
    assert session.busy is False


async def test_the_state_names_the_running_turn_and_refuses_switches():
    session, gate = gated_session(input_orbitals=[ScriptedOrbital(name="guard")])
    assert session.state()["running"] is None
    queue = session.start_turn("hello")
    await asyncio.sleep(0.05)
    assert session.state()["running"] == {"n": 1, "message": "hello"}
    for change in (
        lambda: session.switch("input", "guard", False),
        lambda: session.edit("input", "guard", {"timeout": 5}),
        lambda: session.move("input", "guard", 0),
        lambda: session.add("input", "TurnLimitOrbital", {"max_turns": 3}),
        lambda: session.delete("input", "guard"),
        lambda: session.reset_orbitals(),
        session.undo,
    ):
        with pytest.raises(StudioBusyError):
            change()
    gate.set()
    await drain(queue)
    assert session.state()["running"] is None


async def test_a_run_that_fails_before_the_engine_opens_a_turn_has_no_report():
    class Broken:
        async def run(self, *args: Any, **kwargs: Any) -> Any:
            raise ValueError("bad history")

    session = Session(make_atom(), source="app:atom")
    session._agent = Broken()
    events = await drain(session.start_turn("hello"))
    turn = events[-1]["turn"]
    assert turn["report"] is None
    assert turn["error"] == {"type": "ValueError", "message": "bad history"}


async def test_an_edit_takes_effect_on_the_next_turn():
    atom = make_atom(input_orbitals=[MessageLengthOrbital(max_words=3)])
    session = Session(atom, source="app:atom")
    blocked = await drain(session.start_turn("one two three four"))
    assert blocked[-1]["turn"]["report"]["final_action"] == "blocked"
    session.edit("input", "message_length", {"max_words": 10})
    answered = await drain(session.start_turn("one two three four"))
    assert answered[-1]["turn"]["report"]["final_action"] == "answered"
    assert atom.input_orbitals[0].max_words == 3  # type: ignore[attr-defined]


def test_a_change_the_chain_refuses_leaves_everything_as_it_was():
    first = ScriptedOrbital(Verdict.allow(annotations={"x": 1}), name="first")
    reader = ScriptedOrbital(name="reader", reads=["first"])
    session = Session(make_atom(input_orbitals=[first, reader]), source="app:atom")
    agent, chains = session._agent, session.chains
    with pytest.raises(ChatomonConfigError, match="reads 'first'"):
        session.move("input", "reader", 0)
    assert session._agent is agent
    assert session.chains is chains


def test_undo_brings_back_what_a_deletion_or_a_reset_removed():
    session = Session(
        make_atom(input_orbitals=[MessageLengthOrbital(max_words=3)]), source="app:atom"
    )

    def input_names() -> list[str]:
        return [orbital["name"] for orbital in session.state()["orbitals"]["input"]]

    assert session.state()["undo"] is None
    session.add("input", "TurnLimitOrbital", {"max_turns": 3})
    session.delete("input", "turn_limit")
    assert session.state()["undo"] == "turn_limit deleted"
    session.undo()
    assert input_names() == ["message_length", "turn_limit"]
    assert session.state()["undo"] is None
    with pytest.raises(RefusedError, match="nothing to undo"):
        session.undo()
    session.edit("input", "message_length", {"max_words": 9})
    session.reset_orbitals("input", "message_length")
    assert session.state()["undo"] == "message_length reset to code"
    session.reset_orbitals()
    assert session.state()["undo"] == "Reset to code"
    assert input_names() == ["message_length"]
    session.undo()
    assert input_names() == ["message_length", "turn_limit"]


def test_any_other_change_drops_the_undo():
    session = Session(
        make_atom(input_orbitals=[MessageLengthOrbital(max_words=3)]), source="app:atom"
    )
    session.add("input", "TurnLimitOrbital", {"max_turns": 3})
    session.delete("input", "turn_limit")
    session.switch("input", "message_length", False)
    assert session.state()["undo"] is None
    with pytest.raises(RefusedError, match="nothing to undo"):
        session.undo()


def test_the_state_describes_what_can_be_edited():
    limit = MessageLengthOrbital(max_words=60)
    session = Session(
        make_atom(input_orbitals=[limit, ScriptedOrbital(name="scope")]), source="app:atom"
    )
    session.edit("input", "message_length", {"max_words": 80})
    state = json.loads(json.dumps(session.state()))
    edited, scope = state["orbitals"]["input"]
    assert {key: edited[key] for key in ("origin", "modified", "editable", "note", "fixed")} == {
        "origin": "code",
        "modified": True,
        "editable": True,
        "note": None,
        "fixed": [],
    }
    assert edited["settings"][0] == {
        "key": "max_words",
        "type": "int",
        "optional": True,
        "required": False,
        "default": None,
        "value": 80,
        "changed": True,
    }
    assert [setting["key"] for setting in edited["settings"]] == [
        "max_words",
        "max_chars",
        "reply",
        "guidance",
        "timeout",
        "on_error",
    ]
    assert edited["settings"][-1]["options"] == ["block", "allow"]
    assert edited["settings"][2]["changed"] is False
    assert edited["fingerprint"] != limit.fingerprint()
    assert (scope["editable"], scope["settings"], scope["fixed"]) == (False, [], [])
    assert scope["note"] == "ScriptedOrbital is your own class: change its settings in your code."
    assert scope["config"]["name"] == "scope"
    assert state["changes"] == [
        {
            "type": "setting",
            "side": "input",
            "name": "message_length",
            "setting": "max_words",
            "text": "message_length.max_words changed",
        }
    ]


def test_the_state_lists_what_can_be_added():
    session = Session(
        make_atom(input_orbitals=[MessageLengthOrbital(max_words=3)]), source="app:atom"
    )
    kinds = session.state()["kinds"]
    assert [kind["kind"] for kind in kinds] == [
        "JudgeOrbital",
        "MessageLengthOrbital",
        "TurnLimitOrbital",
    ]
    judge, length, turns = kinds
    assert judge["sides"] == ["input", "output"]
    assert turns["sides"] == ["input"]
    assert judge["summary"] == "A small LLM agent that judges messages following your instructions."
    assert (judge["name"], length["name"]) == ("judge", "message_length_2")
    assert judge["settings"][0] == {
        "key": "model",
        "type": "text",
        "optional": False,
        "required": True,
        "default": None,
    }


def test_the_state_names_the_objects_an_orbital_takes_from_code():
    own = PydanticAIConnector(TestModel())
    judge = JudgeOrbital(model="test:judge", instructions="Check facts.", connector=own)
    state = Session(make_atom(output_orbitals=[judge]), source="app:atom").state()
    assert state["orbitals"]["output"][0]["fixed"] == [
        {"key": "connector", "type": "PydanticAIConnector", "name": None}
    ]


async def test_an_instance_in_both_chains_is_edited_as_one():
    both = MessageLengthOrbital(max_words=60)
    session = Session(make_atom(input_orbitals=[both], output_orbitals=[both]), source="app:atom")
    session.edit("input", "message_length", {"max_words": 80})
    orbitals = session.state()["orbitals"]
    assert orbitals["input"][0]["fingerprint"] == orbitals["output"][0]["fingerprint"]
    assert orbitals["output"][0]["settings"][0]["value"] == 80
    events = await drain(session.start_turn("hello"))
    assert events[-1]["turn"]["report"]["final_action"] == "answered"


def test_the_session_writes_its_code_and_names_objects_from_the_namespace():
    own = PydanticAIConnector(TestModel())
    judge = JudgeOrbital(model="test:judge", instructions="Check facts.", connector=own)
    atom = make_atom(input_orbitals=[MessageLengthOrbital(max_words=60)], output_orbitals=[judge])
    session = Session(atom, source="app:atom", namespace={"spare": own, "facts": judge})
    session.edit("input", "message_length", {"max_words": 80})
    assert session.state()["orbitals"]["output"][0]["fixed"] == [
        {"key": "connector", "type": "PydanticAIConnector", "name": "spare"}
    ]
    code = session.code()
    assert sorted(code) == ["changes", "code", "notes", "placeholders"]
    assert code["changes"] == ["message_length.max_words changed"]
    assert "MessageLengthOrbital(max_words=80)" in code["code"]
    assert "output_orbitals=[facts]," in code["code"]
    assert (code["placeholders"], code["notes"]) == ([], [])
