import asyncio
import json
import math
import sys
import textwrap
import threading
from typing import Any

import pytest
from langchain_core.tools import tool
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from chatomon import Atom, ChatomonConfigError, Verdict
from chatomon.connectors.pydantic_ai import PydanticAIConnector
from chatomon.orbitals import JudgeOrbital, MessageLengthOrbital
from chatomon.schemas import DEFAULT_BLOCKED_PLACEHOLDER
from chatomon.studio import _loading as loading
from chatomon.studio._chains import RefusedError, UnknownOrbitalError
from chatomon.studio._loading import load, namespace
from chatomon.studio._session import Reloaded, Session, StudioBusyError, UnknownTurnError
from tests.contract.harness import ToolCall
from tests.fakes import FakeConnector, ScriptedOrbital
from tests.studio import frameworks
from tests.studio.frameworks import Reply, StudioHarness


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


def test_a_connector_the_studio_has_no_runner_for_is_refused():
    with pytest.raises(ChatomonConfigError, match="the studio has no runner for FakeConnector"):
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
    assert session.conversation is None


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
    assert session.conversation is None
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


class RemoteOrbital(ScriptedOrbital):
    """A scripted orbital that says it calls a model."""

    calls_model = True


async def test_a_turn_records_how_the_session_differed_from_the_code():
    atom = make_atom(
        input_orbitals=[MessageLengthOrbital(max_words=60), ScriptedOrbital(name="guard")]
    )
    session = Session(atom, source="app:atom")
    await drain(session.start_turn("hello"))
    session.edit("input", "message_length", {"max_words": 80})
    session.switch("input", "guard", False)
    events = await drain(session.start_turn("hello again"))
    session.reset_orbitals()  # what the turn recorded does not follow the session
    first, second = session.turns
    assert first.changes == []
    assert second.changes == ["message_length.max_words changed", "guard turned off"]
    assert events[-1]["turn"]["changes"] == second.changes
    assert [turn["changes"] for turn in session.state()["turns"]] == [[], second.changes]


async def test_a_turn_records_which_of_its_orbitals_call_a_model():
    atom = make_atom(
        input_orbitals=[ScriptedOrbital(name="local"), RemoteOrbital(name="remote")],
        output_orbitals=[RemoteOrbital(name="judge")],
    )
    session = Session(atom, source="app:atom")
    await drain(session.start_turn("hello"))
    session.switch("input", "remote", False)
    await drain(session.start_turn("hello again"))
    assert session.turns[0].models == frozenset({"remote", "judge"})
    assert session.turns[1].models == frozenset({"judge"})


async def test_a_turn_says_whether_it_can_be_saved_as_a_test():
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model down")

    failing = Session(
        make_atom(connector=PydanticAIConnector(FunctionModel(respond))), source="app:atom"
    )
    failed = (await drain(failing.start_turn("hello")))[-1]["turn"]
    assert failed["cannot_save"] == "A turn that failed cannot be saved as a test."

    answered = (await drain(Session(make_atom(), source="app:atom").start_turn("hello")))[-1]
    assert answered["turn"]["cannot_save"] is None


def test_the_session_knows_the_directory_it_was_started_from(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Session(make_atom(), source="app:atom").root == tmp_path.resolve()


LONG = "one two three four"
SAVED = "tests/studio/test_one_two_three_four.py"


def saving_session() -> Session:
    """A session started from a module, in the current directory."""
    atom = make_atom(input_orbitals=[MessageLengthOrbital(max_words=3, reply="Too long.")])
    return Session(atom, source="app:atom", namespace={"atom": atom})


async def test_the_session_shows_a_saved_test_and_writes_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = saving_session()
    await drain(session.start_turn(LONG))
    test = session.saved_test(1)
    assert set(test) == {
        "code",
        "path",
        "root",
        "writable",
        "integration",
        "earlier",
        "notes",
        "warnings",
        "turn",
    }
    assert test["code"].startswith('"""Saved from chatomon studio: app:atom, turn 1."""\n')
    assert (test["path"], test["root"]) == (SAVED, str(tmp_path.resolve()))
    assert (test["writable"], test["integration"], test["earlier"]) == (True, False, 0)
    assert list(tmp_path.iterdir()) == []  # showing a test writes nothing

    assert session.write_test(1, SAVED) == SAVED
    assert (tmp_path / SAVED).read_text(encoding="utf-8") == test["code"]
    assert session.saved_test(1)["path"] == "tests/studio/test_one_two_three_four_2.py"


async def test_the_turn_alone_or_with_the_turns_before_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = saving_session()
    await drain(session.start_turn("hello"))
    await drain(session.start_turn(LONG))
    whole, alone = session.saved_test(2), session.saved_test(2, alone=True)
    assert "hello" in whole["code"] and "hello" not in alone["code"]
    assert (whole["integration"], alone["integration"]) == (True, False)
    session.write_test(2, "a.py")
    session.write_test(2, "b.py", alone=True)
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == whole["code"]
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == alone["code"]


async def test_started_from_python_the_test_is_shown_and_not_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = Session(make_atom(), source="serve()")
    await drain(session.start_turn("hello"))
    test = session.saved_test(1)
    assert (test["writable"], test["path"], test["root"]) == (False, None, None)
    with pytest.raises(RefusedError, match="the studio was started from Python"):
        session.write_test(1, SAVED)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("n", [0, 1, -1, 99])
def test_a_turn_that_does_not_exist_is_reported(n):
    session = Session(make_atom(), source="app:atom")
    with pytest.raises(UnknownTurnError, match=f"there is no turn {n} in this conversation"):
        session.saved_test(n)
    with pytest.raises(UnknownTurnError):
        session.write_test(n, SAVED)


def test_an_orbital_that_never_said_whether_it_calls_a_model_stops_the_studio_at_the_start():
    class Bare(ScriptedOrbital):
        def __init__(self) -> None:  # never calls Orbital.__init__
            self.name, self.reads, self.timeout, self.on_error = "bare", (), 30.0, "block"

    del_calls_model = type("Undeclared", (Bare,), {})
    orb = del_calls_model()
    orb.calls_model = "sometimes"  # type: ignore[assignment]
    with pytest.raises(ChatomonConfigError, match="Undeclared must say whether it calls a model"):
        Session(make_atom(input_orbitals=[orb]), source="app:atom")


def framework_atom(framework: StudioHarness, *replies: Reply, **kwargs: Any) -> Atom:
    return Atom(
        name="speedypack",
        instructions="Be brief.",
        connector=framework.connector(*replies),
        **kwargs,
    )


async def test_every_framework_carries_the_conversation_and_keeps_the_placeholder(
    framework: StudioHarness,
):
    guard = ScriptedOrbital(Verdict.block("No."), Verdict.allow(), name="guard")
    session = Session(framework_atom(framework, "Hi", input_orbitals=[guard]), source="app:atom")
    await drain(session.start_turn("secret"))
    await drain(session.start_turn("hello"))
    assert [turn.reply for turn in session.turns] == ["No.", "Hi"]
    assert [message.content for message in guard.contexts[1].history] == [
        DEFAULT_BLOCKED_PLACEHOLDER,
        "No.",
    ]


async def test_every_framework_keeps_the_conversation_through_an_orbital_change(
    framework: StudioHarness,
):
    guard = ScriptedOrbital(name="guard")
    extra = ScriptedOrbital(name="extra")
    atom = framework_atom(framework, "one", "two", "three", input_orbitals=[guard, extra])
    session = Session(atom, source="app:atom")
    await drain(session.start_turn("hello"))
    session.switch("input", "extra", False)  # the studio builds the agent again
    await drain(session.start_turn("again"))
    assert [(m.role, m.content) for m in guard.contexts[1].history] == [
        ("user", "hello"),
        ("assistant", "one"),
    ]
    session.reset_conversation()
    await drain(session.start_turn("fresh"))
    assert guard.contexts[2].history == ()


async def test_every_framework_leaves_a_failed_turn_out_of_the_conversation(
    framework: StudioHarness,
):
    guard = ScriptedOrbital(name="guard")
    atom = framework_atom(
        framework, "one", RuntimeError("model down"), "three", input_orbitals=[guard]
    )
    session = Session(atom, source="app:atom")
    await drain(session.start_turn("hello"))
    events = await drain(session.start_turn("boom"))
    assert events[-1]["turn"]["error"] == {"type": "RuntimeError", "message": "model down"}
    assert events[-1]["turn"]["report"]["final_action"] == "aborted"
    await drain(session.start_turn("again"))
    assert [(m.role, m.content) for m in guard.contexts[2].history] == [
        ("user", "hello"),
        ("assistant", "one"),
    ]


def test_every_framework_names_its_connector_and_model_in_the_state(framework: StudioHarness):
    atom = framework_atom(framework, "Hi")
    described = Session(atom, source="app:atom").state()["atom"]
    assert described["connector"] == type(atom.connector).__name__
    assert (
        described["model"]
        == {
            "pydantic_ai": "function:respond:",
            "langchain": "ScriptedChatModel",
            "maf": "ScriptedClient",
            "maf_foundry": "remote-agent (version 3)",
        }[framework.name]
    )


def test_the_notes_of_the_atoms_framework_are_kept_for_the_start(framework: StudioHarness):
    session = Session(framework_atom(framework, "Hi"), source="app:atom")
    foundry = "the nucleus of this atom is a Foundry agent"
    assert [note[: len(foundry)] for note in session.notes] == (
        [foundry] if framework.name == "maf_foundry" else []
    )


@pytest.mark.parametrize("name", ["maf", "maf_foundry"])
async def test_a_turn_the_nucleus_leaves_without_a_reply_stays_out_of_the_conversation(
    monkeypatch: pytest.MonkeyPatch, name: str
):
    guard = ScriptedOrbital(name="guard")
    atom = framework_atom(
        frameworks.harness(name, monkeypatch), "one", "", "three", input_orbitals=[guard]
    )
    session = Session(atom, source="app:atom")
    await drain(session.start_turn("hello"))
    events = await drain(session.start_turn("say nothing"))
    turn = events[-1]["turn"]
    # No exception: the framework returned, and the engine closed the turn as aborted.
    assert (turn["error"], turn["report"]["final_action"]) == (None, "aborted")
    assert turn["cannot_save"] == "A turn that failed cannot be saved as a test."
    await drain(session.start_turn("again"))
    assert [(m.role, m.content) for m in guard.contexts[2].history] == [
        ("user", "hello"),
        ("assistant", "one"),
    ]


@tool(return_direct=True)
def lookup() -> str:
    """Look up the opening hours."""
    return "Monday to Friday, 9-18"


async def test_a_langchain_turn_that_ends_on_a_tool_stays_out_of_the_conversation(
    monkeypatch: pytest.MonkeyPatch,
):
    guard = ScriptedOrbital(name="guard")
    connector = frameworks.harness("langchain", monkeypatch).connector(
        ToolCall(), "two", tools=[lookup]
    )
    atom = Atom(
        name="speedypack", instructions="Be brief.", connector=connector, input_orbitals=[guard]
    )
    session = Session(atom, source="app:atom")
    events = await drain(session.start_turn("what are your hours?"))
    # The graph ended on the tool's result: no reply of the nucleus, so the turn is aborted.
    assert events[-1]["turn"]["report"]["final_action"] == "aborted"
    await drain(session.start_turn("again"))
    assert guard.contexts[1].history == ()


# --- Reload from code -----------------------------------------------------------------------

SHOP = """
    from pydantic_ai.models.test import TestModel
    from chatomon import Atom
    from chatomon.connectors.pydantic_ai import PydanticAIConnector
    from chatomon.orbitals import MessageLengthOrbital
    from settings import INSTRUCTIONS, WORDS

    limit = MessageLengthOrbital(max_words=WORDS, reply="Too long.")
    faithfulness = MessageLengthOrbital(name="faithfulness", max_words=500)

    def make():
        return Atom(
            name="shop",
            instructions=INSTRUCTIONS,
            connector=PydanticAIConnector(TestModel(custom_output_text="Hi")),
            input_orbitals=[limit],
            output_orbitals=[faithfulness],
        )

    atom = make()
"""
SETTINGS = 'INSTRUCTIONS = "Be brief."\nWORDS = 60\n'


def started(spec: str = "shop:atom") -> Session:
    """A session as the command starts it: the atom and the namespace of its module."""
    return Session(load(spec), source=spec, namespace=namespace(spec))


@pytest.fixture
def shop(sources):
    """A project with a `shop` module that reads its values from `settings.py`."""
    sources("shop.py", SHOP)
    sources("settings.py", SETTINGS)
    return sources


async def say(session: Session, message: str) -> Any:
    await drain(session.start_turn(message))
    return session.turns[-1]


@pytest.mark.parametrize("spec", ["shop:atom", "shop:make"])
async def test_a_reload_takes_the_atom_the_code_gives_now(shop, spec):
    session = started(spec)
    assert (await say(session, "one two three")).reply == "Hi"
    shop("settings.py", 'INSTRUCTIONS = "Be kind."\nWORDS = 2\n')
    assert await session.reload() == Reloaded(dropped=[], restarted=None, notes=[])
    state = session.state()
    assert state["atom"]["instructions"] == "Be kind."
    assert state["changes"] == [] and state["undo"] is None
    assert session.atom.input_orbitals[0] is sys.modules["shop"].limit
    assert (await say(session, "one two three")).reply == "Too long."


async def test_after_a_reload_names_are_read_from_the_module_that_was_loaded_again(shop):
    session = started()
    assert "input_orbitals=[limit]," in session.code()["code"]
    shop("shop.py", SHOP.replace("limit", "cap"))
    await session.reload()
    assert session.namespace is not None
    assert session.namespace["cap"] is sys.modules["shop"].cap
    assert "input_orbitals=[cap]," in session.code()["code"]
    await say(session, "hello")
    assert "from shop import atom\n" in session.saved_test(1)["code"]


async def test_the_changes_of_the_studio_are_carried_and_the_dropped_ones_are_listed(shop):
    session = started()
    session.edit("input", "message_length", {"max_words": 80})
    session.switch("output", "faithfulness", False)
    session.add("input", "TurnLimitOrbital", {"max_turns": 3})
    shop("shop.py", SHOP.replace("output_orbitals=[faithfulness],", ""))
    shop("settings.py", 'INSTRUCTIONS = "Be kind."\nWORDS = 50\n')
    result = await session.reload()
    assert result.dropped == [
        "'faithfulness' is no longer in your code: what the studio changed on it was dropped."
    ]
    state = session.state()
    assert [change["text"] for change in state["changes"]] == [
        "message_length.max_words changed",
        "turn_limit added",
    ]
    assert state["orbitals"]["output"] == []
    values = {setting["key"]: setting for setting in state["orbitals"]["input"][0]["settings"]}
    assert (values["max_words"]["value"], values["max_words"]["default"]) == (80, None)


async def test_a_reload_that_fails_leaves_the_atom_that_was_running(shop):
    session = started()
    session.edit("input", "message_length", {"max_words": 80})
    await say(session, "hello")
    before = (session.state(), session._agent, session.namespace, sys.modules["shop"])
    shop("settings.py", "INSTRUCTIONS = \n")
    with pytest.raises(ChatomonConfigError, match="could not import 'shop'") as error:
        await session.reload()
    assert isinstance(error.value.__cause__, SyntaxError)
    assert (session.state(), session._agent, session.namespace, sys.modules["shop"]) == before
    assert (await say(session, "again")).reply == "Hi"
    shop("settings.py", SETTINGS.replace("brief", "kind"))
    await session.reload()
    assert session.state()["atom"]["instructions"] == "Be kind."


async def test_an_atom_the_code_gets_wrong_fails_with_its_own_error(shop):
    session = started()
    before = sys.modules["shop"]
    shop("shop.py", SHOP.replace("input_orbitals=[limit]", "input_orbitals=[limit, limit]"))
    with pytest.raises(ChatomonConfigError, match="appears twice") as error:
        await session.reload()
    assert "the studio's changes" not in str(error.value)
    assert sys.modules["shop"] is before
    assert session.atom is before.atom


async def test_changes_the_new_code_cannot_take_refuse_the_reload(shop):
    session = started()
    reader = {
        "name": "tone",
        "model": "test:judge",
        "instructions": "Kind.",
        "reads": ["faithfulness"],
    }
    session.add("output", "JudgeOrbital", reader)
    before = (session.state(), sys.modules["shop"])
    shop("shop.py", SHOP.replace("output_orbitals=[faithfulness],", ""))
    with pytest.raises(ChatomonConfigError) as error:
        await session.reload()
    assert str(error.value).startswith(
        "the reloaded atom does not build with the studio's changes: "
    )
    assert str(error.value).endswith(". Change them or reset to code, then reload again")
    assert "faithfulness" in str(error.value)
    assert (session.state(), sys.modules["shop"]) == before
    session.delete("output", "tone")
    assert (await session.reload()).dropped == []
    assert session.state()["orbitals"]["output"] == []


async def test_a_reload_clears_what_could_be_undone_and_cannot_be_undone(shop):
    session = started()
    session.add("input", "TurnLimitOrbital", {"max_turns": 3})
    session.delete("input", "turn_limit")
    assert session.state()["undo"] == "turn_limit deleted"
    await session.reload()
    assert session.state()["undo"] is None
    with pytest.raises(RefusedError, match="nothing to undo"):
        session.undo()


async def test_a_reload_is_refused_while_a_turn_runs(shop):
    session = started()
    queue = session.start_turn("hello")
    with pytest.raises(StudioBusyError, match="a turn is running"):
        await session.reload()
    await drain(queue)
    await session.reload()


async def test_nothing_else_happens_while_a_reload_runs(shop, monkeypatch):
    session = started()
    release = threading.Event()
    real = loading.load

    def slow(spec: str) -> Atom:
        release.wait(timeout=5)
        return real(spec)

    monkeypatch.setattr(loading, "load", slow)
    reload = asyncio.create_task(session.reload())
    await asyncio.sleep(0.05)
    for attempt in (
        lambda: session.start_turn("hello"),
        lambda: session.switch("input", "message_length", False),
        session.reset_conversation,
    ):
        with pytest.raises(StudioBusyError, match="the atom is being reloaded"):
            attempt()
    with pytest.raises(StudioBusyError, match="the atom is being reloaded"):
        await session.reload()
    release.set()
    assert (await reload).dropped == []
    session.switch("input", "message_length", False)


async def test_a_reload_goes_on_when_the_page_that_asked_for_it_goes_away(shop, monkeypatch):
    session = started()
    release = threading.Event()
    real = loading.load

    def slow(spec: str) -> Atom:
        release.wait(timeout=5)
        return real(spec)

    monkeypatch.setattr(loading, "load", slow)
    shop("settings.py", SETTINGS.replace("brief", "kind"))
    request = asyncio.create_task(session.reload())
    await asyncio.sleep(0.05)
    request.cancel()
    release.set()
    for _ in range(100):
        if session.state()["atom"]["instructions"] == "Be kind.":
            break
        await asyncio.sleep(0.02)
    assert session.state()["atom"]["instructions"] == "Be kind."
    assert session.atom is sys.modules["shop"].atom
    session.switch("input", "message_length", False)  # the session is free again


async def test_a_studio_started_from_python_cannot_reload():
    session = Session(make_atom(), source="serve()")
    with pytest.raises(RefusedError) as error:
        await session.reload()
    assert str(error.value) == (
        "the studio was started from Python and does not know where the atom lives: "
        "restart it to load your changes"
    )


async def test_an_atom_whose_module_is_outside_the_directory_says_it_was_not_reloaded(
    sources, tmp_path_factory
):
    """Started from one directory with the atom's module on the path from another."""
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    module = textwrap.dedent(SHOP).replace(
        "from settings import INSTRUCTIONS, WORDS", 'INSTRUCTIONS, WORDS = "Be brief.", 60'
    )
    (elsewhere / "outside_shop.py").write_text(module, encoding="utf-8")
    sys.path.append(str(elsewhere))
    try:
        session = started("outside_shop:atom")
        before = (session.state(), sys.modules["outside_shop"])
        (elsewhere / "outside_shop.py").write_text(
            module.replace("Be brief.", "Be very kind."), encoding="utf-8"
        )
        with pytest.raises(ChatomonConfigError) as error:
            await session.reload()
        assert str(error.value) == (
            "'outside_shop' is outside the directory the studio was started from, so it is "
            "not loaded again: restart the studio to load its changes"
        )
        assert (session.state(), sys.modules["outside_shop"]) == before
        assert session.reloads == 0
    finally:
        sys.modules.pop("outside_shop", None)


async def test_a_module_that_runs_a_loop_of_its_own_when_imported_can_be_reloaded(shop):
    """As a module does to fetch its tools: at start no loop is running, in the server one is."""
    loop = "import asyncio\n\nasync def fetch():\n    return 60\n\n"
    shop("settings.py", loop + 'INSTRUCTIONS = "Be brief."\nWORDS = asyncio.run(fetch())\n')
    session = await asyncio.to_thread(started)
    shop("settings.py", loop + 'INSTRUCTIONS = "Be kind."\nWORDS = asyncio.run(fetch())\n')
    await session.reload()
    assert session.state()["atom"]["instructions"] == "Be kind."


def test_a_module_that_asks_for_the_event_loop_when_imported_can_be_reloaded(shop):
    """Older code: `asyncio.get_event_loop().run_until_complete(...)` at the top of a module."""
    code = "import asyncio\n\nasync def fetch():\n    return 60\n\n"
    words = "WORDS = asyncio.get_event_loop().run_until_complete(fetch())\n"
    shop("settings.py", code + 'INSTRUCTIONS = "Be brief."\n' + words)
    main = asyncio.new_event_loop()
    asyncio.set_event_loop(main)  # what the main thread has when the command starts
    try:
        session = started()
        shop("settings.py", code + 'INSTRUCTIONS = "Be kind."\n' + words)
        asyncio.run(session.reload())
    finally:
        asyncio.set_event_loop(None)
        main.close()
    assert session.state()["atom"]["instructions"] == "Be kind."


async def test_the_thread_of_a_reload_does_not_hold_the_studios_exit(shop, monkeypatch):
    """Code that hangs while it is loaded must not keep Ctrl+C from stopping the studio."""
    session = started()
    threads: list[threading.Thread] = []
    real = loading.load

    def seen(spec: str) -> Atom:
        threads.append(threading.current_thread())
        return real(spec)

    monkeypatch.setattr(loading, "load", seen)
    await session.reload()
    (thread,) = threads
    assert thread is not threading.main_thread()
    assert thread.daemon


async def test_code_that_exits_while_it_is_loaded_does_not_stop_the_studio(shop):
    """A guard a developer adds: `if "KEY" not in os.environ: sys.exit("set KEY")`."""
    session = started()
    before = (session.state(), sys.modules["shop"])
    shop("settings.py", 'import sys\n\nsys.exit("set SHOP_KEY first")\n')
    with pytest.raises(ChatomonConfigError) as error:
        await session.reload()
    assert str(error.value) == (
        "the code called sys.exit('set SHOP_KEY first') while it was loaded"
    )
    assert isinstance(error.value.__cause__, SystemExit)
    assert (session.state(), sys.modules["shop"]) == before
    assert (await say(session, "hello")).reply == "Hi"


async def test_the_notes_the_reloaded_atom_adds_are_told_once(shop):
    session = started()
    assert session.notes == []
    shop(
        "shop.py",
        SHOP.replace('custom_output_text="Hi")', 'custom_output_text="Hi"), deps_type=int'),
    )
    first = await session.reload()
    assert len(first.notes) == 1 and "without deps" in first.notes[0]
    assert session.notes == first.notes
    second = await session.reload()
    assert second.notes == [] and session.notes == first.notes


async def test_a_reload_leaves_the_developers_new_atom_untouched(shop):
    session = started()
    await session.reload()
    atom = sys.modules["shop"].atom
    assert session.atom is atom
    assert (atom.on_turn, atom.tracer_provider, atom.content_policy.mode) == (None, None, "none")


class Listening(ScriptedOrbital):
    """An input orbital that keeps the conversation it was given at every call."""

    def heard(self) -> list[tuple[str, str]]:
        return [(message.role, message.content) for message in self.contexts[-1].history]


def loads(monkeypatch: pytest.MonkeyPatch, *atoms: Atom) -> None:
    """Make the next reloads give these atoms, as code that changed would."""
    queue = list(atoms)
    monkeypatch.setattr(loading, "load", lambda spec: queue.pop(0))


async def test_the_conversation_goes_on_after_a_reload(framework: StudioHarness, monkeypatch):
    first = Listening(Verdict.allow(), Verdict.block("No."), name="guard")
    before = make_atom(connector=framework.connector("one"), input_orbitals=[first])
    session = Session(before, source="shop:atom", namespace={})
    await say(session, "hello")
    await say(session, "secret")
    guard = Listening(name="guard")
    loads(monkeypatch, make_atom(connector=framework.connector("three"), input_orbitals=[guard]))
    assert (await session.reload()).restarted is None
    assert [turn.n for turn in session.turns] == [1, 2]
    assert (await say(session, "and now?")).reply == "three"
    assert guard.heard() == [
        ("user", "hello"),
        ("assistant", "one"),
        ("user", DEFAULT_BLOCKED_PLACEHOLDER),
        ("assistant", "No."),
    ]
    assert [turn.reloads for turn in session.turns] == [0, 0, 1]


class Keyless(PydanticAIConnector):
    """A connector whose framework refuses to build, as with a model whose key is not set."""

    def build(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the API key is not set")


async def test_an_error_of_the_framework_while_building_is_a_failed_reload(monkeypatch):
    session = Session(make_atom(), source="shop:atom", namespace={})
    before = (session.state(), session._agent)
    loads(monkeypatch, make_atom(connector=Keyless(TestModel())))
    with pytest.raises(ChatomonConfigError) as error:
        await session.reload()
    assert str(error.value) == (
        "the reloaded atom could not be built: RuntimeError: the API key is not set"
    )
    assert isinstance(error.value.__cause__, RuntimeError)
    assert (session.state(), session._agent) == before
    assert (await say(session, "hello")).reply == "Hi"


async def test_code_on_another_framework_starts_a_new_conversation(monkeypatch):
    here = frameworks.harness("pydantic_ai", monkeypatch)
    there = frameworks.harness("langchain", monkeypatch)
    session = Session(make_atom(connector=here.connector("one")), source="shop:atom", namespace={})
    session.add("input", "TurnLimitOrbital", {"max_turns": 5})
    await say(session, "hello")
    guard = Listening(name="guard")
    loads(monkeypatch, make_atom(connector=there.connector("two"), input_orbitals=[guard]))
    result = await session.reload()
    assert result.restarted == (
        "The atom is built with LangChainConnector now, another framework: "
        "a new conversation was started."
    )
    assert session.turns == [] and session.conversation is None
    assert [change["text"] for change in session.state()["changes"]] == ["turn_limit added"]
    assert session.state()["atom"]["connector"] == "LangChainConnector"
    turn = await say(session, "and now?")
    assert (turn.n, turn.reply, guard.heard()) == (1, "two", [])


async def test_the_two_connectors_of_one_framework_share_the_conversation(monkeypatch):
    here = frameworks.harness("maf", monkeypatch)
    there = frameworks.harness("maf_foundry", monkeypatch)
    session = Session(make_atom(connector=here.connector("one")), source="shop:atom", namespace={})
    await say(session, "hello")
    guard = Listening(name="guard")
    loads(monkeypatch, make_atom(connector=there.connector("two"), input_orbitals=[guard]))
    assert (await session.reload()).restarted is None
    assert (await say(session, "and now?")).reply == "two"
    assert guard.heard() == [("user", "hello"), ("assistant", "one")]
