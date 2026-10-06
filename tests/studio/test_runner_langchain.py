"""What the LangChain runner adds to the contract every runner keeps."""

from dataclasses import dataclass
from typing import Any

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from chatomon import Atom
from chatomon.connectors.langchain import LangChainConnector
from chatomon.studio._runners import InterruptedTurnError
from chatomon.studio._runners.langchain import describe, notes, run_turn, saved_test
from tests.contract.harness import ToolCall
from tests.contract.harness_langchain import ScriptedChatModel
from tests.fakes import ScriptedOrbital

CONTEXT = (
    "this atom declares context_schema; the studio runs turns without a context, "
    "so tools and middleware that read it fail"
)
CHECKPOINTER = (
    "this atom has a checkpointer; the studio keeps the conversation itself and "
    "writes every turn to the checkpointer as a thread of its own"
)
THREAD = '{"configurable": {"thread_id": uuid4().hex}}'
ASYNC = (
    "The studio runs a LangChain agent asynchronously, and so does the test: it needs "
    "`pytest-asyncio` 0.24 or later."
)
MIDDLEWARE = (
    "This atom has middleware of its own: one that chooses the model can still reach a "
    "real model in this test."
)


@dataclass
class Shopper:
    name: str


class Named(ScriptedChatModel):
    model_name: str = ""
    model: str = ""


def atom(connector: LangChainConnector, **kwargs: Any) -> Atom:
    return Atom(name="t", instructions="Be brief.", connector=connector, **kwargs)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai:gpt-5", "openai:gpt-5"),
        (Named(model_name="gpt-5"), "gpt-5"),
        (Named(model="claude-sonnet"), "claude-sonnet"),
        (ScriptedChatModel(), "ScriptedChatModel"),
        # A model bound to arguments or to a configuration is still that model.
        (Named(model_name="gpt-5").bind(stop=["."]), "gpt-5"),
        (ScriptedChatModel().with_config(tags=["shop"]), "ScriptedChatModel"),
    ],
)
def test_describe_names_the_model(model: Any, expected: str):
    assert describe(LangChainConnector(model), None) == {
        "connector": "LangChainConnector",
        "model": expected,
    }


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, []),
        ({"context_schema": Shopper}, [CONTEXT]),
        ({"checkpointer": InMemorySaver()}, [CHECKPOINTER]),
        ({"context_schema": Shopper, "checkpointer": InMemorySaver()}, [CONTEXT, CHECKPOINTER]),
        ({"context_schema": None, "checkpointer": False}, []),
    ],
)
def test_the_notes_follow_what_the_connector_gives_create_agent(kwargs: Any, expected: list[str]):
    assert notes(LangChainConnector(ScriptedChatModel(), **kwargs)) == expected


async def test_a_reply_written_in_blocks_is_shown_as_its_text():
    blocks: list[str | dict[Any, Any]] = [
        {"type": "text", "text": "We open "},
        {"type": "text", "text": "at 9."},
    ]
    model = ScriptedChatModel(script=[AIMessage(content=blocks)])
    agent = atom(LangChainConnector(model)).build()
    assert (await run_turn(agent, "hello", None)).reply == "We open at 9."


async def test_with_a_checkpointer_every_turn_has_a_thread_and_a_failed_one_leaves_nothing():
    saver = InMemorySaver()
    guard = ScriptedOrbital(name="guard")
    model = ScriptedChatModel(script=["one", RuntimeError("model down"), "three"])
    agent = atom(LangChainConnector(model, checkpointer=saver), input_orbitals=[guard]).build()
    first = await run_turn(agent, "hello", None)
    with pytest.raises(RuntimeError, match="model down"):
        await run_turn(agent, "boom", first.conversation)
    third = await run_turn(agent, "again", first.conversation)
    assert third.reply == "three"
    # The nucleus is not shown the message of the failed turn, which one thread would keep.
    assert [(m.role, m.content) for m in guard.contexts[2].history] == [
        ("user", "hello"),
        ("assistant", "one"),
    ]
    threads = {item.config.get("configurable", {}).get("thread_id") for item in saver.list(None)}
    assert len(threads) == 3


@tool
def lookup() -> str:
    """Look up the opening hours."""
    return str(interrupt("May I?"))


@pytest.mark.parametrize("kwargs", [{}, {"checkpointer": InMemorySaver()}])
async def test_an_interrupt_is_a_failed_turn_and_the_conversation_goes_on(kwargs: Any):
    model = ScriptedChatModel(script=["one", ToolCall(), "three"])
    agent = atom(LangChainConnector(model, tools=[lookup], **kwargs)).build()
    first = await run_turn(agent, "hello", None)
    with pytest.raises(InterruptedTurnError, match="stopped to wait for an answer"):
        await run_turn(agent, "what are your hours?", first.conversation)
    assert (await run_turn(agent, "again", first.conversation)).reply == "three"


def test_a_saved_test_awaits_its_turns_and_carries_the_messages():
    plain = saved_test(LangChainConnector(ScriptedChatModel()))
    # The studio runs the agent with ainvoke: so does the test, or tools and middleware
    # that are asynchronous only would work in the studio and fail in the test.
    assert plain.run == "await agent.ainvoke"
    assert plain.awaited
    assert plain.marker == '@pytest.mark.asyncio(loop_scope="session")'
    assert plain.notes == (ASYNC,)
    assert (plain.first.before, plain.first.after) == ('{"messages": [HumanMessage(', ")]}")
    assert plain.later.before == '{"messages": [*result["messages"], HumanMessage('
    assert plain.first.arguments == plain.later.arguments == ()
    assert plain.imports == ("from langchain_core.messages import HumanMessage",)
    assert plain.reply == 'result["messages"][-1].content'
    assert plain.offline.after_build == ()

    kept = saved_test(LangChainConnector(ScriptedChatModel(), checkpointer=InMemorySaver()))
    assert kept.first.arguments == kept.later.arguments == (THREAD,)
    assert kept.imports[-1] == "from uuid import uuid4"


def test_the_stand_in_of_an_offline_test_takes_the_atoms_tools_and_calls_none():
    offline = saved_test(LangChainConnector(ScriptedChatModel())).offline
    assert offline.definitions[0] == "class OfflineNucleus(FakeListChatModel):"
    assert offline.before_build == (
        'monkeypatch.setattr(atom.connector, "model", OfflineNucleus(responses=["offline"]))',
    )
    assert offline.notes == ()
    scope: dict[str, Any] = {}
    exec("\n".join([*offline.imports, "", "", *offline.definitions]), scope)
    stand_in = scope["OfflineNucleus"](responses=["offline"])
    # LangChain's own fake refuses tools with an error, which a fallback middleware
    # takes as a reason to call another model.
    assert stand_in.bind_tools([lookup]) is stand_in
    assert stand_in.invoke("hello").tool_calls == []


class Quiet(AgentMiddleware):
    pass


@pytest.mark.parametrize(
    ("middleware", "expected"), [(None, ()), ([], ()), ([Quiet()], (MIDDLEWARE,))]
)
def test_an_atom_with_middleware_is_told_what_an_offline_test_cannot_promise(
    middleware: Any, expected: tuple[str, ...]
):
    kwargs = {} if middleware is None else {"middleware": middleware}
    connector = LangChainConnector(ScriptedChatModel(), **kwargs)
    assert saved_test(connector).offline.notes == expected
