"""Studio turns on atoms built with `LangChainConnector`."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from chatomon.connectors.base import Connector
from chatomon.studio._runners import History, InterruptedTurnError, RunOutcome
from chatomon.studio._testgen import Call, Framework, StandIn

_INTERRUPTED = (
    "the agent stopped to wait for an answer, which the studio cannot give: "
    "the turn is left unfinished"
)
_CONTEXT = (
    "this atom declares context_schema; the studio runs turns without a context, "
    "so tools and middleware that read it fail"
)
_CHECKPOINTER = (
    "this atom has a checkpointer; the studio keeps the conversation itself and "
    "writes every turn to the checkpointer as a thread of its own"
)
_THREAD = '{"configurable": {"thread_id": uuid4().hex}}'
_ASYNC = (
    "The studio runs a LangChain agent asynchronously, and so does the test: it needs "
    "`pytest-asyncio` 0.24 or later."
)
_MIDDLEWARE = (
    "This atom has middleware of its own: one that chooses the model can still reach a "
    "real model in this test."
)
# A middleware reaches the model through the connector as the agent does, so in a test that
# repeats the replies it gets the stand-in, which has only those replies to give.
_CALLS_THE_MODEL = (
    "A middleware that calls the model itself, as one that selects tools or summarises the "
    "conversation does, gets the stand-in in place of the model: with one, this test fails "
    "on the unchanged atom."
)
# LangChain's fake chat model refuses tools with an error, which a fallback middleware
# takes as a reason to call another model: the stand-in takes them and calls none.
_OFFLINE_NUCLEUS = (
    "class OfflineNucleus(FakeListChatModel):",
    '    """Stands in for the nucleus: it calls no model and no tool."""',
    "",
    "    def bind_tools(self, tools, **kwargs):",
    "        return self",
)
# LangChain's fake chat model over a list starts again from its first reply when they are
# over; this one reads an iterator and ends with it.
_SCRIPTED_NUCLEUS = (
    "class ScriptedNucleus(GenericFakeChatModel):",
    '    """Stands in for the nucleus: it repeats its replies, with no model and no tool."""',
    "",
    "    def bind_tools(self, tools, **kwargs):",
    "        return self",
)


def _set(connector: Connector[Any], key: str) -> bool:
    """Tell whether the connector passes `key` to `create_agent` with a value.

    Returns:
        False when the argument is absent, `None` or `False`.
    """
    return getattr(connector, "agent_kwargs", {}).get(key) not in (None, False)


async def run_turn(agent: Any, message: str, conversation: Any) -> RunOutcome:
    """Run one turn with `ainvoke`, on a thread of its own.

    The studio keeps the conversation: every message goes in again, and a
    checkpointer never carries it from one turn to the next.

    Args:
        agent: The compiled LangGraph agent built from the studio's copy of the atom.
        message: The user message.
        conversation: The LangChain messages so far, or `None` in the first turn.

    Returns:
        The reply and every message of the conversation, blocked ones already redacted.

    Raises:
        InterruptedTurnError: If the graph stopped at an interrupt.
    """
    messages = [*(conversation or ()), HumanMessage(content=message)]
    config = {"configurable": {"thread_id": uuid4().hex}}
    state = await agent.ainvoke({"messages": messages}, config)
    if state.get("__interrupt__"):
        raise InterruptedTurnError(_INTERRUPTED)
    return RunOutcome(reply=state["messages"][-1].text, conversation=state["messages"])


async def run_from_history(agent: Any, history: History, message: str) -> str:
    """Run one turn on LangChain messages made from the texts, on a thread of its own.

    Args:
        agent: The compiled LangGraph agent built from the atom.
        history: What was said before the message, oldest first.
        message: The user message.

    Returns:
        The reply.

    Raises:
        InterruptedTurnError: If the graph stopped at an interrupt.
    """  # noqa: DOC502 - raised by run_turn()
    past: list[BaseMessage] = [
        HumanMessage(content=text) if role == "user" else AIMessage(content=text)
        for role, text in history
    ]
    return (await run_turn(agent, message, past)).reply


def describe(connector: Connector[Any], agent: Any) -> dict[str, str]:
    """Name the connector and its model for the atom card.

    Args:
        connector: A `LangChainConnector`.
        agent: The native agent; not read.

    Returns:
        `{"connector": ..., "model": ...}`.
    """
    model = getattr(connector, "model", None)
    # A model bound to arguments or to a configuration is named after the model it wraps.
    while not isinstance(model, str) and getattr(model, "bound", None) is not None:
        model = getattr(model, "bound", None)
    if isinstance(model, str):
        name = model
    else:
        named = (getattr(model, key, None) for key in ("model_name", "model"))
        name = next((n for n in named if isinstance(n, str) and n), type(model).__name__)
    return {"connector": type(connector).__name__, "model": name}


def notes(connector: Connector[Any]) -> list[str]:
    """Say what the developer should know before trying this atom in the studio.

    Args:
        connector: A `LangChainConnector`.

    Returns:
        One sentence for a context the turns run without and one for a checkpointer.
    """
    found: list[str] = []
    if _set(connector, "context_schema"):
        found.append(_CONTEXT)
    if _set(connector, "checkpointer"):
        found.append(_CHECKPOINTER)
    return found


def saved_test(connector: Connector[Any]) -> Framework:
    """Return what a saved test needs from LangChain.

    The test awaits `ainvoke`, as the studio does: tools and middleware that are
    asynchronous only would work in the studio and fail under `invoke`.

    Args:
        connector: A `LangChainConnector`; with a checkpointer every call takes a thread.

    Returns:
        The awaited call that runs a turn around the conversation's messages, the
        reply, and the chat models the test defines to stand in for the nucleus:
        one offline, one that repeats its replies.
    """
    thread = (_THREAD,) if _set(connector, "checkpointer") else ()
    middleware = getattr(connector, "agent_kwargs", {}).get("middleware")
    return Framework(
        imports=(
            "from langchain_core.messages import HumanMessage",
            *(("from uuid import uuid4",) if thread else ()),
        ),
        run="await agent.ainvoke",
        first=Call('{"messages": [HumanMessage(', ")]}", thread),
        later=Call('{"messages": [*result["messages"], HumanMessage(', ")]}", thread),
        reply='result["messages"][-1].content',
        marker='@pytest.mark.asyncio(loop_scope="session")',
        notes=(_ASYNC,),
        scripted=StandIn(
            imports=(
                "from langchain_core.language_models.fake_chat_models import GenericFakeChatModel",
            ),
            definitions=_SCRIPTED_NUCLEUS,
            before_build=(
                "nucleus = ScriptedNucleus(messages=nucleus_replies())",
                'monkeypatch.setattr(atom.connector, "model", nucleus)',
            ),
            notes=(_MIDDLEWARE, _CALLS_THE_MODEL) if middleware else (),
        ),
        offline=StandIn(
            label="a chat model of its own, `OfflineNucleus`,",
            imports=(
                "from langchain_core.language_models.fake_chat_models import FakeListChatModel",
            ),
            definitions=_OFFLINE_NUCLEUS,
            comment="# No model is called in this test: OfflineNucleus keeps the nucleus offline.",
            before_build=(
                'monkeypatch.setattr(atom.connector, "model", '
                'OfflineNucleus(responses=["offline"]))',
            ),
            notes=(_MIDDLEWARE,) if middleware else (),
        ),
    )
