"""Studio turns on atoms built with `LangChainConnector`."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage

from chatomon.connectors.base import Connector
from chatomon.studio._runners import InterruptedTurnError, RunOutcome
from chatomon.studio._testgen import Call, Framework, Offline

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
_NO_TOOLS = (
    "If your code lets a message through, the test fails; on an atom with tools it fails "
    "with `NotImplementedError`, because `FakeListChatModel` cannot take them."
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


def describe(connector: Connector[Any], agent: Any) -> dict[str, str]:
    """Name the connector and its model for the atom card.

    Args:
        connector: A `LangChainConnector`.
        agent: The native agent; not read.

    Returns:
        `{"connector": ..., "model": ...}`.
    """
    model = getattr(connector, "model", None)
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

    Args:
        connector: A `LangChainConnector`; with a checkpointer every call takes a thread.

    Returns:
        The call that runs a turn around the conversation's messages, the reply, and
        the fake chat model that stands in for the nucleus offline.
    """
    thread = (_THREAD,) if _set(connector, "checkpointer") else ()
    return Framework(
        imports=(
            "from langchain_core.messages import HumanMessage",
            *(("from uuid import uuid4",) if thread else ()),
        ),
        run="agent.invoke",
        first=Call('{"messages": [HumanMessage(', ")]}", thread),
        later=Call('{"messages": [*result["messages"], HumanMessage(', ")]}", thread),
        reply='result["messages"][-1].content',
        offline=Offline(
            label="LangChain's `FakeListChatModel`",
            imports=(
                "from langchain_core.language_models.fake_chat_models import FakeListChatModel",
            ),
            before_build=(
                "monkeypatch.setattr(",
                '    atom.connector, "model", FakeListChatModel(responses=["offline"])',
                ")",
            ),
            notes=(_NO_TOOLS,),
        ),
    )
