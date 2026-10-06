"""Studio turns on atoms built with `MAFConnector` or `MAFFoundryConnector`."""

from __future__ import annotations

from typing import Any

from agent_framework import AgentSession, HistoryProvider, InMemoryHistoryProvider

from chatomon.connectors.base import Connector
from chatomon.connectors.maf import MAFFoundryConnector
from chatomon.studio._runners import InterruptedTurnError, RunOutcome
from chatomon.studio._testgen import Call, Framework, Offline

_INTERRUPTED = (
    "the agent stopped to wait for an approval, which the studio cannot give: "
    "the turn is left unfinished"
)
_HISTORY = "this atom keeps its history in {name}; the studio's conversations are written there"
_FOUNDRY = (
    "the nucleus of this atom is a Foundry agent; every turn that passes the input chain "
    "calls it with your credentials, and the tools hosted there run"
)
_ASYNC = "Microsoft Agent Framework is asynchronous: the test needs `pytest-asyncio` 0.24 or later."
_OWN_CLASS = (
    "`OfflineNucleus` is written on Microsoft Agent Framework's client classes: "
    "if they change, save the test again."
)
_OFFLINE_NUCLEUS = (
    "class OfflineNucleus(ChatMiddlewareLayer, BaseChatClient):",
    '    """Stands in for the nucleus: it calls no model and no tool."""',
    "",
    "    def _inner_get_response(self, *, messages, stream, options, **kwargs):",
    "        async def respond() -> ChatResponse:",
    '            return ChatResponse(messages=[Message("assistant", ["offline"])])',
    "",
    "        return respond()",
)


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


async def run_turn(agent: Any, message: str, conversation: Any) -> RunOutcome:
    """Run one turn on a copy of the conversation's session.

    The framework writes to the session it is given, also in a turn that ends with
    no reply or with a request for an approval. The copy keeps the session of the
    turn before as it was, whatever this turn does.

    Args:
        agent: The native agent built from the studio's copy of the atom.
        message: The user message.
        conversation: The `AgentSession` of the conversation, or `None` in the first turn.

    Returns:
        The reply and the session after the turn, which holds blocked messages
        already redacted.

    Raises:
        InterruptedTurnError: If a tool asked for an approval.
    """
    if conversation is None:
        session = agent.create_session()
    else:
        session = AgentSession.from_dict(conversation.to_dict())
    response = await agent.run(message, session=session)
    if response.user_input_requests:
        raise InterruptedTurnError(_INTERRUPTED)
    return RunOutcome(reply=response.text, conversation=session)


def describe(connector: Connector[Any], agent: Any) -> dict[str, str]:
    """Name the connector and its model, or the Foundry agent, for the atom card.

    Args:
        connector: A `MAFConnector` or a `MAFFoundryConnector`.
        agent: The native agent: its client knows what Foundry agent it calls.

    Returns:
        `{"connector": ..., "model": ...}`.
    """
    client = getattr(agent, "client", None)
    if isinstance(connector, MAFFoundryConnector):
        name = _text(getattr(client, "agent_name", None)) or "unknown"
        version = _text(getattr(client, "agent_version", None))
        model = f"{name} (version {version})" if version else name
    else:
        model = _text(getattr(client, "model", None)) or type(client).__name__
    return {"connector": type(connector).__name__, "model": model}


def notes(connector: Connector[Any]) -> list[str]:
    """Say what the developer should know before trying this atom in the studio.

    Args:
        connector: A `MAFConnector` or a `MAFFoundryConnector`.

    Returns:
        One sentence for a history provider of the developer's and one for Foundry.
    """
    foundry = isinstance(connector, MAFFoundryConnector)
    kwargs = getattr(connector, "foundry_kwargs" if foundry else "agent_kwargs", {})
    found = [
        _HISTORY.format(name=type(provider).__name__)
        for provider in kwargs.get("context_providers") or ()
        if isinstance(provider, HistoryProvider)
        and not isinstance(provider, InMemoryHistoryProvider)
    ]
    return [*found, _FOUNDRY] if foundry else found


def saved_test(connector: Connector[Any]) -> Framework:
    """Return what a saved test needs from Microsoft Agent Framework.

    Args:
        connector: A `MAFConnector` or a `MAFFoundryConnector`; not read.

    Returns:
        The awaited call that runs a turn on a session, the reply, and the client the
        test defines to stand in for the nucleus offline, put on the built agent.
    """
    turn = Call(arguments=("session=session",))
    return Framework(
        run="await agent.run",
        first=turn,
        later=turn,
        reply="result.text",
        setup=("session = agent.create_session()",),
        marker='@pytest.mark.asyncio(loop_scope="session")',
        notes=(_ASYNC,),
        offline=Offline(
            label="a client of its own, `OfflineNucleus`,",
            imports=(
                "from agent_framework import "
                "BaseChatClient, ChatMiddlewareLayer, ChatResponse, Message",
            ),
            definitions=_OFFLINE_NUCLEUS,
            comment="# No model is called in this test: OfflineNucleus keeps the nucleus offline.",
            after_build=("agent.client = OfflineNucleus()",),
            notes=(_OWN_CLASS,),
        ),
    )
