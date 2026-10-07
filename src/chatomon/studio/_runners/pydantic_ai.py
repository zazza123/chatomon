"""Studio turns on atoms built with `PydanticAIConnector`."""

from __future__ import annotations

from typing import Any

from chatomon.connectors.base import Connector
from chatomon.studio._runners import RunOutcome
from chatomon.studio._testgen import Call, Framework, StandIn


async def run_turn(agent: Any, message: str, conversation: Any) -> RunOutcome:
    """Run one turn without streaming, as the connector requires with output orbitals.

    Args:
        agent: The native `pydantic_ai.Agent` built from the studio's copy of the atom.
        message: The user message.
        conversation: The PydanticAI messages so far, or `None` in the first turn.

    Returns:
        The reply and every message of the conversation, blocked ones already redacted.
    """
    result = await agent.run(message, message_history=list(conversation or ()))
    return RunOutcome(reply=result.output, conversation=result.all_messages())


def describe(connector: Connector[Any], agent: Any) -> dict[str, str]:
    """Name the connector and its model for the atom card.

    Args:
        connector: A `PydanticAIConnector`.
        agent: The native agent; not read.

    Returns:
        `{"connector": ..., "model": ...}`.
    """
    model = getattr(connector, "model", None)
    if isinstance(model, str):
        name = model
    elif model is None:
        name = "unknown"
    else:
        name = str(getattr(model, "model_name", type(model).__name__))
    return {"connector": type(connector).__name__, "model": name}


def notes(connector: Connector[Any]) -> list[str]:
    """Say what the developer should know before trying this atom in the studio.

    Args:
        connector: A `PydanticAIConnector`.

    Returns:
        One sentence per limit that applies to this atom.
    """
    if "deps_type" in getattr(connector, "agent_kwargs", {}):
        return [
            "this atom declares deps_type; the studio runs turns without deps, "
            "so tools and instructions that use them fail"
        ]
    return []


def saved_test(connector: Connector[Any]) -> Framework:
    """Return what a saved test needs from PydanticAI.

    Args:
        connector: A `PydanticAIConnector`; not read.

    Returns:
        The call that runs a turn, the argument that carries the conversation, the
        reply, and the test model that stands in for the nucleus offline.
    """
    return Framework(
        run="agent.run_sync",
        first=Call(),
        later=Call(arguments=("message_history=result.all_messages()",)),
        reply="result.output",
        offline=StandIn(
            label="PydanticAI's `TestModel`",
            imports=("from pydantic_ai.models.test import TestModel",),
            # No tool is called: by default the test model calls every tool the atom has.
            before_build=(
                'monkeypatch.setattr(atom.connector, "model", TestModel(call_tools=[]))',
            ),
        ),
    )
