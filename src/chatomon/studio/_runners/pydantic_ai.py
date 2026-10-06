"""Studio turns on atoms built with `PydanticAIConnector`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai.messages import ModelMessage

from chatomon.connectors.base import Connector
from chatomon.studio._testgen import Framework


@dataclass(frozen=True)
class RunOutcome:
    """The reply of a turn and the conversation history after it."""

    reply: str
    history: list[ModelMessage]


async def run_turn(agent: Any, message: str, history: Sequence[ModelMessage]) -> RunOutcome:
    """Run one turn without streaming, as the connector requires with output orbitals.

    Args:
        agent: The native `pydantic_ai.Agent` built from the studio's copy of the atom.
        message: The user message.
        history: The conversation so far, as PydanticAI messages.

    Returns:
        The reply and the full history, blocked messages already redacted.
    """
    result = await agent.run(message, message_history=list(history))
    return RunOutcome(reply=result.output, history=result.all_messages())


def describe(connector: Connector[Any]) -> dict[str, str]:
    """Name the connector and its model for the atom card.

    Args:
        connector: A `PydanticAIConnector`.

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


def saved_test() -> Framework:
    """Return what a saved test needs from PydanticAI.

    Returns:
        The test model and the line that puts it in place of the nucleus, the call
        that runs a turn, the argument that carries the conversation and the reply.
    """
    return Framework(
        test_model=("pydantic_ai.models.test", "TestModel"),
        test_model_label="PydanticAI's `TestModel`",
        # No tool is called: by default the test model calls every tool the atom has.
        offline='monkeypatch.setattr(atom.connector, "model", TestModel(call_tools=[]))',
        run="agent.run_sync",
        resume="message_history=result.all_messages()",
        reply="result.output",
    )
